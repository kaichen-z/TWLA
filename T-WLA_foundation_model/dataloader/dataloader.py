"""The dataset: one (episode, phase) unit -> one training sample of L consecutive control steps.

Only the last step of a sample is supervised; `sampling.py` draws L and the start. Hand-pose
geometry (anchor-camera frame, MANO-space keypoints, interpolation) comes from
`vitra_core/episodic_dataset.py`. Source frames are 30 Hz, control steps 10 Hz (every 3rd
frame), actions 80 Hz (16 per chunk, 0.2 s).
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .sampling import Window, phase_anchors, sample_window, supervisable_anchors

UPSAMPLE = 8.0 / 3.0   # 30 fps -> 80 Hz for the hand domains; see `_upsample`


def _upsample(cfg) -> float:
    """action_hz / source_fps: action steps per source frame. 8/3 at 30 fps and 80 Hz; 1.0 when
    a config sets source_fps = action_hz, so that recorded frames are used as steps."""
    return float(cfg.action.action_hz) / float(cfg.action.source_fps)


class JpegFrameStore:
    """Random access to the pre-extracted JPEG frames of a frame store.

    A frame is one `pread` into the blob of its video, addressed by episode ordinal and frame
    index within the episode. Frames are decoded with OpenCV and returned as RGB.
    """

    def __init__(self, root: str, episode_ids, frame_counts: np.ndarray):
        z = np.load(os.path.join(root, "index.npz"), allow_pickle=True)
        self.root = root
        self.video_idx = z["video_idx"]
        self.offset = z["offset"]
        self.length = z["length"]
        self.episode_start = z["episode_start"]
        self.videos = [str(v) for v in z["videos"]]

        # The store is addressed by episode ordinal, so its episodes must match the
        # dataset's in number, order and length.
        if len(self.episode_start) != len(episode_ids) + 1:
            raise ValueError(f"store has {len(self.episode_start)-1} episodes, dataset "
                             f"has {len(episode_ids)}")
        stored = np.diff(self.episode_start)
        if not np.array_equal(stored, frame_counts):
            bad = int(np.flatnonzero(stored != frame_counts)[0])
            raise ValueError(f"episode {bad} ({episode_ids[bad]}) has {frame_counts[bad]} "
                             f"frames but the store holds {stored[bad]}")
        self._fd: dict[int, int] = {}

    def complete_episodes(self) -> np.ndarray:
        """Bool per episode: True when none of its frames is stored with length 0, which is
        how the frame store records a frame it could not decode."""
        ok = self.length > 0
        return np.minimum.reduceat(ok, self.episode_start[:-1]).astype(bool)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_fd"] = {}          # file descriptors are not pickled; each worker opens its own
        return s

    def frame(self, episode_ordinal: int, local_frame: int) -> np.ndarray:
        """(H, W, 3) uint8 RGB as stored: long side at the packed size, not padded."""
        import cv2
        r = int(self.episode_start[episode_ordinal]) + int(local_frame)
        vi = int(self.video_idx[r])
        fd = self._fd.get(vi)
        if fd is None:
            # Blob file name: the video name with "/" replaced by "__".
            blob = self.videos[vi].replace("/", "__")
            fd = os.open(os.path.join(self.root, f"{blob}.bin"), os.O_RDONLY)
            self._fd[vi] = fd
        # pread takes an explicit offset, so one descriptor can be shared across threads.
        buf = os.pread(fd, int(self.length[r]), int(self.offset[r]))
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"row {r} ({self.videos[vi]}) did not decode")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@dataclass
class Unit:
    """One sampling unit: a single phase of a single episode of one source."""
    source: int
    episode: str
    phase: int
    n_frames: int


@dataclass(frozen=True)
class Source:
    """One domain's files.

    Episode ids look like `<prefix>_<video>_<ep|c30>_...`. `_video_of` extracts the video
    name, which is what a video-level split is keyed by.
    """
    name: str
    video_root: str
    annotation_file: str
    label_folder: str
    jpeg_root: str | None = None
    # "hand": MANO hand episodes (`vitra_core`); "robot": joint-space episodes (`robot_core`).
    kind: str = "hand"
    # Mixture weight: the source contributes int(weight * units) samples per epoch. The
    # normalisation statistics must be computed with the same weights.
    weight: float = 1.0


def sources_from_config(source_configs: Sequence[dict]) -> list[Source]:
    """Build the `Source` list from the config's `data.sources`. Training, statistics and
    evaluation all go through here, so they see the same mixture weights."""
    return [Source(name=s["name"], video_root=s["video_root"],
                   annotation_file=s["annotation_file"], label_folder=s["label_folder"],
                   jpeg_root=s.get("jpeg_root"), weight=float(s.get("weight", 1.0)),
                   kind=str(s.get("kind", "hand")))
            for s in source_configs]


class SequenceDataset(torch.utils.data.Dataset):
    """One item is one (source, episode, phase) unit; its window (L, start) is redrawn on every
    access. The units of all sources share one flat index, so a batch mixes domains."""

    def __init__(self, sources: Sequence[Source], cfg, *,
                 video_whitelist: dict[str, set[str]] | None = None,
                 tokenizer=None, seed: int = 0, load_images: bool = True,
                 training: bool = True, deterministic: bool = False):
        from .vitra_core.episodic_dataset import EpisodicDatasetCore

        self.sources = list(sources)
        if not self.sources:
            raise ValueError("at least one Source is required")
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"source names must be unique, got {names}")

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.load_images = load_images
        self.stride = cfg.data.stride
        self.seed = int(seed)
        # deterministic: window and paraphrase are drawn from (seed, index), so every
        # evaluation sees the same samples whatever `num_workers` is.
        self.deterministic = bool(deterministic)

        # Augmentation is applied in training only.
        self.augmentation = bool(cfg.data.geometric_augmentation and training)

        # World head (optional). With `cfg.world` unset no future frame is read and no extra
        # key is emitted.
        self.world = getattr(cfg, "world", None)
        if self.world is not None:
            # `horizon_frames` is in source frames; the loader needs anchor offsets.
            self._world_offsets = self.world.loss.anchor_offsets(self.stride)
            self._world_reserve = int(self.world.loss.reserve_anchors)
            if not load_images:
                raise ValueError(
                    "the world head needs target frames but load_images is False. It is "
                    "supervised on images; without them the loss would be computed "
                    "against nothing and `L_world` would read as a healthy zero.")
            if self.augmentation:
                raise ValueError(
                    "geometric_augmentation is not supported with the world head: the "
                    "conditioning frame would be augmented and the target frames would not. "
                    "Turn augmentation off.")
        else:
            self._world_offsets = ()
            self._world_reserve = 0
        if self.augmentation and not load_images:
            raise ValueError("geometric augmentation needs images: it transforms the image "
                             "and the geometry together, and there is nothing to transform")

        # Augmentation transforms image, intrinsics, actions, state and instruction
        # together, so with augmentation on the JPEG store is bypassed and the source video
        # is decoded.
        self.core_load_images: list[bool] = []
        self.cores = []
        self.stores: list[JpegFrameStore | None] = []
        self._ordinals: list[dict[str, int]] = []

        # clip_len=None: the videos are whole files. Text blanking and state dropout are off
        # in the core; the model applies state dropout itself.
        for src in self.sources:
            use_store = bool(src.jpeg_root and load_images and not self.augmentation)
            core_imgs = bool(load_images and not use_store)
            self.core_load_images.append(core_imgs)
            if src.kind == "robot":
                from .robot_core import RobotEpisodicCore
                if load_images and not use_store:
                    raise ValueError(f"{src.name}: robot sources need a jpeg_root store "
                                     "(no video decode path) and no geometric augmentation")
                core = RobotEpisodicCore(src.annotation_file, src.label_folder,
                                         chunk_size=cfg.action.chunk_size,
                                         action_hz=cfg.action.action_hz,
                                         source_fps=cfg.action.source_fps,
                                         image_size=cfg.vlm.image_size)
                self.cores.append(core)
                store, ordinal = None, {}
                if use_store:                       # no store when images are not loaded
                    ids = [str(e) for e in core.index_to_episode_id]
                    counts = np.bincount(core.index_frame_pair[:, 0].astype(np.int64), minlength=len(ids))
                    store = JpegFrameStore(src.jpeg_root, ids, counts)
                    ordinal = {e: i for i, e in enumerate(ids)}
                self.stores.append(store); self._ordinals.append(ordinal)
                continue
            elif src.kind != "hand":
                raise ValueError(f"{src.name}: unknown source kind {src.kind!r}")
            core = EpisodicDatasetCore(
                video_root=src.video_root,
                annotation_file=src.annotation_file,
                label_folder=src.label_folder,
                statistics_path=None,      # normalisation happens in the model
                augmentation=self.augmentation,
                # 1.0 means a horizontal flip with probability 0.5 (hands, actions, state and
                # instruction are swapped with it).
                flip_augmentation=1.0,
                set_none_ratio=0.0,
                action_type="keypoints",
                use_rel=False,
                upsample_factor=_upsample(cfg),
                target_image_height=cfg.vlm.image_size,
                clip_len=None,
                state_mask_prob=0.0,       # state dropout is done in the model
                action_past_window_size=0,
                action_future_window_size=cfg.action.chunk_size - 1,
                image_past_window_size=0,
                image_future_window_size=0,
                rel_mode="step",
                # With a JPEG store the core does not decode video; it then takes H and W
                # from the intrinsics.
                load_images=core_imgs,
            )
            self.cores.append(core)

            # Built before the units: `_build_units` asks the store which episodes are complete.
            store, ordinal = None, {}
            if use_store:
                ids = [str(e) for e in core.index_to_episode_id]
                counts = np.bincount(core.index_frame_pair[:, 0].astype(np.int64),
                                     minlength=len(ids))
                store = JpegFrameStore(src.jpeg_root, ids, counts)
                ordinal = {e: i for i, e in enumerate(ids)}
            elif src.jpeg_root is not None and load_images and self.augmentation:
                print(f"[dataloader] {src.name}: geometric augmentation is on; decoding "
                      f"source frames so image and geometry stay aligned (JPEG store "
                      f"bypassed)", flush=True)
            self.stores.append(store)
            self._ordinals.append(ordinal)

        self.units = self._build_units(video_whitelist)
        # Evaluation-only override set by `set_eval_plan`; None during training.
        self._eval_plan: list[tuple[int, int, int, int]] | None = None
        if not self.units:
            raise RuntimeError("no usable (episode, phase) units")

    # ------------------------------------------------------------------ index

    def _build_units(self, video_whitelist: dict[str, set[str]] | None) -> list[Unit]:
        """List the (episode, phase) units of every source.

        Episode lengths come from `index_frame_pair`, which has one (episode_idx, frame) row
        per frame. `video_whitelist` maps a source name to the entries of its split file.
        """
        units: list[Unit] = []
        for si, src in enumerate(self.sources):
            core, store = self.cores[si], self.stores[si]
            ids = core.index_to_episode_id
            lengths = np.bincount(core.index_frame_pair[:, 0].astype(np.int64),
                                  minlength=len(ids))

            # Episodes with an undecodable frame are dropped and counted.
            complete = store.complete_episodes() if store is not None else None
            keep = None if video_whitelist is None else video_whitelist.get(src.name)
            if video_whitelist is not None and keep is None:
                raise KeyError(f"no split given for source {src.name!r}; the whitelist "
                               f"has {sorted(video_whitelist)}")
            # A split file lists either video names or episode ids. It is read as episode ids
            # when any entry equals an id of this source.
            by_episode = keep is not None and not keep.isdisjoint(str(e) for e in ids)

            n0, dropped = len(units), 0
            for i, eid in enumerate(ids):
                if complete is not None and not complete[i]:
                    dropped += 1
                    continue
                if keep is not None:
                    if str(eid) not in keep if by_episode else _video_of(str(eid)) not in keep:
                        continue
                T = int(lengths[i])
                for ph in range(self.stride):
                    # A phase is a unit if at least one of its anchors can be supervised
                    # (L = 1 is a valid sample: an empty bank, as at the start of an episode).
                    _anc = phase_anchors(T, ph, self.stride)
                    if self._n_supervisable(_anc, T) >= 1:
                        units.append(Unit(si, str(eid), ph, T))
            msg = (f"[dataloader] {src.name}: {len(units) - n0:,} units"
                   + (" (split names episodes)" if keep is not None and by_episode else ""))
            if dropped:
                msg += f" ({dropped:,} episodes dropped for undecodable frames)"
            print(msg, flush=True)
        return units

    def __len__(self) -> int:
        return len(self.units) if self._eval_plan is None else len(self._eval_plan)

    # ------------------------------------------------------------------ item

    def set_eval_plan(self, plan) -> None:
        """Replace sampling by an explicit list of windows (evaluation only; None restores it).

        `plan[i] = (unit_index, start, length, pair_seed)`. The per-sample generator is seeded
        from `pair_seed`, so one anchor evaluated at several depths draws the same paraphrase.
        """
        if plan is not None and not self.deterministic:
            raise RuntimeError("an eval plan needs deterministic=True")
        self._eval_plan = plan

    def __getitem__(self, i: int) -> dict:
        plan = self._eval_plan
        if plan is not None:
            ui, _p_start, _p_len, _p_seed = plan[i]
            u = self.units[ui]
        else:
            u = self.units[i]
        cfg = self.cfg
        core, core_imgs = self.cores[u.source], self.core_load_images[u.source]
        anchors = phase_anchors(u.n_frames, u.phase, self.stride)

        # Anchors whose chunk runs past the episode are kept, with the tail masked; only an
        # anchor on the final frame cannot be supervised.
        n_sup = self._n_supervisable(anchors, u.n_frames)
        if plan is not None:
            torch_rng = self._generator(_p_seed)
            win = Window(start=_p_start, length=_p_len)
            if win.supervised >= n_sup:
                raise ValueError(
                    f"eval plan asks for anchor {win.supervised} but unit {ui} "
                    f"({u.episode} phase {u.phase}) has only {n_sup} supervisable anchors")
        else:
            torch_rng = self._generator(i)
            win = sample_window(n_sup, cfg.train.max_seq_len, cfg.train.length_alpha,
                                generator=torch_rng)

        # The core interpolates the phase once and each step slices it. `anchor_step` is the
        # anchor's index within the phase.
        steps = []
        for k in range(win.length):
            d = core.get_item_frame(
                u.episode, anchors[win.start + k],
                action_past_window_size=0,
                action_future_window_size=cfg.action.chunk_size - 1,
                image_past_window_size=0, image_future_window_size=0,
                rel_mode="step", load_images=core_imgs,
                phase=u.phase, stride=self.stride, anchor_step=win.start + k)
            steps.append(d)

        # Pad to 192 / 212 without normalising (the model normalises). Called once per step:
        # `transform_trajectory` rewrites the dict in place.
        packed = [core.transform_trajectory(d, normalization=False) for d in steps]
        sup = packed[-1]

        # A supervised step with an all-false action mask would contribute no loss at all.
        if not np.asarray(sup["action_mask"], bool).any():
            raise RuntimeError(
                f"episode {u.episode} phase {u.phase} anchor {anchors[win.supervised]}: "
                f"the supervised step has an all-false action_mask, so this sample would "
                f"train on nothing. The window landed entirely inside a stretch where the "
                f"annotated hand is untracked -- check `kept_frames` for this episode.")
        state = np.stack([np.asarray(p["current_state"]) for p in packed])
        smask = np.stack([np.asarray(p["current_state_mask"]) for p in packed])

        out = {
            "length": win.length,
            "state": state.astype(np.float32),
            "state_mask": smask.astype(bool),
            "fov": np.stack([np.asarray(d["fov"], np.float32) for d in steps]),
            "actions": np.asarray(sup["action_list"], np.float32),
            "action_mask": np.asarray(sup["action_mask"], bool),
            "instruction": self._pick_text(steps[-1]["instruction"], torch_rng),
            "episode": u.episode,
            "phase": u.phase,
            "source": self.sources[u.source].name,
            # Intrinsics of the supervised step as returned by the core. Not used in training.
            "intrinsics": np.asarray(sup["intrinsics"], np.float32),
        }
        if self.load_images:
            store = self.stores[u.source]
            if store is not None:
                ordinal = self._ordinals[u.source][u.episode]
                frames = [store.frame(ordinal, anchors[win.start + k])
                          for k in range(win.length)]
            else:
                frames = [np.asarray(d["image_list"])[0] for d in steps]
            out["images"] = self._to_square(np.stack(frames))

            # World-head targets: the frames at the anchors `_world_offsets` after the
            # supervised one, in the same phase.
            if self._world_offsets:
                sup, future, fvalid = win.supervised, [], []
                for off in self._world_offsets:
                    a = sup + off
                    ok = a < len(anchors)
                    # Past the end of the phase a finite placeholder (the last frame) is
                    # used and flagged invalid. Only the JPEG store can serve a later frame;
                    # without one the supervised step's own frame is returned.
                    if ok:
                        img = (store.frame(ordinal, anchors[a]) if store is not None
                               else np.asarray(steps[-1]["image_list"])[0])
                    else:
                        img = frames[-1]
                    future.append(img)
                    fvalid.append(ok)
                out["future_images"] = self._to_square(np.stack(future))
                out["future_valid"] = np.asarray(fvalid, bool)
        return out

    # ------------------------------------------------------------------ helpers

    def _n_supervisable(self, anchors, n_frames: int) -> int:
        """Number of a phase's anchors that may be the supervised step.

        `supervisable_anchors` gives the action head's limit. The world head also needs
        `reserve_anchors` later anchors of the same phase as targets, so those are held back.
        `_build_units`, `__getitem__` and `planned_window` must all use this count.
        """
        n = supervisable_anchors(anchors, n_frames)
        if self._world_reserve:
            n = min(n, len(anchors) - self._world_reserve)
        return n

    def _generator(self, index: int):
        """A generator seeded from (seed, index) in deterministic mode, else None (global RNG)."""
        if not self.deterministic:
            return None
        return torch.Generator().manual_seed((self.seed + int(index)) % (2 ** 63 - 1))

    def planned_window(self, index: int):
        """The window unit `index` will draw, computed without loading data (deterministic
        mode only). Evaluation uses it to group units by bank depth."""
        if not self.deterministic:
            raise RuntimeError("planned_window needs deterministic=True")
        u = self.units[index]
        anchors = phase_anchors(u.n_frames, u.phase, self.stride)
        return sample_window(self._n_supervisable(anchors, u.n_frames),
                             self.cfg.train.max_seq_len, self.cfg.train.length_alpha,
                             generator=self._generator(index))

    def _pick_text(self, instruction, generator=None) -> str:
        """Return the instruction. If it is a pool of paraphrases, draw one, or take the first
        when `random_paraphrase` is off."""
        if isinstance(instruction, (list, tuple)):
            pool = [str(x) for x in instruction if x]
            if not pool:
                return ""
            if not self.cfg.data.random_paraphrase:
                return str(pool[0]).strip()
            # torch's global RNG is reseeded per DataLoader worker, so workers draw different
            # paraphrases. `generator`, when given, is used instead.
            return pool[int(torch.randint(len(pool), (1,), generator=generator).item())].strip()
        return str(instruction or "").strip()

    def _to_square(self, imgs: np.ndarray) -> np.ndarray:
        """(L, H, W, 3) uint8 -> (L, S, S, 3): resize the long side to S, keeping the aspect
        ratio, and pad on the left and top, so that the picture sits in the bottom-right
        corner (the layout of `vlm.resize_with_pad`)."""
        import cv2
        n, h, w = imgs.shape[:3]
        S = self.cfg.vlm.image_size
        if (h, w) == (S, S):
            return imgs
        r = min(S / w, S / h)
        nh, nw = max(1, int(h * r)), max(1, int(w * r))
        out = np.zeros((n, S, S, 3), np.uint8)
        for i in range(n):
            # Frames from the JPEG store already have their long side at S: no resize.
            out[i, S - nh:, S - nw:] = imgs[i] if (nh, nw) == (h, w) else cv2.resize(
                imgs[i], (nw, nh), interpolation=cv2.INTER_AREA)
        return out


def to_observations(batch: dict, device, observation_cls, dtype=torch.float32):
    """Collated batch -> (steps, step_valid, actions, action_mask), the arguments of a model's `loss`.

    `steps` is a list of L `observation_cls` instances, one per control step. Images go from
    uint8 [B, L, H, W, 3] to float [B, 1, 3, H, W] in [-1, 1] per step.
    """
    Observation = observation_cls
    B, L = batch["step_valid"].shape
    steps = []
    for t in range(L):
        img = batch["images"][:, t].to(device)                      # [B,H,W,3] uint8
        img = img.permute(0, 3, 1, 2).to(dtype).div_(255).mul_(2).sub_(1)
        steps.append(Observation(
            images=img[:, None],                                    # single camera
            lang_ids=batch["lang_ids"].to(device),
            lang_mask=batch["lang_mask"].to(device),
            state=batch["state"][:, t].to(device, dtype),
            state_mask=batch["state_mask"][:, t].to(device),
            fov=batch["fov"][:, t].to(device, dtype)[:, None],       # one FoV per camera
        ))
    return (steps,
            batch["step_valid"].to(device),
            batch["actions"].to(device, dtype),
            batch["action_mask"].to(device))


def _video_of(episode_id: str) -> str:
    """`epic_kitchens_P01_01_ep_000000` -> `P01_01`."""
    parts = episode_id.split("_")
    for i, p in enumerate(parts):
        if p in ("ep", "c30"):
            return "_".join(parts[i - 2:i])
    return "_".join(parts[2:4]) if len(parts) > 3 else episode_id


class MixtureSampler(torch.utils.data.Sampler):
    """Index stream that honours each source's mixture weight.

    Per epoch, source i contributes int(weight_i * n_i) indices from fresh permutations of
    its units (tiled when the weight exceeds 1). With all weights 1 this is a plain shuffle.
    For DDP the stream is strided by rank; with `drop_last` it is first truncated to a
    multiple of `world`, so that every rank runs the same number of iterations.
    """

    def __init__(self, dataset, *, shuffle: bool = True, seed: int = 0,
                 rank: int = 0, world: int = 1, drop_last: bool = True):
        self.by_source: dict[int, list[int]] = defaultdict(list)
        for i, u in enumerate(dataset.units):
            self.by_source[u.source].append(i)
        self.weights = [s.weight for s in dataset.sources]
        for si, w in enumerate(self.weights):
            if w <= 0:
                raise ValueError(f"source {dataset.sources[si].name!r} has weight {w}; "
                                 f"drop it from the config instead of weighting it to zero")
        self.shuffle, self.seed = shuffle, int(seed)
        self.rank, self.world, self.drop_last = rank, world, drop_last
        self.epoch = 0
        self._per_epoch = sum(int(self.weights[si] * len(ix))
                              for si, ix in self.by_source.items())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self._per_epoch // self.world
        # length of range(rank, N, world)
        return max(0, -(-(self._per_epoch - self.rank) // self.world))

    def __iter__(self):
        out: list[int] = []
        for si, ix in sorted(self.by_source.items()):
            take = int(self.weights[si] * len(ix))
            g = torch.Generator().manual_seed(
                (self.seed + self.epoch) * 1_000_003 + si)
            perm: list[int] = []
            while len(perm) < take:
                order = (torch.randperm(len(ix), generator=g).tolist() if self.shuffle
                         else list(range(len(ix))))
                perm += [ix[j] for j in order]
            out += perm[:take]
        if self.shuffle:
            g = torch.Generator().manual_seed((self.seed + self.epoch) * 7_919)
            out = [out[j] for j in torch.randperm(len(out), generator=g).tolist()]
        if self.drop_last:
            out = out[:(len(out) // self.world) * self.world]
        out = out[self.rank::self.world]
        return iter(out)


def load_video_list(path: str) -> set[str]:
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip()}


def make_collate(cfg, tokenizer=None):
    """Collate that pads sequences on the left, so the supervised step is always the last one."""
    S, A, C = cfg.action.state_dim, cfg.action.dim, cfg.action.chunk_size
    Lmax = cfg.vlm.max_language_tokens

    def collate(batch: list[dict]) -> dict:
        L = max(b["length"] for b in batch)
        B = len(batch)
        valid = torch.zeros(B, L, dtype=torch.bool)
        state = torch.zeros(B, L, S)
        smask = torch.zeros(B, L, S, dtype=torch.bool)
        fov = torch.zeros(B, L, 2)
        imgs = None
        if batch[0].get("images") is not None:
            h, w = batch[0]["images"].shape[1:3]
            imgs = torch.zeros(B, L, h, w, 3, dtype=torch.uint8)

        for i, b in enumerate(batch):
            n = b["length"]
            valid[i, L - n:] = True
            state[i, L - n:] = torch.from_numpy(b["state"])
            smask[i, L - n:] = torch.from_numpy(b["state_mask"])
            fov[i, L - n:] = torch.from_numpy(b["fov"])
            if imgs is not None:
                imgs[i, L - n:] = torch.from_numpy(b["images"])

        # World-head targets: a fixed number of horizons per item; `future_valid` marks the
        # real ones.
        fut = fmask = None
        if batch[0].get("future_images") is not None:
            fut = torch.from_numpy(np.stack([b["future_images"] for b in batch]))
            fmask = torch.from_numpy(np.stack([b["future_valid"] for b in batch]))

        out = {
            "step_valid": valid, "state": state, "state_mask": smask, "fov": fov,
            "images": imgs,
            "future_images": fut, "future_valid": fmask,
            "actions": torch.from_numpy(np.stack([b["actions"] for b in batch])),
            "action_mask": torch.from_numpy(np.stack([b["action_mask"] for b in batch])),
            "instruction": [b["instruction"] for b in batch],
            "episode": [b["episode"] for b in batch],
            "source": [b["source"] for b in batch],
            "length": torch.tensor([b["length"] for b in batch]),
        }
        if tokenizer is not None:
            # Padded to a fixed length, not to the batch maximum, so the language prefix
            # always has the same shape.
            tok = tokenizer(out["instruction"], padding="max_length", truncation=True,
                            max_length=Lmax, return_tensors="pt")
            out["lang_ids"] = tok["input_ids"]
            out["lang_mask"] = tok["attention_mask"].bool()
        return out

    return collate
