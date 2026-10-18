"""The dataloader: one (episode, phase) unit -> one sample of L consecutive control steps.

The last step of a sample is the supervised one. Source frames are at 30 Hz, control steps
at 10 Hz (every 3rd frame), actions at 80 Hz (16 per chunk, 0.2 s). `sampling.py` draws L
and the start; the geometry is in `vitra_core/episodic_dataset.py`.
A copy of the main model's `T-WLA_foundation_model/dataloader/dataloader.py`, fixed to the
keypoints action layout.
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

UPSAMPLE = 8.0 / 3.0       # 30 Hz source frames -> 80 Hz actions


def _upsample(cfg) -> float:
    """action_hz / source_fps: action steps per source frame, 8/3 for 30 fps data."""
    return float(cfg.action.action_hz) / float(cfg.action.source_fps)


class JpegFrameStore:
    """Random access to the pre-extracted JPEG frames of a frame store.

    Each video has one blob file; a frame is one `pread` from it. Frames are returned as RGB.
    """

    def __init__(self, root: str, episode_ids, frame_counts: np.ndarray):
        z = np.load(os.path.join(root, "index.npz"), allow_pickle=True)
        self.root = root
        self.video_idx = z["video_idx"]
        self.offset = z["offset"]
        self.length = z["length"]
        self.episode_start = z["episode_start"]
        self.videos = [str(v) for v in z["videos"]]

        # The store is addressed by episode ordinal, so it must hold the dataset's episodes
        # in the dataset's order: check the count and the length of every episode.
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
        """Bool per episode: True when all of its frames decoded. A frame that could not be
        decoded is stored with length 0."""
        ok = self.length > 0
        return np.minimum.reduceat(ok, self.episode_start[:-1]).astype(bool)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_fd"] = {}          # each worker opens its own descriptors
        return s

    def frame(self, episode_ordinal: int, local_frame: int) -> np.ndarray:
        """(H, W, 3) uint8 RGB, long side already at `image_size`, not padded."""
        import cv2
        r = int(self.episode_start[episode_ordinal]) + int(local_frame)
        vi = int(self.video_idx[r])
        fd = self._fd.get(vi)
        if fd is None:
            # a video name may contain directories; the blob name replaces "/" by "__"
            blob = self.videos[vi].replace("/", "__")
            fd = os.open(os.path.join(self.root, f"{blob}.bin"), os.O_RDONLY)
            self._fd[vi] = fd
        # `pread` does not use the descriptor's offset, so the descriptor can be shared
        buf = os.pread(fd, int(self.length[r]), int(self.offset[r]))
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"row {r} ({self.videos[vi]}) did not decode")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # cv2 decodes to BGR


@dataclass
class Unit:
    """One sampling unit: a single phase of a single episode of one source."""
    source: int
    episode: str
    phase: int
    n_frames: int


@dataclass(frozen=True)
class Source:
    """One dataset's files. Episode ids have the form `<prefix>_<video>_<ep|c30>_...`,
    from which `_video_of` takes the video."""
    name: str
    video_root: str
    annotation_file: str
    label_folder: str
    jpeg_root: str | None = None
    # Mixture weight: the source contributes `int(weight * units)` samples per epoch.
    # The normalisation statistics are weighted by the same number.
    weight: float = 1.0


def sources_from_config(source_configs: Sequence[dict]) -> list[Source]:
    """The `data.sources` entries of a config -> a list of `Source`."""
    return [Source(name=s["name"], video_root=s["video_root"],
                   annotation_file=s["annotation_file"], label_folder=s["label_folder"],
                   jpeg_root=s.get("jpeg_root"), weight=float(s.get("weight", 1.0)))
            for s in source_configs]


class SequenceDataset(torch.utils.data.Dataset):
    """(source, episode, phase) -> one sample of L steps.

    The units of all sources form one flat list; each source keeps its own core, JPEG
    store and split. In training the window (L, start) is drawn anew on every access.
    """

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
        # deterministic: the window and the paraphrase are drawn from (seed, index), so
        # every evaluation sees the same samples
        self.deterministic = bool(deterministic)

        # training only
        self.augmentation = bool(cfg.data.geometric_augmentation and training)
        if self.augmentation and not load_images:
            raise ValueError("geometric augmentation needs images: it transforms the image "
                             "and the geometry together, and there is nothing to transform")

        # The augmentation transforms image and geometry together, so with it on the
        # source frame is decoded and the JPEG store is not used.
        self.core_load_images: list[bool] = []
        self.cores = []
        self.stores: list[JpegFrameStore | None] = []
        self._ordinals: list[dict[str, int]] = []

        for src in self.sources:
            use_store = bool(src.jpeg_root and load_images and not self.augmentation)
            core_imgs = bool(load_images and not use_store)
            self.core_load_images.append(core_imgs)
            core = EpisodicDatasetCore(
                video_root=src.video_root,
                annotation_file=src.annotation_file,
                label_folder=src.label_folder,
                statistics_path=None,      # the model normalises
                augmentation=self.augmentation,
                flip_augmentation=1.0,     # with augmentation on: flip with probability 0.5
                set_none_ratio=0.0,        # the text of a hand is never blanked
                action_type="keypoints",   # the layout of this baseline
                # also emit the joint-angle layout of the same window (`*_alt`); off here
                dual_layout=getattr(cfg.action, "dual_layout", False),
                use_rel=False,
                upsample_factor=_upsample(cfg),
                target_image_height=cfg.vlm.image_size,
                clip_len=None,             # the videos are whole files
                state_mask_prob=0.0,       # no state dropout in the loader
                action_past_window_size=0,
                action_future_window_size=cfg.action.chunk_size - 1,
                image_past_window_size=0,
                image_future_window_size=0,
                rel_mode="step",
                # with a JPEG store the core decodes no video and takes the image size
                # from the intrinsics
                load_images=core_imgs,
            )
            self.cores.append(core)

            # built before the units: `_build_units` asks the store which episodes are complete
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
        # explicit list of windows for evaluation, see `set_eval_plan`; None in training
        self._eval_plan: list[tuple[int, int, int, int]] | None = None
        if not self.units:
            raise RuntimeError("no usable (episode, phase) units")

    # ------------------------------------------------------------------ index

    def _build_units(self, video_whitelist: dict[str, set[str]] | None) -> list[Unit]:
        """List the (episode, phase) units of every source.

        Episode lengths are counted from `index_frame_pair`, (episode index, frame) per
        frame, so no annotation file is read. `video_whitelist` maps a source name to the
        names in its split file.
        """
        units: list[Unit] = []
        for si, src in enumerate(self.sources):
            core, store = self.cores[si], self.stores[si]
            ids = core.index_to_episode_id
            lengths = np.bincount(core.index_frame_pair[:, 0].astype(np.int64),
                                  minlength=len(ids))

            # episodes with undecodable frames are dropped and counted in the log line
            complete = store.complete_episodes() if store is not None else None
            keep = None if video_whitelist is None else video_whitelist.get(src.name)
            if video_whitelist is not None and keep is None:
                raise KeyError(f"no split given for source {src.name!r}; the whitelist "
                               f"has {sorted(video_whitelist)}")
            # A split file lists video names or episode ids. It is read as episode ids
            # when any of its entries is one.
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
                    # one supervisable anchor is enough: a sample of length 1
                    if supervisable_anchors(phase_anchors(T, ph, self.stride), T) >= 1:
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
        """Replace the sampling with an explicit list of windows, for evaluation.

        `plan[i] = (unit_index, start, length, pair_seed)`. `__len__` follows the plan and
        the generator of a sample is seeded from `pair_seed`, so one (unit, anchor) draws
        the same paraphrase at every depth. `None` restores the sampling.
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

        # A chunk that runs past the end of the episode is kept: its tail repeats the last
        # pose and is off in `action_mask`. Only an anchor on the final frame is excluded.
        n_sup = supervisable_anchors(anchors, u.n_frames)
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

        # The phase is interpolated once and every step slices it; `anchor_step` is the
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

        # Pad to 192 / 212 without normalising; the model normalises. Called once per
        # step, because `transform_trajectory` rewrites its argument in place.
        packed = [core.transform_trajectory(d, normalization=False) for d in steps]
        sup = packed[-1]

        # a supervised step with an empty action mask would give a loss of zero and no gradient
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
            # camera intrinsics of the supervised step; not collated
            "intrinsics": np.asarray(sup["intrinsics"], np.float32),
        }
        if "action_list_alt" in sup:
            # the joint-angle layout of the same window, present with `dual_layout`
            out["actions_alt"] = np.asarray(sup["action_list_alt"], np.float32)
            out["action_mask_alt"] = np.asarray(sup["action_mask_alt"], bool)
            out["state_alt"] = np.stack(
                [np.asarray(p["current_state_alt"]) for p in packed]).astype(np.float32)
        if self.load_images:
            store = self.stores[u.source]
            if store is not None:
                ordinal = self._ordinals[u.source][u.episode]
                frames = [store.frame(ordinal, anchors[win.start + k])
                          for k in range(win.length)]
            else:
                frames = [np.asarray(d["image_list"])[0] for d in steps]
            out["images"] = self._to_square(np.stack(frames))
        return out

    # ------------------------------------------------------------------ helpers

    def _generator(self, index: int):
        """A generator seeded from (seed, index) in deterministic mode, None otherwise."""
        if not self.deterministic:
            return None
        return torch.Generator().manual_seed((self.seed + int(index)) % (2 ** 63 - 1))

    def planned_window(self, index: int):
        """The window that unit `index` will draw, without reading data. Deterministic
        mode only, where the draw depends on (seed, index) alone."""
        if not self.deterministic:
            raise RuntimeError("planned_window needs deterministic=True")
        u = self.units[index]
        anchors = phase_anchors(u.n_frames, u.phase, self.stride)
        return sample_window(supervisable_anchors(anchors, u.n_frames),
                             self.cfg.train.max_seq_len, self.cfg.train.length_alpha,
                             generator=self._generator(index))

    def _pick_text(self, instruction, generator=None) -> str:
        """`instruction` is a string or a list of paraphrases. One is drawn, or the first
        taken when `data.random_paraphrase` is off."""
        if isinstance(instruction, (list, tuple)):
            pool = [str(x) for x in instruction if x]
            if not pool:
                return ""
            if not self.cfg.data.random_paraphrase:
                return str(pool[0]).strip()
            # torch's RNG is reseeded in every DataLoader worker; a `random.Random` made
            # in __init__ would be copied to all of them in the same state
            return pool[int(torch.randint(len(pool), (1,), generator=generator).item())].strip()
        return str(instruction or "").strip()

    def _to_square(self, imgs: np.ndarray) -> np.ndarray:
        """(L, H, W, 3) uint8 -> (L, S, S, 3): the long side is resized to S with the
        aspect kept, and the image is padded on the left and top, as SmolVLA does."""
        import cv2
        n, h, w = imgs.shape[:3]
        S = self.cfg.vlm.image_size
        if (h, w) == (S, S):
            return imgs
        r = min(S / w, S / h)
        nh, nw = max(1, int(h * r)), max(1, int(w * r))
        out = np.zeros((n, S, S, 3), np.uint8)
        for i in range(n):
            # frames from the JPEG store already have their long side at S
            out[i, S - nh:, S - nw:] = imgs[i] if (nh, nw) == (h, w) else cv2.resize(
                imgs[i], (nw, nh), interpolation=cv2.INTER_AREA)
        return out


def to_observations(batch: dict, device, observation_cls, dtype=torch.float32):
    """Collated batch -> (steps, step_valid, actions, action_mask), the arguments of a
    model's `loss`.

    `steps` is a list of L objects of the model's `Observation` class, `observation_cls`.
    Images go from uint8 [B, L, H, W, 3] to float [B, 1, 3, H, W] in [-1, 1] per step.
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
    """Index stream that follows the mixture weight of each source.

    Per epoch, source i contributes `int(weight_i * n_i)` indices from permutations of its
    own units. For DDP the stream is strided by rank; with `drop_last` it is first cut to a
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
    """Collate that pads the step sequences on the left, so that the last step is the
    supervised one for every item."""
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

        alt = torch.zeros(B, L, S) if batch[0].get("state_alt") is not None else None

        for i, b in enumerate(batch):
            n = b["length"]
            valid[i, L - n:] = True
            state[i, L - n:] = torch.from_numpy(b["state"])
            if alt is not None:
                alt[i, L - n:] = torch.from_numpy(b["state_alt"])
            smask[i, L - n:] = torch.from_numpy(b["state_mask"])
            fov[i, L - n:] = torch.from_numpy(b["fov"])
            if imgs is not None:
                imgs[i, L - n:] = torch.from_numpy(b["images"])

        out = {
            "step_valid": valid, "state": state, "state_mask": smask, "fov": fov,
            "images": imgs,
            "actions": torch.from_numpy(np.stack([b["actions"] for b in batch])),
            "action_mask": torch.from_numpy(np.stack([b["action_mask"] for b in batch])),
            "instruction": [b["instruction"] for b in batch],
            "episode": [b["episode"] for b in batch],
            "source": [b["source"] for b in batch],
            "length": torch.tensor([b["length"] for b in batch]),
        }
        if alt is not None:                                   # the joint-angle layout
            out["state_alt"] = alt
            out["actions_alt"] = torch.from_numpy(
                np.stack([b["actions_alt"] for b in batch]))
            out["action_mask_alt"] = torch.from_numpy(
                np.stack([b["action_mask_alt"] for b in batch]))
        if tokenizer is not None:
            # padded to a fixed length, so the language tokens have the same shape in every batch
            tok = tokenizer(out["instruction"], padding="max_length", truncation=True,
                            max_length=Lmax, return_tensors="pt")
            out["lang_ids"] = tok["input_ids"]
            out["lang_mask"] = tok["attention_mask"].bool()
        return out

    return collate
