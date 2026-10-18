#!/usr/bin/env python
"""Training entry point. Runs as a single process or as DDP under torchrun:

    python train.py --config configs/train.json
    torchrun --nproc_per_node=4 train.py --config configs/train.json

The json selects the model variant and supplies the data paths and training hyperparameters.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GROUPS = ("left_translation", "left_rotation", "left_articulation",
          "right_translation", "right_rotation", "right_articulation")


# --------------------------------------------------------------------- setup

def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*a) -> None:
    if is_main():
        print(*a, flush=True)



def validate_run_config(C: dict) -> None:
    """Check the keys this script reads itself (batch size, schedule, data sources).
    The model config's own `validate()` does not cover them."""
    required = {"batch_size", "num_workers", "max_steps", "log_every", "val_every",
                "save_every", "output_dir", "data", "variant"}
    missing = sorted(required - set(C))
    if missing:
        raise KeyError(f"run config is missing {missing}")

    if str(C["variant"]).strip().lower() not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {C['variant']!r}")
    if not isinstance(C["output_dir"], str) or not C["output_dir"].strip():
        raise ValueError(f"output_dir must be a non-empty string, got {C['output_dir']!r}")

    # bool is a subclass of int, so a json `true` would otherwise pass as 1
    def _int(name, value, lo):
        if isinstance(value, bool) or not isinstance(value, int) or value < lo:
            raise ValueError(f"{name} must be an integer >= {lo}, got {value!r}")

    for name in ("batch_size", "max_steps", "log_every", "val_every", "save_every"):
        _int(name, C[name], 1)
    _int("num_workers", C["num_workers"], 0)
    if "prefetch_factor" in C:
        _int("prefetch_factor", C["prefetch_factor"], 1)
    if "val_max_batches" in C:
        _int("val_max_batches", C["val_max_batches"], 1)
    if "seed" in C and (isinstance(C["seed"], bool) or not isinstance(C["seed"], int)):
        raise ValueError(f"seed must be an integer, got {C['seed']!r}")
    if C["max_steps"] < C["val_every"]:
        raise ValueError(f"val_every={C['val_every']} exceeds max_steps={C['max_steps']}, "
                         f"so validation would never run.")

    D = C["data"]
    if not isinstance(D, dict):
        raise TypeError(f"data must be a JSON object, got {type(D).__name__}")
    for key in ("root", "sources", "statistics"):
        if key not in D:
            raise KeyError(f"data is missing {key!r}")
    srcs = D["sources"]
    if not isinstance(srcs, list) or not srcs:
        raise ValueError("data.sources must be a non-empty list, one entry per domain")

    seen: set[str] = set()
    for i, src in enumerate(srcs):
        where = f"data.sources[{i}]"
        if not isinstance(src, dict):
            raise TypeError(f"{where} must be a JSON object, got {type(src).__name__}")
        for key in ("name", "video_root", "annotation_file", "label_folder", "split_dir"):
            if key not in src:
                raise KeyError(f"{where} is missing {key!r}")
            if not isinstance(src[key], str) or not src[key].strip():
                raise ValueError(f"{where}.{key} must be a non-empty string, "
                                 f"got {src[key]!r}")
        if src.get("jpeg_root") is not None and not isinstance(src["jpeg_root"], str):
            raise ValueError(f"{where}.jpeg_root must be a string or null, "
                             f"got {src['jpeg_root']!r}")
        # the name keys the split lists and the per-source metrics, so it must be unique
        if src["name"] in seen:
            raise ValueError(f"{where}.name {src['name']!r} is already used by "
                             f"an earlier source")
        seen.add(src["name"])
        w = src.get("weight", 1.0)
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not w > 0:
            raise ValueError(f"{where}.weight must be a number > 0, got {w!r}. "
                             f"Drop the source instead of weighting it to zero.")
        unknown = set(src) - {"name", "video_root", "annotation_file", "label_folder",
                              "split_dir", "jpeg_root", "weight",
                              # the split the statistics were measured on, if not split_dir
                              "statistics_split_dir", "kind"}
        if unknown:
            raise KeyError(f"{where} has unknown keys {sorted(unknown)}; a misspelled key "
                           f"would otherwise be dropped and the default used instead")


def resolve_paths(C: dict) -> dict:
    """Expand `${VAR}` and `~`, then join the relative paths of `data` onto `data.root`.

    Absolute entries are kept as they are. `output_dir` is expanded but not joined to the root.
    """
    def expand(v):
        return os.path.expanduser(os.path.expandvars(v)) if isinstance(v, str) else v

    C = dict(C)
    C["output_dir"] = expand(C["output_dir"])
    D = dict(C.get("data", {}))
    root = expand(D.pop("root", ""))
    if "${" in root or "${" in C["output_dir"]:
        raise KeyError(
            f"unset environment variable in a path: root={root!r}, "
            f"output_dir={C['output_dir']!r}. Set it (e.g. `export VLA_DATA=...`) or "
            f"write an absolute path into the config.")
    def rooted(v):
        if v is None:
            return None
        v = expand(v)
        return v if os.path.isabs(v) else os.path.join(root, v)

    # the keys of a source that hold paths; `name`, `weight` and the rest are left alone
    SOURCE_PATHS = {"video_root", "annotation_file", "label_folder", "jpeg_root",
                    "split_dir", "statistics_split_dir"}
    srcs = D.pop("sources", None)
    for k, v in D.items():
        D[k] = rooted(v)
    if srcs is not None:
        D["sources"] = [{k: (rooted(v) if k in SOURCE_PATHS else v) for k, v in s.items()}
                        for s in srcs]
    C["data"] = D
    return C


from config_validation import REQUIRED, set_parameter


def apply_overrides(cfg, overrides: dict) -> None:
    """Apply `{"optim": {"lr": 3e-5}}` as `cfg.optim.lr = 3e-5`. Unknown sections and fields raise."""
    for section, values in overrides.items():
        if not hasattr(cfg, section):
            raise KeyError(f"config has no section '{section}'")
        obj = getattr(cfg, section)
        if obj is None:
            # an optional section that is absent by default (`world`): assign the dict
            # whole, and `cfg.validate()` builds the dataclass from it
            setattr(cfg, section, values)
            continue
        for k, v in values.items():
            if not hasattr(obj, k):
                raise KeyError(f"'{section}' has no field '{k}'")
            cur = getattr(obj, k)
            if cur is REQUIRED:
                set_parameter(obj, k, v)     # no default: cast by the declared type
                continue
            # otherwise cast to the type of the default; a None default gives no type
            setattr(obj, k, v if (cur is None or isinstance(v, (list, dict)))
                    else type(cur)(v))


# ----------------------------------------------------------------- pre-flight

def _timed_loader(loader, acc: list):
    """Yield from `loader`, adding to `acc[0]` the seconds spent waiting for each batch."""
    import time as _t
    it = iter(loader)
    while True:
        t0 = _t.perf_counter()
        try:
            b = next(it)
        except StopIteration:
            return
        acc[0] += _t.perf_counter() - t0
        yield b


def _mem_probe() -> str:
    """One short string for the step log: /dev/shm used and total, this process's RSS,
    and the utilisation of each GPU. Never raises."""
    try:
        import shutil
        t, u = shutil.disk_usage("/dev/shm")[0], shutil.disk_usage("/dev/shm")[1]
        rss = 0
        with open("/proc/self/status") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    rss = int(ln.split()[1]) * 1024
                    break
        # one instantaneous utilisation reading per GPU
        util = ""
        try:
            import subprocess
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            vals = [x.strip() for x in r.stdout.split() if x.strip().isdigit()]
            if vals:
                util = " gpu " + "/".join(vals) + "%"
        except Exception:
            pass
        return f"shm {u/2**30:.0f}/{t/2**30:.0f}G rss {rss/2**30:.1f}G{util}"
    except Exception:
        return "mem ?"


def preflight(model, dataset, collate, cfg, obs_cls, n_batches: int = 6,
              batch_size: int = 8, seed: int = 0) -> None:
    """Measure the first-step loss of each component group on randomly drawn units.

    The action head is zero-initialised and the flow target is `noise - a`, both terms
    with unit variance once normalised, so every group should come out near 2.0. A group
    well below that is being divided by a sigma that is too large, for example statistics
    measured at a different `action_hz`. Groups are averaged over their valid entries, so
    a hand that is absent from a batch does not pull its mean towards zero.
    """
    from dataloader import to_observations
    dev = next(model.parameters()).device
    core = model.module if isinstance(model, DDP) else model

    g = torch.Generator().manual_seed(seed)
    n = min(len(dataset), n_batches * batch_size)
    pick = torch.randperm(len(dataset), generator=g)[:n].tolist()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=pick, collate_fn=collate, num_workers=4)

    num = {g_: 0.0 for g_ in GROUPS}
    den = {g_: 0.0 for g_ in GROUPS}
    seen = 0
    with torch.no_grad():
        for batch in loader:
            steps, valid, act, amask = to_observations(batch, dev, obs_cls, torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.optim.bf16
                                and dev.type == "cuda"):
                _, parts = core.loss(steps, valid, act, amask)
            for g_ in GROUPS:
                c = float(parts[g_ + "_n"])
                num[g_] += float(parts[g_]) * c
                den[g_] += c
            seen += 1

    overall = (sum(num.values()) / sum(den.values())) if sum(den.values()) else float("nan")
    log(f"\n  pre-flight ({seen} random batches, {n} units): loss "
        f"{overall:.3f}  (target 2.00)")
    bad = []
    for g_ in GROUPS:
        if den[g_] <= 0:
            log(f"    {g_:<20} {'no data':>6}          -- absent from the draw")
            continue
        v = num[g_] / den[g_]
        ok = 1.5 < v < 2.6
        log(f"    {g_:<20} {v:6.3f}  over {int(den[g_]):>9,} entries  {'ok' if ok else 'OFF'}")
        if not ok:
            bad.append((g_, v))
    if bad:
        log("\n  WARNING: components far from 2.0 mean their sigma does not match the "
            "data. Wrist groups low by ~2.7x is the signature of statistics measured at "
            "the source rate instead of action_hz.")
        log("    " + ", ".join(f"{g_}={v:.2f}" for g_, v in bad))


# ---------------------------------------------------------------------- train


VARIANTS = {"d": "TWLA"}      # variant name -> model package


def load_variant(name: str):
    """Variant name -> (config class, model class, Observation class) of its package."""
    key = str(name).strip().lower()
    if key not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; expected one of {sorted(VARIANTS)}")
    import importlib
    mod = importlib.import_module(VARIANTS[key])
    return (mod.TWLAConfig, mod.TWLAModel, mod.Observation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None, metavar="CKPT",
                    help="Start a NEW run from another run's weights: load the model, "
                         "leave the optimiser empty, and begin at step 0. Distinct from "
                         "--resume, which continues a run and keeps its step and "
                         "schedule.")
    ap.add_argument("--renormalize", action="store_true",
                    help="With --init-from: after loading the checkpoint's weights, REPLACE its normalisation "
                         "buffers (action/state mean and std) with this run's statistics file. Needed when the "
                         "run trains on a different action space than the checkpoint (e.g. robot fine-tuning): "
                         "a checkpoint carries its own mu/sigma, which would otherwise be applied to the new "
                         "targets.")
    ap.add_argument("--fresh", default=None, metavar="PREFIXES",
                    help="With --init-from: comma-separated state-dict prefixes (e.g. "
                         "`memory.`) whose tensors are NOT taken from the checkpoint and "
                         "keep their fresh initialisation. World stage 2a starts this "
                         "way: trained VLM + expert (+ world head) from a stage-1 "
                         "checkpoint, memory module from scratch.")
    ap.add_argument("--init-world-from", default=None, metavar="CKPT",
                    help="Stage 2: load ONLY the `world.*` tensors from CKPT (a stage-1 "
                         "checkpoint, which also carries the frozen backbone it was "
                         "trained against) and leave every other tensor at its fresh "
                         "initialisation. `--init-from` would import that backbone too, "
                         "and the run would not be 'from scratch'. Step 0, empty "
                         "optimiser, like --init-from.")
    ap.add_argument("--wait-for-step", type=int, default=None, metavar="N",
                    help="With --resume: start up (CUDA context, VRAM claim, dataset, model) "
                         "NOW, then block until the --resume file reports step >= N and the "
                         "previous run has released its VRAM, then top up the claim and "
                         "continue. One phase of training can hand its GPUs to the next "
                         "this way without leaving them free in between.")
    ap.add_argument("--wait-free-gb", type=float, default=45.0,
                    help="with --wait-for-step: free VRAM that signals the previous run is "
                         "gone")
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="load the model weights but NOT the optimiser state. Required "
                         "when resuming into a different trainable set (a stage "
                         "transition); without it the mismatch raises, which is the "
                         "correct default -- a silently skipped optimiser state is a run "
                         "that says it resumed and did not.")
    ap.add_argument("--reserve-vram-gb", type=float, default=0.0,
                    help="Claim this card, leaving at most N GB free. The memory goes into OUR allocator pool (freed reference, never `empty_cache`), so nvidia-smi counts it as ours and other processes cannot take it, while our own training still uses it. Topped up every `vram_topup_every` steps so memory another process releases is reclaimed immediately. 0 disables.")
    ap.add_argument("--validate-only", default=None, metavar="CKPT",
                    help="load CKPT, run ONE full validation pass, write --val-out and "
                         "exit without training. Lets validation run on its own GPUs "
                         "instead of stopping the training ranks.")
    ap.add_argument("--val-out", default=None, metavar="JSON",
                    help="where --validate-only writes its result")
    ap.add_argument("--preflight", action="store_true",
                    help="measure the first-step loss against the 2.0 target; run "
                         "after changing statistics, action_hz, chunk_size or the "
                         "dataloader geometry")
    a = ap.parse_args()
    if a.renormalize and not a.init_from:
        ap.error("--renormalize only means something with --init-from")
    if a.resume and (a.init_from or a.init_world_from):
        ap.error("--resume continues a run; it cannot be combined with --init-from / --init-world-from.")
    # --init-from and --init-world-from may be combined: the backbone comes from one
    # checkpoint and the world head from another
    if a.fresh and not a.init_from:
        ap.error("--fresh only means something with --init-from")


    with open(a.config) as f:
        C = json.load(f)
    validate_run_config(C)          # fail early on a malformed run config
    C = resolve_paths(C)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    from dataloader import (SequenceDataset, MixtureSampler, make_collate,
                            load_video_list, sources_from_config,
                            to_observations, load_v4_statistics)

    variant = C.get("variant")
    if variant is None:
        raise KeyError(f'{a.config} has no "variant" field; expected one of '
                       f'{sorted(VARIANTS)}. It selects which model package to import, '
                       f'and defaulting it would silently train the wrong architecture.')
    ConfigCls, ModelCls, obs_cls = load_variant(variant)
    log(f"variant {str(variant).upper()}  ({VARIANTS[str(variant).lower()]})")

    cfg = ConfigCls()
    apply_overrides(cfg, C.get("config", {}))
    cfg.validate()                  # also raises if a REQUIRED parameter was left unset
    torch.manual_seed(C.get("seed", 0) + local_rank)

    log(f"world={world}  device={dev}  batch/gpu={C['batch_size']}  "
        f"total batch={C['batch_size'] * world}")

    # ---- VRAM reservation (--reserve-vram-gb)
    # Allocating a tensor and dropping the reference leaves the block in PyTorch's caching
    # allocator: other processes cannot take it and this process can still use it.
    # `torch.cuda.empty_cache()` would return it to the driver, so it is never called.
    def _claim_vram(target_free_gb: float, chunk_gb: float = 2.0) -> float:
        """Grow the allocator pool until at most `target_free_gb` is left free on this device."""
        if dev.type != "cuda":
            return 0.0
        # query the driver once, then allocate chunk by chunk; an OOM ends the loop early
        got = 0.0
        free, _tot = torch.cuda.mem_get_info()
        n = int((free / 1024 ** 3 - target_free_gb) // chunk_gb)
        for _ in range(max(0, n)):
            try:
                t = torch.empty(int(chunk_gb * 1024 ** 3), dtype=torch.uint8, device=dev)
            except torch.cuda.OutOfMemoryError:
                break
            del t                      # the block stays in the allocator pool
            got += chunk_gb
        return got

    _reserve_leave_free = float(a.reserve_vram_gb) if a.reserve_vram_gb > 0 else 0.0
    if _reserve_leave_free > 0 and dev.type == "cuda":
        g = _claim_vram(_reserve_leave_free)
        free, total = torch.cuda.mem_get_info()
        log(f"vram: claimed {g:.1f} GB into our pool; {free / 1024 ** 3:.1f} GB still free "
            f"on {dev} (target {_reserve_leave_free:.1f})  "
            f"reserved={torch.cuda.memory_reserved() / 1024 ** 3:.1f} GB")

    # ---- data
    tok = AutoTokenizer.from_pretrained(cfg.vlm.model_id)
    D = C["data"]
    # one source per domain; the split lists are per source because video names are
    # only unique within a domain
    sources = sources_from_config(D["sources"])
    tr_vids = {s["name"]: load_video_list(os.path.join(s["split_dir"], "train.txt"))
               for s in D["sources"]}
    va_vids = {s["name"]: load_video_list(os.path.join(s["split_dir"], "val.txt"))
               for s in D["sources"]}
    for n in tr_vids:
        shared = tr_vids[n] & va_vids[n]
        assert not shared, f"{n}: train and val share {len(shared)} videos"

    def build(vids, seed, training, deterministic=False):
        return SequenceDataset(
            sources, cfg=cfg, video_whitelist=vids, load_images=True, seed=seed,
            training=training, deterministic=deterministic)

    # Validation uses no augmentation, and with `deterministic=True` each unit's window
    # depends only on (seed, unit index). The instruction paraphrase, the flow time and the
    # noise are still drawn at random.
    train_ds = build(tr_vids, C.get("seed", 0), True)
    val_ds = build(va_vids, 12345, False, deterministic=True)
    log(f"units: train {len(train_ds)}  val {len(val_ds)}")

    collate = make_collate(cfg, tok)
    # with every source weight at 1 the mixture sampler is a plain shuffle
    sampler = MixtureSampler(train_ds, shuffle=True, seed=C.get("seed", 0),
                             rank=rank, world=world, drop_last=True)
    log("mixture per epoch: " + "  ".join(
        f"{s.name}={int(s.weight * sum(u.source == i for u in train_ds.units)):,}"
        f"{'' if s.weight == 1.0 else f' (x{s.weight})'}"
        for i, s in enumerate(sources)))
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=C["batch_size"], shuffle=False, sampler=sampler,
        num_workers=C["num_workers"], collate_fn=collate, drop_last=True,
        persistent_workers=C["num_workers"] > 0,
        prefetch_factor=C.get("prefetch_factor", 4) if C["num_workers"] > 0 else None)
    if len(train_dl) == 0:
        # with drop_last there is no batch at all, and the loop below would never advance
        raise RuntimeError(f"train loader is EMPTY: {len(sampler)} indices on this rank per epoch < batch_size "
                           f"{C['batch_size']} (units {len(train_ds.units)}, world {world}). Tile the source "
                           f"with a `weight` > 1 or lower batch_size.")
    val_plan = depth_grouped_batches(val_ds, C["batch_size"], rank, world)
    # a full pass unless `val_max_batches` is set (for smoke runs); the cap has to be
    # applied before the loader is built, because the loader keeps the list it was given
    cap = C.get("val_max_batches")
    if cap is not None:
        val_plan = val_plan[:int(cap)]
    # No persistent workers here: they would hold their /dev/shm buffers for the whole
    # run on top of the training workers', which can exhaust shared memory.
    _val_workers = max(1, C["num_workers"] // 2)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_sampler=val_plan, num_workers=_val_workers, collate_fn=collate)
    if len(val_dl) != len(val_plan):
        raise AssertionError(f"val loader has {len(val_dl)} batches but the plan has "
                             f"{len(val_plan)} -- the cap was applied after construction")
    log(f"val: {sum(len(b) for b in val_plan)} of {len(val_ds)} units on this rank in "
        f"{len(val_plan)} batches, one source and bank depth each"
        + (f"  ** TRUNCATED by val_max_batches={cap}; not a full-set number **"
           if cap is not None else ""))

    # ---- model
    stats = load_v4_statistics(D["statistics"])
    model = ModelCls(cfg, load_weights=True, dtype=torch.float32).to(dev)

    # ---- world head
    # The frozen DC-AE only encodes the target frames, so it is kept outside the model
    # and out of the checkpoints.
    ae = None
    if cfg.world is not None:
        from world_model import dc_ae
        ae = dc_ae.build(cfg.world.dcae, dtype=torch.float32).to(dev)
        if cfg.world.freeze_v4:
            n = model.freeze_for_world_stage1()
            log(f"world stage 1: {n['frozen']:,} parameters frozen, "
                f"{n['trainable']:,} trainable (adapter + SANA + horizon)")
            log("  NOTE: a falling L_world shows the head learned to use the memory that "
                "is already there, NOT that the memory carries the future. Zero m_f "
                "afterwards and see how much L_world rises -- that is the real test.")
        elif cfg.world.freeze_head:
            n = model.freeze_world_head()
            log(f"world stage 2: head FROZEN ({n:,} parameters), a fixed critic; "
                f"L_world shapes m_f only")
    if cfg.train.freeze_expert:
        n = model.freeze_expert()
        log(f"expert FROZEN ({n:,} parameters): a fixed reader of m_f; L_action still "
            f"reaches the memory through it. Unfreeze in a new process (stage 2b).")
    if cfg.train.freeze_extra:
        hit = [(nm, p) for nm, p in model.named_parameters()
               if any(nm.startswith(pre) for pre in cfg.train.freeze_extra)]
        if not hit:
            raise RuntimeError(f"train.freeze_extra {cfg.train.freeze_extra} matched nothing")
        for _, p in hit:
            p.requires_grad_(False)
        log(f"freeze_extra {list(cfg.train.freeze_extra)}: {len(hit)} tensors, "
            f"{sum(p.numel() for _, p in hit):,} parameters frozen")
    _by = {}                        # trainable parameters per top-level module
    for nm, p in model.named_parameters():
        if p.requires_grad:
            k = nm.split(".")[0]
            _by[k] = _by.get(k, 0) + p.numel()
    log("TRAINABLE: " + "  ".join(f"{k}={v:,}" for k, v in _by.items())
        + f"  total={sum(_by.values()):,}")
    model.set_normalisation(**{k: torch.tensor(v) for k, v in stats.items()
                               if not k.startswith("_")})
    core = model
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    opt = core.make_optimizer()

    def lam_world_at(step: int) -> float:
        """lambda in `L_total = L_action + lambda * L_world`; 0 without a world head and
        in stage 1, where the action path is frozen."""
        if cfg.world is None:
            return 0.0
        from world_model.loss import lam_at
        return 0.0 if cfg.world.freeze_v4 else lam_at(step, cfg.world.loss)

    # ---- per-domain multiplier on lambda (WorldLossConfig.domain_lam_*)
    # cos_d is the cosine between the action and the world gradient on the memory
    # parameters, over the samples of domain d. Its EMA, the observation count and the
    # multiplier of each domain are saved in every checkpoint under `domain_lam`.
    DOMS = [s["name"] for s in D["sources"]]
    _wl = cfg.world.loss if cfg.world is not None else None
    dom_mode = _wl.domain_lam_mode if _wl is not None else "off"
    dom_ema = {n: 0.0 for n in DOMS}
    dom_n = {n: 0 for n in DOMS}
    dom_mult = {n: 1.0 for n in DOMS}
    if dom_mode == "fixed":
        bad = [k for k in _wl.domain_lam_fixed if k not in DOMS]
        if bad:
            raise ValueError(f"domain_lam_fixed names unknown domains {bad}; known {DOMS}")
        dom_mult.update({k: float(v) for k, v in _wl.domain_lam_fixed.items()})
        log("domain lambda FIXED: " + " ".join(f"{k}={dom_mult[k]:g}" for k in DOMS))
    elif dom_mode == "adaptive":
        log(f"domain lambda ADAPTIVE: rule {_wl.domain_lam_rule} threshold {_wl.domain_lam_threshold:g} "
            f"scale {_wl.domain_lam_scale:g} min {_wl.domain_lam_min:g} ema {_wl.domain_lam_ema:g}; "
            f"multipliers applied from step {_wl.domain_lam_start}; cos_d probed every {C['log_every']} steps")

    def dom_rule(ema: float) -> float:
        if _wl.domain_lam_rule == "sign":
            return 1.0 if ema >= _wl.domain_lam_threshold else _wl.domain_lam_min
        return float(min(1.0, max(_wl.domain_lam_min,
                                  (ema - _wl.domain_lam_threshold) / max(_wl.domain_lam_scale, 1e-9))))

    def lam_for_batch(batch, step: int, lam_now: float):
        """Scalar lambda, or a per-sample tensor [B] when the domain multipliers apply."""
        if dom_mode == "off" or lam_now <= 0.0:
            return lam_now
        if dom_mode == "adaptive" and step < _wl.domain_lam_start:
            return lam_now
        return torch.tensor([lam_now * dom_mult[s_] for s_ in batch["source"]],
                            device=dev, dtype=torch.float32)

    def dom_state() -> dict:
        return {"domain_lam": {"ema": dict(dom_ema), "n": dict(dom_n), "mult": dict(dom_mult)}}

    def load_model_state(state, *, what: str, fresh: tuple[str, ...] = ()):
        """Strict load, except that `world.*` tensors may be missing from the checkpoint
        (a world head added to a model trained without one). Extra keys raise.

        `fresh`: state-dict prefixes whose tensors are not loaded and keep their
        initialisation (`--fresh`).
        """
        if fresh:
            dropped = [k for k in state if any(k.startswith(p) for p in fresh)]
            if not dropped:
                raise RuntimeError(f"{what}: --fresh {fresh} matched no tensor in the "
                                   f"checkpoint; check the prefix (e.g. `memory.`)")
            state = {k: v for k, v in state.items() if k not in dropped}
            log(f"{what}: --fresh {','.join(fresh)}: {len(dropped)} checkpoint tensors "
                f"dropped; those modules keep their fresh initialisation.")
        want, have = set(core.state_dict()), set(state)
        if fresh:
            want = {k for k in want if not any(k.startswith(p) for p in fresh)}
        extra, missing = have - want, want - have
        if extra:
            raise RuntimeError(
                f"{what}: {len(extra)} tensors in the checkpoint have no home in this "
                f"model, e.g. {sorted(extra)[:4]}. If these are a world head, the config "
                "is missing its `world` section and the head would be thrown away.")
        if missing:
            new_head = {k for k in missing if k.startswith("world.")}
            if new_head != missing:
                raise RuntimeError(
                    f"{what}: {len(missing - new_head)} tensors are missing from the "
                    f"checkpoint and are NOT part of the world head, e.g. "
                    f"{sorted(missing - new_head)[:4]}")
            log(f"{what}: {len(new_head)} `world.*` tensors are new and keep their "
                f"initialisation (inherited SANA blocks + zero-init adapter); every "
                f"other tensor came from the checkpoint.")
        core.load_state_dict(state, strict=False)

    step0, epoch0, hist0 = 0, None, {}
    # best validation loss so far; restored on resume so `best.pt` is not overwritten
    # by a worse model
    best_val = float("inf")
    # Reduce-on-plateau state (OptimConfig.plateau_*). `mult` scales every group's lr;
    # `since` is the step of the last drop, and results at or before it are not counted.
    plat = {"mult": 1.0, "best": float("inf"), "bad": 0, "since": 0, "seen": [], "drops": 0}
    if a.init_from:
        # weights only: no optimiser state, step, epoch, length histogram or `best_val`
        ck = torch.load(a.init_from, map_location="cpu")
        fresh = tuple(p for p in (a.fresh or "").split(",") if p)
        load_model_state(ck["model"], what="init-from", fresh=fresh)
        log(f"init-from: {a.init_from} (was step {ck.get('step', '?')}); this run starts "
            f"at step 0 with a fresh optimiser.")
        if a.renormalize:
            core.set_normalisation(**{k: torch.tensor(v) for k, v in stats.items()
                                      if not k.startswith("_")})
            log(f"init-from: normalisation buffers REPLACED by this run's statistics "
                f"({C['data']['statistics']}); the checkpoint's own mu/sigma were discarded.")
    if a.init_world_from:
        if cfg.world is None:
            raise RuntimeError("--init-world-from needs a `world` section in the config")
        ck = torch.load(a.init_world_from, map_location="cpu")
        n_head = core.load_world_head(ck["model"])
        n_rest = sum(1 for k in ck["model"] if not k.startswith("world."))
        log(f"init-world-from: {a.init_world_from} (was step {ck.get('step', '?')}): "
            f"{n_head} `world.*` tensors loaded into the head; the checkpoint's other "
            f"{n_rest} tensors IGNORED -- backbone is "
            + ("what --init-from loaded" if a.init_from else "its fresh initialisation")
            + ". Step 0, fresh optimiser.")
    elif a.resume:
        if a.wait_for_step is not None:
            # the model is already on the GPU, so the card stays occupied while waiting
            log(f"wait-for-step: holding {torch.cuda.memory_reserved() / 2**30:.1f} GB on "
                f"{dev}; waiting for {a.resume} to reach step {a.wait_for_step} ...")
            seen_mtime, ck_step = None, None
            while True:
                try:
                    mt = os.path.getmtime(a.resume)
                    if mt != seen_mtime:
                        seen_mtime = mt
                        ck_step = int(torch.load(a.resume, map_location="cpu")["step"])
                        log(f"wait-for-step: {a.resume} is at step {ck_step}")
                except (OSError, RuntimeError, KeyError, EOFError):
                    pass                                # absent or mid-write; try again
                if ck_step is not None and ck_step >= a.wait_for_step:
                    break
                time.sleep(5)
            if dev.type == "cuda":
                t_w = time.time()
                while torch.cuda.mem_get_info()[0] / 2**30 < a.wait_free_gb:
                    time.sleep(2.0)
                # the previous run has exited: claim the memory it released
                got = _claim_vram(_reserve_leave_free if _reserve_leave_free > 0 else 0.0) \
                    if _reserve_leave_free > 0 else 0.0
                log(f"wait-for-step: previous run released its VRAM after "
                    f"{time.time() - t_w:.1f}s; claimed {got:.0f} GB more "
                    f"(reserved now {torch.cuda.memory_reserved() / 2**30:.1f} GB)")
            if world > 1:
                dist.barrier()                           # all ranks past the wait together
        ck = torch.load(a.resume, map_location="cpu")
        load_model_state(ck["model"], what="resume")
        if a.reset_optimizer:
            # For a resume into a different trainable set (a stage transition): AdamW
            # indexes its state by position within each parameter group, so the saved
            # state would not fit. The moments restart from zero.
            log("resume: optimiser state DISCARDED (--reset-optimizer). Model weights "
                "loaded; AdamW moments start at zero for every parameter.")
        else:
            opt.load_state_dict(ck["optim"])
        step0 = ck["step"] + 1
        epoch0 = ck.get("epoch")        # if absent, derived from the step further down
        hist0 = dict(ck.get("length_histogram") or {})
        # `best.pt` does not carry `best_val`: after resuming from it, the first
        # validation overwrites `best.pt` whatever it scores. Resume from `last.pt`.
        best_val = float(ck.get("best_val", float("inf")))
        # a stage transition (--reset-optimizer) starts the plateau state afresh
        if cfg.optim.plateau_factor > 0 and ck.get("plateau") and not a.reset_optimizer:
            plat = dict(ck["plateau"])
            log(f"resume: plateau state restored: lr x{plat['mult']:.3g}, {plat['drops']} drops, "
                f"best {plat['best']:.4f}, bad {plat['bad']}")
        if dom_mode == "adaptive" and ck.get("domain_lam"):
            dl_ = ck["domain_lam"]
            dom_ema.update({k: float(v) for k, v in dl_.get("ema", {}).items() if k in dom_ema})
            dom_n.update({k: int(v) for k, v in dl_.get("n", {}).items() if k in dom_n})
            dom_mult.update({k: float(v) for k, v in dl_.get("mult", {}).items() if k in dom_mult})
            log("resume: domain lambda state restored: " + " ".join(
                f"{k}:{dom_ema[k]:+.3f}/{dom_mult[k]:g}" for k in DOMS))
        log(f"resumed from {a.resume} at step {step0}"
            + (f", epoch {epoch0}" if epoch0 is not None else ", epoch not in checkpoint")
            + (f", best val {best_val:.4f}" if best_val < float("inf")
               else ", no best val in checkpoint"))

    # The statistics must have been measured with this config's action geometry
    # (action_hz, chunk size, ...) and on the sources this run trains on.
    from dataloader import check_convention, check_sources
    check_convention(stats, cfg)
    check_sources(stats, D["sources"])
    # --validate-only: one pass with the same loader, plan and `evaluate` as training
    if a.validate_only:
        ck = torch.load(a.validate_only, map_location="cpu")
        load_model_state(ck["model"], what="validate-only")
        step_of = int(ck.get("step", -1))
        log(f"validate-only: {a.validate_only} (step {step_of})")
        t = time.time()
        core.eval()
        v = evaluate(core, val_dl, cfg, dev, obs_cls, world)
        if is_main():
            log(f"  [val] step {step_of}  loss {v['loss']:.4f}  " +
                " ".join(f"{g}={v[g]:.3f}" for g in GROUPS))
            log("        per bank depth  " + "  ".join(
                f"{d}:{b['loss']:.3f}" for d, b in sorted(v["depth"].items())))
            log("        per source      " + "  ".join(
                f"{n}:{m['loss']:.3f}" for n, m in v["source"].items()))
            if a.val_out:
                os.makedirs(os.path.dirname(os.path.abspath(a.val_out)), exist_ok=True)
                tmp = a.val_out + ".partial"
                with open(tmp, "w") as f:
                    json.dump({"step": step_of, "checkpoint": a.validate_only,
                               "seconds": round(time.time() - t, 1), "val": v}, f)
                os.replace(tmp, a.val_out)      # atomic: a reader never sees half a file
                log(f"  wrote {a.val_out}")
        if world > 1:
            dist.destroy_process_group()
        return

    if a.preflight and is_main():
        preflight(model, val_ds, collate, cfg, obs_cls,
                  batch_size=C["batch_size"], seed=C.get("seed", 0))

    out_dir = C["output_dir"]
    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config_used.json"), "w") as f:
            json.dump(C, f, indent=2)

    # ---- loop
    max_steps = C["max_steps"]
    log_every, val_every, save_every = C["log_every"], C["val_every"], C["save_every"]
    accum = cfg.optim.grad_accum_steps
    trainable = [p for p in core.parameters() if p.requires_grad]

    hist = defaultdict(int, hist0)   # histogram of the realised sequence lengths L
    run = defaultdict(list)
    # seconds spent waiting for batches in the current log window, and the window's start
    wait_acc, t_win = [0.0], time.time()
    step = step0
    # `step` counts optimiser updates; `micro` counts the batches accumulated towards
    # the next one (`grad_accum_steps` per update)
    micro = 0
    t0 = time.time()
    model.train()
    # `MixtureSampler` derives its order from seed + epoch. A resume restarts at the
    # beginning of the checkpoint's epoch, so up to one epoch of data is seen again.
    epoch = epoch0 if epoch0 is not None else (step0 // max(1, len(train_dl)))
    if step0:
        log(f"  data order resumes at epoch {epoch} "
            f"({len(train_dl)} steps/epoch, so at most one epoch of overlap)")
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        # only a pass that ran to the end advances the epoch; a pass cut short by
        # `max_steps` must not, or a resume would skip an epoch of data
        epoch_complete = True
        for batch in _timed_loader(train_dl, wait_acc):
            if step >= max_steps:
                epoch_complete = False
                break
            # scale each group's own `base_lr`: the world head's groups have their own rates
            lr = core.lr_at(step) * plat["mult"]
            scale = core.lr_scale_at(step) * plat["mult"]
            e_scale = core.expert_warmup_scale_at(step)
            for g in opt.param_groups:
                g["lr"] = (g.get("base_lr", cfg.optim.lr) * scale
                           * (e_scale if g.get("expert") else 1.0))

            steps_, valid, act, amask = to_observations(batch, dev, obs_cls, torch.float32)
            # under accumulation only the last backward of an update needs DDP's
            # gradient all-reduce; the earlier ones run under `no_sync`
            sync = (micro + 1) % accum == 0
            with contextlib.nullcontext() if (sync or world == 1) else model.no_sync():
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=cfg.optim.bf16 and dev.type == "cuda"):
                    wt = wv = None
                    if ae is not None:
                        # [B, K, H, W, 3] uint8 -> [B, K, 3, H, W] in [-1, 1], the same
                        # convention `to_observations` applies to the conditioning frame
                        f = batch["future_images"].to(dev, non_blocking=True)
                        f = f.permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
                        wt = dc_ae.encode_targets(ae, f, cfg.world.dcae)
                        wv = batch["future_valid"].to(dev, non_blocking=True)
                    # the gradient probes run only on logged steps
                    lam_now = lam_world_at(step)
                    on_log = sync and step % log_every == 0
                    dprobe = {} if (dom_mode == "adaptive" and on_log and lam_now > 0.0
                                    and wt is not None) else None
                    # call the DDP wrapper, not `model.module.loss(...)`, which would
                    # skip the gradient synchronisation
                    loss, parts = model(steps_, valid, act, amask,
                                        world_targets=wt, world_valid=wv,
                                        lam_world=lam_for_batch(batch, step, lam_now),
                                        probe_mf_grad=on_log,
                                        sample_source=(list(batch["source"]) if dprobe is not None else None),
                                        domain_probe=dprobe)
                (loss / accum).backward()
                if dprobe is not None:
                    # Sum each domain's two memory-parameter gradients across ranks (zeros
                    # where the domain is absent from a rank's batch), then update the EMA
                    # of cos_d and the multiplier.
                    P_ = sum(q.numel() for q in core.memory.parameters() if q.requires_grad)
                    for name in DOMS:
                        va, vw, cnt = dprobe.get(name, (None, None, 0))
                        va = torch.zeros(P_, device=dev) if va is None else va
                        vw = torch.zeros(P_, device=dev) if vw is None else vw
                        ct = torch.tensor([float(cnt)], device=dev)
                        if world > 1:
                            dist.all_reduce(va); dist.all_reduce(vw); dist.all_reduce(ct)
                        if ct.item() <= 0 or va.norm() == 0 or vw.norm() == 0:
                            continue
                        cos_d = float(torch.nn.functional.cosine_similarity(va, vw, dim=0))
                        r_d = float(vw.norm() / va.norm().clamp_min(1e-12))
                        e_ = _wl.domain_lam_ema
                        dom_ema[name] = cos_d if dom_n[name] == 0 else e_ * dom_ema[name] + (1 - e_) * cos_d
                        dom_n[name] += 1
                        dom_mult[name] = dom_rule(dom_ema[name])
                        run[f"dom_cos_{name}"].append(cos_d); run[f"dom_r_{name}"].append(r_d)
                    del dprobe

            if sync:
                gn = torch.nn.utils.clip_grad_norm_(trainable, cfg.optim.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                run["grad"].append(float(gn))

            run["loss"].append(float(loss.detach()))
            if "world" in parts:
                run["world"].append(float(parts["world"]))
                run["action"].append(float(parts["action"]))
                for hk in [k for k in parts if k.startswith("world_h") and not k.endswith("_n")]:
                    run[hk].append(float(parts[hk]))
                for pk in ("mf_gA", "mf_gW", "mf_r", "mf_cos", "mem_gA", "mem_gW", "mem_r", "mem_cos"):
                    if pk in parts:
                        run[pk].append(float(parts[pk]))
            for g in GROUPS:
                run[g].append(float(parts[g]))
            for n in batch["length"].tolist():
                hist[n] += 1

            micro += 1
            if not sync:
                # still accumulating: no update yet, so nothing to log, validate or save
                continue
            micro = 0

            if is_main() and step % log_every == 0:
                el = time.time() - t0
                sps = (step - step0 + 1) * C["batch_size"] * world * accum / max(el, 1e-9)
                # share of wall time this rank spent waiting for a batch; a large value
                # means the dataloader workers are the bottleneck
                el_win = max(time.time() - t_win, 1e-9)
                wait_pct = 100.0 * wait_acc[0] / el_win
                wait_acc[0], t_win = 0.0, time.time()

                wl = ""
                if run.get("world"):
                    hs = sorted(k for k in run if k.startswith("world_h"))   # one per horizon
                    wl = (f"  Lw {np.mean(run['world']):.4f}  "
                          + " ".join(f"{k[6:]}={np.mean(run[k]):.4f}" for k in hs))
                    lam_now = lam_world_at(step)
                    if lam_now > 0.0:
                        # `loss` is the combined objective and `La` the action term alone.
                        # `gA`/`gW` are the gradient norms of the action and world terms
                        # at m_f (the world one includes lambda); `r` is their ratio.
                        wl = (f"  La {np.mean(run['action']):.4f}  lam {lam_now:.3f}"
                              + wl)
                        if run.get("mf_r"):
                            wl += (f"  gA {np.mean(run['mf_gA']):.3e} "
                                   f"gW {np.mean(run['mf_gW']):.3e} "
                                   f"r {np.mean(run['mf_r']):.3f} "
                                   f"cos {np.mean(run['mf_cos']):+.2f}")
                        if run.get("mem_r"):
                            # the same quantities on the memory module's parameters
                            wl += (f"  memA {np.mean(run['mem_gA']):.2f} "
                                   f"memW {np.mean(run['mem_gW']):.3f} "
                                   f"memr {np.mean(run['mem_r']):.3f} "
                                   f"memcos {np.mean(run['mem_cos']):+.2f}")
                if dom_mode != "off" and run.get("world"):
                    # per domain: cos_d of this window (EMA in brackets) and the multiplier in force
                    wl += "  dom[" + " ".join(
                        f"{n_[:4]}:{np.mean(run['dom_cos_' + n_]):+.2f}({dom_ema[n_]:+.2f})/{dom_mult[n_]:g}"
                        if run.get("dom_cos_" + n_) else f"{n_[:4]}:-/{dom_mult[n_]:g}"
                        for n_ in DOMS) + "]"
                lr_e = ("" if cfg.optim.lr_expert is None else
                        f" lrE {cfg.optim.lr_expert * scale * e_scale:.2e}")
                log(f"step {step:>6}  loss {np.mean(run['loss']):.4f}{wl}  "
                    f"[wait {wait_pct:4.1f}% {_mem_probe()}]  "
                    f"lr {lr:.2e}{lr_e}  |g| {np.mean(run['grad'] or [0]):.2f}  "
                    f"{sps:.1f} samp/s  "
                    + " ".join(f"{g.split('_')[0][0]}{g.split('_')[1][:3]}="
                               f"{np.mean(run[g]):.3f}" for g in GROUPS))
                run.clear()

            if step and step % val_every == 0:
                v = evaluate(core, val_dl, cfg, dev, obs_cls, world)
                log(f"  [val] step {step}  loss {v['loss']:.4f}  " +
                    " ".join(f"{g}={v[g]:.3f}" for g in GROUPS))
                log("        per bank depth  " + "  ".join(
                    f"{d}:{b['loss']:.3f}" for d, b in sorted(v["depth"].items())))
                log("        per source      " + "  ".join(
                    f"{name}:{metrics['loss']:.3f}"
                    for name, metrics in v["source"].items()))
                # `evaluate` all-reduces, so every rank compares the same number
                if v["loss"] < best_val:
                    log(f"        best so far ({best_val:.4f} -> {v['loss']:.4f})")
                    best_val = v["loss"]
                    if is_main():
                        save(core, opt, step, out_dir, hist, epoch, "best",
                             {"val_loss": best_val})
                model.train()

            # Reduce-on-plateau: rank 0 decides from the validator's result files and
            # broadcasts, so all ranks apply the same multiplier. Done before saving, so
            # the checkpoint carries the decision.
            plateau_stop = False
            if cfg.optim.plateau_factor > 0 and step and step % save_every == 0:
                if is_main():
                    plateau_stop = plateau_update(plat, os.path.join(out_dir, "val"),
                                                  cfg.optim, log)
                if world > 1:
                    t = torch.tensor([plat["mult"], 1.0 if plateau_stop else 0.0], device=dev)
                    dist.broadcast(t, src=0)
                    plat["mult"], plateau_stop = float(t[0]), bool(t[1] > 0.5)

            if is_main() and step and step % save_every == 0:
                # one file per save (`step_<step>.pt`; these accumulate) and `last.pt`,
                # which is overwritten each time
                save(core, opt, step, out_dir, hist, epoch, None,
                     {**dom_state(), "best_val": best_val, "plateau": plat})
                save(core, opt, step, out_dir, hist, epoch, "last",
                     {**dom_state(), "best_val": best_val, "plateau": plat})
                # optional copy under `snapshots/`, for validation in a separate process
                if C.get("snapshot_for_validator", False):
                    save(core, opt, step, os.path.join(out_dir, "snapshots"), hist, epoch,
                         f"snap_{step:07d}", {**dom_state(), "best_val": best_val})

            if plateau_stop:
                log(f"plateau at the lr floor ({cfg.optim.plateau_min_lr:.1e}): stopping "
                    f"at step {step}")
                step += 1
                break

            step += 1
        if epoch_complete:
            epoch += 1

    if is_main():
        save(core, opt, step, out_dir, hist, epoch, "last",
             {**dom_state(), "best_val": best_val, "plateau": plat})
        log(f"\ndone. {step} steps in {(time.time()-t0)/3600:.2f} h")
        log("realised L histogram: " +
            " ".join(f"{k}:{100*v/sum(hist.values()):.1f}%" for k, v in sorted(hist.items())))
    if world > 1:
        dist.destroy_process_group()


def depth_grouped_batches(ds, batch_size: int, rank: int, world: int) -> list[list[int]]:
    """Plan the validation batches so that each holds one source and one bank depth (L - 1).

    The depth of a unit comes from `planned_window`, which needs `deterministic=True` and
    reads nothing from disk. Each rank takes the units `range(rank, N, world)`: the shards
    are disjoint and no unit is repeated, so `evaluate` can sum across ranks exactly. The
    (source, depth) groups are interleaved so that a truncated plan still covers several
    sources and depths.
    """
    by_source_depth: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(rank, len(ds), world):
        source = ds.units[i].source
        depth = ds.planned_window(i).length - 1
        by_source_depth[source, depth].append(i)
    source_ids = sorted({key[0] for key in by_source_depth})
    depth_ids = sorted({key[1] for key in by_source_depth})
    keys = [(source, depth_ids[(round_ + si) % len(depth_ids)])
            for round_ in range(len(depth_ids))
            for si, source in enumerate(source_ids)
            if (source, depth_ids[(round_ + si) % len(depth_ids)]) in by_source_depth]
    per_group = [[by_source_depth[key][j:j + batch_size]
                  for j in range(0, len(by_source_depth[key]), batch_size)]
                 for key in keys]
    return [b for row in itertools.zip_longest(*per_group) for b in row if b is not None]


@torch.no_grad()
def evaluate(core, loader, cfg, dev, obs_cls, world: int) -> dict:
    """One validation pass: the loss overall, per source and per bank depth.

    Sums each group's squared error and its count of valid entries over all batches (and
    ranks) and divides once at the end.
    """
    from dataloader import to_observations
    core.eval()
    names = [g[0] for g in cfg.flow.component_groups]
    if names != list(GROUPS):
        raise ValueError(f"GROUPS {list(GROUPS)} must match cfg.flow.component_groups "
                         f"{names} in order -- they index the same weight vector")

    source_names = [s.name for s in loader.dataset.sources]
    source_index = {name: i for i, name in enumerate(source_names)}
    depths = cfg.train.max_seq_len
    num = torch.zeros(len(source_names), depths, len(GROUPS),
                      dtype=torch.float64, device=dev)
    den = torch.zeros_like(num)
    for batch in loader:
        lens = batch["length"]
        lo, hi = int(lens.min()), int(lens.max())
        if lo != hi:
            raise AssertionError(f"a validation batch must hold one depth, got L in "
                                 f"[{lo}, {hi}] -- was the plan built with shuffling?")
        batch_sources = set(batch["source"])
        if len(batch_sources) != 1:
            raise AssertionError("a validation batch must hold one source, got "
                                 f"{sorted(batch_sources)}")
        source = next(iter(batch_sources))
        if source not in source_index:
            raise KeyError(f"validation batch names unknown source {source!r}")
        si = source_index[source]
        steps_, valid, act, amask = to_observations(batch, dev, obs_cls, torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=cfg.optim.bf16 and dev.type == "cuda"):
            _, parts = core.loss(steps_, valid, act, amask)
        for j, g in enumerate(GROUPS):
            n = parts[g + "_n"].double()
            num[si, lo - 1, j] += parts[g].double() * n
            den[si, lo - 1, j] += n

    # the ranks hold disjoint shards, so the sums add up to the whole-set totals
    if world > 1:
        dist.all_reduce(num)
        dist.all_reduce(den)

    w = torch.tensor(cfg.flow.component_weights, dtype=torch.float64, device=dev)

    def combine(nu, de):
        """Weighted mean over the groups that have data; nan if none has."""
        live = de > 0
        if not bool(live.any()):
            return float("nan")
        return float((w[live] * nu[live] / de[live]).sum() / w[live].sum())

    def summary(nu: torch.Tensor, de: torch.Tensor) -> dict:
        # leading axes are source and/or depth; the last axis is the component group
        nu = nu.reshape(-1, len(GROUPS)).sum(0)
        de = de.reshape(-1, len(GROUPS)).sum(0)
        ans = {"loss": combine(nu, de), "entries": float(de.sum())}
        for j, g in enumerate(GROUPS):
            total = float(de[j])
            ans[g] = float(nu[j]) / total if total > 0 else float("nan")
        return ans

    out = summary(num, den)
    out["depth"] = {d: summary(num[:, d], den[:, d])
                    for d in range(depths) if float(den[:, d].sum()) > 0}
    out["source"] = {}
    for si, source in enumerate(source_names):
        src = summary(num[si], den[si])
        src["depth"] = {d: summary(num[si, d], den[si, d])
                        for d in range(depths) if float(den[si, d].sum()) > 0}
        out["source"][source] = src
    return out


def plateau_update(plat: dict, val_dir: str, o, log) -> bool:
    """Reduce-on-plateau over the validator's `<val_dir>/<step>.json` files.

    Reads every result not yet seen, in step order, skipping steps at or before the last
    drop. A result improves when it beats `best` by `plateau_threshold` (absolute). After
    `plateau_patience` results in a row without improvement the lr multiplier is scaled by
    `plateau_factor`. Returns True (stop) when that happens with the lr already at
    `plateau_min_lr`. Mutates `plat` in place.
    """
    import glob, json
    new = []
    for p in glob.glob(os.path.join(val_dir, "*.json")):
        try:
            st = int(os.path.basename(p)[:-5])
        except ValueError:
            continue
        if st in plat["seen"] or st <= plat["since"]:
            continue
        try:
            new.append((st, float(json.load(open(p))["val"]["loss"])))
        except Exception:                      # half-written file: pick it up next time
            continue
    for st, loss in sorted(new):
        plat["seen"].append(st)
        if loss < plat["best"] - o.plateau_threshold:
            plat["best"], plat["bad"] = loss, 0
            log(f"  [plateau] val {loss:.4f} @ {st}: new best")
        else:
            plat["bad"] += 1
            log(f"  [plateau] val {loss:.4f} @ {st}: no improvement over {plat['best']:.4f} "
                f"by {o.plateau_threshold:.0e} ({plat['bad']}/{o.plateau_patience})")
        if plat["bad"] >= o.plateau_patience:
            cur = o.lr * plat["mult"]
            if cur <= o.plateau_min_lr * 1.001:
                return True
            plat["mult"] = max(plat["mult"] * o.plateau_factor, o.plateau_min_lr / o.lr)
            plat["bad"], plat["since"], plat["drops"] = 0, st, plat["drops"] + 1
            log(f"  [plateau] lr {cur:.2e} -> {o.lr * plat['mult']:.2e} (drop #{plat['drops']}; "
                f"results up to step {st} no longer counted)")
    return False


def save(core, opt, step, out_dir, hist, epoch: int = 0, name: str | None = None,
         extra: dict | None = None) -> None:
    """Write a checkpoint: model state (which includes the normalisation buffers),
    optimiser state, step, epoch and length histogram, plus `extra`.

    `name` gives a fixed file name that is overwritten (`last`, `best`); without it the
    file is `step_<step>.pt`. The file is written under a temporary name and renamed.
    """
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{name}.pt" if name else f"step_{step:07d}.pt")
    tmp = p + ".partial"
    payload = {"model": core.state_dict(), "optim": opt.state_dict(), "step": step,
               "epoch": int(epoch), "length_histogram": dict(hist)}
    payload.update(extra or {})
    torch.save(payload, tmp)
    os.replace(tmp, p)                      # atomic within one filesystem
    log(f"  saved {p}")


if __name__ == "__main__":
    main()
