#!/usr/bin/env python
"""Test the MemoryVLA baseline.

    python baselines/memoryvla_official/test.py --out results/memoryvla.json
    torchrun --nproc_per_node=4 baselines/memoryvla_official/test.py --out results/memoryvla.json

Defaults: the released weights at `<repository root>/checkpoints/MemoryVLA.pt` and `configs/test.json` (the test split
of the three fine-tuning datasets). The checkpoint holds only the trained parts: the frozen VLM is rebuilt from
CogACT-Large (`config.vlm.pretrained_checkpoint`), the Llama-2 config and tokenizer from `config.vlm.llama_local_dir`.

The metric is `metric.py`, a copy of the main model's. The prediction is keypoints: it is denormalised with the
statistics stored in the checkpoint and scored against the data's `joints_manospace`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(HERE)
for _p in (BASELINES, HERE):                 # HERE first: `data`, `metric`, `dataloader` are this package's own
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(BASELINES)), "checkpoints", "MemoryVLA.pt")   # where the released weights go
DEFAULT_CONFIG = os.path.join(HERE, "configs", "test.json")

from data import dataset_class                                            # noqa: E402
from metric import (ANCHORS_PER_UNIT, GROUPS, MAX_DEPTH, VIEWS, build_plan,  # noqa: E402
                    errors, limit_plan_per_source, stride_plan_per_source)
from v4_bridge import (load_video_list, make_collate, resolve_paths,      # noqa: E402
                       sources_from_config, to_observations)
from memoryvla_official import (Config, Model, Observation,  # noqa: E402
                                build_tokenizer, from_json)

del ANCHORS_PER_UNIT, MAX_DEPTH          # not used in this script


def load_config(path: str) -> tuple[dict, Config]:
    """Read the run json -> (json with resolved paths, Config)."""
    with open(path) as f:
        C = resolve_paths(json.load(f))
    overrides = dict(C.get("config", {}))
    overrides["data"] = {**overrides.get("data", {}),
                         "stats_path": C["data"]["statistics"]}
    cfg = from_json(overrides)
    for sec, field in (("vlm", "model_id"), ("init", "pretrain_path")):
        v = getattr(getattr(cfg, sec), field)
        if v and "$" in v:
            v = os.path.expanduser(os.path.expandvars(v))
            if "${" in v:
                raise KeyError(f"unset environment variable in {sec}.{field}: {v!r}")
            setattr(getattr(cfg, sec), field, v)
    if cfg.action.action_type != "keypoints":
        # The prediction is scored directly against keypoints; joint angles would need forward kinematics first.
        raise SystemExit(
            f"this baseline is keypoints-only, got action_type={cfg.action.action_type!r}")
    return C, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--depths", default="0,1,2,3,4,5,6,7,8",
                    help="memory depths to test, comma separated. The (unit, anchor) pairs are the "
                         "same whatever depths are chosen.")
    a = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    log = (lambda *m: print(*m, flush=True)) if rank == 0 else (lambda *m: None)

    C, cfg = load_config(a.config)
    src_cfgs = C["data"]["sources"]
    log(f"action_type {cfg.action.action_type}   images {cfg.image.mode}   "
        f"statistics {cfg.data.stats_path}")

    ds = dataset_class(cfg)(
        sources_from_config(src_cfgs), cfg=cfg,
        video_whitelist={s["name"]: load_video_list(
            os.path.join(s["split_dir"], "test.txt")) for s in src_cfgs},
        load_images=True, seed=a.seed, training=False, deterministic=True)

    plan = build_plan(ds)
    pair_source = {pid: ds.sources[ds.units[ui].source].name
                   for ui, _s, _l, pid in plan}
    all_pairs_by_source = collections.Counter(pair_source.values())
    all_n_pairs = len(pair_source)
    log(f"  test units {len(ds):,}  ->  {all_n_pairs:,} (unit, anchor) pairs")
    if a.limit:
        plan = limit_plan_per_source(plan, pair_source, a.limit)
        log(f"  --limit {a.limit} per source: {len(plan):,} evaluations")
    if a.stride > 1:
        plan = stride_plan_per_source(plan, pair_source, a.stride)
        log(f"  --stride {a.stride}: {len(plan):,} evaluations")
    want = {int(s) for s in a.depths.split(",") if s.strip() != ""}
    plan = [x for x in plan if (x[2] - 1) in want]
    if not plan:
        raise SystemExit("the plan is empty after filtering")
    selected = {x[3] for x in plan}
    selected_by_source = collections.Counter(pair_source[p] for p in selected)
    BASE_DEPTH = min(x[2] - 1 for x in plan)
    n_at_base = sum(1 for x in plan if x[2] - 1 == BASE_DEPTH)
    if n_at_base != len(selected):
        raise AssertionError(f"depth {BASE_DEPTH} holds {n_at_base} entries but there are "
                             f"{len(selected)} pairs -- `samples` would double-count")
    log(f"  --depths {sorted(want)}: {len(plan):,} evaluations, "
        f"{len(selected):,} pairs; `samples` counted once, at depth {BASE_DEPTH}")
    ds.set_eval_plan(plan)

    # load_weights=True rebuilds the frozen VLM from CogACT-Large; the checkpoint holds only the trained parts.
    model = Model(cfg, load_weights=True).to(dev)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    # The normalisation statistics are buffers of the checkpoint; no statistics file is read here.
    model.load_state_dict(ck["model"])
    model.eval()
    log(f"  checkpoint {a.checkpoint}")

    collate = make_collate(cfg, build_tokenizer(cfg))
    by_depth: dict[int, list[int]] = {}
    for i, (_u, _s, ln, _p) in enumerate(plan):
        by_depth.setdefault(ln - 1, []).append(i)
    # A batch holds a single depth, so every row has the same window length.
    batches = [g[j:j + a.batch_size] for d in sorted(by_depth)
               for g in [by_depth[d]] for j in range(0, len(g), a.batch_size)]
    batches = batches[rank::world]
    log(f"  {len(batches):,} batches of up to {a.batch_size} on this rank")

    dl = torch.utils.data.DataLoader(ds, batch_sampler=batches,
                                     num_workers=a.num_workers, collate_fn=collate)

    def new_acc():
        return {v: {d: {g[0]: [0.0, 0] for g in GROUPS}
                    for d in sorted(by_depth)} for v in VIEWS}

    source_names = [s.name for s in ds.sources]
    # Per-sample errors (pair, episode, source, depth, view and the three errors), so the results can be regrouped
    # without a rerun.
    per_rows = []
    acc = new_acc()
    src_acc = {n: new_acc() for n in source_names}
    nsamp = {v: 0 for v in VIEWS}
    src_nsamp = {n: {v: 0 for v in VIEWS} for n in source_names}

    every = max(1, len(batches) // 20)
    t0 = time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for bi, batch in enumerate(dl):
        steps, _valid, act, amask = to_observations(batch, dev, Observation, torch.float32)
        sources = batch["source"]
        depths = [int(l) - 1 for l in batch["length"]]
        if len(set(depths)) != 1:
            raise AssertionError(f"a batch must hold one depth, got {sorted(set(depths))}")
        d = depths[0]
        pair_ids = [plan[i][3] for i in batches[bi]]
        n = len(depths)

        # Prediction and ground truth are both in the metric's keypoint layout.
        t_all = act.cpu().numpy()
        m_all = amask.cpu().numpy()

        # One noise per (unit, anchor) pair, identical at every depth and seeded as in the main model's test.
        noise = torch.stack([
            torch.randn(cfg.action.chunk_size, cfg.action.dim,
                        generator=torch.Generator().manual_seed(
                            a.seed * 1_000_003 + pid))
            for pid in pair_ids]).to(dev)
        with torch.no_grad():
            # Every step is valid here (one depth per batch); the mask keeps padded steps out of the memory.
            pred = model.predict_from_noise(steps, noise, step_valid=_valid)
            p_all = model.unnormalise_action(pred.float()).cpu().numpy()

        for k in range(n):
            source = sources[k]
            for view in VIEWS:
                e = errors(p_all[k], t_all[k], m_all[k], view)
                if e is None:
                    continue
                per_rows.append((int(pair_ids[k]), str(batch["episode"][k]), source, d, view,
                                 float(e.get("wrist_translation", float("nan"))),
                                 float(e.get("wrist_rotation", float("nan"))),
                                 float(e.get("fingers", float("nan")))))
                if d == BASE_DEPTH:
                    nsamp[view] += 1
                    src_nsamp[source][view] += 1
                for name, v in e.items():
                    if not np.isnan(v):
                        acc[view][d][name][0] += v; acc[view][d][name][1] += 1
                        src_acc[source][view][d][name][0] += v
                        src_acc[source][view][d][name][1] += 1

        if rank == 0 and bi % every == 0:
            done = sum(len(b) for b in batches[:bi + 1])
            tot = sum(len(b) for b in batches)
            mem = torch.cuda.max_memory_allocated() / 1024**3 if dev.type == "cuda" else 0
            log(f"    {done:,}/{tot:,}  {done / max(time.time() - t0, 1e-9):.1f} eval/s  "
                f"peak {mem:.1f} GB")

    if per_rows:
        # The per-sample file is a by-product: failing to write it must not lose the summary.
        _p = os.path.splitext(a.out)[0] + f".per_sample_rank{rank}.npz"
        os.makedirs(os.path.dirname(_p), exist_ok=True)
        try:
            np.savez_compressed(
                _p,
                pair=np.asarray([r[0] for r in per_rows], np.int32),
                episode=np.asarray([r[1] for r in per_rows]),
                source=np.asarray([r[2] for r in per_rows]),
                depth=np.asarray([r[3] for r in per_rows], np.int8),
                view=np.asarray([r[4] for r in per_rows]),
                wrist_translation=np.asarray([r[5] for r in per_rows], np.float32),
                wrist_rotation=np.asarray([r[6] for r in per_rows], np.float32),
                fingers=np.asarray([r[7] for r in per_rows], np.float32))
            log(f"  per-sample errors -> {_p}  ({len(per_rows):,} rows)")
        except Exception as _e:
            log(f"  could not write the per-sample errors ({type(_e).__name__}: {_e}); the summary is unaffected")

    if world > 1:
        def red_depth(store):
            for v in VIEWS:
                for dd in store[v]:
                    for nm in store[v][dd]:
                        t = torch.tensor(store[v][dd][nm], dtype=torch.float64, device=dev)
                        dist.all_reduce(t); store[v][dd][nm] = t.tolist()

        def red_n(store):
            for v in VIEWS:
                t = torch.tensor([store[v]], dtype=torch.float64, device=dev)
                dist.all_reduce(t); store[v] = int(t.item())

        red_depth(acc); red_n(nsamp)
        for nm in source_names:
            red_depth(src_acc[nm]); red_n(src_nsamp[nm])

    if rank != 0:
        return

    div = lambda v: (v[0] / v[1] if v[1] else None)

    def views(accum, samples):
        return {v: {"samples": samples[v],
                    "depth": {str(dd): {n: div(x) for n, x in accum[v][dd].items()}
                              for dd in accum[v]}}
                for v in VIEWS}

    res = {
        "checkpoint": a.checkpoint, "seed": a.seed,
        "model": "memoryvla_official", "action_type": cfg.action.action_type,
        "statistics": cfg.data.stats_path,
        "candidate_pairs": all_n_pairs, "pairs": len(selected), "evaluations": len(plan),
        "units": {g[0]: g[4] for g in GROUPS},
        "note": ("keypoints in, keypoints out, no conversion. The truth is the data's own "
                 "joints_manospace, the same truth the main model is scored against."),
        "views": views(acc, nsamp),
        "source": {nm: {"candidate_pairs": all_pairs_by_source[nm],
                        "pairs": selected_by_source[nm],
                        "views": views(src_acc[nm], src_nsamp[nm])}
                   for nm in source_names},
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)

    w = 24
    log(f"\n  {'':<22}" + "".join(f"{g[0] + ' (' + g[4] + ')':>{w}}" for g in GROUPS))
    for v in VIEWS:
        r = res["views"][v]
        log(f"  view {v:<17}" + f"  n={r['samples']:,}")
        rows = []
        for dd in sorted(r["depth"]):
            rows.append((f"model depth {dd}", r["depth"][dd]))
        for label, row in rows:
            log(f"    {label:<20}" + "".join(
                f"{(f'{row[g[0]]:.6f}' if row.get(g[0]) is not None else '-'):>{w}}"
                for g in GROUPS))
    log(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
