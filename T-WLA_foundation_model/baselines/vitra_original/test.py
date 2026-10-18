#!/usr/bin/env python
"""Test the VITRA baseline with the metric of the main model (`metric.py`).

    python baselines/vitra_original/test.py --out results/vitra.json
    torchrun --nproc_per_node=4 baselines/vitra_original/test.py --out results/vitra.json

The model predicts MANO joint angles. The prediction is converted to keypoints by forward
kinematics (`angle_fk.py`) and scored against the data's own `joints_manospace`.
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
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(BASELINES)), "checkpoints", "VITRA.pt")   # where the released weights go
DEFAULT_CONFIG = os.path.join(HERE, "configs", "test.json")

from data import dataset_class                                            # noqa: E402
from metric import (ANCHORS_PER_UNIT, GROUPS, MAX_DEPTH, VIEWS, build_plan,  # noqa: E402
                    errors, limit_plan_per_source, stride_plan_per_source)
from v4_bridge import (load_video_list, make_collate, resolve_paths,      # noqa: E402
                       sources_from_config, to_observations)
from vitra_original import (Config, Observation, PaliGemmaPromptTokenizer,  # noqa: E402
                            VitraOriginalModel, from_json)

del ANCHORS_PER_UNIT, MAX_DEPTH          # imported but not used here


def load_config(path: str) -> tuple[dict, Config]:
    """The run json -> (resolved json, Config). Environment variables in the two weight paths are expanded."""
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
    if cfg.action.action_type == "angle":
        # The model takes its state in the angle layout and the metric is scored on
        # keypoints, so each sample has to carry both.
        cfg.action.dual_layout = True
    return C, cfg


def to_keypoints(angle: np.ndarray, betas_l, betas_r, fk) -> np.ndarray:
    from angle_fk import angle_to_keypoints
    return angle_to_keypoints(angle, betas_l, betas_r, fk)


class Betas:
    """MANO shape (left, right) per episode, read from the annotation file and cached.

    The loader's state does not carry the betas, and forward kinematics needs them. A
    hand the episode does not annotate gets zeros; the metric masks it out.
    """

    def __init__(self, label_folder: str):
        self.root = label_folder
        self.cache: dict[str, tuple] = {}

    def __call__(self, episode: str):
        hit = self.cache.get(episode)
        if hit is None:
            d = np.load(os.path.join(self.root, f"{episode}.npy"),
                        allow_pickle=True).item()
            out = []
            for side in ("left", "right"):
                h = d.get(side)
                out.append(np.asarray(h["beta"], np.float64)
                           if isinstance(h, dict) and "beta" in h else np.zeros(10))
            hit = self.cache[episode] = tuple(out)
        return hit


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
    ap.add_argument("--baseline-only", action="store_true",
                    help="no model, no GPU: the two hand-still baselines only. This is the "
                         "run that checks `metric.py` against the main model's numbers.")
    ap.add_argument("--mano-weights",
                    default=os.path.expandvars("${VLA_WEIGHTS}"),
                    help="parent of `mano/`; read only when action_type is 'angle'")
    ap.add_argument("--depths", default="0",
                    help="default '0': this model has no memory, so every depth returns "
                         "the same number. The sample set is unaffected -- "
                         "`MIN_ANCHOR = MAX_DEPTH = 8` selects the same (unit, anchor) "
                         "pairs whatever depths are run, so these numbers sit on exactly "
                         "the pairs the main model's depth-0 column was measured on.")
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
    angle = cfg.action.action_type == "angle"
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
                             f"{len(selected)} pairs -- the baseline would double-count")
    log(f"  --depths {sorted(want)}: {len(plan):,} evaluations, "
        f"{len(selected):,} pairs; baseline accumulates at depth {BASE_DEPTH}")
    ds.set_eval_plan(plan)

    model = None
    if not a.baseline_only:
        model = VitraOriginalModel(cfg, load_weights=False, dtype=torch.float32).to(dev)
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        log(f"  checkpoint {a.checkpoint}")

    fk = None
    # One per source: a batch mixes episodes of several datasets, and each has its own label folder.
    betas_by_src = {s["name"]: Betas(s["label_folder"]) for s in src_cfgs}
    if angle:
        from angle_fk import ManoFK
        fk = ManoFK(a.mano_weights)

    collate = make_collate(cfg, PaliGemmaPromptTokenizer(cfg))
    by_depth: dict[int, list[int]] = {}
    for i, (_u, _s, ln, _p) in enumerate(plan):
        by_depth.setdefault(ln - 1, []).append(i)
    # Batches are cut within a depth, so one batch holds one depth.
    batches = [g[j:j + a.batch_size] for d in sorted(by_depth)
               for g in [by_depth[d]] for j in range(0, len(g), a.batch_size)]
    if a.baseline_only:
        batches = [b for b in batches if plan[b[0]][2] - 1 == BASE_DEPTH]
    batches = batches[rank::world]
    log(f"  {len(batches):,} batches of up to {a.batch_size} on this rank")

    dl = torch.utils.data.DataLoader(ds, batch_sampler=batches,
                                     num_workers=a.num_workers, collate_fn=collate)

    def new_acc():
        return {v: {d: {g[0]: [0.0, 0] for g in GROUPS}
                    for d in sorted(by_depth)} for v in VIEWS}

    def new_base():
        return {v: {g[0]: [0.0, 0] for g in GROUPS} for v in VIEWS}

    source_names = [s.name for s in ds.sources]
    # Per-sample errors (pair, episode, source, depth, view and the three errors), so the results can be regrouped
    # without a rerun.
    per_rows = []
    acc, base, base_fk = new_acc(), new_base(), new_base()
    src_acc = {n: new_acc() for n in source_names}
    src_base = {n: new_base() for n in source_names}
    src_base_fk = {n: new_base() for n in source_names}
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

        if angle:
            # The truth in the metric's keypoints layout, from the same window. `act` and
            # `amask` above are in the angle layout.
            act = batch["actions_alt"].to(dev, torch.float32)
            amask = batch["action_mask_alt"].to(dev)
            st_metric = batch["state_alt"]
            bl_ = [betas_by_src[src](e)
                   for src, e in zip(batch["source"], batch["episode"])]
            b_l = np.stack([x[0] for x in bl_])
            b_r = np.stack([x[1] for x in bl_])
        else:
            st_metric = batch["state"]
        t_all = act.cpu().numpy()
        m_all = amask.cpu().numpy()
        st_angle = batch["state"].numpy()
        st_kp = st_metric.numpy()

        # Static reference of an angle model: the state's pose held for the whole chunk,
        # so its fingers are FK of the state's `pose_euler`. The wrist channels passed in
        # are the state's absolute pose; only the finger block of the result is read.
        kp_still = None
        if angle and d == BASE_DEPTH:
            kp_still = to_keypoints(
                np.concatenate([st_angle[:, -1, 0:102][:, None, :],
                                np.zeros((n, 1, 90), np.float64)], axis=2),
                b_l, b_r, fk)[:, 0]

        p_all = None
        if model is not None:
            # One noise per (unit, anchor), the same at every depth and seeded as in the
            # main model's test.
            noise = torch.stack([
                torch.randn(cfg.action.chunk_size, cfg.action.dim,
                            generator=torch.Generator().manual_seed(
                                a.seed * 1_000_003 + pid))
                for pid in pair_ids]).to(dev)
            with torch.no_grad():
                pred = model.predict_from_noise(steps, noise)
                p_all = model.unnormalise_action(pred.float()).cpu().numpy()
            if angle:
                p_all = to_keypoints(p_all, b_l, b_r, fk)

        for k in range(n):
            source = sources[k]
            if p_all is not None:
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
            if d != BASE_DEPTH:
                continue

            # The two static references. The wrist channels are increments, so "does not
            # move" is zero. The fingers are copied from the state's `joints_manospace`
            # (`baseline_static`) or come from FK of the state's pose
            # (`baseline_static_fk`); the difference is the error floor of the conversion.
            bl = np.zeros_like(t_all[k])
            bl[:, 6:69] = st_kp[k, -1][6:69]
            bl[:, 75:138] = st_kp[k, -1][75:138]
            for view in VIEWS:
                e = errors(bl, t_all[k], m_all[k], view)
                if e is None:
                    continue
                if p_all is None:
                    nsamp[view] += 1
                    src_nsamp[source][view] += 1
                for name, v in e.items():
                    if not np.isnan(v):
                        base[view][name][0] += v; base[view][name][1] += 1
                        src_base[source][view][name][0] += v
                        src_base[source][view][name][1] += 1

            if angle:
                kp = kp_still[k]
                blf = np.zeros_like(t_all[k])
                blf[:, 6:69] = kp[6:69]
                blf[:, 75:138] = kp[75:138]
                for view in VIEWS:
                    e = errors(blf, t_all[k], m_all[k], view)
                    if e is None:
                        continue
                    for name, v in e.items():
                        if not np.isnan(v):
                            base_fk[view][name][0] += v; base_fk[view][name][1] += 1
                            src_base_fk[source][view][name][0] += v
                            src_base_fk[source][view][name][1] += 1

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

        def red_flat(store):
            for v in VIEWS:
                for nm in store[v]:
                    t = torch.tensor(store[v][nm], dtype=torch.float64, device=dev)
                    dist.all_reduce(t); store[v][nm] = t.tolist()

        def red_n(store):
            for v in VIEWS:
                t = torch.tensor([store[v]], dtype=torch.float64, device=dev)
                dist.all_reduce(t); store[v] = int(t.item())

        red_depth(acc); red_flat(base); red_flat(base_fk); red_n(nsamp)
        for nm in source_names:
            red_depth(src_acc[nm]); red_flat(src_base[nm]); red_flat(src_base_fk[nm])
            red_n(src_nsamp[nm])

    if rank != 0:
        return

    div = lambda v: (v[0] / v[1] if v[1] else None)

    def views(accum, b, bfk, samples):
        return {v: {"samples": samples[v],
                    "baseline_static": {n: div(x) for n, x in b[v].items()},
                    "baseline_static_fk": ({n: div(x) for n, x in bfk[v].items()}
                                           if angle else None),
                    "depth": {str(dd): {n: div(x) for n, x in accum[v][dd].items()}
                              for dd in accum[v]}}
                for v in VIEWS}

    res = {
        "checkpoint": a.checkpoint, "seed": a.seed,
        "model": "vitra_original", "action_type": cfg.action.action_type,
        "statistics": cfg.data.stats_path,
        "candidate_pairs": all_n_pairs, "pairs": len(selected), "evaluations": len(plan),
        "units": {g[0]: g[4] for g in GROUPS},
        "note": ("prediction converted angle->keypoints by MANO FK; truth is the data's "
                 "own joints_manospace, the same truth the main model is scored against. "
                 "baseline_static copies the state's joints_manospace (the main model's "
                 "form); baseline_static_fk is FK(state's pose_euler), which "
                 "is what zero motion means for an angle model. Their difference is the "
                 "FK floor.") if angle else None,
        "views": views(acc, base, base_fk, nsamp),
        "source": {nm: {"candidate_pairs": all_pairs_by_source[nm],
                        "pairs": selected_by_source[nm],
                        "views": views(src_acc[nm], src_base[nm], src_base_fk[nm],
                                       src_nsamp[nm])}
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
        # The two static references are written to the json (the FK floor is their difference) and not printed.
        rows = []
        for dd in sorted(r["depth"]):
            rows.append((f"model depth {dd}", r["depth"][dd]))
        for label, row in rows:
            log(f"    {label:<20}" + "".join(
                f"{(f'{row[g[0]]:.6f}' if row.get(g[0]) is not None else '-'):>{w}}"
                for g in GROUPS))
    if False and angle:                      # disabled: the floor is in the json
        c = res["views"]["all"]["baseline_static"]["fingers"]
        k = res["views"]["all"]["baseline_static_fk"]["fingers"]
        # `errors()` already applied the scale of GROUPS, so these are centimetres.
        log(f"\n  FK floor on this sample set: {k - c:.4f} cm ({10 * (k - c):.3f} mm) "
            f"-- the fingers column of an angle model carries it; the main model does not. "
            f"(copy {c:.6f} vs FK {k:.6f})")
    log(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
