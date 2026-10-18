"""Test-set evaluation of the predicted actions, at memory-bank depths 0 to 8.

    torchrun --nproc_per_node=2 eval_test.py --config configs/train.json \
        --checkpoint ../checkpoints/T-WLA-FD.pt --out test_depth.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist

import train
from dataloader import (SequenceDataset, load_video_list, make_collate,
                        sources_from_config, to_observations)
from dataloader.sampling import phase_anchors, supervisable_anchors

MAX_DEPTH = 8
MIN_ANCHOR = MAX_DEPTH          # the earliest anchor that can carry a full bank
ANCHORS_PER_UNIT = 4

# (name, left slice, right slice, scale to the reported unit, unit). Wrist: increments per
# action step, in metres and radians (xyz Euler). Fingers: absolute keypoints in metres.
GROUPS = [
    ("wrist_translation", slice(0, 3), slice(69, 72), 100.0, "cm"),
    ("wrist_rotation", slice(3, 6), slice(72, 75), None, "deg"),
    ("fingers", slice(6, 69), slice(75, 138), 100.0, "cm"),
]


def build_plan(ds) -> list[tuple[int, int, int, int]]:
    """One entry per (unit, anchor, depth): (unit index, window start, window length, pair id).
    The pair id is shared by all depths of one (unit, anchor) and seeds its starting noise."""
    plan, pairs = [], []
    for ui, u in enumerate(ds.units):
        n_sup = supervisable_anchors(phase_anchors(u.n_frames, u.phase, ds.stride),
                                     u.n_frames)
        lo, hi = MIN_ANCHOR, n_sup - 1
        if hi < lo:
            continue
        n = hi - lo + 1
        picks = (list(range(lo, hi + 1)) if n <= ANCHORS_PER_UNIT else
                 sorted({int(round(x)) for x in
                         np.linspace(lo, hi, ANCHORS_PER_UNIT)}))
        for a in picks:
            pairs.append((ui, a))
    # ordered by depth, so that a batch can hold a single depth
    for d in range(MAX_DEPTH + 1):
        for pid, (ui, a) in enumerate(pairs):
            plan.append((ui, a - d, d + 1, pid))
    return plan


def limit_plan_per_source(plan: list[tuple[int, int, int, int]],
                          pair_source: dict[int, str],
                          limit: int) -> list[tuple[int, int, int, int]]:
    """Keep at most `limit` pairs from every source, at all their depths."""
    if limit <= 0:
        return plan
    keep, seen = set(), collections.Counter()
    for pid in sorted({x[3] for x in plan}):
        source = pair_source[pid]
        if seen[source] < limit:
            keep.add(pid)
            seen[source] += 1
    return [x for x in plan if x[3] in keep]


def stride_plan_per_source(plan: list[tuple[int, int, int, int]],
                           pair_source: dict[int, str],
                           stride: int) -> list[tuple[int, int, int, int]]:
    """Keep every `stride`-th pair within each source."""
    if stride <= 1:
        return plan
    by_source: dict[str, list[int]] = collections.defaultdict(list)
    for pid in sorted({x[3] for x in plan}):
        by_source[pair_source[pid]].append(pid)
    keep = {pid for pids in by_source.values() for pid in pids[::stride]}
    return [x for x in plan if x[3] in keep]


@torch.no_grad()
def predict(model, variant: str, steps, noise: torch.Tensor) -> torch.Tensor:
    """Fill the bank with `steps[:-1]`, then integrate the last step from `noise`.
    Returns the normalised actions [B, T, 192]."""
    from importlib import import_module
    pkg = train.VARIANTS[variant]
    is_dp = ".ablation." in pkg                 # DDPM action head instead of flow matching
    flow = import_module(f"{pkg}." + ("diffusion" if is_dp else "flow"))
    cfg = model.cfg
    dev = steps[-1].images.device
    b = steps[-1].images.shape[0]
    model.reset(b, dev)
    bank = model._bank

    for obs in steps[:-1]:                      # fill the bank
        out = model.encode(obs, bank)
        bank.append(out if (is_dp or variant == "d") else out[0])

    obs = steps[-1]
    if is_dp:
        m_f = model.encode(obs, bank)
        ctx = model.expert.prepare(
            m_f, model.state_with_mask(obs.state, obs.state_mask)
                 if cfg.action.state_in_sequence else None)
    elif variant == "d":
        m_f = model.encode(obs, bank)
        state_cond = (model.state_with_mask(obs.state, obs.state_mask)
                      if cfg.action.direct_state_to_expert else None)
        ctx = model.expert.prepare(m_f, state_cond)
    else:
        m_f, cond, pad = model.encode(obs, bank)
        ctx = model.expert.prepare(cond, pad)

    mask = flow.component_action_mask(b, cfg.action.chunk_size, cfg.action.dim,
                                      cfg.flow, dev)
    x = noise.to(dev, m_f.dtype).clone()
    x.masked_fill_(~mask, 0)                    # padding slots start at 0, as in select_action
    if is_dp:
        return flow.ddpm_sample(lambda z, ts: model.expert.epsilon(ctx, z, mask, ts),
                                x, model.alpha_bar, cfg.flow.num_inference_steps)
    return flow.euler_integrate(lambda z, tau: model.expert.velocity(ctx, z, mask, tau),
                                x, cfg.flow.num_inference_steps)


# Three views of one run. A chunk near the end of an episode has fewer than 16 valid steps.
#   "all"  every valid step, which is what the training loss averages over
#   "8"    the first 8 steps (the executed part), over the samples that have them
#   "16"   the whole chunk, over the samples that have all 16 steps
VIEWS = ("all", "8", "16")


KP_KEEP = {"left": list(range(1, 21)), "right": list(range(1, 21))}


def set_kp_keep(action_std: np.ndarray, log=print) -> None:
    """Exclude from the finger metric the keypoints with constant coordinates: their sigma is
    the placeholder 1.0, so their unnormalised prediction is not on the scale of the others."""
    for hand, sl in (("left", GROUPS[2][1]), ("right", GROUPS[2][2])):
        s = np.asarray(action_std[sl], np.float64).reshape(-1, 3)
        const = [i for i in range(s.shape[0]) if bool((s[i] == 1.0).all())]
        KP_KEEP[hand] = [i for i in range(s.shape[0]) if i not in const]
        if const != [0]:
            log(f"[fingers] {hand}: keypoints {const} have the sigma placeholder (constant "
                f"coordinates) and are excluded; {len(KP_KEEP[hand])} points measured")


def errors(pred: np.ndarray, true: np.ndarray, mask: np.ndarray,
           view: str) -> dict[str, float] | None:
    """Mean error per group for one view, or None when the sample has fewer valid steps than
    the view needs. `pred` and `true` are [T, 192], unnormalised. The wrist increments are
    accumulated first: translation error is a distance, rotation error a geodesic angle."""
    from scipy.spatial.transform import Rotation as R
    horizon = pred.shape[0] if view == "all" else int(view)
    out = {}
    for name, ls, rs, scale, _unit in GROUPS:
        vals, n = 0.0, 0
        for sl in (ls, rs):
            m = mask[:horizon, sl]
            if not m.any():
                continue                       # no valid step for this hand
            rows = m.all(1)
            if view != "all" and not rows.all():
                return None                    # the chunk is shorter than this view needs
            if not rows.any():
                continue
            p, t = pred[:horizon, sl], true[:horizon, sl]
            if name == "wrist_translation":
                e = np.linalg.norm(np.cumsum(p, 0) - np.cumsum(t, 0), axis=1) * scale
            elif name == "wrist_rotation":
                ep = R.from_euler("xyz", p); et = R.from_euler("xyz", t)
                acc_p = acc_t = R.identity()
                e = np.empty(p.shape[0])
                for k in range(p.shape[0]):
                    acc_p = ep[k] * acc_p
                    acc_t = et[k] * acc_t
                    e[k] = np.degrees((acc_p * acc_t.inv()).magnitude())
            else:
                # constant keypoints are left out; by default keypoint 0, the wrist
                keep = KP_KEEP["left" if sl is ls else "right"]
                d = (p - t).reshape(p.shape[0], -1, 3)[:, keep, :]
                e = np.linalg.norm(d, axis=2).mean(1) * scale
            vals += float(e[rows].sum()); n += int(rows.sum())
        out[name] = vals / n if n else float("nan")
    return out


@torch.no_grad()
def run(a) -> None:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        # the final all_reduce waits for the slowest rank; EVAL_DIST_TIMEOUT_MIN is in minutes
        import datetime
        dist.init_process_group("nccl", timeout=datetime.timedelta(
            minutes=int(os.environ.get("EVAL_DIST_TIMEOUT_MIN", "10"))))
        torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    log = (lambda *m: print(*m, flush=True)) if rank == 0 else (lambda *m: None)

    with open(a.config) as f:
        C = json.load(f)
    train.validate_run_config(C)
    C = train.resolve_paths(C)
    variant = C["variant"]
    Cfg, Model, Obs = train.load_variant(variant)
    cfg = Cfg()
    # no world head: its config is dropped here, its `world.*` tensors are skipped below
    C.get("config", {}).pop("world", None)
    train.apply_overrides(cfg, C.get("config", {}))
    cfg.validate()
    source_configs = C["data"]["sources"]

    ds = SequenceDataset(
        sources_from_config(source_configs), cfg=cfg,
        video_whitelist={s["name"]: load_video_list(
            os.path.join(s["split_dir"], "test.txt")) for s in source_configs},
        load_images=True, seed=a.seed, training=False, deterministic=True)

    plan = build_plan(ds)
    pair_source = {pid: ds.sources[ds.units[ui].source].name
                   for ui, _start, _length, pid in plan}
    all_pair_ids = set(pair_source)
    all_n_pairs = len(all_pair_ids)
    all_pairs_by_source = collections.Counter(pair_source.values())
    log(f"  test units {len(ds):,}  ->  {all_n_pairs:,} (unit, anchor) pairs "
        f"x {MAX_DEPTH + 1} depths = {len(plan):,} evaluations")
    log("  pairs by source: " + "  ".join(
        f"{name}={all_pairs_by_source[name]:,}" for name in all_pairs_by_source))
    if a.limit:
        plan = limit_plan_per_source(plan, pair_source, a.limit)
        log(f"  --limit {a.limit} per source: {len(plan):,} evaluations")
    if a.stride > 1:
        before_stride = len({x[3] for x in plan})
        plan = stride_plan_per_source(plan, pair_source, a.stride)
        log(f"  --stride {a.stride}: {len({x[3] for x in plan}):,} pairs kept "
            f"(1/{a.stride} of {before_stride:,} per source), "
            f"{len(plan):,} evaluations")
    want = None
    if a.depths:
        want = {int(s) for s in a.depths.split(",") if s.strip() != ""}
        bad = want - set(range(MAX_DEPTH + 1))
        if bad:
            raise SystemExit(f"--depths out of range: {sorted(bad)}")
        plan = [x for x in plan if (x[2] - 1) in want]
        log(f"  --depths {sorted(want)}: {len(plan):,} evaluations")
    if not plan:
        raise SystemExit("the plan is empty after filtering")
    selected_pair_ids = {x[3] for x in plan}
    selected_pairs_by_source = collections.Counter(pair_source[p] for p in selected_pair_ids)
    # every pair appears exactly once at the smallest depth: baseline and counts are taken there
    BASE_DEPTH = min(x[2] - 1 for x in plan)
    n_at_base = sum(1 for x in plan if x[2] - 1 == BASE_DEPTH)
    if n_at_base != len(selected_pair_ids):
        raise AssertionError(f"depth {BASE_DEPTH} holds {n_at_base} entries but there are "
                             f"{len(selected_pair_ids)} pairs -- the baseline would "
                             f"double-count or miss pairs")
    log(f"  baseline & sample counts accumulate at depth {BASE_DEPTH} "
        f"({n_at_base:,} pairs, exactly once each)")
    ds.set_eval_plan(plan)

    if a.baseline_only:
        model, ck = None, {"step": -1}
        log(f"  --baseline-only: no model, no GPU. Depth {BASE_DEPTH} batches only -- the "
            f"baseline is depth-independent and every pair appears there exactly once.")
    else:
        model = Model(cfg, load_weights=False, dtype=torch.float32).to(dev)
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v for k, v in ck["model"].items()
                               if not k.startswith("world.")})
        model.eval()
        set_kp_keep(model.action_std.cpu().numpy(), log)
        log(f"  checkpoint {a.checkpoint}")

    from transformers import AutoTokenizer
    collate = make_collate(cfg, AutoTokenizer.from_pretrained(cfg.vlm.model_id))
    # one depth per batch; the batches are then shared out across the ranks
    by_depth: dict[int, list[int]] = {}
    for i, (_ui, _st, ln, _pid) in enumerate(plan):
        by_depth.setdefault(ln - 1, []).append(i)
    batches = [g[j:j + a.batch_size] for d in sorted(by_depth)
               for g in [by_depth[d]] for j in range(0, len(g), a.batch_size)]
    if a.baseline_only:
        batches = [b for b in batches if plan[b[0]][2] - 1 == BASE_DEPTH]
    batches = batches[rank::world]
    log(f"  {len(batches):,} batches of up to {a.batch_size} on this rank")

    dl = torch.utils.data.DataLoader(
        ds, batch_sampler=batches, num_workers=a.num_workers, collate_fn=collate)

    # [view][depth][group] -> [sum of errors, count], over all samples and per source
    def new_acc():
        return {v: {d: {g[0]: [0.0, 0] for g in GROUPS}
                    for d in range(MAX_DEPTH + 1)} for v in VIEWS}

    def new_base():
        return {v: {g[0]: [0.0, 0] for g in GROUPS} for v in VIEWS}

    source_names = [s.name for s in ds.sources]
    source_to_index = {name: i for i, name in enumerate(source_names)}
    acc, base = new_acc(), new_base()
    source_acc = {name: new_acc() for name in source_names}
    source_base = {name: new_base() for name in source_names}
    nsamp = {v: 0 for v in VIEWS}          # samples that have enough steps for each view
    source_nsamp = {name: {v: 0 for v in VIEWS} for name in source_names}
    # raw dump: normalised predictions per (pair, depth); truth and mask once per pair
    raw = {"pred": [], "pid": [], "depth": [],
           "truth": {}, "mask": {}, "source": {}} if not a.no_raw else None

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    every = max(1, len(batches) // 20)          # about 20 progress lines
    t0 = time.time()

    for bi, batch in enumerate(dl):
        steps, valid, act, amask = to_observations(batch, dev, Obs, torch.float32)
        sources = batch["source"]
        depths = [int(l) - 1 for l in batch["length"]]
        if len(set(depths)) != 1:
            raise AssertionError(f"a batch must hold one depth, got {sorted(set(depths))} "
                                 f"-- was the plan re-ordered or shuffled?")
        d = depths[0]
        pair_ids = [plan[i][3] for i in batches[bi]]
        if a.baseline_only:
            t_all = act.cpu().numpy(); m_all = amask.cpu().numpy()
            for k in range(len(depths)):
                source = sources[k]
                st = batch["state"][k, -1].numpy()
                bl = np.zeros_like(t_all[k])
                bl[:, 6:69] = st[6:69]; bl[:, 75:138] = st[75:138]
                for view in VIEWS:
                    e = errors(bl, t_all[k], m_all[k], view)
                    if e is None:
                        continue
                    nsamp[view] += 1
                    source_nsamp[source][view] += 1
                    for name, v in e.items():
                        if not np.isnan(v):
                            base[view][name][0] += v; base[view][name][1] += 1
                            source_base[source][view][name][0] += v
                            source_base[source][view][name][1] += 1
            if rank == 0 and bi % every == 0:
                dn = sum(len(b) for b in batches[:bi + 1])
                log(f"    {dn:,}/{sum(len(b) for b in batches):,}  "
                    f"{dn / max(time.time() - t0, 1e-9):.0f} pair/s")
            continue

        # the same noise at every depth of a (unit, anchor)
        noise = torch.stack([
            torch.randn(cfg.action.chunk_size, cfg.action.dim,
                        generator=torch.Generator().manual_seed(a.seed * 1_000_003 + pid))
            for pid in pair_ids]).to(dev)
        pred = predict(model, variant, steps, noise)
        p_all = model.unnormalise_action(pred.float()).cpu().numpy()
        t_all = act.cpu().numpy()
        m_all = amask.cpu().numpy()

        if raw is not None:
            raw["pred"].append(pred.float().cpu().numpy().astype(np.float16))
            raw["pid"].extend(pair_ids); raw["depth"].extend([d] * len(pair_ids))
            for k, pid in enumerate(pair_ids):
                if pid not in raw["truth"]:
                    raw["truth"][pid] = t_all[k].astype(np.float16)
                    raw["mask"][pid] = m_all[k]
                    raw["source"][pid] = source_to_index[sources[k]]

        for k in range(len(depths)):
            source = sources[k]
            for view in VIEWS:
                e = errors(p_all[k], t_all[k], m_all[k], view)
                if e is None:
                    continue
                if d == BASE_DEPTH:
                    nsamp[view] += 1
                    source_nsamp[source][view] += 1
                for name, v in e.items():
                    if not np.isnan(v):
                        acc[view][d][name][0] += v; acc[view][d][name][1] += 1
                        source_acc[source][view][d][name][0] += v
                        source_acc[source][view][d][name][1] += 1
            # static-hand baseline, counted once per pair
            if d == BASE_DEPTH:
                st = batch["state"][k, -1].numpy()
                bl = np.zeros_like(t_all[k])
                bl[:, 6:69] = st[6:69]                       # copy the current hand
                bl[:, 75:138] = st[75:138]                   # wrist deltas stay zero
                for view in VIEWS:
                    e = errors(bl, t_all[k], m_all[k], view)
                    if e is None:
                        continue
                    for name, v in e.items():
                        if not np.isnan(v):
                            base[view][name][0] += v; base[view][name][1] += 1
                            source_base[source][view][name][0] += v
                            source_base[source][view][name][1] += 1
        if rank == 0 and bi % every == 0:
            done = sum(len(b) for b in batches[:bi + 1])
            tot = sum(len(b) for b in batches)
            mem = torch.cuda.max_memory_allocated() / 1024**3 if dev.type == "cuda" else 0
            log(f"    {done:,}/{tot:,}  {done / max(time.time() - t0, 1e-9):.1f} eval/s  "
                f"peak {mem:.1f} GB")

    if world > 1:
        def reduce_acc(store):
            for view in VIEWS:
                for d in store[view]:
                    for name in store[view][d]:
                        tt = torch.tensor(store[view][d][name],
                                          dtype=torch.float64, device=dev)
                        dist.all_reduce(tt)
                        store[view][d][name] = tt.tolist()

        def reduce_base(store):
            for view in VIEWS:
                for name in store[view]:
                    tt = torch.tensor(store[view][name], dtype=torch.float64, device=dev)
                    dist.all_reduce(tt)
                    store[view][name] = tt.tolist()

        def reduce_samples(store):
            for view in VIEWS:
                tt = torch.tensor([store[view]], dtype=torch.float64, device=dev)
                dist.all_reduce(tt)
                store[view] = int(tt.item())

        reduce_acc(acc)
        reduce_base(base)
        reduce_samples(nsamp)
        for source in source_names:
            reduce_acc(source_acc[source])
            reduce_base(source_base[source])
            reduce_samples(source_nsamp[source])

    if raw is not None and raw["pred"]:
        pids = sorted(raw["truth"])
        out = os.path.splitext(a.out)[0] + f".raw_rank{rank}.npz"
        np.savez_compressed(
            out,
            pred=np.concatenate(raw["pred"]),               # [N, T, A] float16, normalised
            pred_pair=np.asarray(raw["pid"], np.int32),
            pred_depth=np.asarray(raw["depth"], np.int8),
            truth_pair=np.asarray(pids, np.int32),
            truth=np.stack([raw["truth"][i] for i in pids]),
            mask=np.stack([raw["mask"][i] for i in pids]),
            truth_source=np.asarray([raw["source"][i] for i in pids], np.int16),
            source_names=np.asarray(source_names),
            action_mean=model.action_mean.cpu().numpy() if model is not None else np.zeros(1),
            action_std=model.action_std.cpu().numpy() if model is not None else np.zeros(1))
        log(f"  raw -> {out}  ({os.path.getsize(out) / 1024**3:.2f} GB)")

    if rank == 0:
        div = lambda v: (v[0] / v[1] if v[1] else None)

        def views_result(accumulator, baseline, samples):
            return {v: {"samples": samples[v],
                        "baseline_static": {n: div(x)
                                            for n, x in baseline[v].items()},
                        "depth": {str(d): {n: div(x)
                                          for n, x in accumulator[v][d].items()}
                                  for d in accumulator[v]}}
                    for v in VIEWS}

        res = {"checkpoint": a.checkpoint, "step": int(ck.get("step", -1)), "variant": variant,
               "seed": a.seed, "candidate_pairs": all_n_pairs,
               "pairs": len(selected_pair_ids), "evaluations": len(plan),
               "units": {g[0]: g[4] for g in GROUPS},
               "views": views_result(acc, base, nsamp),
               "source": {
                   source: {
                       "candidate_pairs": all_pairs_by_source[source],
                       "pairs": selected_pairs_by_source[source],
                       "evaluations": sum(
                           pair_source[x[3]] == source for x in plan),
                       "views": views_result(source_acc[source], source_base[source],
                                             source_nsamp[source]),
                   }
                   for source in source_names
               }}
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        for v in VIEWS:
            lbl = {"all": "all valid steps (as in training)", "8": "first 8 steps (executed)",
                   "16": "full 16 steps"}[v]
            log(f"\n  === {lbl} · {nsamp[v]:,} samples ===")
            log(f"  {'':<10}" + "".join(f"{g[0]+' ('+g[4]+')':>24}" for g in GROUPS))
            b = res["views"][v]["baseline_static"]
            log(f"  {'baseline':<10}" +
                "".join(f"{b[g[0]]:>24.4f}" if b[g[0]] is not None else f"{'-':>24}"
                        for g in GROUPS))
            for d in range(MAX_DEPTH + 1):
                row = res["views"][v]["depth"][str(d)]
                # skip the depths that were not run
                if all(row[g[0]] is None for g in GROUPS):
                    continue
                log(f"  {'depth ' + str(d):<10}" +
                    "".join(f"{row[g[0]]:>24.4f}" if row[g[0]] is not None else f"{'-':>24}"
                            for g in GROUPS))
        log("\n  === per-source · first 8 steps · deepest evaluated bank ===")
        deepest = max(x[2] - 1 for x in plan)
        for source in source_names:
            src = res["source"][source]
            row = src["views"]["8"]["depth"][str(deepest)]
            values = "  ".join(
                f"{g[0]}={row[g[0]]:.4f}" if row[g[0]] is not None else f"{g[0]}=-"
                for g in GROUPS)
            log(f"  {source:<16} pairs={src['pairs']:,}  {values}")
        log(f"\n  -> {a.out}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-raw", action="store_true",
                    help="skip the raw dump of the predictions; it is written by default, "
                         "so that the metrics can be recomputed offline")
    ap.add_argument("--baseline-only", action="store_true",
                    help="skip the model and compute only the static-hand baseline; needs no GPU")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N (unit, anchor) pairs PER SOURCE; for a "
                         "multi-source smoke run")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth (unit, anchor) pair WITHIN EACH SOURCE. Pairs "
                         "are enumerated in unit order and units are grouped by episode, "
                         "so this spreads the subset over every source and video.")
    ap.add_argument("--depths", default="",
                    help="comma-separated depths to run, e.g. '8' for the full bank only. "
                         "Default: all 0..8. The baseline and the per-view sample count "
                         "are accumulated on the smallest depth present, where every pair "
                         "appears exactly once.")
    run(ap.parse_args())
