"""Shared-editor experiment protocol, training cache validation, and physical scoring."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .action_layout import ActionNormalizer, FULL_ACTION_GROUPS
from .direct_residual import DirectTactileResidualEditor
from .editor_backends import MemoryFreeTactileEditor, TactileEditorRouter

MODELS = ("smolvla", "vitra", "twla")
OPTIONAL_MODELS = ("memoryvla",)


def cache_models(cache):
    return tuple(name for name in (*MODELS, *OPTIONAL_MODELS) if name in cache["base_actions"])

METRICS = ("translation_cm", "rotation_deg", "joint_mpjpe_cm")


@dataclass(frozen=True)
class SharedEditorProtocol:
    dataset: str
    seed: int = 0
    batch_size: int = 64
    max_steps: int = 10000
    val_every: int = 250
    learning_rate: float = 1e-4
    primary_horizon: int = 8
    residual_bases: int = 8
    mode: str = "history_delta"

    def __post_init__(self):
        if self.dataset not in ("egotouch", "tachin"):
            raise ValueError("dataset must be egotouch or tachin")
        if self.primary_horizon not in (8, 16):
            raise ValueError("primary_horizon must be 8 or 16")
        if min(self.batch_size, self.max_steps, self.val_every, self.residual_bases, self.learning_rate) <= 0:
            raise ValueError("training budget and learning rate must be positive")

    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def paired_noise(sample_ids, seed, *, device="cpu"):
    """Stable noise independent of model, source ordering, and batch composition."""
    samples = []
    for key in sample_ids:
        number = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "little") % (2**63)
        samples.append(torch.randn(16, 192, generator=torch.Generator().manual_seed(number)))
    return torch.stack(samples).to(device)


def validate_cache(cache):
    """Reject data leakage, missing predictions, and unknown sensor conventions."""
    if cache.get("schema") != "table1.shared.v1":
        raise ValueError("unsupported cache schema")
    ids = cache["sample_id"]
    n = len(ids)
    if not n or len(set(ids)) != n:
        raise ValueError("sample_id must be nonempty and unique")
    shapes = {"target_action": (n, 16, 192), "action_mask": (n, 16, 192),
              "pressure": (n, 8, 2, 21, 21), "temporal_valid": (n, 8, 2),
              "pressure_seconds": (n, 8), "anchor_seconds": (n,),
              "pressure_masks": (2, 21, 21)}
    for key, shape in shapes.items():
        value = cache[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{key} must be a tensor of shape {shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values")
    for key in ("action_mask", "temporal_valid", "pressure_masks"):
        if cache[key].dtype != torch.bool:
            raise ValueError(f"{key} must be boolean")
    if cache["action_mask"][..., 138:].any():
        raise ValueError("padding cannot be valid")
    if not cache["action_mask"].flatten(1).any(1).all():
        raise ValueError("every sample must contain a valid action target")
    for start in (0, 69):
        m = cache["action_mask"][..., start:start + 69]
        if (m.any(-1) != m.all(-1)).any():
            raise ValueError("a hand must have a complete channel mask")
        live = m.all(-1)
        if (live[:, 1:] & ~live[:, :-1]).any():
            raise ValueError("valid action steps must form a prefix")
    valid_touch = cache["temporal_valid"].any(-1)
    if ((cache["pressure_seconds"] > cache["anchor_seconds"][:, None] + 1e-7) & valid_touch).any():
        raise ValueError("future tactile input leaks into the editor")
    if (cache["pressure_seconds"][:, 1:] < cache["pressure_seconds"][:, :-1]).any():
        raise ValueError("tactile history must be ordered by time")
    if cache["pressure_convention"] not in ("egotouch_raw", "canonical_right"):
        raise ValueError("declare egotouch_raw or canonical_right pressure convention")
    if cache["dataset"] == "tachin" and cache["pressure_convention"] != "canonical_right":
        raise ValueError("Tachin requires a documented mapping to canonical_right grids")
    if not cache.get("pressure_mapping_id"):
        raise ValueError("pressure_mapping_id must identify the sensor preprocessing")
    for key in ("sequence_id", "environment", "split"):
        if len(cache[key]) != n:
            raise ValueError(f"{key} must have one entry per sample")
    if set(cache["split"]) != {"train", "val", "test"}:
        raise ValueError("cache must contain train, val, and test")
    seen = {}
    for seq, split, env in zip(cache["sequence_id"], cache["split"], cache["environment"]):
        if seq in seen and seen[seq] != (split, env):
            raise ValueError("sequence crosses splits or environments")
        seen[seq] = (split, env)
    names = set(cache["base_actions"])
    if not set(MODELS).issubset(names) or names - set((*MODELS, *OPTIONAL_MODELS)):
        raise ValueError("shared comparison requires predictions for all three models")
    for model_id, action in cache["base_actions"].items():
        if tuple(action.shape) != (n, 16, 192) or not torch.isfinite(action).all():
            raise ValueError(f"invalid {model_id} physical keypoint predictions")
        if not cache.get("provenance", {}).get(model_id):
            raise ValueError(f"missing checkpoint provenance for {model_id}")
        if cache["provenance"][model_id].get("human_action_trained") is False:
            raise ValueError("untrained human action heads are smoke-only, not Table 1 predictions")
    return cache


def cache_identity(cache):
    """Hash data, splits, and prediction sources for checkpoint/evaluation matching."""
    h = hashlib.sha256()
    for key in ("dataset", "sample_id", "sequence_id", "environment", "split",
                "pressure_convention", "pressure_mapping_id", "provenance"):
        h.update(json.dumps(cache[key], sort_keys=True).encode())
    for key in ("target_action", "action_mask", "pressure", "temporal_valid",
                "pressure_seconds", "anchor_seconds", "pressure_masks"):
        a = cache[key].detach().cpu().contiguous()
        h.update(str(a.dtype).encode()); h.update(a.numpy().tobytes())
    for name in cache_models(cache):
        h.update(cache["base_actions"][name].detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def fit_normalizer(cache):
    """One normalizer from training targets, never model predictions or held-out data."""
    train = torch.tensor([s == "train" for s in cache["split"]])
    x = cache["target_action"][train].float()
    m = cache["action_mask"][train].bool()
    count = m.sum((0, 1))
    mean = (x * m).sum((0, 1)) / count.clamp_min(1)
    var = ((x - mean).square() * m).sum((0, 1)) / count.clamp_min(1)
    std = var.sqrt()
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    digest = hashlib.sha256(mean.numpy().tobytes() + std.numpy().tobytes()).hexdigest()
    return ActionNormalizer(mean, std, "train-keypoints:" + digest)


def build_shared_editor(protocol, pressure_masks, pressure_convention):
    # Inputs and targets are already normalized. Do not normalize a second time.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(protocol.seed)
        editor = DirectTactileResidualEditor(
            torch.zeros(192), torch.ones(192), pressure_masks.cpu(), torch.zeros(1, 8, 2, 21, 21),
            mode=protocol.mode, residual_bases=protocol.residual_bases,
            articulation_only=False, flip_left=pressure_convention == "egotouch_raw")
    return TactileEditorRouter(None, MemoryFreeTactileEditor(editor), route="shared")


def grouped_action_loss(pred, target, mask):
    """Equal group weight, reduced over valid hand-steps. Padding contributes nothing."""
    total, count = pred.sum() * 0, 0
    for start, stop in FULL_ACTION_GROUPS:
        live = mask[..., start:stop].all(-1)
        square = (pred[..., start:stop] - target[..., start:stop]).square().mean(-1)
        total = total + square[live].sum()
        count = count + live.sum()
    if int(count) == 0:
        raise ValueError("no valid action targets")
    return total / count


def physical_window_errors(pred, target, mask, *, horizon):
    """Accumulated wrist motion and palm-local joints 1..20, in cm/degrees.

    A strict horizon excludes each hand whose valid prefix is too short.
    Joint 0 is a fixed origin and is not re-subtracted from predicted local joints.
    """
    if horizon not in (8, 16):
        raise ValueError("horizon must be 8 or 16")
    p, t, m = [np.asarray(x) for x in (pred, target, mask)]
    if p.ndim != 3 or p.shape[1:] != (16, 192) or t.shape != p.shape or m.shape != p.shape:
        raise ValueError("actions and mask must be [N,16,192]")
    n = len(p)
    sums, counts = np.zeros((n, 3)), np.zeros(n)
    for lo in (0, 69):
        valid = m[:, :horizon, lo:lo + 69].all((1, 2))
        if not valid.any():
            continue
        a, b = p[valid, :horizon, lo:lo + 69], t[valid, :horizon, lo:lo + 69]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("non-finite prediction or target on valid frames")
        trans = np.linalg.norm(np.cumsum(a[..., :3] - b[..., :3], axis=1), axis=-1) * 100
        rp = Rotation.from_euler("xyz", a[..., 3:6].reshape(-1, 3)).as_matrix().reshape(-1, horizon, 3, 3)
        rt = Rotation.from_euler("xyz", b[..., 3:6].reshape(-1, 3)).as_matrix().reshape(-1, horizon, 3, 3)
        ap = np.broadcast_to(np.eye(3), (len(a), 3, 3)).copy(); at = ap.copy()
        rotation = np.empty((len(a), horizon))
        for k in range(horizon):
            ap, at = rp[:, k] @ ap, rt[:, k] @ at
            cosine = ((np.trace(np.swapaxes(ap, -1, -2) @ at, axis1=-2, axis2=-1) - 1) / 2).clip(-1, 1)
            rotation[:, k] = np.degrees(np.arccos(cosine))
        fingers = np.linalg.norm((a[..., 6:] - b[..., 6:]).reshape(-1, horizon, 21, 3)[:, :, 1:], axis=-1).mean(-1) * 100
        sums[valid] += np.stack((trans.sum(1), rotation.sum(1), fingers.sum(1)), axis=-1)
        counts[valid] += horizon
    return sums, counts


def summarize_physical(pred, target, mask, sequence_id, environment, *, horizon, dataset):
    sums, counts = physical_window_errors(pred, target, mask, horizon=horizon)
    seqs, envs = np.asarray(sequence_id), np.asarray(environment)
    per_seq, by_env = {}, {}
    for seq in sorted(set(seqs)):
        rows = seqs == seq
        if counts[rows].sum() == 0:
            continue
        group = set(envs[rows])
        if len(group) != 1:
            raise ValueError("sequence belongs to multiple environments")
        value = sums[rows].sum(0) / counts[rows].sum()
        per_seq[str(seq)] = dict(zip(METRICS, value.tolist()))
        by_env.setdefault(str(next(iter(group))), []).append(value)
    if not per_seq:
        raise ValueError("no sequences have the requested horizon")
    means = {name: np.mean(values, axis=0) for name, values in by_env.items()}
    # EgoTouch environments have equal weight. Tachin sequences have equal weight.
    overall = np.mean(list(means.values()), axis=0) if dataset == "egotouch" else np.mean(
        [list(v.values()) for v in per_seq.values()], axis=0)
    return {"overall": dict(zip(METRICS, overall.tolist())),
            "environments": {k: dict(zip(METRICS, v.tolist())) for k, v in means.items()},
            "sequences": per_seq, "valid_sequences": len(per_seq),
            "valid_hand_frames": int(counts.sum()), "horizon": horizon,
            "complete_egotouch_environments": len(means) == 5 if dataset == "egotouch" else None}


def summarize_seeds(reports):
    """Mean and sample standard deviation for one model on an unchanged test cache."""
    if not reports:
        raise ValueError("at least one seed report is required")
    first = reports[0]
    expected = {k: v for k, v in first["protocol"].items() if k != "seed"}
    seeds = []
    for report in reports:
        protocol = {k: v for k, v in report["protocol"].items() if k != "seed"}
        if report["model_id"] != first["model_id"] or report["cache_id"] != first["cache_id"] or protocol != expected:
            raise ValueError("seed reports must use one model, cache, and protocol")
        seeds.append(report["protocol"]["seed"])
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed reports contain duplicate seeds")
    output = {"model_id": first["model_id"], "seeds": seeds, "views": {}}
    def statistics(values):
        return {"mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    for view in ("8", "16"):
        rows = {}
        for mode in ("base", "editor", "improvement"):
            rows[mode] = {}
            for metric in METRICS:
                values = [r["views"][view][mode][metric] if mode == "improvement"
                          else r["views"][view][mode]["overall"][metric] for r in reports]
                rows[mode][metric] = statistics(values)
        environments = set(first["views"][view]["base"]["environments"])
        if any(set(r["views"][view][m]["environments"]) != environments
               for r in reports for m in ("base", "editor")):
            raise ValueError("seed reports have different environment coverage")
        rows["environments"] = {}
        for env in sorted(environments):
            cell = {}
            for mode in ("base", "editor", "improvement"):
                cell[mode] = {}
                for metric in METRICS:
                    values = []
                    for r in reports:
                        v = r["views"][view]
                        if mode == "improvement":
                            values.append(v["base"]["environments"][env][metric] - v["editor"]["environments"][env][metric])
                        else:
                            values.append(v[mode]["environments"][env][metric])
                    cell[mode][metric] = statistics(values)
            rows["environments"][env] = cell
        output["views"][view] = rows
    return output
