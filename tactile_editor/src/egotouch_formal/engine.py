"""Losses and evaluation shared by formal training and standalone evaluation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .model import FormalEncoderModel, articulation_target, symmetric_infonce


def physical_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    masks: torch.Tensor,
    *,
    contact_threshold: float,
    contact_pos_weight: float,
) -> dict[str, torch.Tensor]:
    valid = masks[batch["side"].long()].to(output["future_pressure"].dtype)
    target_pressure = batch["future_touch"]
    target_dynamics = target_pressure - batch["touch"][:, -1]
    target_contact = (target_pressure > contact_threshold).to(target_pressure.dtype)
    denominator = valid.sum().clamp_min(1.0)
    pressure = (F.smooth_l1_loss(output["future_pressure"], target_pressure, reduction="none") * valid).sum() / denominator
    dynamics = (F.smooth_l1_loss(output["future_dynamics"], target_dynamics, reduction="none") * valid).sum() / denominator
    weights = torch.where(target_contact > 0, contact_pos_weight, 1.0)
    contact = (
        F.binary_cross_entropy_with_logits(output["future_contact_logits"], target_contact, reduction="none")
        * valid
        * weights
    ).sum() / (valid * weights).sum().clamp_min(1.0)
    return {"pressure": pressure, "dynamics": dynamics, "contact": contact}


def compute_loss(
    model: FormalEncoderModel,
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    task = config["task"]
    pieces: dict[str, torch.Tensor] = {}
    if task in {"p1", "p3"}:
        pieces["retrieval"] = symmetric_infonce(
            output["pose_features"], output["tactile_features"], output["logit_scale"], batch["recording"]
        )
        auxiliary = config.get("fingertip_aux_weights", {})
        if auxiliary:
            tip_touch = output["aux_tip_tactile_features"]
            tip_pose = output["aux_tip_pose_features"]
            pieces["retrieval_tip_touch_full_pose"] = symmetric_infonce(
                output["pose_features"], tip_touch, output["logit_scale"], batch["recording"]
            )
            pieces["retrieval_full_touch_tip_pose"] = symmetric_infonce(
                tip_pose, output["tactile_features"], output["logit_scale"], batch["recording"]
            )
            pieces["retrieval_tip_touch_tip_pose"] = symmetric_infonce(
                tip_pose, tip_touch, output["logit_scale"], batch["recording"]
            )
    if model.physical_aux:
        physical = physical_losses(
            output,
            batch,
            model.pressure_masks,
            contact_threshold=float(config["contact_threshold"]),
            contact_pos_weight=float(config["contact_pos_weight"]),
        )
        pieces.update({f"physical_{key}": value for key, value in physical.items()})
    if model.predictive_aux:
        target = articulation_target(batch["pose"], batch["future_pose"])
        pieces["predictive"] = F.smooth_l1_loss(output["future_articulation"], target)

    if task == "p0":
        total = (
            float(config["pressure_weight"]) * pieces["physical_pressure"]
            + float(config["contact_weight"]) * pieces["physical_contact"]
            + float(config["dynamics_weight"]) * pieces["physical_dynamics"]
        )
    elif task == "p1":
        total = pieces["retrieval"]
        for key in (
            "retrieval_tip_touch_full_pose",
            "retrieval_full_touch_tip_pose",
            "retrieval_tip_touch_tip_pose",
        ):
            if key in pieces:
                total = total + float(config["fingertip_aux_weights"][key]) * pieces[key]
    elif task == "p2":
        total = pieces["predictive"]
    else:
        total = pieces["retrieval"]
        if model.physical_aux:
            physical_total = (
                float(config["pressure_weight"]) * pieces["physical_pressure"]
                + float(config["contact_weight"]) * pieces["physical_contact"]
                + float(config["dynamics_weight"]) * pieces["physical_dynamics"]
            )
            total = total + float(config["physical_aux_weight"]) * physical_total
        if model.predictive_aux:
            total = total + float(config["predictive_aux_weight"]) * pieces["predictive"]
    scalars = {key: float(value.detach()) for key, value in pieces.items()}
    scalars["total"] = float(total.detach())
    return total, scalars


def retrieval_metrics(
    query: torch.Tensor, target: torch.Tensor, recording: torch.Tensor, device: torch.device
) -> dict[str, float]:
    query, target, recording = query.to(device), target.to(device), recording.to(device)
    similarity = query @ target.T
    same = recording[:, None].eq(recording[None, :])
    diagonal = torch.eye(len(recording), dtype=torch.bool, device=device)
    similarity.masked_fill_(same & ~diagonal, -torch.inf)
    labels = torch.arange(len(query), device=device)
    order = similarity.argsort(dim=1, descending=True)
    ranks = (order == labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    return {
        "r1": float((ranks <= 1).float().mean()),
        "r5": float((ranks <= 5).float().mean()),
        "r10": float((ranks <= 10).float().mean()),
        "map": float((1.0 / ranks.float()).mean()),
        "median_rank": float(ranks.float().median()),
    }


@torch.no_grad()
def evaluate(model: FormalEncoderModel, loader, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model.eval()
    task = config["task"]
    pose_features, tactile_features, recordings, sides = [], [], [], []
    pressure_abs = dynamics_abs = valid_count = 0.0
    contact_tp = contact_fp = contact_fn = 0.0
    predictive_abs = predictive_count = 0.0
    loss_total = samples = 0.0
    for raw in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(batch["touch"], batch["pose"], batch["side"])
            loss, _ = compute_loss(model, output, batch, config)
        b = len(batch["side"])
        loss_total += float(loss) * b
        samples += b
        if task in {"p1", "p3"}:
            pose_features.append(output["pose_features"].float().cpu())
            tactile_features.append(output["tactile_features"].float().cpu())
            recordings.append(batch["recording"].cpu())
            sides.append(batch["side"].cpu())
        if model.physical_aux:
            valid = model.pressure_masks[batch["side"].long()]
            target_pressure = batch["future_touch"]
            target_dynamics = target_pressure - batch["touch"][:, -1]
            pressure_abs += float(((output["future_pressure"] - target_pressure).abs() * valid).sum())
            dynamics_abs += float(((output["future_dynamics"] - target_dynamics).abs() * valid).sum())
            valid_count += float(valid.sum())
            truth = (target_pressure > float(config["contact_threshold"])) & valid
            pred = (output["future_contact_logits"] > 0) & valid
            contact_tp += float((truth & pred).sum())
            contact_fp += float((~truth & pred & valid).sum())
            contact_fn += float((truth & ~pred).sum())
        if model.predictive_aux:
            target = articulation_target(batch["pose"], batch["future_pose"])
            predictive_abs += float((output["future_articulation"] - target).abs().sum())
            predictive_count += float(target.numel())

    result: dict[str, Any] = {"num_samples": int(samples), "loss": loss_total / max(samples, 1)}
    if pose_features:
        pose = torch.cat(pose_features)
        touch = torch.cat(tactile_features)
        recording = torch.cat(recordings)
        p2t = retrieval_metrics(pose, touch, recording, device)
        t2p = retrieval_metrics(touch, pose, recording, device)
        result["p2t"] = p2t
        result["t2p"] = t2p
        result["mean_map"] = 0.5 * (p2t["map"] + t2p["map"])
    if valid_count:
        precision = contact_tp / max(contact_tp + contact_fp, 1.0)
        recall = contact_tp / max(contact_tp + contact_fn, 1.0)
        result["physical"] = {
            "pressure_mae": pressure_abs / valid_count,
            "dynamics_mae": dynamics_abs / valid_count,
            "contact_precision": precision,
            "contact_recall": recall,
            "contact_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        }
    if predictive_count:
        result["predictive"] = {"articulation_mae": predictive_abs / predictive_count}
    return result
