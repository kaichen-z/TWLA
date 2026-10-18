"""Shared models for the P0/P1/P2/P3 EgoTouch experiment matrix."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGERTIP_POSE_INDICES = (4, 8, 12, 16, 20)
FINGERTIP_TACTILE_SLICES = (
    {  # Transformed left-hand cache grid.
        "thumb": (slice(6, 10), slice(1, 4)),
        "index": (slice(2, 6), slice(5, 8)),
        "middle": (slice(2, 6), slice(9, 12)),
        "ring": (slice(2, 6), slice(13, 16)),
        "pinky": (slice(2, 6), slice(17, 20)),
    },
    {  # Right-hand cache grid.
        "thumb": (slice(6, 10), slice(17, 20)),
        "index": (slice(2, 6), slice(13, 16)),
        "middle": (slice(2, 6), slice(9, 12)),
        "ring": (slice(2, 6), slice(5, 8)),
        "pinky": (slice(2, 6), slice(1, 4)),
    },
)


def finger_tactile_masks_by_side() -> torch.Tensor:
    """Return [left/right, finger, 21, 21] masks in cache-grid coordinates."""
    result = torch.zeros(2, len(FINGER_NAMES), 21, 21, dtype=torch.bool)
    for side, regions in enumerate(FINGERTIP_TACTILE_SLICES):
        for finger_index, finger in enumerate(FINGER_NAMES):
            rows, columns = regions[finger]
            result[side, finger_index, rows, columns] = True
    return result


def canonical_finger_pose_masks() -> torch.Tensor:
    """Return five one-hot MANO/OpenPose-21 fingertip masks in FINGER_NAMES order."""
    result = torch.zeros(len(FINGER_NAMES), 21, dtype=torch.bool)
    for finger_index, joint_index in enumerate(FINGERTIP_POSE_INDICES):
        result[finger_index, joint_index] = True
    return result


def _region_indices(region: str) -> tuple[torch.Tensor | None, bool]:
    """Resolve a region name to finger indices and whether it is a complement."""
    if region == "full":
        return None, False
    if region == "fingertips":
        return torch.arange(len(FINGER_NAMES)), False
    if region in FINGER_NAMES:
        return torch.tensor([FINGER_NAMES.index(region)]), False
    prefix = "fingertips_without_"
    if region.startswith(prefix) and region.removeprefix(prefix) in FINGER_NAMES:
        finger = region.removeprefix(prefix)
        return torch.tensor([FINGER_NAMES.index(finger)]), True
    allowed = ("full", "fingertips", *FINGER_NAMES, *(f"fingertips_without_{x}" for x in FINGER_NAMES))
    raise ValueError(f"unknown hand region {region!r}; expected one of {allowed}")


class TactileEncoder(nn.Module):
    """OpenTouch tactile CNN + two-layer bidirectional GRU."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.gru = nn.GRU(128, 120, num_layers=2, bidirectional=True)
        self.projection = nn.Linear(240, embed_dim)

    def forward(self, pressure: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        if pressure.ndim != 4 or pressure.shape[-2:] != (21, 21):
            raise ValueError(f"touch must be [B,T,21,21], got {tuple(pressure.shape)}")
        b, t = pressure.shape[:2]
        validity = validity[:, None].expand(b, t, 21, 21).to(pressure.dtype)
        frames = torch.stack((pressure, validity), dim=2).reshape(b * t, 2, 21, 21)
        encoded = self.cnn(frames)
        if encoded.shape[-1] != 128:
            raise RuntimeError(f"unexpected tactile frame dimension: {encoded.shape[-1]}")
        sequence = encoded.reshape(b, t, 128).transpose(0, 1)
        self.gru.flatten_parameters()
        output, _ = self.gru(sequence)
        temporal = torch.cat((output[-1, :, :120], output[0, :, 120:]), dim=-1)
        return self.projection(F.relu(temporal))


class PoseEncoder(nn.Module):
    """OpenTouch frame MLP with either mean or Bash BiGRU aggregation."""

    def __init__(self, embed_dim: int = 64, temporal: str = "mean") -> None:
        super().__init__()
        if temporal not in {"mean", "gru"}:
            raise ValueError(f"unknown pose temporal encoder {temporal!r}")
        self.temporal = temporal
        self.encoder = nn.Sequential(
            nn.Linear(63, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
        )
        if temporal == "gru":
            self.gru = nn.GRU(128, 120, num_layers=2, bidirectional=True)
            self.projection = nn.Linear(240, embed_dim)
        else:
            self.projection = nn.Linear(128, embed_dim)

    @staticmethod
    def normalize(landmarks: torch.Tensor) -> torch.Tensor:
        centered = landmarks - landmarks[:, :, 0:1]
        d1 = torch.linalg.vector_norm(centered[:, :, 5] - centered[:, :, 1], dim=-1)
        d2 = torch.linalg.vector_norm(centered[:, :, 9] - centered[:, :, 1], dim=-1)
        scale = (0.5 * (d1 + d2)).mean(dim=1).clamp_min(1e-6)
        return centered / scale[:, None, None, None]

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        if landmarks.ndim != 4 or landmarks.shape[-2:] != (21, 3):
            raise ValueError(f"pose must be [B,T,21,3], got {tuple(landmarks.shape)}")
        normalized = self.normalize(landmarks)
        return self.encode_normalized(normalized)

    def encode_normalized(self, normalized: torch.Tensor) -> torch.Tensor:
        """Encode already wrist-centered and hand-scale-normalized landmarks."""
        if normalized.ndim != 4 or normalized.shape[-2:] != (21, 3):
            raise ValueError(f"normalized pose must be [B,T,21,3], got {tuple(normalized.shape)}")
        b, t = normalized.shape[:2]
        frames = self.encoder(normalized.reshape(b * t, 63)).reshape(b, t, 128)
        if self.temporal == "mean":
            temporal = frames.mean(dim=1)
        else:
            sequence = frames.transpose(0, 1)
            self.gru.flatten_parameters()
            output, _ = self.gru(sequence)
            temporal = torch.cat((output[-1, :, :120], output[0, :, 120:]), dim=-1)
            temporal = F.relu(temporal)
        return self.projection(temporal)


class FormalEncoderModel(nn.Module):
    def __init__(
        self,
        pressure_masks: torch.Tensor,
        *,
        task: str,
        pose_temporal: str = "mean",
        embed_dim: int = 64,
        use_touch_prediction: bool = True,
        physical_aux: bool = False,
        predictive_aux: bool = False,
        tactile_region: str = "full",
        pose_region: str = "full",
        focus_mode: str = "full",
        context_weight: float = 0.1,
        fingertip_auxiliary: bool = False,
        fingertip_residual: bool = False,
        tactile_residual_weight: float = 0.25,
        pose_residual_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if task not in {"p0", "p1", "p2", "p3"}:
            raise ValueError(f"unknown task {task!r}")
        self.task = task
        self.use_touch_prediction = use_touch_prediction
        self.physical_aux = physical_aux or task == "p0"
        self.predictive_aux = predictive_aux or task == "p2"
        if focus_mode not in {"full", "hard", "soft"}:
            raise ValueError(f"unknown focus mode {focus_mode!r}")
        if context_weight < 0:
            raise ValueError("context_weight must be non-negative")
        _region_indices(tactile_region)
        _region_indices(pose_region)
        self.tactile_region = tactile_region
        self.pose_region = pose_region
        self.focus_mode = focus_mode
        self.context_weight = float(context_weight)
        self.fingertip_auxiliary = bool(fingertip_auxiliary)
        self.fingertip_residual = bool(fingertip_residual)
        self.tactile_residual_weight = float(tactile_residual_weight)
        self.pose_residual_weight = float(pose_residual_weight)
        if min(self.tactile_residual_weight, self.pose_residual_weight) < 0:
            raise ValueError("residual weights must be non-negative")
        self.tactile = TactileEncoder(embed_dim)
        self.pose = PoseEncoder(embed_dim, temporal=pose_temporal)
        if self.fingertip_residual:
            self.tactile_tip = TactileEncoder(embed_dim)
            self.pose_tip = PoseEncoder(embed_dim, temporal=pose_temporal)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self.register_buffer("pressure_masks", pressure_masks.to(torch.bool), persistent=True)
        finger_tactile = finger_tactile_masks_by_side()
        if not torch.all(finger_tactile.sum(dim=(2, 3)) == 12):
            raise RuntimeError("each fingertip tactile region must contain 12 cells")
        if torch.any(finger_tactile & ~self.pressure_masks[:, None]):
            raise ValueError("side-specific fingertip cells must be subsets of the pressure masks")
        self.register_buffer("finger_tactile_masks", finger_tactile, persistent=False)
        self.register_buffer("finger_pose_masks", canonical_finger_pose_masks(), persistent=False)
        if self.physical_aux:
            self.pressure_head = nn.Linear(embed_dim, 441)
            self.contact_head = nn.Linear(embed_dim, 441)
            self.dynamics_head = nn.Linear(embed_dim, 441)
        if self.predictive_aux:
            prediction_input = embed_dim * (2 if use_touch_prediction else 1)
            self.predictive_head = nn.Sequential(
                nn.Linear(prediction_input, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 60)
            )

    def forward(
        self,
        touch: torch.Tensor,
        pose: torch.Tensor,
        side: torch.Tensor,
        *,
        shuffle_touch: bool = False,
        tactile_region_override: str | None = None,
        pose_region_override: str | None = None,
        focus_mode_override: str | None = None,
        fingertip_auxiliary_override: bool | None = None,
        fingertip_residual_override: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        need_tactile = self.task in {"p0", "p1", "p3"} or self.use_touch_prediction
        tactile_region = tactile_region_override or self.tactile_region
        pose_region = pose_region_override or self.pose_region
        focus_mode = focus_mode_override or self.focus_mode
        use_auxiliary = (
            self.fingertip_auxiliary
            if fingertip_auxiliary_override is None
            else fingertip_auxiliary_override
        )
        use_residual = (
            self.fingertip_residual
            if fingertip_residual_override is None
            else fingertip_residual_override
        )
        tactile_raw = (
            self.encode_tactile_view(touch, side, tactile_region, focus_mode) if need_tactile else None
        )
        pose_raw = self.encode_pose_view(pose, pose_region, focus_mode) if self.task != "p0" else None
        if use_residual:
            assert tactile_raw is not None and pose_raw is not None
            tip_tactile_raw = self._encode_tactile_view(
                self.tactile_tip, touch, side, "fingertips", "hard"
            )
            tip_pose_raw = self._encode_pose_view(self.pose_tip, pose, "fingertips", "hard")
            tactile_raw = self._soft_fuse(
                tactile_raw, tip_tactile_raw, self.tactile_residual_weight
            )
            pose_raw = self._soft_fuse(pose_raw, tip_pose_raw, self.pose_residual_weight)
        result: dict[str, torch.Tensor] = {"logit_scale": self.logit_scale.exp()}
        if tactile_raw is not None:
            result["tactile_features"] = F.normalize(tactile_raw, dim=-1)
        if pose_raw is not None:
            result["pose_features"] = F.normalize(pose_raw, dim=-1)
        if use_auxiliary:
            assert tactile_raw is not None and pose_raw is not None
            result["aux_tip_tactile_features"] = F.normalize(
                self.encode_tactile_view(touch, side, "fingertips", "hard"), dim=-1
            )
            result["aux_tip_pose_features"] = F.normalize(
                self.encode_pose_view(pose, "fingertips", "hard"), dim=-1
            )
        if self.physical_aux:
            assert tactile_raw is not None
            result.update(
                future_pressure=self.pressure_head(tactile_raw).reshape(-1, 21, 21),
                future_contact_logits=self.contact_head(tactile_raw).reshape(-1, 21, 21),
                future_dynamics=self.dynamics_head(tactile_raw).reshape(-1, 21, 21),
            )
        if self.predictive_aux:
            assert pose_raw is not None
            if self.use_touch_prediction:
                predictive_touch = tactile_raw
                if shuffle_touch and len(predictive_touch) > 1:
                    predictive_touch = predictive_touch.roll(1, dims=0)
                fused = torch.cat((pose_raw, predictive_touch), dim=-1)
            else:
                fused = pose_raw
            result["future_articulation"] = self.predictive_head(fused).reshape(-1, 20, 3)
        return result

    def _tactile_region_mask(self, region: str, side: torch.Tensor) -> torch.Tensor:
        indices, complement = _region_indices(region)
        full = self.pressure_masks[side.long()]
        if indices is None:
            return full
        finger_masks = self.finger_tactile_masks[side.long()]
        selected = finger_masks[:, indices.to(self.finger_tactile_masks.device)].any(dim=1)
        if complement:
            all_tips = finger_masks.any(dim=1)
            selected = all_tips & ~selected
        return full & selected

    def _pose_region_mask(self, region: str, batch_size: int) -> torch.Tensor:
        indices, complement = _region_indices(region)
        if indices is None:
            return torch.ones(batch_size, 1, 21, 1, dtype=torch.bool, device=self.finger_pose_masks.device)
        selected = self.finger_pose_masks[indices.to(self.finger_pose_masks.device)].any(dim=0)
        if complement:
            all_tips = self.finger_pose_masks.any(dim=0)
            selected = all_tips & ~selected
        return selected[None, None, :, None].expand(batch_size, 1, 21, 1)

    @staticmethod
    def _soft_fuse(focus: torch.Tensor, context: torch.Tensor, context_weight: float) -> torch.Tensor:
        focus = F.normalize(focus, dim=-1)
        context = F.normalize(context, dim=-1)
        return F.normalize(focus + context_weight * context, dim=-1)

    def encode_tactile_view(
        self, touch: torch.Tensor, side: torch.Tensor, region: str, focus_mode: str
    ) -> torch.Tensor:
        return self._encode_tactile_view(self.tactile, touch, side, region, focus_mode)

    def _encode_tactile_view(
        self,
        encoder: TactileEncoder,
        touch: torch.Tensor,
        side: torch.Tensor,
        region: str,
        focus_mode: str,
    ) -> torch.Tensor:
        full_validity = self.pressure_masks[side.long()]
        if region == "full":
            return encoder(touch, full_validity)
        selected = self._tactile_region_mask(region, side)
        focus_touch = touch * selected[:, None].to(touch.dtype)
        if focus_mode in {"full", "hard"}:
            return encoder(focus_touch, selected)
        if focus_mode != "soft":
            raise ValueError(f"unknown focus mode {focus_mode!r}")
        context = full_validity & ~selected
        context_touch = touch * context[:, None].to(touch.dtype)
        combined_touch = torch.cat((focus_touch, context_touch), dim=0)
        combined_validity = torch.cat((selected, context), dim=0)
        encoded = encoder(combined_touch, combined_validity)
        focus_encoded, context_encoded = encoded.chunk(2, dim=0)
        return self._soft_fuse(focus_encoded, context_encoded, self.context_weight)

    def encode_pose_view(self, pose: torch.Tensor, region: str, focus_mode: str) -> torch.Tensor:
        return self._encode_pose_view(self.pose, pose, region, focus_mode)

    def _encode_pose_view(
        self, encoder: PoseEncoder, pose: torch.Tensor, region: str, focus_mode: str
    ) -> torch.Tensor:
        if region == "full":
            return encoder(pose)
        normalized = encoder.normalize(pose)
        selected = self._pose_region_mask(region, len(pose))
        focus_pose = normalized * selected.to(normalized.dtype)
        if focus_mode in {"full", "hard"}:
            return encoder.encode_normalized(focus_pose)
        if focus_mode != "soft":
            raise ValueError(f"unknown focus mode {focus_mode!r}")
        context_pose = normalized * (~selected).to(normalized.dtype)
        combined_pose = torch.cat((focus_pose, context_pose), dim=0)
        encoded = encoder.encode_normalized(combined_pose)
        focus_encoded, context_encoded = encoded.chunk(2, dim=0)
        return self._soft_fuse(focus_encoded, context_encoded, self.context_weight)

    @torch.no_grad()
    def clamp_temperature(self) -> None:
        self.logit_scale.clamp_(math.log(1.0), math.log(100.0))

    def initialize_from_opentouch(
        self,
        path: str | Path,
        *,
        reset_temperature: bool = True,
        skip_prefixes: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Load compatible OpenTouch weights, optionally excluding named submodules.

        ``skip_prefixes`` is used by controlled architecture comparisons.  For
        example, skipping ``pose.gru`` and ``pose.projection`` lets mean and
        BiGRU pose models share the exact same pretrained frame encoder while
        training their architecture-specific temporal aggregation from scratch.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        source = checkpoint.get("state_dict", checkpoint)
        target = self.state_dict()
        loaded: list[str] = []
        skipped: list[str] = []
        for key, value in source.items():
            if key == "logit_scale" and reset_temperature:
                skipped.append("logit_scale: reset for target-domain comparison")
                continue
            if any(key.startswith(prefix) for prefix in skip_prefixes):
                skipped.append(f"{key}: excluded by controlled initialization policy")
                continue
            if key not in target:
                continue
            if key == "tactile.cnn.0.weight" and value.shape[1] == 1 and target[key].shape[1] == 2:
                adapted = torch.zeros_like(target[key])
                adapted[:, :1].copy_(value)
                target[key] = adapted
                loaded.append(key + " [pressure copied; validity channel zero-init]")
            elif target[key].shape == value.shape:
                target[key] = value
                loaded.append(key)
            else:
                skipped.append(f"{key}: source={tuple(value.shape)} target={tuple(target[key].shape)}")
        self.load_state_dict(target, strict=True)
        if self.fingertip_residual:
            self.tactile_tip.load_state_dict(self.tactile.state_dict(), strict=True)
            self.pose_tip.load_state_dict(self.pose.state_dict(), strict=True)
            loaded.extend(
                [
                    "tactile_tip.* [cloned from initialized full tactile encoder]",
                    "pose_tip.* [cloned from initialized full pose encoder]",
                ]
            )
        return {
            "checkpoint": str(path),
            "source_epoch": checkpoint.get("epoch"),
            "source_task_type": checkpoint.get("task_type"),
            "skip_prefixes": list(skip_prefixes),
            "loaded": loaded,
            "skipped": skipped,
        }


def symmetric_infonce(
    pose: torch.Tensor,
    touch: torch.Tensor,
    logit_scale: torch.Tensor,
    recording: torch.Tensor,
) -> torch.Tensor:
    logits = logit_scale * pose @ touch.T
    # Adjacent windows from one recording are not valid negatives.  Keep the
    # synchronized diagonal positive and mask other same-recording pairs.
    same_recording = recording[:, None].eq(recording[None, :])
    diagonal = torch.eye(len(recording), dtype=torch.bool, device=recording.device)
    logits = logits.masked_fill(same_recording & ~diagonal, -torch.inf)
    labels = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def articulation_target(history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    """Future change of the 20 wrist-relative joints, normalized by hand size."""
    centered = history - history[:, :, 0:1]
    d1 = torch.linalg.vector_norm(centered[:, :, 5] - centered[:, :, 1], dim=-1)
    d2 = torch.linalg.vector_norm(centered[:, :, 9] - centered[:, :, 1], dim=-1)
    scale = (0.5 * (d1 + d2)).mean(dim=1).clamp_min(1e-6)
    current_articulation = centered[:, -1, 1:] / scale[:, None, None]
    future_centered = future - future[:, 0:1]
    future_articulation = future_centered[:, 1:] / scale[:, None, None]
    return future_articulation - current_articulation
