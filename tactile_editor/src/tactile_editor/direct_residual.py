"""End-to-end causal tactile-to-action-residual models.

The tactile trunk is trained only by downstream action supervision. It does not
use a tactile-to-pose retrieval objective. The action branch creates residual
bases without tactile conditioning; the tactile branch creates coefficients.
They meet only in the final weighted sum.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .action_layout import (
    ACTION_DIM,
    ARTICULATION_GROUPS,
    FULL_ACTION_GROUPS,
)

DIRECT_INPUT_MODES = ("current", "current_delta", "history", "history_delta")
COEFFICIENT_SCOPES = ("global", "hand_local")


def canonical_tactile_channels(
    pressure: torch.Tensor,
    temporal_valid: torch.Tensor,
    pressure_masks: torch.Tensor,
    *,
    include_delta: bool,
    flip_left: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical pressure/difference/mask channels and latest validity."""

    if pressure.ndim != 5 or tuple(pressure.shape[2:]) != (2, 21, 21):
        raise ValueError("pressure must be [B,T,2,21,21]")
    if tuple(temporal_valid.shape) != tuple(pressure.shape[:3]):
        raise ValueError("temporal_valid must be [B,T,2]")
    if tuple(pressure_masks.shape) != (2, 21, 21):
        raise ValueError("pressure_masks must be [2,21,21]")

    value = torch.nan_to_num(pressure.float()).clone()
    # EgoTouch stores the left sensor in the opposite horizontal convention.
    if flip_left:
        value[:, :, 0] = value[:, :, 0].flip(-1)
    valid = temporal_valid.bool()
    spatial_valid = (
        valid[..., None, None]
        & pressure_masks.bool()[None, None].to(device=value.device)
    )
    value = value * spatial_valid.to(value.dtype)
    difference = torch.zeros_like(value)
    if include_delta and value.shape[1] > 1:
        pair_valid = valid[:, 1:] & valid[:, :-1]
        difference[:, 1:] = (
            value[:, 1:] - value[:, :-1]
        ) * pair_valid[..., None, None].to(value.dtype)
    channels = torch.stack(
        (value, difference, spatial_valid.to(value.dtype)), dim=3
    )
    return channels, valid[:, -1]


class DirectTactileTrunk(nn.Module):
    """Map raw causal tactile history directly to residual-basis coefficients."""

    def __init__(
        self,
        pressure_masks: torch.Tensor,
        reference_pressure: torch.Tensor,
        *,
        mode: str,
        coefficient_scope: str = "global",
        residual_bases: int = 8,
        frame_dim: int = 128,
        temporal_hidden: int = 128,
        dropout: float = 0.1,
        flip_left: bool = True,
    ) -> None:
        super().__init__()
        if mode not in DIRECT_INPUT_MODES:
            raise ValueError(f"unknown direct tactile input mode: {mode}")
        if coefficient_scope not in COEFFICIENT_SCOPES:
            raise ValueError(f"unknown coefficient scope: {coefficient_scope}")
        if tuple(reference_pressure.shape) != (1, 8, 2, 21, 21):
            raise ValueError("reference_pressure must be [1,8,2,21,21]")
        if residual_bases < 1:
            raise ValueError("residual_bases must be positive")
        self.mode = mode
        self.flip_left = flip_left
        self.coefficient_scope = coefficient_scope
        self.residual_bases = residual_bases
        self.register_buffer("pressure_masks", pressure_masks.bool(), persistent=True)
        self.register_buffer(
            "reference_pressure", reference_pressure.float(), persistent=True
        )
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128, frame_dim),
            nn.GELU(),
        )
        self.temporal = (
            nn.GRU(
                frame_dim,
                temporal_hidden,
                num_layers=1,
                batch_first=True,
            )
            if mode in ("history", "history_delta")
            else None
        )
        output_dim = temporal_hidden if self.temporal is not None else frame_dim
        coefficient_input_dim = (
            output_dim * 2 if coefficient_scope == "global" else output_dim
        )
        self.coefficient_head = nn.Sequential(
            nn.LayerNorm(coefficient_input_dim),
            nn.Dropout(dropout),
            nn.Linear(coefficient_input_dim, residual_bases, bias=False),
        )

    def _encode(
        self, pressure: torch.Tensor, temporal_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        channels, latest_valid = canonical_tactile_channels(
            pressure,
            temporal_valid,
            self.pressure_masks,
            include_delta=self.mode in ("current_delta", "history_delta"),
            flip_left=self.flip_left,
        )
        batch, history, hands = channels.shape[:3]
        frames = self.frame_encoder(channels.reshape(batch * history * hands, 3, 21, 21))
        frames = frames.reshape(batch, history, hands, -1).permute(0, 2, 1, 3)
        frame_valid = temporal_valid.bool().permute(0, 2, 1)[..., None]
        frames = frames * frame_valid.to(frames.dtype)
        if self.temporal is None:
            history_features = frames
            encoded = history_features[:, :, -1]
        else:
            sequence = frames.reshape(batch * hands, history, -1)
            self.temporal.flatten_parameters()
            history_features = self.temporal(sequence)[0]
            history_features = history_features.reshape(batch, hands, history, -1)
            encoded = history_features[:, :, -1]
        return encoded, latest_valid, history_features

    def forward(
        self, pressure: torch.Tensor, temporal_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(pressure)
        reference_pressure = self.reference_pressure.to(device=pressure.device).expand(
            batch, -1, -1, -1, -1
        )
        reference_valid = torch.ones(
            batch,
            self.reference_pressure.shape[1],
            2,
            dtype=torch.bool,
            device=pressure.device,
        )
        # Encode matched and reference inputs in one call. Identical tensors then
        # produce bitwise-identical features even when convolution algorithms are
        # sensitive to batch shape.
        combined_pressure = torch.cat((pressure, reference_pressure), dim=0)
        combined_valid = torch.cat((temporal_valid, reference_valid), dim=0)
        combined, combined_hand_valid, combined_history = self._encode(
            combined_pressure, combined_valid
        )
        encoded, reference = combined.split(batch, dim=0)
        encoded_history, reference_history = combined_history.split(batch, dim=0)
        hand_valid = combined_hand_valid[:batch]
        centered = (encoded - reference).mul(hand_valid[..., None].to(encoded.dtype))
        history_valid = temporal_valid.bool().permute(0, 2, 1)[..., None]
        centered_history = (encoded_history - reference_history).mul(
            history_valid.to(encoded_history.dtype)
        )
        if self.coefficient_scope == "global":
            coefficients = self.coefficient_head(centered.flatten(1))
            active = hand_valid.any(1, keepdim=True) & (
                centered.abs().sum((1, 2), keepdim=False)[:, None] > 0
            )
            coefficients = coefficients * active.to(coefficients.dtype)
        else:
            coefficients = self.coefficient_head(centered)
            active = hand_valid & (centered.abs().sum(-1) > 0)
            coefficients = coefficients * active[..., None].to(coefficients.dtype)
        return coefficients, hand_valid, centered, centered_history


class ActionResidualBasisTrunk(nn.Module):
    """Create action-conditioned bases without access to tactile inputs."""

    def __init__(
        self,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        *,
        residual_bases: int = 8,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if tuple(action_mean.shape) != (192,) or tuple(action_std.shape) != (192,):
            raise ValueError("action statistics must have shape [192]")
        self.residual_bases = residual_bases
        self.register_buffer("action_mean", action_mean.float(), persistent=True)
        self.register_buffer(
            "action_std", action_std.float().clamp_min(1e-3), persistent=True
        )
        self.action_projection = nn.Linear(192, hidden)
        self.position = nn.Parameter(torch.randn(1, 16, hidden) * 0.02)
        self.time_projection = nn.Linear(1, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.basis_head = nn.Linear(hidden, residual_bases * 192)
        nn.init.zeros_(self.basis_head.weight)
        nn.init.zeros_(self.basis_head.bias)

    def forward(self, anchor: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor:
        if tuple(anchor.shape[1:]) != (16, 192):
            raise ValueError("anchor must be [B,16,192]")
        hidden = self.action_projection(
            (anchor.detach() - self.action_mean) / self.action_std
        )
        hidden = hidden + self.position + self.time_projection(
            elapsed.float()[:, None, None] * 10
        )
        hidden = self.trunk(hidden)
        bases = self.basis_head(hidden).reshape(
            len(anchor), 16, self.residual_bases, 192
        )
        return bases.permute(0, 2, 1, 3)


class DirectTactileResidualEditor(nn.Module):
    """Predict a tactile-only low-rank correction to a frozen action anchor."""

    def __init__(
        self,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        pressure_masks: torch.Tensor,
        reference_pressure: torch.Tensor,
        *,
        mode: str,
        residual_bases: int = 8,
        articulation_only: bool = True,
        flip_left: bool = True,
    ) -> None:
        super().__init__()
        self.tactile = DirectTactileTrunk(
            pressure_masks,
            reference_pressure,
            mode=mode,
            residual_bases=residual_bases,
            flip_left=flip_left,
        )
        self.action = ActionResidualBasisTrunk(
            action_mean, action_std, residual_bases=residual_bases
        )
        component_mask = torch.zeros(1, 1, ACTION_DIM, dtype=torch.bool)
        groups = ARTICULATION_GROUPS if articulation_only else FULL_ACTION_GROUPS
        for start, stop in groups:
            component_mask[..., start:stop] = True
        self.register_buffer("component_mask", component_mask, persistent=True)
        self.residual_bases = residual_bases

    def forward(
        self,
        anchor: torch.Tensor,
        edit_mask: torch.Tensor,
        pressure: torch.Tensor,
        temporal_valid: torch.Tensor,
        elapsed: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coefficients, hand_valid, tactile_features, tactile_history_features = self.tactile(
            pressure, temporal_valid
        )
        bases = self.action(anchor, elapsed)
        raw_delta = (
            coefficients[:, :, None, None] * bases
        ).sum(1) / math.sqrt(self.residual_bases)
        allowed = edit_mask.bool() & self.component_mask
        # The action layout is [left 69 | right 69 | padding].
        allowed = allowed.clone()
        allowed[..., :69] &= hand_valid[:, 0, None, None]
        allowed[..., 69:138] &= hand_valid[:, 1, None, None]
        delta = raw_delta * allowed.to(raw_delta.dtype)
        return anchor.detach() + delta, {
            "effective_delta": delta,
            "coefficients": coefficients,
            "bases": bases,
            "allowed_mask": allowed,
            "hand_valid": hand_valid,
            "tactile_features": tactile_features,
            "tactile_history_features": tactile_history_features,
        }
