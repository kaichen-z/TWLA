"""Model-independent action output contract for Table 1 evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .action_layout import (
    ACTION_DIM,
    ARTICULATION_GROUPS,
    CHUNK_SIZE,
    DEFAULT_ACTION_LAYOUT,
    FULL_ACTION_GROUPS,
)


@dataclass(frozen=True)
class NormalizedActionOutput:
    """A model prediction in the frozen VLA normalized action space.

    Every baseline row (native VLA, tactile editor, or adapted model) can be
    exported in this format before denormalization and physical metric scoring.
    ``valid_mask`` describes available hand/action values; padding is rejected
    at construction time so it cannot enter losses or Table 1 metrics.
    """

    actions: torch.Tensor
    valid_mask: torch.Tensor
    model_id: str
    normalization_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.actions.ndim != 3 or tuple(self.actions.shape[1:]) != (
            CHUNK_SIZE,
            ACTION_DIM,
        ):
            raise ValueError("actions must be [B,16,192]")
        if self.valid_mask.shape != self.actions.shape:
            raise ValueError("valid_mask must match actions")
        if not self.actions.is_floating_point() or not torch.isfinite(self.actions).all():
            raise ValueError("actions must be finite floating-point values")
        if self.valid_mask[..., 138:].any():
            raise ValueError("padding dimensions 138:192 must be invalid")
        if not self.model_id or not self.normalization_id:
            raise ValueError("model_id and normalization_id are required")

    @property
    def physical_mask(self) -> torch.Tensor:
        return self.valid_mask.bool() & DEFAULT_ACTION_LAYOUT.physical_mask(
            tuple(self.valid_mask.shape), device=self.valid_mask.device
        )

    def editable(self, *, articulation_only: bool = False) -> torch.Tensor:
        groups = ARTICULATION_GROUPS if articulation_only else FULL_ACTION_GROUPS
        return self.valid_mask.bool() & DEFAULT_ACTION_LAYOUT.physical_mask(
            tuple(self.valid_mask.shape),
            device=self.valid_mask.device,
            groups=groups,
        )

    def denormalize(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if tuple(mean.shape) != (ACTION_DIM,) or tuple(std.shape) != (ACTION_DIM,):
            raise ValueError("normalization statistics must be [192]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("normalization statistics must be finite and positive")
        mean = mean.to(device=self.actions.device, dtype=self.actions.dtype)
        std = std.to(device=self.actions.device, dtype=self.actions.dtype)
        return self.actions * std + mean
