"""Canonical Method-D action layout shared by training and evaluation.

The VLA emits 192 values per action step.  The first 138 values are the two
69-dimensional hand actions; the final 54 values are padding and are never
editable or scored.  Keeping this contract in one module prevents model rows
from silently using different masks for Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


ACTION_DIM = 192
CHUNK_SIZE = 16
VALID_ACTION_DIM = 138
HAND_ACTION_DIM = 69

LEFT_TRANSLATION = slice(0, 3)
LEFT_ROTATION = slice(3, 6)
LEFT_ARTICULATION = slice(6, 69)
RIGHT_TRANSLATION = slice(69, 72)
RIGHT_ROTATION = slice(72, 75)
RIGHT_ARTICULATION = slice(75, 138)

TRANSLATION_GROUPS = ((0, 3), (69, 72))
ROTATION_GROUPS = ((3, 6), (72, 75))
ARTICULATION_GROUPS = ((6, 69), (75, 138))
FULL_ACTION_GROUPS = (
    (0, 3),
    (3, 6),
    (6, 69),
    (69, 72),
    (72, 75),
    (75, 138),
)
FULL_HAND_GROUPS = ((0, 69), (69, 138))


@dataclass(frozen=True)
class ActionLayout:
    """Shape and physical component metadata for a normalized action chunk."""

    chunk_size: int = CHUNK_SIZE
    action_dim: int = ACTION_DIM
    valid_action_dim: int = VALID_ACTION_DIM

    def validate(self, action: torch.Tensor, *, name: str = "action") -> None:
        if action.ndim < 2 or tuple(action.shape[-2:]) != (
            self.chunk_size,
            self.action_dim,
        ):
            raise ValueError(
                f"{name} must end in [{self.chunk_size},{self.action_dim}], "
                f"got {tuple(action.shape)}"
            )

    def physical_mask(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | None = None,
        groups: tuple[tuple[int, int], ...] = FULL_ACTION_GROUPS,
    ) -> torch.Tensor:
        if len(shape) < 2 or shape[-2:] != (self.chunk_size, self.action_dim):
            raise ValueError("mask shape must end in [16,192]")
        result = torch.zeros(shape, dtype=torch.bool, device=device)
        for start, stop in groups:
            if not 0 <= start < stop <= self.valid_action_dim:
                raise ValueError("action group is outside the valid action range")
            result[..., start:stop] = True
        return result

    def apply_edit_scope(
        self,
        action_mask: torch.Tensor,
        *,
        groups: tuple[tuple[int, int], ...] = FULL_ACTION_GROUPS,
    ) -> torch.Tensor:
        """Intersect a runtime mask with physical components and padding mask."""

        self.validate(action_mask, name="action_mask")
        return action_mask.bool() & self.physical_mask(
            tuple(action_mask.shape), device=action_mask.device, groups=groups
        )


DEFAULT_ACTION_LAYOUT = ActionLayout()


@dataclass(frozen=True)
class ActionNormalizer:
    """Frozen affine statistics used by every model row."""

    mean: torch.Tensor
    std: torch.Tensor
    identity: str = "unlabelled"

    def __post_init__(self) -> None:
        if tuple(self.mean.shape) != (ACTION_DIM,) or tuple(self.std.shape) != (
            ACTION_DIM,
        ):
            raise ValueError("action statistics must have shape [192]")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("action statistics must be finite")
        if (self.std <= 0).any():
            raise ValueError("action standard deviations must be positive")
        if not self.identity:
            raise ValueError("normalizer identity must be non-empty")

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        DEFAULT_ACTION_LAYOUT.validate(action)
        mean = self.mean.to(device=action.device, dtype=action.dtype)
        std = self.std.to(device=action.device, dtype=action.dtype)
        return (action - mean) / std

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        DEFAULT_ACTION_LAYOUT.validate(action)
        mean = self.mean.to(device=action.device, dtype=action.dtype)
        std = self.std.to(device=action.device, dtype=action.dtype)
        return action * std + mean
