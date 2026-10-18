"""Shared model-to-tactile-editor interface.

VLA implementations differ in their internal memory representation.  They all
expose the same normalized action context here, while editor backends decide
whether to use optional memory/image tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from .action_layout import ACTION_DIM, CHUNK_SIZE, DEFAULT_ACTION_LAYOUT


@dataclass(frozen=True)
class BaseActionContext:
    """Frozen output of a model-specific VLA adapter."""

    base_action: torch.Tensor
    action_mask: torch.Tensor
    model_id: str
    normalization_id: str
    memory_tokens: torch.Tensor | None = None
    image_tokens: torch.Tensor | None = None
    edit_mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        DEFAULT_ACTION_LAYOUT.validate(self.base_action, name="base_action")
        if self.action_mask.shape != self.base_action.shape:
            raise ValueError("action_mask must match base_action")
        if self.edit_mask is not None and self.edit_mask.shape != self.base_action.shape:
            raise ValueError("edit_mask must match base_action")
        if self.action_mask[..., 138:].any():
            raise ValueError("action_mask cannot mark padding dimensions valid")
        if self.edit_mask is not None and self.edit_mask[..., 138:].any():
            raise ValueError("edit_mask cannot mark padding dimensions editable")
        if not self.base_action.is_floating_point() or not torch.isfinite(self.base_action).all():
            raise ValueError("base_action must be finite floating-point values")
        if self.memory_tokens is not None:
            if self.memory_tokens.ndim != 3 or self.memory_tokens.shape[0] != self.base_action.shape[0]:
                raise ValueError("memory_tokens must be [B,N,D]")
        if self.image_tokens is not None and self.image_tokens.shape[0] != self.base_action.shape[0]:
            raise ValueError("image_tokens batch size must match base_action")
        if not self.model_id or not self.normalization_id:
            raise ValueError("model_id and normalization_id are required")

    @property
    def effective_edit_mask(self) -> torch.Tensor:
        mask = self.action_mask.bool()
        if self.edit_mask is not None:
            mask = mask & self.edit_mask.bool()
        return mask

    @property
    def has_memory(self) -> bool:
        return self.memory_tokens is not None


@dataclass(frozen=True)
class EditorOutput:
    """Common output of every tactile editor backend."""

    final_action: torch.Tensor
    delta_action: torch.Tensor
    allowed_mask: torch.Tensor
    backend: str
    auxiliary_outputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.final_action.shape != self.delta_action.shape:
            raise ValueError("final_action and delta_action must match")
        if self.allowed_mask.shape != self.final_action.shape:
            raise ValueError("allowed_mask must match final_action")
        DEFAULT_ACTION_LAYOUT.validate(self.final_action, name="final_action")
        if not self.backend:
            raise ValueError("backend name is required")


class TactileEditorBackend(Protocol):
    """Structural protocol implemented by memory and memory-free backends."""

    requires_memory: bool

    def edit(
        self,
        context: BaseActionContext,
        pressure: torch.Tensor,
        temporal_valid: torch.Tensor,
        decision_seconds: torch.Tensor,
        **kwargs: Any,
    ) -> EditorOutput: ...
