"""Unified tactile-editor router and backend adapters."""

from __future__ import annotations

from typing import Any
from dataclasses import replace

import torch
from torch import nn

from .editor_interface import BaseActionContext, EditorOutput
from .direct_residual import DirectTactileResidualEditor
from .unified_world import UnifiedWorldTactileEditor


class MemoryFreeTactileEditor(nn.Module):
    """Action-conditioned tactile residual editor for models without memory."""

    requires_memory = False

    def __init__(self, editor: DirectTactileResidualEditor) -> None:
        super().__init__()
        self.editor = editor

    def edit(
        self,
        context: BaseActionContext,
        pressure: torch.Tensor,
        temporal_valid: torch.Tensor,
        decision_seconds: torch.Tensor,
        **_: Any,
    ) -> EditorOutput:
        if context.has_memory:
            raise ValueError("MemoryFreeTactileEditor cannot consume memory_tokens")
        final_action, details = self.editor(
            context.base_action,
            context.effective_edit_mask,
            pressure,
            temporal_valid,
            decision_seconds,
        )
        return EditorOutput(
            final_action=final_action,
            delta_action=details["effective_delta"],
            allowed_mask=details["allowed_mask"],
            backend="memory_free",
            auxiliary_outputs=details,
        )


class NativeFinalActionEditor(nn.Module):
    """Pass through a model that already performed its native tactile update.

    T-Rex (and eventually a native T-WLA predictor) returns a tactile-refined
    action, rather than a vision-only anchor plus reusable memory tokens.  The
    pass-through keeps the common ``EditorOutput`` contract without applying a
    second tactile residual editor.
    """

    requires_memory = False

    def edit(
        self,
        context: BaseActionContext,
        pressure: torch.Tensor,
        temporal_valid: torch.Tensor,
        decision_seconds: torch.Tensor,
        **_: Any,
    ) -> EditorOutput:
        del pressure, temporal_valid, decision_seconds
        return EditorOutput(
            final_action=context.base_action,
            delta_action=torch.zeros_like(context.base_action),
            allowed_mask=context.effective_edit_mask,
            backend="native_tactile",
            auxiliary_outputs={"native_tactile_policy": True},
        )


class MemoryW0W3Editor(nn.Module):
    """Adapter exposing the existing memory-conditioned W0--W3 editor."""

    requires_memory = True

    def __init__(self, editor: UnifiedWorldTactileEditor) -> None:
        super().__init__()
        self.editor = editor

    def edit(
        self,
        context: BaseActionContext,
        pressure: torch.Tensor,
        temporal_valid: torch.Tensor,
        decision_seconds: torch.Tensor,
        **kwargs: Any,
    ) -> EditorOutput:
        if not context.has_memory:
            raise ValueError("MemoryW0W3Editor requires memory_tokens")
        # Runtime action editing cannot consume future supervision. Training
        # callers can explicitly enable auxiliary world predictions.
        kwargs.setdefault("compute_auxiliary", False)
        output = self.editor(
            base_action=context.base_action,
            edit_mask=context.effective_edit_mask,
            pressure=pressure,
            temporal_valid=temporal_valid,
            decision_seconds=decision_seconds,
            final_memory=context.memory_tokens,
            image_tokens=context.image_tokens,
            **kwargs,
        )
        auxiliary = {
            "tactile_features": output.tactile_features,
            "hand_valid": output.hand_valid,
            "tactile_velocity": output.tactile_velocity,
            "tactile_target": output.tactile_target,
            "tactile_valid": output.tactile_valid,
            "rgb": output.rgb,
        }
        return EditorOutput(
            final_action=output.final_action,
            delta_action=output.delta_action,
            allowed_mask=output.allowed_mask,
            backend=f"memory_{self.editor.world_mode.value}",
            auxiliary_outputs=auxiliary,
        )


class TactileEditorRouter(nn.Module):
    """Select an explicit shared/memory experiment, or preserve legacy auto routing."""

    def __init__(
        self,
        memory_backend: MemoryW0W3Editor | None,
        memory_free_backend: MemoryFreeTactileEditor | None,
        native_final_backend: NativeFinalActionEditor | None = None,
        *,
        route: str = "auto",
    ) -> None:
        super().__init__()
        if memory_backend is None and memory_free_backend is None:
            raise ValueError("at least one tactile-editor backend is required")
        self.memory_backend = memory_backend
        self.memory_free_backend = memory_free_backend
        self.native_final_backend = native_final_backend or NativeFinalActionEditor()
        if route not in ("auto", "shared", "memory_ablation"):
            raise ValueError("route must be auto, shared, or memory_ablation")
        self.route = route

    def edit(
        self,
        context: BaseActionContext,
        pressure: torch.Tensor,
        temporal_valid: torch.Tensor,
        decision_seconds: torch.Tensor,
        **kwargs: Any,
    ) -> EditorOutput:
        if self.route == "shared":
            if self.memory_free_backend is None:
                raise ValueError("shared route requires a memory-free backend")
            # The base retains its native memory. Only the editor loses access.
            context = replace(context, memory_tokens=None, image_tokens=None)
            return self.memory_free_backend.edit(
                context, pressure, temporal_valid, decision_seconds, **kwargs
            )
        if self.route == "memory_ablation":
            if context.model_id != "twla" or not context.has_memory or self.memory_backend is None:
                raise ValueError("memory_ablation requires twla, memory tokens, and a memory backend")
            return self.memory_backend.edit(
                context, pressure, temporal_valid, decision_seconds, **kwargs
            )
        if context.metadata.get("action_is_final", False):
            return self.native_final_backend.edit(
                context, pressure, temporal_valid, decision_seconds, **kwargs
            )
        if context.has_memory:
            if self.memory_backend is None:
                raise ValueError("memory_tokens were supplied but no memory backend is configured")
            return self.memory_backend.edit(
                context, pressure, temporal_valid, decision_seconds, **kwargs
            )
        if self.memory_free_backend is None:
            raise ValueError("no memory-free backend is configured for this context")
        return self.memory_free_backend.edit(
            context, pressure, temporal_valid, decision_seconds, **kwargs
        )

    def forward(
        self,
        context: BaseActionContext,
        pressure: torch.Tensor,
        temporal_valid: torch.Tensor,
        decision_seconds: torch.Tensor,
        **kwargs: Any,
    ) -> EditorOutput:
        return self.edit(context, pressure, temporal_valid, decision_seconds, **kwargs)
