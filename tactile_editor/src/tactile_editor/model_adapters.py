"""Model adapters that normalize heterogeneous VLA outputs to BaseActionContext.

The adapters intentionally do not know dataset layouts.  A dataset/inference
loader supplies the observation object; the adapter only calls the model and
normalizes its returned action/memory fields.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from .action_layout import ACTION_DIM, CHUNK_SIZE, DEFAULT_ACTION_LAYOUT
from .editor_interface import BaseActionContext


ActionPredictor = Callable[..., Any]


@dataclass(frozen=True)
class AdapterSpec:
    """Output convention for one base model checkpoint."""

    model_id: str
    normalization_id: str
    output_is_normalized: bool = True
    action_mean: torch.Tensor | None = None
    action_std: torch.Tensor | None = None
    action_key: str = "base_action"
    mask_key: str = "action_mask"
    memory_key: str = "memory_tokens"
    image_key: str = "image_tokens"

    def __post_init__(self) -> None:
        if not self.model_id or not self.normalization_id:
            raise ValueError("model_id and normalization_id are required")
        if not self.output_is_normalized:
            if self.action_mean is None or self.action_std is None:
                raise ValueError("raw action adapters require action_mean and action_std")
        for name, value in (("action_mean", self.action_mean), ("action_std", self.action_std)):
            if value is not None and tuple(value.shape) != (ACTION_DIM,):
                raise ValueError(f"{name} must have shape [192]")
        if self.action_std is not None and (self.action_std <= 0).any():
            raise ValueError("action_std must be positive")


def _first(mapping: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


class ModelAdapter:
    """Call a model and convert its output to a validated BaseActionContext."""

    ACTION_NAMES = ("base_action", "action", "actions", "action_chunk", "norm_action", "normalized_action")
    MASK_NAMES = ("action_mask", "mask", "valid_mask")
    MEMORY_NAMES = ("memory_tokens", "final_memory", "memory", "m_f")
    IMAGE_NAMES = ("image_tokens", "vision_tokens", "image_features")

    def __init__(
        self,
        model: Any,
        spec: AdapterSpec,
        *,
        predictor: ActionPredictor | None = None,
        action_converter: Callable[[torch.Tensor], torch.Tensor] | None = None,
        method_names: tuple[str, ...] = (
            "predict_base_action",
            "sample_action_context",
            "select_action_context",
            "predict_action",
            "sample_action",
            "select_action",
        ),
    ) -> None:
        self.model = model
        self.spec = spec
        self.predictor = predictor
        self.action_converter = action_converter
        self.method_names = method_names

    def _predict(self, observation: Any, **model_kwargs: Any) -> Any:
        if self.predictor is not None:
            return self.predictor(observation, **model_kwargs)
        if callable(self.model) and not any(hasattr(self.model, name) for name in self.method_names):
            return self.model(observation, **model_kwargs)
        for name in self.method_names:
            method = getattr(self.model, name, None)
            if method is not None:
                return method(observation, **model_kwargs)
        raise AttributeError(
            f"{self.spec.model_id} adapter found no supported inference method; "
            f"expected one of {self.method_names}"
        )

    def _unpack(self, result: Any) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
        if isinstance(result, BaseActionContext):
            return result.base_action, result.memory_tokens, result.image_tokens, {
                "action_mask": result.action_mask,
                "edit_mask": result.edit_mask,
                "metadata": result.metadata,
            }
        if isinstance(result, Mapping):
            action = _first(result, (self.spec.action_key, *self.ACTION_NAMES))
            memory = _first(result, (self.spec.memory_key, *self.MEMORY_NAMES))
            image = _first(result, (self.spec.image_key, *self.IMAGE_NAMES))
            extras = dict(result)
            return action, memory, image, extras
        if isinstance(result, (tuple, list)):
            if not result:
                raise ValueError("model adapter received an empty output tuple")
            action = result[0]
            memory = result[1] if len(result) > 1 else None
            image = result[2] if len(result) > 2 else None
            return action, memory, image, {}
        return result, None, None, {}

    def __call__(
        self,
        observation: Any,
        *,
        action_mask: torch.Tensor | None = None,
        edit_mask: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> BaseActionContext:
        result = self._predict(observation, **model_kwargs)
        action, memory, image, extras = self._unpack(result)
        if action is None:
            raise ValueError(
                f"{self.spec.model_id} predictor did not return an action; "
                f"expected one of {(self.spec.action_key, *self.ACTION_NAMES)}"
            )
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action)
        if self.action_converter is not None:
            action = self.action_converter(action)
        if action.ndim != 3 or tuple(action.shape[1:]) != (CHUNK_SIZE, ACTION_DIM):
            raise ValueError(
                f"{self.spec.model_id} must emit [B,{CHUNK_SIZE},{ACTION_DIM}], "
                f"got {tuple(action.shape)}; provide an explicit action converter"
            )
        action = action.float()
        if not self.spec.output_is_normalized:
            mean = self.spec.action_mean.to(action.device, action.dtype)
            std = self.spec.action_std.to(action.device, action.dtype)
            action = (action - mean) / std

        supplied_mask = _first(extras, (self.spec.mask_key, *self.MASK_NAMES))
        mask = action_mask if action_mask is not None else supplied_mask
        if mask is None:
            mask = DEFAULT_ACTION_LAYOUT.physical_mask(
                tuple(action.shape), device=action.device
            )
        mask = mask.to(device=action.device, dtype=torch.bool)
        supplied_edit = _first(extras, ("edit_mask", "future_mask", "editable_mask"))
        if edit_mask is None:
            edit_mask = supplied_edit
        return BaseActionContext(
            base_action=action,
            action_mask=mask,
            edit_mask=edit_mask,
            model_id=self.spec.model_id,
            normalization_id=self.spec.normalization_id,
            memory_tokens=memory,
            image_tokens=image,
            metadata={"adapter_output": extras},
        )


class SmolVLAAdapter(ModelAdapter):
    """Adapter for the local V4-A/B/C/D SmolVLA-derived models."""

    def __init__(self, model: Any, spec: AdapterSpec, **kwargs: Any) -> None:
        super().__init__(model, spec, **kwargs)


class VITRAAdapter(ModelAdapter):
    """Adapter for VITRA or VITRA student action predictors."""


class TRexAdapter(ModelAdapter):
    """Adapter for the native T-Rex slow/fast predictor.

    T-Rex does not expose the ``[B, 16, 192]`` hand-action contract used by
    this repository.  Its released inference server returns a denormalized
    native chunk, normally ``[8, 62]`` or ``[16, 62]`` (31 dimensions per
    arm: 9-D delta EEF pose + 22-D hand target).  The Table 1 evaluator uses
    a two-hand MANO action space, so converting that native output is a
    task-specific operation and must be supplied through ``action_converter``.

    The adapter nevertheless handles the complete interface around that
    operation: unbatched server responses, the ``actions`` response key,
    native-shape validation, optional raw-action normalization, and metadata
    recording.  If a predictor already emits the canonical action, no
    converter is needed.
    """

    NATIVE_ACTION_DIMS = (29, 31, 58, 62)
    NATIVE_HORIZONS = (8, 16)

    def __init__(self, model: Any, spec: AdapterSpec, **kwargs: Any) -> None:
        native_converter = kwargs.pop("action_converter", None)

        def convert(action: torch.Tensor) -> torch.Tensor:
            # The ZMQ server returns a single chunk as [T, D]; model adapters
            # use a batch dimension everywhere else.
            if action.ndim == 2:
                action = action.unsqueeze(0)

            if action.ndim == 3 and tuple(action.shape[1:]) == (CHUNK_SIZE, ACTION_DIM):
                return action

            native_shape = tuple(action.shape[-2:]) if action.ndim == 3 else None
            if native_shape is None or native_shape[0] not in self.NATIVE_HORIZONS or native_shape[1] not in self.NATIVE_ACTION_DIMS:
                raise ValueError(
                    "T-Rex predictor must emit canonical [B,16,192] or a "
                    "native [T,D] / [B,T,D] chunk with T in "
                    f"{self.NATIVE_HORIZONS} and D in {self.NATIVE_ACTION_DIMS}; "
                    f"got {tuple(action.shape)}"
                )
            if native_converter is None:
                raise ValueError(
                    "T-Rex emits native EEF/hand actions "
                    f"{native_shape}; provide action_converter to map them to "
                    "the canonical [B,16,192] MANO action space"
                )
            converted = native_converter(action)
            if not isinstance(converted, torch.Tensor):
                converted = torch.as_tensor(converted, device=action.device)
            if converted.ndim == 2:
                converted = converted.unsqueeze(0)
            return converted

        super().__init__(model, spec, action_converter=convert, **kwargs)

    def __call__(
        self,
        observation: Any,
        *,
        action_mask: torch.Tensor | None = None,
        edit_mask: torch.Tensor | None = None,
    ) -> BaseActionContext:
        context = super().__call__(
            observation, action_mask=action_mask, edit_mask=edit_mask
        )
        metadata = dict(context.metadata)
        metadata.update(
            {
                "native_interface": "trex_slow_fast_cascaded_flow",
                "native_action_dims": self.NATIVE_ACTION_DIMS,
                "native_action_horizons": self.NATIVE_HORIZONS,
                "has_external_memory_tokens": False,
                "action_is_final": True,
            }
        )
        # T-Rex's cached KV is an internal inference state, not a reusable
        # memory-token tensor.  Keeping it absent routes the model through the
        # memory-free editor when a separate editor is explicitly requested.
        return replace(context, memory_tokens=None, metadata=metadata)


class TWLAAdapter(ModelAdapter):
    """Adapter for the local T-WLA/V4-D checkpoint family.

    The current T-WLA checkpoint exposes ``select_action_context`` and returns
    the canonical full action together with ``final_memory``/``m_f``.  This is
    intentionally a named adapter so Table 1 configuration can distinguish
    T-WLA from the MemoryVLA comparison row while sharing the same validated
    contract.
    """


class MemoryVLAAdapter(SmolVLAAdapter):
    """Adapter configuration for the official MemoryVLA baseline.

    Official MemoryVLA checkpoints emit robot-native end-effector actions and
    keep cognitive/perceptual memories inside the policy. They do not emit
    the canonical human-MANO action or reusable ``[B,N,D]`` memory tensor.
    Therefore this adapter must be constructed with a predictor that includes
    the task-specific human action head; it must not silently reuse the local
    V4-D/T-WLA checkpoint.
    """
