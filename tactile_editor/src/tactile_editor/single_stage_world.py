"""Single-stage memory/tactile action editing with a tactile world objective.

The frozen Method-D VLA is outside this module.  Its cached ``m_f`` and action
plan are immutable inputs.  One tactile trunk is shared by two trainable heads:

* the action head edits the frozen plan from ``m_f + a_base + z_touch``;
* the world head predicts tactile pressure 0.1 s later from
  ``m_f + z_touch + stopgrad(a_edited)``.

The world path follows Kai's VLA-WM training contract where it matters for
optimisation: a deterministic frozen target codec, rectified-flow supervision,
four trainable blocks, a faster condition adapter, and an explicit gradient
boundary at the frozen VLA.  SANA itself is not reused because its 2-D latent
geometry and pretrained RGB decoder are specific to image latents, not the two
21x21 pressure grids used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .action_objectives import per_sample_action_loss
from .action_layout import ARTICULATION_GROUPS, FULL_ACTION_GROUPS
from .direct_residual import DirectTactileTrunk


def _validate_shape(
    tensor: torch.Tensor, shape: tuple[int | None, ...], name: str
) -> None:
    if tensor.ndim != len(shape):
        raise ValueError(
            f"{name} must have {len(shape)} dimensions, got {tuple(tensor.shape)}"
        )
    for actual, expected in zip(tensor.shape, shape):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )


def gather_action_window(
    action: torch.Tensor,
    decision_seconds: torch.Tensor,
    *,
    steps: int = 8,
    action_hz: float = 80.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the next plan-relative actions at each tactile decision.

    The first index uses the same float-tolerant ceil rule as
    ``build_future_action_mask``.  Windows near the plan end are zero padded and
    accompanied by a validity mask.
    """

    _validate_shape(action, (None, None, None), "action")
    if decision_seconds.shape != (len(action),):
        raise ValueError("decision_seconds must be [B]")
    if steps < 1 or not math.isfinite(action_hz) or action_hz <= 0:
        raise ValueError("steps and action_hz must be positive")
    seconds = decision_seconds.to(device=action.device, dtype=torch.float64)
    if not torch.isfinite(seconds).all() or (seconds < 0).any():
        raise ValueError("decision_seconds must be finite and non-negative")
    tolerance = 1e-6 if decision_seconds.dtype != torch.float64 else 1e-9
    start = torch.ceil(seconds * action_hz - tolerance).long()
    offsets = torch.arange(steps, device=action.device)[None]
    indices = start[:, None] + offsets
    valid = indices < action.shape[1]
    safe = indices.clamp(max=max(action.shape[1] - 1, 0))
    batch = torch.arange(len(action), device=action.device)[:, None]
    window = action[batch, safe]
    return window * valid[..., None].to(window.dtype), valid


class TactileTargetCodec(nn.Module):
    """Frozen deterministic analogue of the RGB DC-AE target encoder.

    It canonicalises the left sensor, removes train-split means, and scales each
    valid taxel by train-split standard deviation.  There are deliberately no
    trainable parameters, so the future target cannot collapse with the online
    tactile encoder.
    """

    def __init__(
        self,
        pressure_masks: torch.Tensor,
        reference_pressure: torch.Tensor,
        pressure_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        if tuple(pressure_masks.shape) != (2, 21, 21):
            raise ValueError("pressure_masks must be [2,21,21]")
        if tuple(reference_pressure.shape) == (1, 8, 2, 21, 21):
            reference_pressure = reference_pressure[0, -1]
        if tuple(reference_pressure.shape) != (2, 21, 21):
            raise ValueError("reference_pressure must be [2,21,21] or [1,8,2,21,21]")
        if tuple(pressure_scale.shape) != (2, 21, 21):
            raise ValueError("pressure_scale must be [2,21,21]")
        # Statistics are stored in the dataset's raw sensor coordinates.  The
        # codec target is canonical, so transform its statistics identically.
        reference_pressure = reference_pressure.float().clone()
        pressure_scale = pressure_scale.float().clone()
        reference_pressure[0] = reference_pressure[0].flip(-1)
        pressure_scale[0] = pressure_scale[0].flip(-1)
        self.register_buffer("pressure_masks", pressure_masks.bool(), persistent=True)
        self.register_buffer("reference_pressure", reference_pressure, persistent=True)
        self.register_buffer(
            "pressure_scale", pressure_scale.clamp_min(1e-3), persistent=True
        )

    def encode(
        self, pressure: torch.Tensor, hand_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_shape(pressure, (None, 2, 21, 21), "future_pressure")
        if hand_valid.shape != (len(pressure), 2):
            raise ValueError("future_hand_valid must be [B,2]")
        value = torch.nan_to_num(pressure.float()).clone()
        value[:, 0] = value[:, 0].flip(-1)
        spatial = hand_valid.bool()[..., None, None] & self.pressure_masks[None]
        normalized = (value - self.reference_pressure) / self.pressure_scale
        normalized = normalized * spatial.to(normalized.dtype)
        return normalized.flatten(2), spatial.flatten(2)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        _validate_shape(tokens, (None, 2, 441), "tactile_tokens")
        value = (
            tokens.reshape(-1, 2, 21, 21) * self.pressure_scale
            + self.reference_pressure
        )
        value = value.clone()
        value[:, 0] = value[:, 0].flip(-1)
        return value


class MemoryConditionedActionBasis(nn.Module):
    """Create residual bases from the frozen plan and explicit Method-D memory."""

    def __init__(
        self,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        *,
        memory_dim: int = 960,
        hidden_dim: int = 256,
        residual_bases: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if tuple(action_mean.shape) != (192,) or tuple(action_std.shape) != (192,):
            raise ValueError("action statistics must be [192]")
        self.residual_bases = residual_bases
        self.register_buffer("action_mean", action_mean.float(), persistent=True)
        self.register_buffer(
            "action_std", action_std.float().clamp_min(1e-3), persistent=True
        )
        self.memory_norm = nn.RMSNorm(memory_dim)
        self.memory_projection = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.action_projection = nn.Linear(192, hidden_dim)
        self.memory_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_position = nn.Parameter(torch.randn(1, 16, hidden_dim) * 0.02)
        self.elapsed_projection = nn.Linear(1, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output = nn.Linear(hidden_dim, residual_bases * 192)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        base_action: torch.Tensor,
        final_memory: torch.Tensor,
        decision_seconds: torch.Tensor,
    ) -> torch.Tensor:
        _validate_shape(base_action, (None, 16, 192), "base_action")
        _validate_shape(final_memory, (len(base_action), 8, None), "final_memory")
        if decision_seconds.shape != (len(base_action),):
            raise ValueError("decision_seconds must be [B]")
        dtype = self.action_projection.weight.dtype
        action = (
            base_action.detach().to(dtype) - self.action_mean.to(dtype)
        ) / self.action_std.to(dtype)
        action = (
            self.action_projection(action) + self.action_type + self.action_position
        )
        action = action + self.elapsed_projection(
            decision_seconds.to(dtype=dtype)[:, None, None] * 10.0
        )
        memory = self.memory_projection(
            self.memory_norm(final_memory.detach().to(dtype))
        )
        memory = memory + self.memory_type
        hidden = self.blocks(torch.cat((memory, action), dim=1))[:, memory.shape[1] :]
        bases = self.output(hidden).reshape(len(action), 16, self.residual_bases, 192)
        return bases.permute(0, 2, 1, 3)


class TactileWorldConditionAdapter(nn.Module):
    """Map memory, shared tactile features, and an optional action window to WM tokens."""

    def __init__(
        self,
        *,
        memory_dim: int = 960,
        tactile_dim: int = 128,
        hidden_dim: int = 256,
        action_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.action_tokens = action_tokens
        # Opt-in experiment: identical conditioning values, scaled backward path.
        # Default preserves the original stop-gradient behavior.
        self.action_gradient_scale = 0.0
        self.memory_norm = nn.RMSNorm(memory_dim)
        self.tactile_norm = nn.RMSNorm(tactile_dim)
        self.memory_projection = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.tactile_projection = nn.Linear(tactile_dim, hidden_dim, bias=False)
        self.action_projection = (
            nn.Linear(192, hidden_dim, bias=False) if action_tokens else None
        )
        self.memory_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.tactile_type = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        self.action_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_position = (
            nn.Parameter(torch.randn(1, action_tokens, hidden_dim) * 0.02)
            if action_tokens
            else None
        )

    def forward(
        self,
        final_memory: torch.Tensor,
        tactile_features: torch.Tensor,
        hand_valid: torch.Tensor,
        action_window: torch.Tensor | None,
        action_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_shape(final_memory, (None, 8, None), "final_memory")
        _validate_shape(
            tactile_features, (len(final_memory), 2, None), "tactile_features"
        )
        if hand_valid.shape != (len(final_memory), 2):
            raise ValueError("hand_valid must be [B,2]")
        dtype = self.memory_projection.weight.dtype
        memory = self.memory_projection(
            self.memory_norm(final_memory.detach().to(dtype))
        )
        memory = memory + self.memory_type
        tactile = self.tactile_projection(self.tactile_norm(tactile_features.to(dtype)))
        tactile = (tactile + self.tactile_type) * hand_valid[..., None].to(dtype)
        tokens = [memory, tactile]
        padding = [
            torch.zeros(
                len(memory), memory.shape[1], dtype=torch.bool, device=memory.device
            ),
            ~hand_valid.bool(),
        ]
        if self.action_projection is not None:
            if action_window is None or action_valid is None:
                raise ValueError("the configured world model requires an action window")
            _validate_shape(
                action_window, (len(memory), self.action_tokens, 192), "action_window"
            )
            if action_valid.shape != (len(memory), self.action_tokens):
                raise ValueError("action_valid has the wrong shape")
            scale = self.action_gradient_scale
            if not 0 <= scale <= 1:
                raise ValueError("action_gradient_scale must be in [0,1]")
            action_input = action_window.detach() + scale * (
                action_window - action_window.detach()
            )
            action = self.action_projection(action_input.to(dtype))
            action = action + self.action_type + self.action_position
            action = action * action_valid[..., None].to(dtype)
            tokens.append(action)
            padding.append(~action_valid.bool())
        return torch.cat(tokens, dim=1), torch.cat(padding, dim=1)


class TactileWorldModel(nn.Module):
    """Four-block rectified-flow model for a future two-hand pressure state."""

    def __init__(
        self,
        *,
        memory_dim: int = 960,
        tactile_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        action_tokens: int = 8,
        dropout: float = 0.1,
        cells: int = 441,
    ) -> None:
        super().__init__()
        if num_layers != 4:
            raise ValueError(
                "the Kai-aligned primary setup uses exactly four world blocks"
            )
        self.adapter = TactileWorldConditionAdapter(
            memory_dim=memory_dim,
            tactile_dim=tactile_dim,
            hidden_dim=hidden_dim,
            action_tokens=action_tokens,
        )
        if cells < 1:
            raise ValueError("cells must be positive")
        self.cells = int(cells)
        self.target_projection = nn.Linear(self.cells, hidden_dim)
        self.target_side = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerDecoderLayer(
            hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm_out = nn.RMSNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, self.cells)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _time_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            torch.linspace(math.log(0.25), math.log(250.0), half, device=tau.device)
        )
        angles = tau.float()[:, None] * frequencies[None] * 2 * math.pi
        value = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return value[:, :dim]

    def velocity(
        self,
        noisy_future: torch.Tensor,
        tau: torch.Tensor,
        *,
        final_memory: torch.Tensor,
        tactile_features: torch.Tensor,
        hand_valid: torch.Tensor,
        action_window: torch.Tensor | None,
        action_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        _validate_shape(noisy_future, (None, 2, self.cells), "noisy_future")
        if tau.shape != (len(noisy_future),):
            raise ValueError("tau must be [B]")
        condition, condition_padding = self.adapter(
            final_memory,
            tactile_features,
            hand_valid,
            action_window,
            action_valid,
        )
        dtype = self.target_projection.weight.dtype
        time = self.time_mlp(self._time_embedding(tau, condition.shape[-1]).to(dtype))
        target = (
            self.target_projection(noisy_future.to(dtype))
            + self.target_side
            + time[:, None]
        )
        hidden = self.blocks(
            tgt=target,
            memory=condition,
            memory_key_padding_mask=condition_padding,
        )
        return self.output(self.norm_out(hidden))


def tactile_flow_pair(
    clean: torch.Tensor,
    valid: torch.Tensor,
    tau: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kai/Method-D convention: tau=0 is data and tau=1 is noise."""

    if clean.shape != noise.shape or valid.shape != clean.shape:
        raise ValueError("clean, noise, and valid must have matching [B,2,441] shapes")
    if tau.shape != (len(clean),):
        raise ValueError("tau must be [B]")
    weight = valid.to(clean.dtype)
    clean = clean * weight
    noise = noise * weight
    t = tau.to(clean.dtype)[:, None, None]
    return ((1.0 - t) * clean + t * noise) * weight, (noise - clean) * weight


def masked_tactile_flow_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or valid.shape != prediction.shape:
        raise ValueError("world tensors must have matching shapes")
    squared = (prediction.float() - target.float()).square().masked_fill(~valid, 0)
    return squared.sum() / valid.sum().clamp_min(1)


@dataclass
class SingleStageWorldOutput:
    final_action: torch.Tensor
    delta_action: torch.Tensor
    coefficients: torch.Tensor
    tactile_features: torch.Tensor
    tactile_history_features: torch.Tensor
    hand_valid: torch.Tensor
    allowed_mask: torch.Tensor
    world_velocity: torch.Tensor | None
    world_target: torch.Tensor | None
    world_valid: torch.Tensor | None


class SingleStageWorldTactileEditor(nn.Module):
    """Joint single-stage editor and future-tactile world model."""

    def __init__(
        self,
        *,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        pressure_masks: torch.Tensor,
        reference_pressure: torch.Tensor,
        pressure_scale: torch.Tensor,
        tactile_mode: str = "history_delta",
        residual_bases: int = 8,
        memory_dim: int = 960,
        hidden_dim: int = 256,
        condition_on_action: bool = True,
        articulation_only: bool = False,
    ) -> None:
        super().__init__()
        self.tactile = DirectTactileTrunk(
            pressure_masks,
            reference_pressure,
            mode=tactile_mode,
            residual_bases=residual_bases,
        )
        self.action = MemoryConditionedActionBasis(
            action_mean,
            action_std,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            residual_bases=residual_bases,
        )
        self.world = TactileWorldModel(
            memory_dim=memory_dim,
            tactile_dim=128,
            hidden_dim=hidden_dim,
            action_tokens=8 if condition_on_action else 0,
        )
        self.target_codec = TactileTargetCodec(
            pressure_masks, reference_pressure, pressure_scale
        )
        component_mask = torch.zeros(1, 1, 192, dtype=torch.bool)
        groups = ARTICULATION_GROUPS if articulation_only else FULL_ACTION_GROUPS
        for start, stop in groups:
            component_mask[..., start:stop] = True
        self.register_buffer("component_mask", component_mask, persistent=True)
        self.residual_bases = residual_bases
        self.condition_on_action = condition_on_action
        self.articulation_only = articulation_only

    def forward(
        self,
        *,
        base_action: torch.Tensor,
        edit_mask: torch.Tensor,
        pressure: torch.Tensor,
        temporal_valid: torch.Tensor,
        decision_seconds: torch.Tensor,
        final_memory: torch.Tensor,
        future_pressure: torch.Tensor | None = None,
        future_valid: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> SingleStageWorldOutput:
        coefficients, hand_valid, tactile_features, tactile_history = self.tactile(
            pressure, temporal_valid
        )
        bases = self.action(base_action, final_memory, decision_seconds)
        raw_delta = (coefficients[:, :, None, None] * bases).sum(1) / math.sqrt(
            self.residual_bases
        )
        allowed = edit_mask.bool() & self.component_mask
        allowed = allowed.clone()
        allowed[..., :69] &= hand_valid[:, 0, None, None]
        allowed[..., 69:138] &= hand_valid[:, 1, None, None]
        delta = raw_delta * allowed.to(raw_delta.dtype)
        final_action = base_action.detach() + delta

        world_velocity = world_target = world_valid = None
        supplied = [
            future_pressure is not None,
            future_valid is not None,
            tau is not None,
        ]
        if any(supplied) and not all(supplied):
            raise ValueError(
                "future_pressure, future_valid, and tau must be supplied together"
            )
        if all(supplied):
            assert (
                future_pressure is not None
                and future_valid is not None
                and tau is not None
            )
            clean, world_valid = self.target_codec.encode(future_pressure, future_valid)
            if noise is None:
                noise = torch.randn_like(clean)
            noisy, world_target = tactile_flow_pair(clean, world_valid, tau, noise)
            if self.condition_on_action:
                action_window, action_valid = gather_action_window(
                    final_action.detach(), decision_seconds, steps=8
                )
            else:
                action_window = action_valid = None
            world_velocity = self.world.velocity(
                noisy,
                tau,
                final_memory=final_memory.detach(),
                tactile_features=tactile_features,
                hand_valid=hand_valid,
                action_window=action_window,
                action_valid=action_valid,
            )
        return SingleStageWorldOutput(
            final_action=final_action,
            delta_action=delta,
            coefficients=coefficients,
            tactile_features=tactile_features,
            tactile_history_features=tactile_history,
            hand_valid=hand_valid,
            allowed_mask=allowed,
            world_velocity=world_velocity,
            world_target=world_target,
            world_valid=world_valid,
        )

    def objective(
        self,
        output: SingleStageWorldOutput,
        target_action: torch.Tensor,
        edit_mask: torch.Tensor,
        *,
        lambda_world: float,
        lambda_delta: float = 1e-4,
        lambda_coefficient: float = 1e-5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if lambda_world < 0 or lambda_delta < 0 or lambda_coefficient < 0:
            raise ValueError("loss weights must be non-negative")
        # Use the exact mask that produced the residual.  In particular, do not
        # charge an irreducible action loss for a hand whose latest touch frame
        # is absent: that loss has no gradient but corrupts model selection.
        mask = edit_mask.bool() & output.allowed_mask
        present = mask.any((1, 2))
        if not present.any():
            raise ValueError("the batch contains no tactile-editable action target")
        action_loss = per_sample_action_loss(
            output.final_action[present],
            target_action[present],
            mask[present],
            groups=(ARTICULATION_GROUPS if self.articulation_only else FULL_ACTION_GROUPS),
        ).mean()
        delta_l2 = output.delta_action.float().square().masked_fill(~mask, 0).sum()
        delta_l2 = delta_l2 / mask.sum().clamp_min(1)
        coefficient_l2 = output.coefficients.float().square().mean()
        if output.world_velocity is None:
            if lambda_world:
                raise ValueError(
                    "world supervision is missing while lambda_world is nonzero"
                )
            world_loss = action_loss.new_zeros(())
        else:
            assert output.world_target is not None and output.world_valid is not None
            world_loss = masked_tactile_flow_loss(
                output.world_velocity, output.world_target, output.world_valid
            )
        total = (
            action_loss
            + lambda_world * world_loss
            + lambda_delta * delta_l2
            + lambda_coefficient * coefficient_l2
        )
        return total, {
            "loss": total.detach(),
            "action_loss": action_loss.detach(),
            "world_loss": world_loss.detach(),
            "delta_l2": delta_l2.detach(),
            "coefficient_l2": coefficient_l2.detach(),
        }

    def optimizer_groups(
        self,
        *,
        action_head_lr: float = 1e-4,
        tactile_gru_lr: float = 1e-5,
        tactile_cnn_lr: float = 3e-6,
        world_adapter_lr: float = 1e-4,
        world_block_lr: float = 1e-5,
    ) -> list[dict]:
        """Kai-style parameter groups with no frozen-VLA parameters."""

        tactile_head = list(self.tactile.coefficient_head.parameters())
        tactile_gru = (
            list(self.tactile.temporal.parameters()) if self.tactile.temporal else []
        )
        groups = [
            {
                "name": "action",
                "params": list(self.action.parameters()) + tactile_head,
                "lr": action_head_lr,
            },
            {
                "name": "tactile.cnn",
                "params": list(self.tactile.frame_encoder.parameters()),
                "lr": tactile_cnn_lr,
            },
            {
                "name": "world.adapter",
                "params": list(self.world.adapter.parameters()),
                "lr": world_adapter_lr,
            },
            {
                "name": "world.blocks",
                "params": [
                    *self.world.target_projection.parameters(),
                    self.world.target_side,
                    *self.world.time_mlp.parameters(),
                    *self.world.blocks.parameters(),
                    *self.world.norm_out.parameters(),
                    *self.world.output.parameters(),
                ],
                "lr": world_block_lr,
            },
        ]
        if tactile_gru:
            groups.insert(
                2, {"name": "tactile.gru", "params": tactile_gru, "lr": tactile_gru_lr}
            )
        parameters = [parameter for group in groups for parameter in group["params"]]
        if len(parameters) != len({id(parameter) for parameter in parameters}):
            raise RuntimeError("a parameter occurs in more than one optimizer group")
        expected = {
            id(parameter) for parameter in self.parameters() if parameter.requires_grad
        }
        if {id(parameter) for parameter in parameters} != expected:
            raise RuntimeError(
                "optimizer groups do not cover every trainable parameter"
            )
        return groups


def ramp_world_weight(step: int, maximum: float, warmup_steps: int) -> float:
    if step < 0 or maximum < 0 or warmup_steps < 0:
        raise ValueError("step and world-weight settings must be non-negative")
    if warmup_steps == 0:
        return float(maximum)
    return float(maximum) * min(1.0, step / warmup_steps)
