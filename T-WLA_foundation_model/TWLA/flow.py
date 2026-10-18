"""Rectified flow: training target, masked loss and Euler sampling.

Convention (as in SmolVLA):
    x_t = tau * noise + (1 - tau) * action        tau = 1 is the noise end
    u   = noise - action                          the constant velocity along that line
"""

from __future__ import annotations

import torch

from .config import FlowConfig


def sample_time_beta(batch: int, device: torch.device, cfg: FlowConfig,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """tau ~ scaled Beta(alpha, beta), returned in [offset, offset + scale]."""
    a = torch.full((batch,), cfg.time_beta_alpha, device=device)
    b = torch.full((batch,), cfg.time_beta_beta, device=device)
    if generator is None:
        t = torch.distributions.Beta(a, b).sample()
    else:
        # Beta.sample() takes no generator, so the seeded draw uses two Gammas:
        # Beta(a, b) = G1 / (G1 + G2) with G1 ~ Gamma(a, 1), G2 ~ Gamma(b, 1).
        g1 = torch._standard_gamma(a, generator=generator)
        g2 = torch._standard_gamma(b, generator=generator)
        t = g1 / (g1 + g2)
    return t * cfg.time_scale + cfg.time_offset


def make_flow_target(actions: torch.Tensor, noise: torch.Tensor,
                     tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (x_t, u). actions and noise are [B, T, A], tau is [B]."""
    t = tau[:, None, None]
    return t * noise + (1.0 - t) * actions, noise - actions


def masked_component_loss(pred: torch.Tensor, target: torch.Tensor,
                          action_mask: torch.Tensor, cfg: FlowConfig,
                          ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked squared error, averaged per component group and combined with
    `cfg.component_weights`.

    Averaging within each group first makes a group's share of the loss independent of
    its width (3 dims for a wrist group, 63 for the keypoints).

    Returns (loss, parts). parts holds each group's value and, under `<name>_n`, the
    mask sum it was averaged over.
    """
    se = (pred - target) ** 2                                  # [B, T, A]
    m = action_mask.to(se.dtype)

    if cfg.horizon_weights is not None:
        hw = torch.as_tensor(cfg.horizon_weights, device=se.device, dtype=se.dtype)
        if hw.numel() != se.shape[1]:
            raise ValueError(f"horizon_weights has {hw.numel()} entries, chunk is {se.shape[1]}")
        m = m * hw[None, :, None]

    parts: dict[str, torch.Tensor] = {}
    total = se.new_zeros(())
    wsum = se.new_zeros(())
    for (name, lo, hi), w in zip(cfg.component_groups, cfg.component_weights):
        num = (se[..., lo:hi] * m[..., lo:hi]).sum()
        den = m[..., lo:hi].sum()
        val = num / den.clamp_min(1.0)
        parts[name] = val.detach()
        parts[name + "_n"] = den.detach()
        total = total + w * val
        # A group with no valid entry in the batch does not count in the divisor.
        wsum = wsum + float(w) * (den > 0).to(se.dtype)
    return total / wsum.clamp_min(1e-8), parts


def component_action_mask(batch: int, horizon: int, action_dim: int, cfg: FlowConfig,
                          device: torch.device) -> torch.Tensor:
    """The action mask to use at inference: True on the component groups, False on padding.

    The mask is part of the expert's input and training never sets the padding dims, so
    an all-ones mask would be out of distribution.
    """
    mask = torch.zeros(batch, horizon, action_dim, dtype=torch.bool, device=device)
    for _, lo, hi in cfg.component_groups:
        mask[..., lo:hi] = True
    return mask


@torch.no_grad()
def euler_integrate(velocity_fn, noise: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Integrate from noise (tau = 1) to data (tau = 0) in `num_steps` Euler steps.

    `velocity_fn(x_t, tau) -> v` is called once per step.
    """
    x = noise
    dt = 1.0 / num_steps
    for i in range(num_steps):
        tau = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device, dtype=x.dtype)
        v = velocity_fn(x, tau)
        x = x - dt * v          # tau decreases, so step against the velocity
    return x


def repeat_for_draws(condition: torch.Tensor, k: int) -> torch.Tensor:
    """Repeat every row k times along the batch axis: [B, ...] -> [B * k, ...]."""
    return condition.repeat_interleave(k, dim=0)
