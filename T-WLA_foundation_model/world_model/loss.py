"""`L_world`: flow matching on the future latents, in the convention of `TWLA/flow.py`.

    z_tau = (1 - tau) * z0 + tau * eps        tau = 1 is the noise end
    u     = eps - z0                          the velocity to regress
    loss  = ||v_pred - u||^2
"""

from __future__ import annotations

import torch

from .config import WorldLossConfig


def sample_tau(shape, cfg: WorldLossConfig, *, device, dtype,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """tau ~ U(tau_low, tau_high), drawn independently for every element of `shape`."""
    u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    return cfg.tau_low + (cfg.tau_high - cfg.tau_low) * u


def world_loss(sana, adapter, image_tokens, final_memory, z0, valid,
               cfg: WorldLossConfig, *, generator: torch.Generator | None = None,
               timestep_scale: float = 1000.0, sample_weight: torch.Tensor | None = None):
    """Flow-matching loss over K target frames. Returns (scalar, dict of log values).

        image_tokens   E_t [B, 64, 960]
        final_memory   m_f [B, 8, 960]; not detached when the loss should reach memory
        z0             target latents [B, K, 32, 16, 16], scaled and detached
        valid          [B, K] bool, whether the target frame exists
        sample_weight  [B], multiplies each sample's squared error (default 1)

    The dict holds the unweighted per-horizon means `h{frame}`, the counts `h{frame}_n`
    and the [B] vectors `per_sample` (weighted) and `per_sample_raw`, with gradient.
    """
    if z0.dim() != 5:
        raise ValueError(f"z0 must be [B, K, C, H, W], got {tuple(z0.shape)}")
    b, k = z0.shape[0], z0.shape[1]
    if k != cfg.num_horizons:
        raise ValueError(f"z0 carries {k} horizons, config says {cfg.num_horizons}")
    if tuple(valid.shape) != (b, k):
        raise ValueError(f"valid must be [{b}, {k}], got {tuple(valid.shape)}")

    adapter_k = getattr(adapter.cfg, "num_horizons", 1)
    if adapter_k not in (1, k):
        raise ValueError(
            f"adapter carries {adapter_k} horizon tags but the loss has {k} horizons. "
            "It must be either k (tags in the condition) or 1 (tags in SANA's timestep "
            "path); anything else means some horizons share a tag.")

    per_frame = z0.shape[2] * z0.shape[3] * z0.shape[4]           # 32*16*16 = 8192
    m = valid.to(z0.dtype)                                        # [B, K]

    # Zero the invalid targets with `where`: a multiply would keep a NaN placeholder,
    # since NaN * 0 is NaN.
    z0 = torch.where(valid[:, :, None, None, None], z0, torch.zeros_like(z0))

    # Horizon-major flattening, [B, K, ...] -> [K*B, ...]: rows 0..B-1 are horizon 0.
    # `horizon.horizon_rows` reads the horizon from this order.
    flat = lambda t: t.transpose(0, 1).reshape(k * b, *t.shape[2:])   # noqa: E731
    z0f = flat(z0)                                                    # [K*B, 32, 16, 16]

    tau = sample_tau((k * b,), cfg, device=z0.device, dtype=z0.dtype, generator=generator)
    eps = torch.randn(z0f.shape, device=z0.device, dtype=z0.dtype, generator=generator)
    t = tau[:, None, None, None]
    z_tau = (1.0 - t) * z0f + t * eps
    u = eps - z0f

    # Same condition for every horizon, unless the adapter carries the horizon tags.
    base = adapter(image_tokens, final_memory)                        # [B, 72, 2304]
    if adapter_k == k and k > 1:
        cond = torch.cat([adapter.for_horizon(base, i) for i in range(k)], dim=0)
    else:
        cond = base.repeat(k, 1, 1)                                   # [K*B, 72, 2304]

    # SANA's timestep embedding is built for diffusers' [0, 1000] range, hence the scale.
    v = sana(hidden_states=z_tau, encoder_hidden_states=cond,
             timestep=tau * timestep_scale, return_dict=False)[0]

    if v.shape != u.shape:
        raise RuntimeError(f"SANA returned {tuple(v.shape)}, target is {tuple(u.shape)}")

    # Squared error per row, summed in fp32, then back to [B, K].
    se = ((v.float() - u.float()) ** 2).flatten(1).sum(dim=1).reshape(k, b).transpose(0, 1)

    weights = cfg.weights()                                           # sums to 1
    total = se.new_zeros(())
    wsum = se.new_zeros(())
    parts: dict[str, torch.Tensor] = {}

    sw = torch.ones(b, device=se.device, dtype=se.dtype) if sample_weight is None \
        else sample_weight.to(se.device, se.dtype)
    if tuple(sw.shape) != (b,):
        raise ValueError(f"sample_weight must be [{b}], got {tuple(sw.shape)}")
    per_sample = se.new_zeros(b)
    per_sample_raw = se.new_zeros(b)
    for i in range(k):
        n = m[:, i].sum()
        # Mean over the valid rows of a horizon first, then the horizon weights.
        contrib = se[:, i] * m[:, i] / (n * per_frame).clamp_min(1.0)      # [B], sums to the mean
        raw = contrib.sum()
        val = (contrib * sw).sum()
        parts[f"h{cfg.horizon_frames[i]}"] = raw.detach()
        parts[f"h{cfg.horizon_frames[i]}_n"] = n.detach()
        total = total + weights[i] * val
        per_sample = per_sample + weights[i] * contrib * sw
        per_sample_raw = per_sample_raw + weights[i] * contrib
        # A horizon with no valid row is left out of the divisor.
        wsum = wsum + weights[i] * (n > 0).to(se.dtype)

    parts["per_sample"] = per_sample / wsum.clamp_min(1e-8)                # attached, [B], weighted
    parts["per_sample_raw"] = per_sample_raw / wsum.clamp_min(1e-8)        # attached, [B], unweighted
    return total / wsum.clamp_min(1e-8), parts


def lam_at(step: int, cfg: WorldLossConfig) -> float:
    """lambda in `L_total = L_action + lambda * L_world` at training step `step`.

    `lam_schedule` is piecewise constant: the value of the last segment that starts at or
    before `step`, ramped linearly from 0 over the first `lam_warmup_steps` steps.
    """
    lam = 0.0
    for start, value in cfg.lam_schedule:
        if step >= start:
            lam = float(value)
    if cfg.lam_warmup_steps > 0:
        lam *= min(1.0, max(0.0, step / cfg.lam_warmup_steps))
    return lam
