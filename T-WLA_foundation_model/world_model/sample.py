"""Sampling from the world head: Euler integration of the velocity field, then decoding.

Not used in training. With z_tau = (1 - tau) * z0 + tau * eps the model predicts
v ~ eps - z0, so sampling starts from noise at tau = 1 and steps down to tau = 0:

    z(tau - h) = z(tau) - h * v(z(tau), tau)
"""

from __future__ import annotations

import torch

from .config import SanaConfig, WorldLossConfig


@torch.no_grad()
def sample_latent(sana, condition: torch.Tensor, *, steps: int = 20,
                  shape: tuple[int, int, int] | None = None,
                  cfg: SanaConfig | None = None,
                  noise: torch.Tensor | None = None,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """Euler-integrate from noise to a predicted latent. Returns [N, C, H, W].

    `condition` is the adapter output, [N, 72, 2304]. With the horizon embedding
    installed the rows must be horizon-major over all K horizons. Pass `noise` to start
    several calls from the same draw.
    """
    cfg = cfg or SanaConfig()
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    b = condition.shape[0]
    c, h, w = shape or (cfg.in_channels, cfg.sample_size, cfg.sample_size)

    if noise is None:
        noise = torch.randn(b, c, h, w, device=condition.device,
                            dtype=torch.float32, generator=generator)
    if tuple(noise.shape) != (b, c, h, w):
        raise ValueError(f"noise must be {(b, c, h, w)}, got {tuple(noise.shape)}")

    z = noise.to(torch.float32)
    step = 1.0 / steps
    for k in range(steps):
        tau = 1.0 - k * step                       # from 1 down to 1 / steps
        t = torch.full((b,), tau * cfg.timestep_scale,
                       device=z.device, dtype=condition.dtype)
        v = sana(hidden_states=z.to(condition.dtype), encoder_hidden_states=condition,
                 timestep=t, return_dict=False)[0].float()
        z = z - step * v                           # towards the data
    return z


@torch.no_grad()
def decode(ae, z: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """Scaled latents [B, 32, 16, 16] -> images [B, 3, 512, 512] in [-1, 1]."""
    if ae.decoder is None:
        raise RuntimeError(
            "this DC-AE was built with keep_decoder=False. Training never needs the "
            "decoder, so it is dropped by default; pass keep_decoder=True to look at "
            "pictures.")
    return ae.decode(z.to(ae.dtype) / scaling_factor).sample


def to_uint8(img: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] in [-1, 1] -> [B, H, W, 3] uint8. Out-of-range values are clamped."""
    return (((img.float().clamp(-1, 1) + 1) * 127.5)
            .round().to(torch.uint8).permute(0, 2, 3, 1).contiguous())


@torch.no_grad()
def predict_frames(head, image_tokens: torch.Tensor, final_memory: torch.Tensor,
                   ae, loss_cfg: WorldLossConfig, *, steps: int = 20,
                   noise: torch.Tensor | None = None) -> dict[int, torch.Tensor]:
    """Sample and decode every horizon. Returns {horizon_frame: [B, H, W, 3] uint8}.

    All horizons go through SANA in one horizon-major batch, as in `world_loss`: the
    horizon embedding reads the horizon from the row position. The same `noise`
    [B, C, H, W] is used for every horizon.
    """
    k = loss_cfg.num_horizons
    b = image_tokens.shape[0]
    cfg = head.cfg

    base = head.adapter(image_tokens, final_memory)                  # [B, 72, 2304]
    if getattr(head.adapter.cfg, "num_horizons", 1) == k and k > 1:
        cond = torch.cat([head.adapter.for_horizon(base, i) for i in range(k)], dim=0)
    else:
        cond = base.repeat(k, 1, 1)                                  # [K*B, 72, 2304]

    c, h, w = cfg.sana.in_channels, cfg.sana.sample_size, cfg.sana.sample_size
    if noise is None:
        noise = torch.randn(b, c, h, w, device=base.device, dtype=torch.float32)
    if tuple(noise.shape) != (b, c, h, w):
        raise ValueError(f"noise must be {(b, c, h, w)}, got {tuple(noise.shape)}")
    noise_k = noise.repeat(k, 1, 1, 1)                               # same draw per horizon

    z = sample_latent(head.sana, cond, steps=steps, cfg=cfg.sana, noise=noise_k)
    z = z.reshape(k, b, c, h, w)                                     # back to [K, B, ...]

    return {hf: to_uint8(decode(ae, z[i], cfg.dcae.scaling_factor))
            for i, hf in enumerate(loss_cfg.horizon_frames)}
