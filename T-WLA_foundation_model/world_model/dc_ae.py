"""The frozen DC-AE: encodes a target RGB frame into the 32 latent channels SANA predicts.

Training uses the encoder only, so `build` drops the decoder unless `keep_decoder=True`.
Everything here runs under `no_grad` and returns detached tensors.
"""

from __future__ import annotations

import torch

from .config import DCAEConfig


def build(cfg: DCAEConfig | None = None, *, dtype=None, keep_decoder: bool = False):
    """Load the autoencoder, freeze it and put it in eval mode."""
    from diffusers import AutoencoderDC

    cfg = cfg or DCAEConfig()
    if not cfg.source_revision:
        raise ValueError(
            "DCAEConfig.source_revision is not pinned. The encoder defines what the world "
            "loss is measured against; if it moves, two runs' L_world numbers stop being "
            "comparable and nothing reports it. Set it to a commit sha.")

    model = AutoencoderDC.from_pretrained(
        cfg.source_repo, subfolder=cfg.subfolder, revision=cfg.source_revision,
        torch_dtype=dtype)

    got = int(model.config.latent_channels)
    if got != cfg.latent_channels:
        raise ValueError(
            f"DC-AE has {got} latent channels, config says {cfg.latent_channels}. "
            "SanaConfig.in_channels must match this exactly.")

    repo_sf = getattr(model.config, "scaling_factor", None)
    if repo_sf is not None and abs(float(repo_sf) - cfg.scaling_factor) > 1e-6:
        raise ValueError(
            f"DCAEConfig.scaling_factor is {cfg.scaling_factor} but the checkpoint says "
            f"{float(repo_sf)}. Fix the config rather than the checkpoint.")

    if not keep_decoder:
        model.decoder = None

    model.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def encode(model, images: torch.Tensor, cfg: DCAEConfig | None = None) -> torch.Tensor:
    """[B, 3, 512, 512] in [-1, 1] -> [B, 32, 16, 16], detached, fp32.

    `images` must be on the same canvas as the conditioning frame: `resize_with_pad`
    (picture in the bottom-right corner), then mapped to [-1, 1].
    """
    cfg = cfg or DCAEConfig()
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(f"expected [B, 3, H, W], got {tuple(images.shape)}")
    if images.shape[-1] != cfg.image_size or images.shape[-2] != cfg.image_size:
        raise ValueError(
            f"expected {cfg.image_size}x{cfg.image_size}, got "
            f"{tuple(images.shape[-2:])}. DC-AE compresses by {cfg.spatial_compression}x "
            f"and SanaConfig.sample_size assumes the result is {cfg.latent_size}.")

    z = model.encode(images.to(model.dtype)).latent
    if cfg.apply_scaling:
        z = z * cfg.scaling_factor

    exp = (cfg.latent_channels, cfg.latent_size, cfg.latent_size)
    if tuple(z.shape[1:]) != exp:
        raise RuntimeError(f"DC-AE returned {tuple(z.shape[1:])}, expected {exp}")

    # The flow-matching target is kept in fp32 whatever dtype the encoder ran in.
    return z.detach().float()


@torch.no_grad()
def check_scaling(model, images: torch.Tensor, cfg: DCAEConfig | None = None) -> dict:
    """Report the std of the latents, to check that `apply_scaling` is set correctly.

    `raw_std` is the encoder output, `scaled_std` the same times `scaling_factor`, and
    `used_std` what `encode` returns; it should fall in `expected_std_range`. If
    `raw_std` already does, the autoencoder scales internally.
    """
    cfg = cfg or DCAEConfig()
    raw = model.encode(images.to(model.dtype)).latent.float()
    raw_std = float(raw.std())
    scaled_std = raw_std * cfg.scaling_factor
    used_std = scaled_std if cfg.apply_scaling else raw_std

    lo, hi = cfg.expected_std_range
    return {
        "raw_std": raw_std,
        "scaled_std": scaled_std,
        "used_std": used_std,
        "apply_scaling": cfg.apply_scaling,
        "ok": lo <= used_std <= hi,
        "verdict": (
            "ok" if lo <= used_std <= hi else
            "apply_scaling should be False -- diffusers already scaled it"
            if raw_std <= hi and cfg.apply_scaling else
            "apply_scaling should be True -- latents are unscaled"
            if not cfg.apply_scaling and raw_std > hi else
            "neither setting lands in range; check the checkpoint"),
    }


@torch.no_grad()
def encode_targets(model, frames: torch.Tensor, cfg: DCAEConfig | None = None,
                   chunk: int = 64) -> torch.Tensor:
    """[B, K, 3, 512, 512] -> [B, K, 32, 16, 16], detached.

    Encodes `chunk` images at a time to bound memory. `frames[:, k]` is the target for
    horizon `k`, on the same canvas as the conditioning frame. Rows without a target must
    hold a finite placeholder; the loss masks them.
    """
    cfg = cfg or DCAEConfig()
    if frames.dim() != 5:
        raise ValueError(f"expected [B, K, 3, H, W], got {tuple(frames.shape)}")
    b, k = frames.shape[0], frames.shape[1]

    if not torch.isfinite(frames).all():
        raise ValueError(
            "target frames contain non-finite values. The loader must supply a finite "
            "placeholder for a missing frame and mark it invalid, not NaN -- DC-AE's "
            "normalisation would spread one NaN across the batch before the mask applies.")

    flat = frames.reshape(b * k, *frames.shape[2:])
    outs = [encode(model, flat[i:i + chunk], cfg) for i in range(0, flat.shape[0], chunk)]
    z = torch.cat(outs, dim=0)
    return z.reshape(b, k, *z.shape[1:])
