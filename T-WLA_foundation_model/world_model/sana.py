"""The 4-block SANA transformer and the loading of its pretrained weights.

The four blocks are copied from a 28-layer SANA checkpoint as an initialisation and
trained from there; no teacher model is run. `remap_keys` and `provenance` are plain
Python, `build` and `load_inherited` need diffusers.
"""

from __future__ import annotations

import re
from typing import Iterable

from .config import SanaConfig

_BLOCK_RE = re.compile(r"^(transformer_blocks\.)(\d+)(\..*)$")


def remap_keys(dst_keys: Iterable[str], source_indices: Iterable[int]) -> dict[str, str]:
    """Map each destination state-dict key to the source key it is loaded from.

    Block indices are rewritten (0, 1, 2, 3 -> 0, 9, 18, 27 by default); every other
    key keeps its name.
    """
    src = list(source_indices)
    out: dict[str, str] = {}
    for k in dst_keys:
        m = _BLOCK_RE.match(k)
        if not m:
            out[k] = k
            continue
        head, idx, tail = m.group(1), int(m.group(2)), m.group(3)
        if idx >= len(src):
            raise ValueError(
                f"destination block {idx} has no entry in source_indices {src} "
                f"(key {k!r})")
        out[k] = f"{head}{src[idx]}{tail}"
    return out


def provenance(cfg: SanaConfig) -> dict:
    """Where the initial weights came from: source repo, revision and block mapping."""
    return {
        "source_repo": cfg.source_repo,
        "source_revision": cfg.source_revision,
        "source_block_indices": list(cfg.source_block_indices),
        "source_num_layers": cfg.source_num_layers,
        "num_layers": cfg.num_layers,
        "in_channels": cfg.in_channels,
        "out_channels": cfg.out_channels,
        "distillation": False,
        "teacher_forward": False,
    }


def build(cfg: SanaConfig | None = None):
    """A `SanaTransformer2DModel` with `cfg.num_layers` blocks, randomly initialised."""
    from diffusers import SanaTransformer2DModel   # here, so `remap_keys` needs no diffusers

    cfg = cfg or SanaConfig()
    model = SanaTransformer2DModel(**cfg.diffusers_kwargs())

    n_blocks = len(model.transformer_blocks)
    if n_blocks != cfg.num_layers:
        raise RuntimeError(f"expected {cfg.num_layers} blocks, built {n_blocks}")
    return model


def load_inherited(model, cfg: SanaConfig | None = None, *, strict: bool = True) -> dict:
    """Copy the selected source blocks and the weights outside the blocks into `model`.

    Tensors are read one at a time from the source safetensors file, so the full
    checkpoint is never held in memory. Returns the provenance dict.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    cfg = cfg or SanaConfig()
    if not cfg.source_revision:
        raise ValueError(
            "source_revision is not pinned. A moving branch changes the initialisation "
            "silently -- the run trains fine and the curve is simply different, with "
            "nothing to point at. Set SanaConfig.source_revision to a commit sha.")

    path = hf_hub_download(cfg.source_repo, "transformer/diffusion_pytorch_model.safetensors",
                           revision=cfg.source_revision)

    dst = model.state_dict()
    mapping = remap_keys(dst.keys(), cfg.source_block_indices)

    loaded = {}
    with safe_open(path, framework="pt") as f:
        available = set(f.keys())
        missing = sorted({v for v in mapping.values()} - available)
        if missing:
            raise KeyError(f"source is missing {len(missing)} tensors, first few: {missing[:5]}")
        for dst_key, src_key in mapping.items():
            t = f.get_tensor(src_key)
            if tuple(t.shape) != tuple(dst[dst_key].shape):
                raise ValueError(
                    f"shape mismatch {dst_key}: destination {tuple(dst[dst_key].shape)} "
                    f"vs source {src_key} {tuple(t.shape)}")
            loaded[dst_key] = t

    model.load_state_dict(loaded, strict=strict)

    n = sum(p.numel() for p in model.parameters())
    if n > cfg.max_parameters:
        raise RuntimeError(f"{n:,} parameters exceeds the {cfg.max_parameters:,} budget")

    prov = provenance(cfg)
    prov["num_parameters"] = n
    prov["tensors_copied"] = len(loaded)
    return prov


def check_timestep_scale(model, cfg: SanaConfig | None = None, *, device=None) -> dict:
    """Check which timestep range the model responds to: tau or tau * `timestep_scale`.

    Runs one latent and condition at tau = 0.1 and tau = 0.9 under each scaling. Returns
    the relative output difference for both and a verdict; the scaling that separates
    the two noise levels is the one to use.
    """
    import torch

    cfg = cfg or SanaConfig()
    device = device or next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    g = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(2, cfg.in_channels, cfg.sample_size, cfg.sample_size,
                    generator=g).to(device=device, dtype=dtype)
    cond = torch.randn(2, 72, cfg.caption_channels, generator=g).to(device=device, dtype=dtype)

    out = {}
    with torch.no_grad():
        for scale in (1.0, cfg.timestep_scale):
            lo = torch.full((2,), 0.1 * scale, device=device, dtype=dtype)
            hi = torch.full((2,), 0.9 * scale, device=device, dtype=dtype)
            a = model(hidden_states=z, encoder_hidden_states=cond, timestep=lo,
                      return_dict=False)[0].float()
            b = model(hidden_states=z, encoder_hidden_states=cond, timestep=hi,
                      return_dict=False)[0].float()
            denom = a.norm().clamp_min(1e-12)
            out[f"rel_diff_at_scale_{scale:g}"] = float((a - b).norm() / denom)

    one = out["rel_diff_at_scale_1"]
    many = out[f"rel_diff_at_scale_{cfg.timestep_scale:g}"]
    out["timestep_scale"] = cfg.timestep_scale
    out["verdict"] = (
        f"use timestep_scale = {cfg.timestep_scale:g} "
        f"({many:.4f} separation vs {one:.4f} at 1.0)" if many > 4 * one else
        "use timestep_scale = 1.0 (tau is already the model's convention)"
        if one > 4 * many else
        "INCONCLUSIVE -- neither scale separates the two taus; the time embedding may not "
        "be wired, which would be a construction bug rather than a convention question")
    return out
