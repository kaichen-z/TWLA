"""Horizon conditioning through SANA's modulation path.

With K target frames the condition (E_t, m_f) is the same for all of them, so the model
has to be told which frame is asked for. As DiT does for class labels, a learned
per-horizon vector is added to the timestep embedding: c = timestep_emb(tau) + horizon_emb(k).
The horizon of a row is read from its position in the batch, which is horizon-major.
"""

from __future__ import annotations

import torch
from torch import nn


def horizon_rows(n: int, num_horizons: int, device=None) -> torch.Tensor:
    """Horizon index of each row in a horizon-major batch of `n = B * K` rows.

    Rows 0..B-1 are horizon 0, B..2B-1 horizon 1: `n = 6, K = 2` gives [0, 0, 0, 1, 1, 1].
    """
    if num_horizons <= 0:
        raise ValueError(f"num_horizons must be positive, got {num_horizons}")
    if n % num_horizons:
        raise ValueError(
            f"batch of {n} rows is not divisible by {num_horizons} horizons. The rows "
            "carry the horizon by position, so a ragged batch would silently mislabel "
            "every row after the first block.")
    per = n // num_horizons
    return torch.arange(n, device=device) // per


class HorizonTimestepEmbedding(nn.Module):
    """Wraps SANA's timestep embedder and adds a learned per-horizon vector, zero at init.

    The wrapped module is `time_embed.emb` of diffusers' `AdaLayerNormSingle`, so the
    vector reaches both the per-block modulation and the output norm. In the state dict
    the original weights move to `time_embed.emb.inner.*`.
    """

    def __init__(self, inner: nn.Module, num_horizons: int, dim: int):
        super().__init__()
        if num_horizons < 2:
            raise ValueError(
                f"num_horizons is {num_horizons}; with fewer than 2 targets there is "
                "nothing to disambiguate and the embedding would be a constant offset "
                "that `linear`'s bias already spans. Do not install the wrapper.")
        self.inner = inner
        self.num_horizons = num_horizons
        self.horizon_emb = nn.Parameter(torch.zeros(num_horizons, dim))

    def forward(self, timestep, *args, **kwargs):
        emb = self.inner(timestep, *args, **kwargs)          # [B*K, dim]
        idx = horizon_rows(emb.shape[0], self.num_horizons, device=emb.device)
        return emb + self.horizon_emb[idx].to(emb.dtype)

    def extra_repr(self) -> str:
        return f"num_horizons={self.num_horizons}, dim={self.horizon_emb.shape[1]}"


def install(sana, num_horizons: int) -> nn.Parameter:
    """Wrap `sana.time_embed.emb` in place. Returns the new parameter, so the caller can
    put it in its own optimiser group."""
    te = getattr(sana, "time_embed", None)
    if te is None or not hasattr(te, "emb"):
        raise AttributeError(
            "expected `sana.time_embed.emb` (diffusers' AdaLayerNormSingle). The hook "
            "point is version-coupled; if diffusers moved it, this must be re-checked "
            "against `AdaLayerNormSingle.forward` rather than patched blindly.")
    if isinstance(te.emb, HorizonTimestepEmbedding):
        raise RuntimeError("the horizon embedding is already installed")

    dim = sana.config.num_attention_heads * sana.config.attention_head_dim
    te.emb = HorizonTimestepEmbedding(te.emb, num_horizons, dim)
    return te.emb.horizon_emb


def is_installed(sana) -> bool:
    te = getattr(sana, "time_embed", None)
    return te is not None and isinstance(getattr(te, "emb", None), HorizonTimestepEmbedding)
