"""WorldAdapter: turns the model's image tokens and memory tokens into SANA's condition.

    E_t [B, 64, 960]  --norm--> +image_type  --+
                                               |--concat--> [B, 72, 960] --Linear--> [B, 72, 2304]
    m_f [B,  8, 960]  --norm--> +memory_type --+
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from ..TWLA.memory_module import RMSNorm
except ImportError:   # imported with T-WLA_foundation_model itself on sys.path
    from TWLA.memory_module import RMSNorm

from .config import AdapterConfig


class WorldAdapter(nn.Module):
    """[B, 64, 960] image tokens + [B, 8, 960] memory tokens -> [B, 72, 2304], the
    width SANA's caption projection expects."""

    def __init__(self, cfg: AdapterConfig | None = None):
        super().__init__()
        self.cfg = cfg or AdapterConfig()
        c = self.cfg

        self.image_norm = RMSNorm(c.in_dim, eps=c.norm_eps)
        self.memory_norm = RMSNorm(c.in_dim, eps=c.norm_eps)

        # Learned tags that tell the two groups apart after normalisation. Zero at init,
        # broadcast over [B, N, 960].
        self.image_type = nn.Parameter(torch.zeros(1, 1, c.in_dim))
        self.memory_type = nn.Parameter(torch.zeros(1, 1, c.in_dim))

        self.proj = nn.Linear(c.in_dim, c.out_dim, bias=c.proj_bias)

        # One tag per target frame, added after the projection.
        self.horizon_type = (
            nn.Parameter(torch.zeros(c.num_horizons, 1, c.out_dim))
            if c.num_horizons > 1 else None)

    def forward(self, image_tokens: torch.Tensor,
                final_memory: torch.Tensor) -> torch.Tensor:
        """`image_tokens` is E_t, the image tokens that enter the decoder; `final_memory`
        is m_f, the decoder's last `num_slots` hidden states. Pass m_f without `detach()`
        when `L_world` should reach the memory."""
        c = self.cfg
        self._check(image_tokens, "image_tokens", c.image_tokens)
        self._check(final_memory, "final_memory", c.memory_tokens)

        if c.norm_first:
            image = self.image_norm(image_tokens)
            memory = self.memory_norm(final_memory)
        else:                      # ablation
            image, memory = image_tokens, final_memory

        image = image + self.image_type.to(image.dtype)
        memory = memory + self.memory_type.to(memory.dtype)

        condition = torch.cat([image, memory], dim=1)         # [B, 72, 960]
        return self.proj(condition)                           # [B, 72, 2304]

    def for_horizon(self, condition: torch.Tensor, k: int) -> torch.Tensor:
        """Add the tag of target frame `k` to the output of `forward`."""
        if not 0 <= k < self.cfg.num_horizons:
            raise ValueError(f"horizon {k} outside [0, {self.cfg.num_horizons})")
        if self.horizon_type is None:
            return condition                     # single horizon: no tag
        return condition + self.horizon_type[k].to(condition.dtype)

    def _check(self, x: torch.Tensor, name: str, tokens: int) -> None:
        """Raise unless `x` is [B, tokens, in_dim]."""
        if x.dim() != 3:
            raise ValueError(f"{name}: expected [B, N, {self.cfg.in_dim}], got {tuple(x.shape)}")
        if x.shape[1] != tokens or x.shape[2] != self.cfg.in_dim:
            raise ValueError(
                f"{name}: expected [B, {tokens}, {self.cfg.in_dim}], got {tuple(x.shape)}")

    @torch.no_grad()
    def scale_report(self, image_tokens: torch.Tensor,
                     final_memory: torch.Tensor) -> dict[str, float]:
        """RMS of the two inputs before and after normalisation, for logging."""
        def rms(t: torch.Tensor) -> float:
            return float(t.float().pow(2).mean().sqrt())

        return {
            "image_in": rms(image_tokens),
            "memory_in": rms(final_memory),
            "ratio_in": rms(image_tokens) / max(rms(final_memory), 1e-12),
            "image_out": rms(self.image_norm(image_tokens)),
            "memory_out": rms(self.memory_norm(final_memory)),
        }
