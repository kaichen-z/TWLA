"""The memory module: learned query slots that attend to the bank.

    q   = M_learn                                               [B, 8, 960]
    kv  = [ M_learn + slot_emb ; bank + slot_emb + age_emb ]    [B, 8 + 64, 960]
    m_t = blocks(q, kv)       the same kv for every block       [B, 8, 960]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MemoryConfig
from .memory_bank import BankRead, age_to_index


class RMSNorm(nn.Module):
    """RMS normalisation, computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward without bias: down(silu(gate(x)) * up(x))."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MemoryBlock(nn.Module):
    """One retrieval step: pre-norm cross-attention and SwiGLU, each behind a tanh gate.

    With `qk_norm`, q and k are normalised per head and the logit scale is learned, which
    bounds the attention logits. With `zero_init_residual_gate` the gates start at zero
    and the block is the identity at initialisation.
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        d, h = cfg.dim, cfg.num_heads
        if d % h:
            raise ValueError(f"dim {d} not divisible by num_heads {h}")
        self.num_heads = h
        self.head_dim = d // h

        self.norm_q = RMSNorm(d)
        self.norm_kv = RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
            # Log of the logit scale, initialised to 1/sqrt(head_dim).
            self.logit_scale = nn.Parameter(torch.tensor(float(self.head_dim) ** -0.5).log())

        self.norm_ffn = RMSNorm(d)
        self.ffn = SwiGLU(d, cfg.ffn_intermediate)

        init = 0.0 if cfg.zero_init_residual_gate else 3.0   # tanh(3) ~= 0.995
        self.g_attn = nn.Parameter(torch.tensor(init))
        self.g_ffn = nn.Parameter(torch.tensor(init))

    def _shape(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, kv: torch.Tensor,
                kv_valid: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, S, D] query stream. kv: [B, N, D]. kv_valid: [B, N] bool."""
        q = self._shape(self.q_proj(self.norm_q(x)))
        kvn = self.norm_kv(kv)
        k = self._shape(self.k_proj(kvn))
        v = self._shape(self.v_proj(kvn))

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
            # The learned scale is folded into q: `scale=` must be a Python float.
            q = q * self.logit_scale.exp().to(q.dtype)
            scale = 1.0
        else:
            scale = self.head_dim ** -0.5

        attn_mask = None
        if kv_valid is not None:
            attn_mask = kv_valid[:, None, None, :]        # [B, 1, 1, N] broadcast over heads

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
        out = out.transpose(1, 2).reshape(x.shape)
        x = x + torch.tanh(self.g_attn) * self.o_proj(out)
        x = x + torch.tanh(self.g_ffn) * self.ffn(self.norm_ffn(x))
        return x


class MemoryModule(nn.Module):
    """Queries: S learned slots. Keys and values: the slots followed by the bank."""

    def __init__(self, cfg: MemoryConfig, num_age_values: int):
        super().__init__()
        self.cfg = cfg
        d, s = cfg.dim, cfg.num_slots

        self.query = nn.Parameter(torch.randn(1, s, d) * 0.02)
        self.slot_emb = nn.Embedding(s, d)
        self.age_emb = nn.Embedding(num_age_values, d)
        nn.init.normal_(self.slot_emb.weight, std=0.02)
        nn.init.normal_(self.age_emb.weight, std=0.02)

        if cfg.age_fusion == "concat_mlp":
            # Fuse content, slot and age with an MLP instead of adding them.
            self.fuse = nn.Sequential(nn.Linear(3 * d, d, bias=False), nn.SiLU(),
                                      nn.Linear(d, d, bias=False))
        else:
            self.fuse = None

        self.blocks = nn.ModuleList(MemoryBlock(cfg) for _ in range(cfg.num_blocks))

    # ------------------------------------------------------------------

    def build_kv(self, bank: BankRead) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the slot and age embeddings to the bank entries and prepend the queries.

        Returns (kv [B, S + C*S, D], kv_valid [B, S + C*S]).
        """
        b = bank.entries.shape[0]
        slot_vec = self.slot_emb(bank.slot_id)                                   # [B, N, D]
        age_vec = self.age_emb(age_to_index(bank.age, self.cfg.bank_capacity))   # [B, N, D]

        if self.fuse is None:
            entries = bank.entries + slot_vec + age_vec
        else:
            entries = self.fuse(torch.cat([bank.entries, slot_vec, age_vec], dim=-1))

        # Zero the empty entries; they are also masked in the attention.
        entries = entries * bank.valid.unsqueeze(-1).to(entries.dtype)

        slot_ids = torch.arange(self.cfg.num_slots, device=bank.entries.device)
        queries = self.query.expand(b, -1, -1) + self.slot_emb(slot_ids)[None]   # [B, S, D]
        q_valid = torch.ones(b, self.cfg.num_slots, dtype=torch.bool,
                             device=bank.entries.device)

        return torch.cat([queries, entries], dim=1), torch.cat([q_valid, bank.valid], dim=1)

    def forward(self, bank: BankRead) -> torch.Tensor:
        """Bank read -> m_t [B, S, D], the memory tokens placed in the VLM prefix."""
        kv, kv_valid = self.build_kv(bank)
        x = self.query.expand(bank.entries.shape[0], -1, -1)
        for blk in self.blocks:
            x = blk(x, kv, kv_valid)
        return x
