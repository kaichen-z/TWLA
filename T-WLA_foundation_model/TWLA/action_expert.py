"""The flow-matching action expert.

Every layer attends to the same 8 memory tokens (`m_f` projected to the expert width).
Even layers (SA): K/V = [memory 8 ; actions 16], actions causal.
Odd layers (CA):  K/V = memory 8 only, so the actions do not see each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExpertConfig
from .memory_module import RMSNorm, SwiGLU


@dataclass
class ExpertContext:
    """What stays fixed across the Euler steps of one control step."""

    mem: torch.Tensor        # [B, S(+1), W]
    sa_mask: torch.Tensor    # [B, 1, T, S(+1)+T]
    ca_mask: torch.Tensor    # [B, 1, T, S(+1)]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Grouped-query attention: [B, H_kv, N, D] -> [B, H_kv * n_rep, N, D]."""
    if n_rep == 1:
        return x
    b, h, n, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, n, d).reshape(b, h * n_rep, n, d)


class ExpertLayer(nn.Module):
    """One layer. With `is_self_attn` the action tokens are part of the K/V."""

    def __init__(self, cfg: ExpertConfig, is_self_attn: bool):
        super().__init__()
        self.cfg = cfg
        self.is_self_attn = is_self_attn
        h, kvh, hd, w = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size
        self.n_rep = h // kvh

        self.norm_attn = RMSNorm(w)
        self.q_proj = nn.Linear(w, h * hd, bias=False)     # 720 -> 960
        self.k_proj = nn.Linear(w, kvh * hd, bias=False)   # 720 -> 320
        self.v_proj = nn.Linear(w, kvh * hd, bias=False)
        self.o_proj = nn.Linear(h * hd, w, bias=False)     # 960 -> 720

        self.norm_ffn = RMSNorm(w)
        self.ffn = SwiGLU(w, _swiglu_hidden(w))

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                attn_mask: torch.Tensor | None) -> torch.Tensor:
        """x: [B, T, W] action tokens. memory: [B, S, W], already projected.
        attn_mask: [B, 1, T, K] additive, K = S (+ T in SA layers)."""
        cfg = self.cfg
        hn = self.norm_attn(x)
        q = _split(self.q_proj(hn), cfg.num_heads, cfg.head_dim)

        # The memory is not in the residual stream; each layer projects it with its own
        # k_proj and v_proj.
        mem_n = self.norm_attn(memory)
        k = _split(self.k_proj(mem_n), cfg.num_kv_heads, cfg.head_dim)
        v = _split(self.v_proj(mem_n), cfg.num_kv_heads, cfg.head_dim)

        if self.is_self_attn:
            ka = _split(self.k_proj(hn), cfg.num_kv_heads, cfg.head_dim)
            va = _split(self.v_proj(hn), cfg.num_kv_heads, cfg.head_dim)
            k = torch.cat([k, ka], dim=2)      # along the sequence axis
            v = torch.cat([v, va], dim=2)

        out = F.scaled_dot_product_attention(
            q, repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep), attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)

        x = x + self.o_proj(out)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class ActionExpert(nn.Module):
    """Predicts the flow velocity of an action chunk, conditioned on the memory tokens."""

    def __init__(self, cfg: ExpertConfig, chunk_size: int, action_dim: int,
                 num_slots: int, state_dim_with_mask: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.chunk_size = chunk_size
        self.num_slots = num_slots
        w = cfg.hidden_size

        # The mask is concatenated to the action (192 -> 384): zero is a valid normalised
        # value, so the network needs the mask to tell real data from padding.
        self.action_in_proj = nn.Linear(2 * action_dim, w, bias=True)
        self.action_out_proj = nn.Linear(w, action_dim, bias=True)

        # tau is fused into every action token by concatenation and an MLP.
        self.time_mlp_in = nn.Linear(2 * w, w, bias=True)
        self.time_mlp_out = nn.Linear(w, w, bias=True)

        # Learned absolute positions for the action tokens.
        self.action_pos = nn.Parameter(torch.zeros(1, chunk_size, w))
        nn.init.normal_(self.action_pos, std=0.02)

        self.register_buffer("time_freqs", _time_frequencies(w), persistent=False)

        self.cond_proj = nn.Linear(cfg.cond_proj_in, w, bias=False)
        self.expert_slot_emb = (nn.Embedding(num_slots, w)
                                if cfg.expert_slot_embedding else None)
        if self.expert_slot_emb is not None:
            nn.init.normal_(self.expert_slot_emb.weight, std=0.02)

        self.direct_state_proj = (nn.Linear(state_dim_with_mask, w, bias=True)
                                  if state_dim_with_mask else None)

        self.layers = nn.ModuleList(
            ExpertLayer(cfg, is_self_attn=(i % cfg.self_attn_every_n_layers == 0))
            for i in range(cfg.num_layers))
        self.norm_out = RMSNorm(w)

        # Zero-initialised output layer: the predicted velocity starts at zero. Train with
        # a learning-rate warmup (`OptimConfig.warmup_steps`).
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

    # ------------------------------------------------------------------

    def embed_memory(self, m_f: torch.Tensor) -> torch.Tensor:
        """m_f [B, S, 960] -> [B, S, W]. Computed once and used by every layer."""
        mem = self.cond_proj(m_f)
        if self.expert_slot_emb is not None:
            ids = torch.arange(self.num_slots, device=m_f.device)
            mem = mem + self.expert_slot_emb(ids)[None]
        return mem

    def embed_actions(self, noisy_actions: torch.Tensor, action_mask: torch.Tensor,
                      tau: torch.Tensor) -> torch.Tensor:
        """noisy_actions [B, T, A], action_mask [B, T, A], tau [B] -> [B, T, W]."""
        x = torch.cat([noisy_actions * action_mask.to(noisy_actions.dtype),
                       action_mask.to(noisy_actions.dtype)], dim=-1)
        a = self.action_in_proj(x)
        t = sinusoidal_time_embedding(tau, a.shape[-1], freqs=self.time_freqs).to(a.dtype)
        t = t[:, None, :].expand_as(a)
        a = self.time_mlp_out(F.silu(self.time_mlp_in(torch.cat([a, t], dim=-1))))
        return a + self.action_pos

    def build_masks(self, batch: int, device: torch.device,
                    dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Additive masks (sa, ca) of shape [B, 1, T, S + T] and [B, 1, T, S].

        Memory columns are always visible. The action block of the SA mask is
        lower-triangular when `causal_actions` is set.
        """
        t, s = self.chunk_size, self.num_slots
        neg = torch.finfo(dtype).min

        ca = torch.zeros(1, 1, t, s, dtype=dtype, device=device)

        act = torch.ones(t, t, dtype=torch.bool, device=device)
        if self.cfg.causal_actions:
            act = torch.tril(act)
        act_add = torch.where(act, torch.zeros((), dtype=dtype, device=device),
                              torch.full((), neg, dtype=dtype, device=device))
        sa = torch.cat([ca.expand(1, 1, t, s), act_add[None, None]], dim=-1)
        return sa.expand(batch, 1, t, s + t), ca.expand(batch, 1, t, s)

    def prepare(self, m_f: torch.Tensor,
                state_with_mask: torch.Tensor | None = None) -> "ExpertContext":
        """Compute what does not change across the Euler steps: memory tokens and masks."""
        mem = self.embed_memory(m_f)
        if self.direct_state_proj is None:
            if state_with_mask is not None:
                raise ValueError("state was passed but direct_state_to_expert is off")
        else:
            if state_with_mask is None:
                raise ValueError("direct_state_to_expert is on but no state was passed")
            mem = torch.cat([mem, self.direct_state_proj(state_with_mask)[:, None]], dim=1)

        sa_mask, ca_mask = self.build_masks(m_f.shape[0], m_f.device, mem.dtype)
        if self.direct_state_proj is not None:
            pad = torch.zeros(*sa_mask.shape[:-1], 1, dtype=sa_mask.dtype, device=sa_mask.device)
            sa_mask = torch.cat([sa_mask[..., :self.num_slots], pad,
                                 sa_mask[..., self.num_slots:]], dim=-1)
            ca_mask = torch.cat([ca_mask, pad], dim=-1)
        return ExpertContext(mem=mem, sa_mask=sa_mask, ca_mask=ca_mask)

    def velocity(self, ctx: "ExpertContext", noisy_actions: torch.Tensor,
                 action_mask: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """One velocity evaluation with a prepared context. Returns [B, T, A]."""
        x = self.embed_actions(noisy_actions, action_mask, tau)
        for layer in self.layers:
            x = layer(x, ctx.mem, ctx.sa_mask if layer.is_self_attn else ctx.ca_mask)
        return self.action_out_proj(self.norm_out(x))

    def forward(self, m_f: torch.Tensor, noisy_actions: torch.Tensor,
                action_mask: torch.Tensor, tau: torch.Tensor,
                state_with_mask: torch.Tensor | None = None) -> torch.Tensor:
        """prepare() followed by velocity(), as used in training."""
        ctx = self.prepare(m_f, state_with_mask)
        return self.velocity(ctx, noisy_actions, action_mask, tau)


# ---------------------------------------------------------------------- helpers

def _split(t: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    b, n, _ = t.shape
    return t.view(b, n, heads, head_dim).transpose(1, 2)


def _swiglu_hidden(dim: int, multiplier: int = 4, multiple_of: int = 256) -> int:
    """SwiGLU hidden width with the parameter count of a `multiplier`x MLP, rounded up
    to a multiple of `multiple_of`. 720 -> 2048."""
    h = int(2 * dim / 3)
    h = multiplier * h
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


def _time_frequencies(dim: int, min_period: float = 4e-3,
                      max_period: float = 4.0) -> torch.Tensor:
    """Log-spaced frequencies, with periods from `max_period` down to `min_period`."""
    half = dim // 2
    return torch.exp(torch.linspace(
        torch.log(torch.tensor(1.0 / max_period)),
        torch.log(torch.tensor(1.0 / min_period)), half))


def sinusoidal_time_embedding(tau: torch.Tensor, dim: int,
                              freqs: torch.Tensor | None = None) -> torch.Tensor:
    """tau [B] -> [B, dim]. Pass the cached `freqs` buffer to avoid rebuilding it."""
    if freqs is None:
        freqs = _time_frequencies(dim).to(tau.device)
    ang = tau[:, None].float() * freqs[None].to(tau.device) * 2 * torch.pi
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb
