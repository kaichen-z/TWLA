"""VITRA's action head, a DiT with adaLN-Zero conditioning, re-expressed for the
transformers version used here; the `timm` layers it uses are defined below.

The blocks see [ z | state | a_0 ... a_15 ]: the projected cognition token, the state as
[state ; state_mask] (424 features) and the noised chunk, each token [x_t ; action_mask]
(384 features). `c = z + t`, one vector per sample, drives every adaLN layer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """timm's `RmsNorm` with its defaults: affine, eps 1e-6, computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class Mlp(nn.Module):
    """timm's `Mlp` as DiT configures it: tanh-approximated GELU, no dropout."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """timm's `Attention` with `qkv_bias=True, qk_norm=True, norm_layer=RmsNorm`.

    `attn_mask` is an additive float mask (0 or -inf) of shape [T, T]. The query and key
    norms act per head, over `head_dim`.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)      # each [B, H, N, hd]
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of the integer diffusion step, then a 2-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000
                           ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period)
                          * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        # cos first, then sin: the order the pretrained weights expect
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(freq.to(next(self.mlp.parameters()).dtype))


class ActionEmbedder(nn.Module):
    """`[x_t ; action_mask]` -> hidden. Linear-GELU-Linear."""

    def __init__(self, action_size: int, hidden_size: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(action_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class StateEmbedder(nn.Module):
    """`[state * mask ; mask]` -> hidden."""

    def __init__(self, state_size: int, hidden_size: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(2 * state_size, 4 * hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class LabelEmbedder(nn.Module):
    """Projects the cognition token, with dropout for classifier-free guidance.

    In training a `dropout_prob` fraction of samples get the learned `uncondition` vector
    in place of their condition, before the projection. At sampling the same vector feeds
    the unconditional branch.
    """

    def __init__(self, in_size: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.linear = nn.Linear(in_size, hidden_size)
        if dropout_prob > 0:
            # [1, in_size], broadcast against the [B, 1, in_size] condition
            self.uncondition = nn.Parameter(torch.empty(1, in_size))

    def token_drop(self, conditions: torch.Tensor,
                   generator: torch.Generator | None = None) -> torch.Tensor:
        drop = torch.rand(conditions.shape[0], device=conditions.device,
                          generator=generator) < self.dropout_prob
        return torch.where(drop[:, None, None], self.uncondition.to(conditions.dtype),
                           conditions)

    def forward(self, conditions: torch.Tensor, train: bool,
                generator: torch.Generator | None = None) -> torch.Tensor:
        if train and self.dropout_prob > 0:
            conditions = self.token_drop(conditions, generator)
        return self.linear(conditions)


class DiTBlock(nn.Module):
    """adaLN-Zero block. `c` [B, D] gives shift, scale and gate for both sub-layers."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                attn_mask: torch.Tensor | None) -> torch.Tensor:
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), sh1, sc1), attn_mask)
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh2, sc2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    """VITRA's `DiT` as its config builds it: no past actions, no learned sigma."""

    def __init__(self, cfg: Config):
        super().__init__()
        d, a, v = cfg.dit, cfg.action, cfg.vlm
        self.cfg = cfg
        self.chunk_size = a.chunk_size
        self.in_channels = a.dim * (2 if d.concat_mask_to_input else 1)
        self.out_channels = a.dim
        self.use_state = v.use_state == "DiT"

        self.x_embedder = ActionEmbedder(self.in_channels, d.hidden_size)
        self.t_embedder = TimestepEmbedder(d.hidden_size)
        self.z_embedder = LabelEmbedder(v.hidden_size, d.hidden_size, d.class_dropout_prob)
        if self.use_state:
            self.state_embedder = StateEmbedder(a.state_dim, d.hidden_size)

        # Learned positions for z (+ state) + the chunk, added to the whole sequence.
        self.seq_len = a.chunk_size + (2 if self.use_state else 1)
        self.positional_embedding = nn.Parameter(
            d.hidden_size ** -0.5 * torch.randn(self.seq_len, d.hidden_size))

        self.blocks = nn.ModuleList(
            [DiTBlock(d.hidden_size, d.num_heads, d.mlp_ratio) for _ in range(d.depth)])
        self.final_layer = FinalLayer(d.hidden_size, self.out_channels)
        self.initialize_weights()

    # ------------------------------------------------------------------ init

    def initialize_weights(self) -> None:
        """Xavier on every Linear, then the per-module overrides; the adaLN and output
        layers end at zero."""

        def basic(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(basic)

        for mod in (self.x_embedder, *( [self.state_embedder] if self.use_state else [] )):
            for layer in mod.projector:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.constant_(layer.bias, 0)

        if self.z_embedder.dropout_prob > 0:
            nn.init.normal_(self.z_embedder.uncondition, std=0.02)
        nn.init.normal_(self.z_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.z_embedder.linear.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ------------------------------------------------------------------ mask

    def _attn_mask(self, device, dtype) -> torch.Tensor | None:
        """Causal mask over the whole sequence, as upstream: the condition token sees only
        itself, the state token the condition and itself."""
        if not self.cfg.dit.causal_attention:
            return None
        n = self.seq_len
        m = torch.triu(torch.ones(n, n, device=device, dtype=torch.float32), 1)
        return m.masked_fill(m == 1, float("-inf")).to(dtype)

    # ------------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor,
                state: torch.Tensor | None = None,
                state_mask: torch.Tensor | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """`x` [B, T, 2*A] already carries the mask; `z` is [B, 1, vlm_hidden].
        Returns [B, T, A]."""
        seq = [self.z_embedder(z, self.training, generator)]     # [B, 1, D]
        if self.use_state:
            if state is None or state_mask is None:
                raise ValueError("use_state='DiT' but no state was passed")
            sm = state_mask.to(state.dtype)
            seq.append(self.state_embedder(torch.cat([state * sm, sm], dim=-1)))
        seq.append(self.x_embedder(x))

        h = torch.cat(seq, dim=1)
        if h.shape[1] != self.seq_len:
            raise ValueError(f"sequence is {h.shape[1]} long, positions table has "
                             f"{self.seq_len}")
        h = h + self.positional_embedding

        c = seq[0].squeeze(1) + self.t_embedder(t)               # [B, D]
        mask = self._attn_mask(h.device, h.dtype)
        for block in self.blocks:
            h = block(h, c, mask)
        h = self.final_layer(h, c)
        return h[:, -self.chunk_size:, :]
