"""Attention masks for the VLM prefix.

`att_masks` [B, N] int marks where a new attention block starts and `pad_masks` [B, N]
bool marks the real positions; together they give the 2D visibility matrix. The state
and action masks used elsewhere mark valid values and do not affect attention.
"""

from __future__ import annotations

import torch


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Block-causal visibility from block-start markers.

    Token i may attend token j iff cumsum(att_masks)[j] <= cumsum(att_masks)[i]: a block
    is bidirectional inside and sees every earlier block, never a later one.

    pad_masks: [B, N] bool, True for real positions.
    att_masks: [B, N] int or bool, 1 where a new block starts.
    Returns [B, N, N] bool; out[b, i, j] is True if query i may attend key j.
    """
    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError(f"expected 2D masks, got {att_masks.shape} and {pad_masks.shape}")
    if att_masks.shape != pad_masks.shape:
        raise ValueError(f"shape mismatch: {att_masks.shape} vs {pad_masks.shape}")

    cumsum = torch.cumsum(att_masks.to(torch.int64), dim=1)          # [B, N]
    visible = cumsum[:, None, :] <= cumsum[:, :, None]               # [B, N, N]
    both_real = pad_masks[:, None, :] & pad_masks[:, :, None]        # [B, N, N]
    out = visible & both_real

    # Every position attends to itself, so that a padding row is not fully masked
    # (softmax over a fully masked row is NaN).
    eye = torch.eye(out.shape[-1], dtype=torch.bool, device=out.device)
    return out | eye


def build_prefix_att_masks(
    n_image: int,
    n_fov: int,
    n_language: int,
    n_state: int,
    n_memory: int,
    *,
    memory_bidirectional: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Block-start markers for one prefix. Returns [N].

        prefix = [ image | FoV | language | state | memory    ]
        marks  = [ 0 ..... 0 ... 0 ...... |   1   | 1 0 ... 0 ]

    Image, FoV and language form one block. The state sees them and is not seen by them.
    The memory slots come last, see everything and see each other; with
    `memory_bidirectional=False` every slot opens its own block and they are causal.
    """
    marks: list[int] = []
    marks += [0] * n_image
    marks += [0] * n_fov
    marks += [0] * n_language
    if n_state:
        marks += [1] + [0] * (n_state - 1)
    if n_memory:
        if memory_bidirectional:
            marks += [1] + [0] * (n_memory - 1)
        else:
            marks += [1] * n_memory
    return torch.tensor(marks, dtype=torch.int64, device=device)


def build_suffix_att_masks(chunk_size: int, *, causal: bool = True,
                           device: torch.device | None = None) -> torch.Tensor:
    """Block-start markers for the action tokens. Returns [chunk_size].

    With `causal`, every action opens its own block: action h sees actions <= h.
    """
    return torch.full((chunk_size,), 1 if causal else 0, dtype=torch.int64, device=device)


def additive_mask(att_2d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Boolean visibility -> additive mask for scaled-dot-product attention."""
    neg = torch.finfo(dtype).min
    return torch.where(att_2d, torch.zeros((), dtype=dtype, device=att_2d.device),
                       torch.full((), neg, dtype=dtype, device=att_2d.device))
