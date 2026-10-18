"""Masked loss over groups of action slots.

Of the 192 action slots 138 carry data, and an episode may annotate one hand only, so the
squared error is averaged over the unmasked slots of each group, not over everything.
"""

from __future__ import annotations

import torch


def masked_group_loss(pred: torch.Tensor, target: torch.Tensor,
                      action_mask: torch.Tensor,
                      groups,
                      ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked squared error of `pred` against `target`, [B, T, D], per group and combined.

    `groups` is an iterable of `(name, lo, hi, weight)`. A group's value is its weight times
    the mean squared error over the unmasked elements of slots `lo:hi`. The total averages
    the group values, each weighted by the number of (sample, step) pairs in which the
    group's first slot is unmasked. Returns (total, dict of values to log).
    """
    return masked_group_loss_sq((pred - target) ** 2, action_mask, groups)


def masked_group_loss_sq(square: torch.Tensor, action_mask: torch.Tensor, groups
                         ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The same, from a per-element squared error [B, T, D] that is already computed."""
    m = action_mask.to(square.dtype)
    square_delta = square * m

    parts: dict[str, torch.Tensor] = {}
    counts: dict[str, torch.Tensor] = {}
    dens: dict[str, torch.Tensor] = {}
    for name, lo, hi, weight in groups:
        n = m[:, :, lo:hi].sum()
        val = square_delta[:, :, lo:hi].sum() / n if n > 0 else square_delta.new_zeros(())
        parts[name] = val * weight
        counts[name] = m[:, :, lo].sum()
        dens[name] = n

    total = sum(counts.values())
    if float(total) == 0.0:
        loss = square_delta[0, 0, 0]        # nothing is unmasked: a zero that stays in the graph
    else:
        loss = sum(parts[k] * counts[k] for k in counts) / total

    logged = {k: v.detach() for k, v in parts.items()}
    # `<group>_n` is the weight of the group in the total; `<group>_den` is the number of
    # unmasked elements its mean was taken over (needed to average means across batches).
    logged.update({f"{k}_n": v.detach() for k, v in counts.items()})
    logged.update({f"{k}_den": v.detach() for k, v in dens.items()})
    return loss, logged
