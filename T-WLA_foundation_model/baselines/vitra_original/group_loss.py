"""Masked group loss over channel ranges of the action.

This baseline uses four groups: a 6-D wrist block and a finger block per hand. The mask
selects the channels that carry data; a sample may annotate only one hand.
"""

from __future__ import annotations

import torch


def masked_group_loss(pred: torch.Tensor, target: torch.Tensor,
                      action_mask: torch.Tensor,
                      groups,
                      ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """VITRA's `DiffusionPolicy.loss` over `groups`, an iterable of `(name, lo, hi, weight)`.

    Per group: the masked mean of the squared error over its channels, times its weight.
    The groups are then averaged, each weighted by the number of (sample, step) pairs in
    which its first channel is unmasked. Returns (loss, dict of values to log).
    """
    return masked_group_loss_sq((pred - target) ** 2, action_mask, groups)


def masked_group_loss_sq(square: torch.Tensor, action_mask: torch.Tensor, groups
                         ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The same, from an already computed per-element squared error."""
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
        # nothing is unmasked: return an element of the graph so backward still runs
        loss = square_delta[0, 0, 0]
    else:
        loss = sum(parts[k] * counts[k] for k in counts) / total

    logged = {k: v.detach() for k, v in parts.items()}
    # `<name>_n` is the group's weight in the total above (unmasked first channels);
    # `<name>_den` is the denominator of the group's own mean, over all its channels.
    logged.update({f"{k}_n": v.detach() for k, v in counts.items()})
    logged.update({f"{k}_den": v.detach() for k, v in dens.items()})
    return loss, logged
