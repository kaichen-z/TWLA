"""How one training sample is drawn.

A sample is L consecutive control steps of one phase of one episode: the first L - 1 only
fill the memory bank, the last one is supervised. With M usable anchors in the phase,
L ~ P(L) proportional to L^alpha over {1 .. min(max_len, M)}, then s ~ Uniform{0 .. M - L}.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Window:
    """One drawn window. Indices are anchor indices within a phase."""

    start: int          # s
    length: int         # L  (>= 1)

    @property
    def supervised(self) -> int:
        """The anchor that is supervised."""
        return self.start + self.length - 1

    @property
    def burn_in(self) -> range:
        """The anchors that only fill the bank. Empty when L == 1."""
        return range(self.start, self.supervised)

    @property
    def bank_depth(self) -> int:
        """Number of earlier steps in the bank when the supervised step runs."""
        return self.length - 1


def length_weights(max_len: int, alpha: float) -> torch.Tensor:
    """Unnormalised P(L) over L = 1 .. max_len, proportional to L^alpha (alpha = 0 is uniform)."""
    if max_len < 1:
        raise ValueError(f"max_len must be >= 1, got {max_len}")
    return torch.arange(1, max_len + 1, dtype=torch.float32) ** alpha


def sample_window(num_anchors: int, max_len: int, alpha: float, *,
                  generator: torch.Generator | None = None) -> Window:
    """Draw (L, s) for one phase.

    Args:
        num_anchors: number of anchors that may be the supervised step
            (see `supervisable_anchors`).
        max_len: cap on L. Reaching a full bank needs max_len = bank capacity + 1.
        alpha: exponent of the length distribution.

    If num_anchors < max_len, L is drawn over {1 .. num_anchors}, which is equivalent to
    redrawing L until it fits. Short phases therefore lower the realised mean of L.
    """
    if num_anchors < 1:
        raise ValueError(f"a phase needs at least one supervisable anchor, got {num_anchors}")
    hi = min(max_len, num_anchors)
    w = length_weights(hi, alpha)
    length = int(torch.multinomial(w, 1, generator=generator).item()) + 1
    start = int(torch.randint(0, num_anchors - length + 1, (1,), generator=generator).item())
    return Window(start=start, length=length)


def supervisable_anchors(anchors: Sequence[int], num_frames: int) -> int:
    """Number of a phase's anchors that may be the supervised step.

    Only an anchor on the episode's final frame is excluded: an action is the difference to
    a later pose, so that anchor has no target. Anchors whose chunk runs past the end are
    kept; the missing tail is masked out in `action_mask`. `anchors` must be sorted.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if any(a < 0 or a >= num_frames for a in anchors):
        raise ValueError("every anchor must lie inside the episode")
    return len(anchors) - int(bool(anchors) and anchors[-1] == num_frames - 1)


def phase_anchors(num_frames: int, phase: int, stride: int) -> list[int]:
    """Frame indices of one phase's anchors. `stride` = source_fps / control_hz.

    The `stride` phases of an episode are separate sampling units, but they overlap almost
    entirely in time: split the data by video, not by unit.
    """
    if not 0 <= phase < stride:
        raise ValueError(f"phase must be in [0, {stride}), got {phase}")
    return list(range(phase, num_frames, stride))
