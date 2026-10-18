"""How one sample is drawn.

A sample is L consecutive control steps of one phase of one episode. Step s+L-1 is the
supervised one; steps s .. s+L-2 are its history, for a model with memory. With M
supervisable anchors, L is drawn over {1 .. min(max_len, M)} with probability proportional
to L^alpha, then s uniformly over {0 .. M - L}.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Window:
    """One drawn sample. Indices are anchor indices within one phase."""

    start: int          # s
    length: int         # L  (>= 1)

    @property
    def supervised(self) -> int:
        """The supervised anchor."""
        return self.start + self.length - 1

    @property
    def burn_in(self) -> range:
        """The anchors before the supervised one. Empty when L == 1."""
        return range(self.start, self.supervised)

    @property
    def bank_depth(self) -> int:
        """Number of history steps before the supervised one."""
        return self.length - 1


def length_weights(max_len: int, alpha: float) -> torch.Tensor:
    """Unnormalised P(L) over L = 1 .. max_len, proportional to L^alpha (alpha = 0 is
    uniform)."""
    if max_len < 1:
        raise ValueError(f"max_len must be >= 1, got {max_len}")
    return torch.arange(1, max_len + 1, dtype=torch.float32) ** alpha


def sample_window(num_anchors: int, max_len: int, alpha: float, *,
                  generator: torch.Generator | None = None) -> Window:
    """Draw (L, s) for one phase.

    num_anchors: how many anchors of the phase may be the supervised step, see
        `supervisable_anchors`.
    max_len: cap on L.
    alpha: exponent of the length distribution.
    When `num_anchors < max_len`, L is drawn over {1 .. num_anchors}.
    """
    if num_anchors < 1:
        raise ValueError(f"a phase needs at least one supervisable anchor, got {num_anchors}")
    hi = min(max_len, num_anchors)
    w = length_weights(hi, alpha)
    length = int(torch.multinomial(w, 1, generator=generator).item()) + 1
    start = int(torch.randint(0, num_anchors - length + 1, (1,), generator=generator).item())
    return Window(start=start, length=length)


def supervisable_anchors(anchors: Sequence[int], num_frames: int) -> int:
    """How many of a phase's anchors may be the supervised step.

    All but an anchor on the episode's final frame, which has no later pose to take a
    difference with. An anchor whose chunk runs past the end is kept; the missing tail is
    masked. `anchors` is sorted, so the usable ones are a prefix.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if any(a < 0 or a >= num_frames for a in anchors):
        raise ValueError("every anchor must lie inside the episode")
    return len(anchors) - int(bool(anchors) and anchors[-1] == num_frames - 1)


def phase_anchors(num_frames: int, phase: int, stride: int) -> list[int]:
    """Frame indices of one phase's anchors: `phase`, `phase + stride`, ...

    `stride` = source_fps / control_hz (3 at 30 fps and 10 Hz). Each phase is a sampling
    unit of its own. The chunks of adjacent phases overlap almost entirely in time, so
    data must be split by video, not by unit.
    """
    if not 0 <= phase < stride:
        raise ValueError(f"phase must be in [0, {stride}), got {phase}")
    return list(range(phase, num_frames, stride))
