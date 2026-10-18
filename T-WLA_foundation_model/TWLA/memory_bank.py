"""The memory bank: a batched ring buffer of past memory tokens.

Every control step writes one entry: `m_f`, the final hidden states at the S memory
positions. Entries are stored raw; the slot and age embeddings are added when the bank
is read. The bank holds `capacity` steps and is reset at every episode boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BankRead:
    """One read of the bank, flattened for attention."""

    entries: torch.Tensor    # [B, capacity * S, D]  ring order, S slots per step
    valid: torch.Tensor      # [B, capacity * S]     bool
    age: torch.Tensor        # [B, capacity * S]     int, 1 = written on the previous step, 0 = empty
    slot_id: torch.Tensor    # [B, capacity * S]     int in [0, S)


class MemoryBank:
    """Fixed-capacity ring of memory writes, stored as [B, capacity, S, D]."""

    def __init__(self, batch_size: int, capacity: int, num_slots: int, dim: int,
                 *, device: torch.device, dtype: torch.dtype = torch.float32):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.num_slots = num_slots
        self.dim = dim
        self.device = device
        self.dtype = dtype

        self.buffer = torch.zeros(batch_size, capacity, num_slots, dim,
                                  device=device, dtype=dtype)
        self.write_step = torch.full((batch_size, capacity), -1,
                                     device=device, dtype=torch.int64)
        self.step = torch.zeros(batch_size, device=device, dtype=torch.int64)

    # ------------------------------------------------------------------ state

    @property
    def batch_size(self) -> int:
        return self.buffer.shape[0]

    def reset_where(self, reset: torch.Tensor) -> None:
        """Clear the bank of the batch items where `reset` is True (episode boundaries)."""
        if reset.shape != (self.batch_size,):
            raise ValueError(f"reset must be [{self.batch_size}], got {reset.shape}")
        if not reset.any():
            return
        m = reset.view(-1, 1, 1, 1)
        self.buffer = torch.where(m, torch.zeros_like(self.buffer), self.buffer)
        self.write_step = torch.where(reset.view(-1, 1),
                                      torch.full_like(self.write_step, -1),
                                      self.write_step)
        self.step = torch.where(reset, torch.zeros_like(self.step), self.step)

    def detach(self) -> "MemoryBank":
        """Detach the stored entries from the graph. Returns self."""
        self.buffer = self.buffer.detach()
        return self

    # ------------------------------------------------------------------ write

    def append(self, m_f: torch.Tensor, valid: torch.Tensor | None = None) -> None:
        """Write one step's `m_f` into the ring.

        m_f:   [B, S, D] final hidden states at the memory positions.
        valid: [B] bool; False leaves that item's bank and step counter unchanged.
        """
        b, s, d = m_f.shape
        if (b, s, d) != (self.batch_size, self.num_slots, self.dim):
            raise ValueError(
                f"m_f must be [{self.batch_size}, {self.num_slots}, {self.dim}], got {m_f.shape}")
        if valid is None:
            valid = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)

        slot = (self.step % self.capacity)                         # [B]
        idx = slot.view(-1, 1, 1, 1).expand(-1, 1, self.num_slots, self.dim)
        written = self.buffer.scatter(1, idx, m_f.unsqueeze(1).to(self.buffer.dtype))
        self.buffer = torch.where(valid.view(-1, 1, 1, 1), written, self.buffer)

        new_ws = self.write_step.scatter(1, slot.view(-1, 1), self.step.view(-1, 1))
        self.write_step = torch.where(valid.view(-1, 1), new_ws, self.write_step)
        self.step = self.step + valid.to(self.step.dtype)

    def append_rows(self, rows: torch.Tensor, m_f: torch.Tensor) -> None:
        """Write one step's `m_f` [K, S, D] for the batch rows `rows` [K], in place.
        The other rows do not advance."""
        k, s, d = m_f.shape
        if k != rows.numel() or (s, d) != (self.num_slots, self.dim):
            raise ValueError(f"m_f must be [{rows.numel()}, {self.num_slots}, "
                             f"{self.dim}], got {tuple(m_f.shape)}")
        slot = self.step[rows] % self.capacity                     # [K]
        self.buffer[rows, slot] = m_f.to(self.buffer.dtype)
        self.write_step[rows, slot] = self.step[rows]
        self.step[rows] += 1

    # ------------------------------------------------------------------ read

    def read(self, rows: torch.Tensor | None = None) -> BankRead:
        """Flatten the bank to [B, capacity * S, D] and attach validity, age and slot id.

        Entries are in ring order: recency is given by `age`, not by position.
        `rows` restricts the read to those batch rows.
        """
        buffer, write_step, step = self.buffer, self.write_step, self.step
        if rows is not None:
            buffer, write_step, step = buffer[rows], write_step[rows], step[rows]
        b, c, s, d = buffer.shape
        entries = buffer.reshape(b, c * s, d)

        valid_rows = write_step >= 0                                       # [B, C]
        valid = valid_rows[:, :, None].expand(b, c, s).reshape(b, c * s)

        # A valid entry has age >= 1: the bank holds earlier steps only.
        age_rows = step[:, None] - write_step                              # [B, C]
        age_rows = torch.where(valid_rows, age_rows, torch.zeros_like(age_rows))
        age = age_rows[:, :, None].expand(b, c, s).reshape(b, c * s)

        slot_id = (torch.arange(s, device=self.device)
                   .view(1, 1, s).expand(b, c, s).reshape(b, c * s))

        return BankRead(entries=entries, valid=valid, age=age, slot_id=slot_id)


def age_to_index(age: torch.Tensor, capacity: int) -> torch.Tensor:
    """Age (1..capacity) -> embedding index (0..capacity-1).

    Empty entries have age 0 and map to index 0; they are masked in the attention.
    """
    return torch.clamp(age, min=1, max=capacity) - 1
