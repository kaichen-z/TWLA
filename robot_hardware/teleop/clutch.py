"""Clutch: the switch that connects the operator to the robot.

Engaging takes two steps: the "enable" switch in the UI must be on, then the space bar
toggles between engaged and disengaged. While disengaged the arms hold their position.

The clutch always falls back to disengaged:
    - turning the enable switch off disengages immediately;
    - the UI sends a heartbeat every 0.2 s; without one for `heartbeat_timeout_s` the clutch
      disengages (dead-man switch: the UI is closed or frozen);
    - `panic()` (Esc / emergency button) disengages and latches until re-enabled;
    - `guard_lock()` (fast-motion lock) disengages and locks; unlocking with the space bar
      re-engages after `UNLOCK_DELAY_S`, so the operator can bring the hand back first.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

__all__ = ["Clutch", "ClutchState"]


@dataclass
class ClutchState:
    """Snapshot of the clutch."""

    engaged: bool = False
    armed: bool = False  # the enable switch
    focused: bool = False
    alive: bool = False  # heartbeat is fresh
    last_beat_ago_s: float = float("inf")
    reason: str = "UI not connected"  # why engaging is not possible now; "" if it is
    locked: bool = False  # locked by the fast-motion guard
    lock_reason: str = ""
    pending_s: float = 0.0  # seconds until re-engaging after an unlock; 0 if none


class Clutch:
    """Thread-safe clutch shared by the control loops and the UI."""

    UNLOCK_DELAY_S = 5.0

    def __init__(self, side: str = "right", heartbeat_timeout_s: float = 8.0) -> None:
        self.side = side  # "left", "right" or "both"
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self._lock = threading.Lock()
        self._engaged = False
        self._armed = False
        self._focused = False
        self._last_beat = 0.0
        self._panic = False
        self._locked = False
        self._lock_reason = ""
        self._pending_until = 0.0  # > 0 while waiting to re-engage after an unlock
        self._log: list = []
        # Optional callback fn(msg, engaged, locked) on every state change (used for recording).
        self.on_event = None

    # -- called by the UI ---------------------------------------------------------------

    def heartbeat(self, focused: bool) -> None:
        """Called by the UI every ~0.2 s; losing focus while engaged disengages."""
        with self._lock:
            self._last_beat = time.monotonic()
            was = self._focused
            self._focused = bool(focused)
            if was and not self._focused and self._engaged:
                self._engaged = False
                self._note("focus lost -> disengaged")

    def set_armed(self, on: bool) -> None:
        """The enable switch; turning it off disengages immediately."""
        with self._lock:
            on = bool(on)
            if on == self._armed:
                return
            self._armed = on
            if not on:
                self._engaged = False
                self._pending_until = 0.0
                self._note("enable switch off -> disengaged")
            else:
                self._panic = False
                self._note("space bar enabled")

    def toggle(self) -> bool:
        """Space bar. Disengaging is always allowed; returns True if now engaged."""
        with self._lock:
            if self._engaged:
                self._engaged = False
                self._note("space -> disengaged")
                return False
            if self._pending_until:
                self._pending_until = 0.0
                self._note("space during the unlock delay -> cancelled, still locked")
                return False
            blocked = self._why_locked()
            if self._locked:
                if blocked:
                    self._note(f"unlock refused: {blocked}")
                    return False
                self._pending_until = time.monotonic() + self.UNLOCK_DELAY_S
                self._note(
                    f"unlocking: re-engaging in {self.UNLOCK_DELAY_S:.0f} s (space to cancel)"
                )
                return False
            if blocked:
                self._note(f"engage refused: {blocked}")
                return False
            self._panic = False
            self._engaged = True
            self._note("space -> engaged")
            return True

    def guard_lock(self, reason: str) -> bool:
        """Fast-motion lock: disengage and lock. Only acts while engaged; returns True if so."""
        with self._lock:
            if not self._engaged:
                return False
            self._engaged = False
            self._locked, self._lock_reason = True, reason
            self._pending_until = 0.0
            self._note(f"locked: {reason}")
            return True

    def panic(self) -> None:
        """Disengage and latch until the enable switch is turned off and on again."""
        with self._lock:
            self._pending_until = 0.0
            self._engaged = False
            self._panic = True
            self._note("PANIC -> disengaged and latched")

    # -- internal (caller holds the lock) -----------------------------------------------

    def _why_locked(self) -> str:
        """Reason why engaging is not allowed now, or ""."""
        if not self._armed:
            return "the enable switch is off"
        if self._panic:
            return "panic latched (turn the enable switch off and on)"
        age = time.monotonic() - self._last_beat
        if age > self.heartbeat_timeout_s:
            return f"UI heartbeat lost for {age:.1f} s"
        return ""

    def _note(self, msg: str) -> None:
        self._log.append((time.strftime("%H:%M:%S"), msg))
        del self._log[:-8]
        cb = self.on_event
        if cb is not None:
            try:
                cb(msg, self._engaged, self._locked)
            except Exception:  # noqa: BLE001
                pass

    def _tick(self) -> None:
        """Apply the unlock delay and the heartbeat timeout."""
        if self._pending_until and time.monotonic() >= self._pending_until:
            self._pending_until = 0.0
            blocked = self._why_locked()
            if blocked:
                self._note(f"unlock delay over but cannot engage: {blocked}")
            else:
                self._engaged, self._locked, self._lock_reason = True, False, ""
                self._note("unlock delay over -> engaged")
        if not self._armed and self._engaged:
            self._engaged = False
        if self._engaged and (time.monotonic() - self._last_beat) > self.heartbeat_timeout_s:
            self._engaged = False
            self._note("heartbeat lost -> disengaged")

    # -- read ---------------------------------------------------------------------------

    def state(self) -> ClutchState:
        """Current state."""
        with self._lock:
            self._tick()
            age = time.monotonic() - self._last_beat if self._last_beat else float("inf")
            return ClutchState(
                engaged=self._engaged,
                armed=self._armed,
                focused=self._focused,
                alive=age <= self.heartbeat_timeout_s,
                last_beat_ago_s=age,
                reason="" if self._engaged else self._why_locked(),
                locked=self._locked,
                lock_reason=self._lock_reason,
                pending_s=(
                    max(0.0, self._pending_until - time.monotonic()) if self._pending_until else 0.0
                ),
            )

    def get_key_value_by_name(self, name: str) -> float:
        """1.0 for "<side>_grip" while engaged, else 0.0 (both sides match when side="both")."""
        ok = ("left_grip", "right_grip") if self.side == "both" else (f"{self.side}_grip",)
        if name not in ok:
            return 0.0
        with self._lock:
            self._tick()
            return 1.0 if self._engaged else 0.0

    def is_engaged(self) -> bool:
        """True while engaged."""
        with self._lock:
            self._tick()
            return self._engaged
