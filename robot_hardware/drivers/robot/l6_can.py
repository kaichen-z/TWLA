#!/usr/bin/env python3
"""CAN interface of the L6 dexterous hand (python-can, socketcan).

Frames carry the hand id as arbitration id (0x27 right, 0x28 left) and
`[command, D1..D6]` as data; a 1-byte frame is a read request, a 7-byte frame a write.

A background thread stores every reply by command code; `snapshot()` returns the latest
complete state without blocking. Sending is serialised with a small gap between frames.
"""

import threading
import time

import can

# Command codes.
CMD_POSITION = 0x01  # write: target position; read: current position
CMD_TORQUE = 0x02  # torque limit
CMD_SPEED = 0x05  # speed
CMD_TEMPERATURE = 0x33  # read only
CMD_FAULT = 0x35  # read only
CMD_CLEAR_FAULT = 0x83  # 0/1 per joint
CMD_SERIAL = 0xC0  # factory serial number
CMD_FIRMWARE = 0xC2  # firmware version
TOUCH_CMDS = (0xB1, 0xB2, 0xB3, 0xB4, 0xB5)  # tactile arrays: thumb, index, middle, ring, little
# 0xB6 (palm tactile array) must not be sent to the L6, which has no palm sensor: after
# repeated 0xB6 requests the hand stops answering until it is power-cycled.
PALM_CMD_DO_NOT_USE = 0xB6
TOUCH_SIZE_12x6 = 0xC6  # size argument of a tactile request (12 rows x 6 columns)

N_JOINTS = 6
TOUCH_ROWS, TOUCH_COLS = 12, 6

JOINT_NAMES = [
    "thumb flexion",
    "thumb abduction",
    "index flexion",
    "middle flexion",
    "ring flexion",
    "little flexion",
]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]

# First byte of a tactile reply: 0, 16, 32, ..., 176 for rows 0..11.
_ROW_CODE = {v: i for i, v in enumerate(range(0, TOUCH_ROWS * 16, 16))}

# Bits of the fault byte (command 0x35).
FAULT_BITS = [
    (6, "MCU over-temperature"),
    (5, "motor over-temperature"),
    (4, "overload 2"),
    (3, "overload 1"),
    (2, "phase A over-current"),
    (1, "phase C over-current"),
    (0, "phase B over-current"),
]


def decode_fault(code):
    """Names of the faults set in one fault byte."""
    if not code:
        return []
    return [name for bit, name in FAULT_BITS if code & (1 << bit)]


class L6Can:
    """One L6 hand on a socketcan interface."""

    def __init__(self, channel="can0", hand_id=0x27, frame_gap=0.002):
        self.hand_id = hand_id
        self.frame_gap = frame_gap
        self.bus = can.interface.Bus(channel=channel, interface="socketcan")

        self._tx_lock = threading.Lock()
        self._st_lock = threading.Lock()

        self.position = [0] * N_JOINTS
        self.temperature = [0] * N_JOINTS
        self.fault = [0] * N_JOINTS
        self.touch = {name: [[0] * TOUCH_COLS for _ in range(TOUCH_ROWS)] for name in FINGER_NAMES}
        self.firmware = None
        self.serial = None
        self._serial_parts = {}

        self.rx_count = 0
        self.malformed = 0
        self.tx_count = 0
        self._rx_window = []  # receive timestamps of the last second
        self._last_frame_time = 0.0

        # Optional callback `fn(cmd, body, t_ns)` for every reply frame, called in the receive
        # thread with `time.monotonic_ns()`; used for recording. Keep it light.
        self.on_frame = None

        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

    def _send(self, cmd, payload=()):
        msg = can.Message(
            arbitration_id=self.hand_id, data=[cmd] + list(payload), is_extended_id=False
        )
        with self._tx_lock:
            try:
                self.bus.send(msg)
                self.tx_count += 1
            except can.CanError:
                return False
            time.sleep(self.frame_gap)
        return True

    def request(self, cmd, payload=()):
        """Send a read request; the reply is handled by the receive thread."""
        return self._send(cmd, payload)

    def write_position(self, values):
        """Set the six target positions (0-255)."""
        assert len(values) == N_JOINTS
        return self._send(CMD_POSITION, [max(0, min(255, int(v))) for v in values])

    def write_torque(self, values):
        """Set the six torque limits (0-255)."""
        assert len(values) == N_JOINTS
        return self._send(CMD_TORQUE, [max(0, min(255, int(v))) for v in values])

    def write_speed(self, values):
        """Set the six speeds (0-255); sent twice, as the vendor SDK does."""
        assert len(values) == N_JOINTS
        ok = self._send(CMD_SPEED, [max(0, min(255, int(v))) for v in values])
        return self._send(CMD_SPEED, [max(0, min(255, int(v))) for v in values]) and ok

    def clear_faults(self, mask=(1, 1, 1, 1, 1, 1)):
        """Clear the faults of the joints selected by mask."""
        return self._send(CMD_CLEAR_FAULT, mask)

    def request_touch(self, finger_idx):
        """Request the tactile array of one finger (0 thumb .. 4 little)."""
        return self._send(TOUCH_CMDS[finger_idx], [TOUCH_SIZE_12x6])

    def _rx_loop(self):
        while self._running:
            try:
                msg = self.bus.recv(timeout=0.2)
            except Exception:
                time.sleep(0.05)
                continue
            if msg is None:
                continue
            if msg.arbitration_id not in (self.hand_id, self.hand_id + 8):
                continue
            data = list(msg.data)
            if not data:
                continue
            t_ns = time.monotonic_ns()
            self._dispatch(data[0], data[1:])
            cb = self.on_frame
            if cb is not None:
                try:
                    cb(data[0], data[1:], t_ns)
                except Exception:
                    pass  # a callback error must not stop the receive thread

    def _dispatch(self, cmd, body):
        now = time.time()
        with self._st_lock:
            self.rx_count += 1
            self._rx_window.append(now)
            cutoff = now - 1.0
            while self._rx_window and self._rx_window[0] < cutoff:
                self._rx_window.pop(0)
            self._last_frame_time = now

            # Status frames must have exactly 6 data bytes. During fast tactile polling, frames
            # with a status command code but a tactile-sized body occasionally arrive; they are
            # dropped and counted instead of being read as false fault codes.
            if cmd == CMD_POSITION and len(body) == N_JOINTS:
                self.position = body[:N_JOINTS]
            elif cmd == CMD_TEMPERATURE and len(body) == N_JOINTS:
                self.temperature = body[:N_JOINTS]
            elif cmd == CMD_FAULT and len(body) == N_JOINTS:
                self.fault = body[:N_JOINTS]
            elif cmd in (CMD_POSITION, CMD_TEMPERATURE, CMD_FAULT) and len(body) > 0:
                self.malformed += 1
            elif cmd == CMD_FIRMWARE and len(body) >= 3:
                self.firmware = ".".join(str(b) for b in body[:3])
            elif cmd == CMD_SERIAL and len(body) >= 1:
                # The serial number arrives in four frames; the first byte is the fragment index.
                self._serial_parts[body[0]] = body[1:]
                try:
                    joined = b"".join(
                        bytes(self._serial_parts[i]) for i in sorted(self._serial_parts)
                    )
                    self.serial = joined.decode("ascii").rstrip("\x00")
                except Exception:
                    pass
            elif cmd in TOUCH_CMDS and len(body) == 1 + TOUCH_COLS:
                row = _ROW_CODE.get(body[0])
                if row is not None:
                    finger = FINGER_NAMES[TOUCH_CMDS.index(cmd)]
                    self.touch[finger][row] = body[1 : 1 + TOUCH_COLS]

    def snapshot(self):
        """Copy of the latest state: positions, temperatures, faults, tactile arrays, counters."""
        with self._st_lock:
            return {
                "position": list(self.position),
                "temperature": list(self.temperature),
                "fault": list(self.fault),
                "fault_text": [decode_fault(c) for c in self.fault],
                "touch": {k: [row[:] for row in v] for k, v in self.touch.items()},
                "firmware": self.firmware,
                "serial": self.serial,
                "rx_hz": len(self._rx_window),
                "tx_count": self.tx_count,
                "rx_count": self.rx_count,
                "malformed": self.malformed,
                "link_age": (
                    time.time() - self._last_frame_time if self._last_frame_time else None
                ),
            }

    def age(self):
        """Seconds since the last reply, or None if none has been received."""
        with self._st_lock:
            if not self._last_frame_time:
                return None
            return time.time() - self._last_frame_time

    def close(self):
        """Stop the receive thread and close the bus."""
        self._running = False
        try:
            self._rx_thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            self.bus.shutdown()
        except Exception:
            pass
