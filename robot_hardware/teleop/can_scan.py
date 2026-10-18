#!/usr/bin/env python3
"""Find which device (left/right arm, left/right hand) is on each CAN bus. Read-only.

CAN interface numbers follow USB enumeration and change when adapters are re-plugged, so
the teleoperation backend runs this scan at start-up.

    hands  a 1-byte read request (firmware version) to ids 0x27 (right) and 0x28 (left)
    arms   an SDK handshake (`A7(side=...)`) without enabling; the arms use different motor
           ids (51-57 right, 61-67 left)

The backend runs it as a subprocess (`--json`): a failed A7 handshake can leak sockets, which
are released when the process exits.

    python teleop/can_scan.py            # human-readable
    python teleop/can_scan.py --json     # for the backend
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

HAND_IDS = {"right_hand": 0x27, "left_hand": 0x28}
CMD_FIRMWARE = 0xC2


def up_can_buses() -> list:
    """Names of the CAN interfaces that are up."""
    out = []
    for name in sorted(os.listdir("/sys/class/net")):
        if not name.startswith("can"):
            continue
        try:
            flags = int(open(f"/sys/class/net/{name}/flags").read().strip(), 16)
        except OSError:
            continue
        if flags & 0x1:  # IFF_UP
            out.append(name)
    return out


def find_hands(bus_name: str, tries: int = 3, listen_s: float = 0.25) -> list:
    """L6 hands answering on one bus."""
    import can

    found = set()
    bus = can.interface.Bus(channel=bus_name, interface="socketcan")
    try:
        for _ in range(tries):
            for key, hid in HAND_IDS.items():
                if key in found:
                    continue
                try:
                    bus.send(
                        can.Message(arbitration_id=hid, data=[CMD_FIRMWARE], is_extended_id=False)
                    )
                except can.CanError:
                    pass
            t_end = time.monotonic() + listen_s
            while time.monotonic() < t_end:
                msg = bus.recv(timeout=0.05)
                if msg is None:
                    continue
                for key, hid in HAND_IDS.items():
                    # Replies come from the hand id or hand id + 8 (as in L6Can).
                    if (
                        msg.arbitration_id in (hid, hid + 8)
                        and msg.data
                        and msg.data[0] == CMD_FIRMWARE
                    ):
                        found.add(key)
            if len(found) == len(HAND_IDS):
                break
    finally:
        bus.shutdown()
    return sorted(found)


def find_arm(bus_name: str) -> list:
    """The arm answering on one bus, if any."""
    from realhand.arm import A7

    found = []
    for side in ("right", "left"):
        h = None
        try:
            h = A7(side=side, interface_name=bus_name)  # handshake only, not enabled
            h.get_angles()
            found.append(f"{side}_arm")
        except Exception:  # noqa: BLE001
            pass
        finally:
            if h is not None:
                try:
                    h.close()
                except Exception:  # noqa: BLE001
                    pass
        if found:
            break
    return found


def scan(buses=None) -> dict:
    """{"buses": {can: [devices]}, "assign": {device: can}, "conflicts": [...], "errors": {}}."""
    buses = buses or up_can_buses()
    res: dict = {b: [] for b in buses}
    errs: dict = {}

    def one(b):
        try:
            got = find_hands(b)
            if not got:  # a bus carries either hands or an arm
                got = find_arm(b)
            res[b] = got
        except Exception as exc:  # noqa: BLE001
            errs[b] = f"{type(exc).__name__}: {exc}"[:160]

    th = [threading.Thread(target=one, args=(b,), daemon=True) for b in buses]
    for t in th:
        t.start()
    for t in th:
        t.join(timeout=40)

    assign, conflicts = {}, []
    for b, devs in res.items():
        for d in devs:
            if d in assign:
                conflicts.append(f"{d} found on both {assign[d]} and {b}")
            else:
                assign[d] = b
    return {"buses": res, "assign": assign, "conflicts": conflicts, "errors": errs}


LABEL = {
    "left_arm": "left arm",
    "right_arm": "right arm",
    "left_hand": "left hand",
    "right_hand": "right hand",
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--json", action="store_true", help="print one JSON line")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(_ROOT, "drivers"))
    t0 = time.monotonic()
    r = scan()
    r["elapsed_s"] = round(time.monotonic() - t0, 2)
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0
    for b, devs in r["buses"].items():
        print(
            f"  {b}: {', '.join(LABEL[d] for d in devs) or '(no reply)'}"
            + (f"   error: {r['errors'][b]}" if b in r["errors"] else "")
        )
    for c in r["conflicts"]:
        print(f"  warning: {c}")
    print(f"  took {r['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
