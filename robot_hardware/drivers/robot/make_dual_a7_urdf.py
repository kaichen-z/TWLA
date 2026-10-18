#!/usr/bin/env python3
"""Merge the single-arm SDK models `a7__left.urdf` and `a7__right.urdf` into `dual_a7.urdf`.

Both files share the same `Base_Link` on the floor, which is kept once; all other links and
joints are copied unchanged (their names carry the Left_/Right_ prefix). The left shoulder
base, whose origin is 0 in the SDK file, is placed at z = 1.217 m like the right one, so that
both arms use the same floor reference. No other value is changed.

The meshes referenced by the SDK files are not shipped with it: load the model without
geometry (`pin.buildModelFromUrdf()`); `tools/sim_viewer.py` attaches the meshes from
`models/a7/meshes/`.

Usage:  python make_dual_a7_urdf.py        # writes dual_a7.urdf next to this file
"""

import os
import re
import sys

try:
    from importlib.resources import files

    SRC = files("realhand.arm.kinetix.urdf")
    LEFT, RIGHT = str(SRC / "a7__left.urdf"), str(SRC / "a7__right.urdf")
except Exception:
    LEFT = RIGHT = None

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dual_a7.urdf")


def split(path):
    """Return (Base_Link element, the rest of the robot body) of one URDF file."""
    s = open(path, encoding="utf-8").read()
    body = s[s.index(">", s.index("<robot")) + 1 : s.rindex("</robot>")]
    base = re.search(r'<link\s+name="Base_Link">.*?</link>', body, re.S).group(0)
    rest = body.replace(base, "", 1)
    return base, rest.strip("\n")


def main():
    """Write dual_a7.urdf."""
    if LEFT is None:
        sys.exit("realhand.arm.kinetix.urdf not found: install the SDK first")
    bl, left = split(LEFT)
    br, right = split(RIGHT)
    assert bl == br, "the two Base_Link definitions differ; cannot merge"
    # Raise the left shoulder base to the height of the right one (see the module docstring).
    m = re.search(
        r'(<joint\s+name="Left_Shoulder_Base_Joint"[^>]*>\s*<origin\s+xyz=")([^"]+)(")', left
    )
    assert m and m.group(2).split() == [
        "0",
        "0",
        "0",
    ], f"left shoulder base origin is not the expected 0 0 0: {m and m.group(2)}"
    left = left[: m.start(2)] + "0 0 1.217" + left[m.end(2) :]
    out = (
        f'<?xml version="1.0" encoding="utf-8"?>\n'
        f"<!-- Dual-arm A7 model, merged from a7__left.urdf + a7__right.urdf "
        f"by make_dual_a7_urdf.py.\n"
        f"     Shared Base_Link on the floor; both shoulder bases at z = 1.217 m. -->\n"
        f'<robot name="dual_a7">\n{bl}\n{left}\n{right}\n</robot>\n'
    )
    open(OUT, "w", encoding="utf-8").write(out)
    print(f"wrote {OUT}  ({len(out)} bytes)")


if __name__ == "__main__":
    main()
