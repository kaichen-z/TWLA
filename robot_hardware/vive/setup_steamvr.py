#!/usr/bin/env python3
"""Configure SteamVR to run without a headset (null driver), for base stations + trackers only.

SteamVR refuses to start without an HMD unless its built-in "null" driver is enabled.
This script sets, after backing up each file (.bak.<timestamp>):

    <SteamVR>/drivers/null/resources/settings/default.vrsettings
        driver_null.enable = true
    <Steam>/config/steamvr.vrsettings
        steamvr.requireHmd = false
        steamvr.forcedDriver = "null"
        steamvr.activateMultipleDrivers = true   # otherwise the null driver replaces lighthouse

Quit SteamVR first: it writes its settings back on exit and would undo the change.

    python vive/setup_steamvr.py           # apply
    python vive/setup_steamvr.py --check   # report only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

STEAM_ROOT_CANDIDATES = [
    Path.home() / ".steam" / "debian-installation",
    Path.home() / ".local" / "share" / "Steam",
    Path.home() / ".steam" / "steam",
]


def find_steam_root() -> Path:
    seen = set()
    for c in STEAM_ROOT_CANDIDATES:
        if not c.exists():
            continue
        real = c.resolve()
        if real in seen:
            continue
        seen.add(real)
        if (real / "steamapps").exists():
            return real
    sys.exit("Steam installation not found; install Steam and start it once")


def find_steamvr(steam_root: Path) -> Path:
    p = steam_root / "steamapps" / "common" / "SteamVR"
    if not p.exists():
        sys.exit(f"SteamVR not found: {p}\ninstall it from the Steam library (filter: Tools)")
    return p


def steamvr_running() -> bool:
    try:
        out = subprocess.run(["pgrep", "-x", "vrserver"], capture_output=True)
        return out.returncode == 0
    except FileNotFoundError:
        return False


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    dst = path.with_suffix(path.suffix + f".bak.{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, dst)
    return dst


def apply(path: Path, section: str, updates: dict, dry: bool) -> list[str]:
    """Set data[section][key] = value for each update; returns one line per key."""
    data = load_json(path)
    sec = data.setdefault(section, {})
    changes = []

    def dump(x):
        return json.dumps(x, ensure_ascii=False)

    for k, v in updates.items():
        cur = sec.get(k, "<missing>")
        if cur == v:
            changes.append(f"    ok       {section}.{k} = {dump(v)}")
        else:
            changes.append(
                f"    {'to change' if dry else 'changed'} {section}.{k}: {dump(cur)} -> {dump(v)}"
            )
            sec[k] = v
    if not dry:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=3, ensure_ascii=False) + "\n", encoding="utf-8")
    return changes


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    args = ap.parse_args()
    dry = args.check

    if not dry and steamvr_running():
        sys.exit("SteamVR (vrserver) is running; quit it first, it rewrites its settings on exit")

    steam_root = find_steam_root()
    steamvr = find_steamvr(steam_root)
    print(f"Steam   : {steam_root}")
    print(f"SteamVR : {steamvr}\n")

    file_a = steamvr / "drivers" / "null" / "resources" / "settings" / "default.vrsettings"
    file_b = steam_root / "config" / "steamvr.vrsettings"

    if not file_a.exists():
        sys.exit(f"null driver settings not found: {file_a}")

    print(f"[A] {file_a}")
    if not dry:
        b = backup(file_a)
        print(f"    backup: {b.name}")
    for line in apply(file_a, "driver_null", {"enable": True}, dry):
        print(line)

    print(f"\n[B] {file_b}")
    if not file_b.exists():
        print("    does not exist yet (SteamVR never started); it will be created")
    if not dry:
        b = backup(file_b)
        if b:
            print(f"    backup: {b.name}")
    for line in apply(
        file_b,
        "steamvr",
        {"requireHmd": False, "forcedDriver": "null", "activateMultipleDrivers": True},
        dry,
    ):
        print(line)

    print("\n" + ("check done (nothing written)" if dry else "done; now start SteamVR"))


if __name__ == "__main__":
    main()
