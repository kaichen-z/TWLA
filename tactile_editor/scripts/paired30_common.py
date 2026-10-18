"""Portable paths for offline EgoTouch editor training."""
from pathlib import Path
import json
import os
import sys

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
ROOT = Path(os.environ.get("PAIRED30_CACHE_ROOT", "data/paired30"))
ENCODER = Path(os.environ.get("PAIRED30_ENCODER_CHECKPOINT", "checkpoints/tactile_encoder.pt"))
WORLD = Path(os.environ.get("PAIRED30_WORLD_CHECKPOINT", "checkpoints/T-WLA-WM.pt"))


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def log(**values):
    print(json.dumps(values, default=str), flush=True)
