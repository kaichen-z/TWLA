#!/usr/bin/env python3
"""Evaluate a selected formal checkpoint on fixed held-out windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import FormalWindowDataset, load_pressure_masks
from .engine import evaluate
from .train import make_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["val", "test_seen", "test_unseen"])
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = make_model(config, load_pressure_masks(config["cache_root"]))
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(args.device)
    model.to(device)
    results = {}
    for split in args.splits:
        dataset = FormalWindowDataset(
            config["cache_root"],
            split,
            config["task"],
            evaluation=True,
            tactile_history=config.get("tactile_history_frames"),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(int(config["batch_size"]), len(dataset)),
            shuffle=False,
            num_workers=int(config["workers"]),
            pin_memory=True,
            persistent_workers=int(config["workers"]) > 0,
        )
        results[split] = evaluate(model, loader, config, device)
        print(split, json.dumps(results[split]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
