#!/usr/bin/env python3
"""Train one configuration from the formal EgoTouch encoder matrix."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .data import FormalWindowDataset, load_pressure_masks
from .engine import compute_loss, evaluate
from .model import FormalEncoderModel, finger_tactile_masks_by_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_cache_protocol(config: dict) -> None:
    expected = config.get("required_cache_format_version")
    if expected is None:
        return
    manifest_path = Path(config["cache_root"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    actual = manifest.get("format_version")
    if actual != int(expected):
        raise ValueError(
            f"cache format mismatch: expected {expected}, found {actual} in {manifest_path}"
        )


def loader(dataset, config: dict, *, train: bool, smoke: bool, sampler=None) -> DataLoader:
    workers = 0 if smoke else int(config["workers"])
    generator = torch.Generator().manual_seed(int(config["seed"]) + (0 if train else 1))
    return DataLoader(
        dataset,
        batch_size=min(int(config["batch_size"]), len(dataset)),
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=train,
        generator=generator,
    )


def contact_balanced_sampler(dataset, config: dict) -> tuple[WeightedRandomSampler, dict]:
    """Sample anchor-contact and non-contact windows at a configured ratio."""
    if isinstance(dataset, Subset):
        root = dataset.dataset
        positions = np.asarray(dataset.indices, dtype=np.int64)
        selected = root.selected[positions]
    else:
        root = dataset
        selected = root.selected
    if not isinstance(root, FormalWindowDataset):
        raise TypeError("contact-balanced sampling requires FormalWindowDataset")
    rows = root.windows[selected]
    anchors = np.asarray(rows["frame"], dtype=np.int64)
    sides = np.asarray(rows["side"], dtype=np.int64)
    masks = finger_tactile_masks_by_side().any(dim=1).numpy()
    threshold = float(config.get("contact_sampling_threshold", config["contact_threshold"]))
    contact = np.zeros(len(selected), dtype=bool)
    chunk_size = 32768
    for start in range(0, len(selected), chunk_size):
        end = min(start + chunk_size, len(selected))
        if root.target_hand_only:
            pressure = np.asarray(root.touch[anchors[start:end]], dtype=np.float32)
        else:
            pressure = np.asarray(root.touch[anchors[start:end], sides[start:end]], dtype=np.float32)
        valid = masks[sides[start:end]]
        contact[start:end] = np.max(np.where(valid, pressure, -np.inf), axis=(1, 2)) > threshold
    num_contact = int(contact.sum())
    num_non_contact = int(len(contact) - num_contact)
    if not num_contact or not num_non_contact:
        raise ValueError(
            f"contact balancing requires both classes; contact={num_contact} non_contact={num_non_contact}"
        )
    target_contact = float(config.get("contact_sampling_fraction", 0.5))
    if not 0 < target_contact < 1:
        raise ValueError("contact_sampling_fraction must be between zero and one")
    weights = np.where(
        contact,
        target_contact / num_contact,
        (1.0 - target_contact) / num_non_contact,
    )
    generator = torch.Generator().manual_seed(int(config["seed"]) + 7)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(weights), replacement=True, generator=generator
    )
    report = {
        "method": "weighted_random_sampler_with_replacement",
        "contact_definition": "maximum fingertip pressure at final history frame exceeds threshold",
        "threshold": threshold,
        "source_contact_windows": num_contact,
        "source_non_contact_windows": num_non_contact,
        "source_contact_fraction": num_contact / len(contact),
        "target_contact_fraction": target_contact,
        "samples_per_epoch": len(weights),
    }
    return sampler, report


def make_model(config: dict, masks: torch.Tensor) -> FormalEncoderModel:
    model = FormalEncoderModel(
        masks,
        task=config["task"],
        pose_temporal=config.get("pose_temporal", "mean"),
        embed_dim=int(config.get("embed_dim", 64)),
        use_touch_prediction=bool(config.get("use_touch_prediction", True)),
        physical_aux=bool(config.get("physical_aux", False)),
        predictive_aux=bool(config.get("predictive_aux", False)),
        tactile_region=str(config.get("tactile_region", "full")),
        pose_region=str(config.get("pose_region", "full")),
        focus_mode=str(config.get("focus_mode", "full")),
        context_weight=float(config.get("context_weight", 0.1)),
        fingertip_auxiliary=bool(config.get("fingertip_aux_weights")),
        fingertip_residual=bool(config.get("fingertip_residual", False)),
        tactile_residual_weight=float(config.get("tactile_residual_weight", 0.25)),
        pose_residual_weight=float(config.get("pose_residual_weight", 0.25)),
    )
    return model


def score(metrics: dict, task: str) -> float:
    if task in {"p1", "p3"}:
        return float(metrics["mean_map"])
    if task == "p2":
        return -float(metrics["predictive"]["articulation_mae"])
    return -float(metrics["loss"])


def save(path: Path, model, optimizer, scheduler, epoch: int, step: int, config: dict, metrics: dict) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    required = {"name", "task", "cache_root", "output_dir", "batch_size", "epochs", "max_steps", "lr", "seed"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing config keys: {sorted(missing)}")
    seed_all(int(config["seed"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    validate_cache_protocol(config)
    tactile_history = config.get("tactile_history_frames")
    pose_history = config.get("pose_history_frames")
    train_set = FormalWindowDataset(
        config["cache_root"],
        "train",
        config["task"],
        tactile_history=tactile_history,
        pose_history=pose_history,
    )
    val_set = FormalWindowDataset(
        config["cache_root"],
        "val",
        config["task"],
        evaluation=True,
        tactile_history=tactile_history,
        pose_history=pose_history,
    )
    if args.smoke:
        train_set = Subset(train_set, range(min(2048, len(train_set))))
        val_set = Subset(val_set, range(min(512, len(val_set))))
    train_sampler = None
    if bool(config.get("contact_balanced_sampling", False)):
        train_sampler, sampling_report = contact_balanced_sampler(train_set, config)
        (output_dir / "sampling_report.json").write_text(json.dumps(sampling_report, indent=2) + "\n")
        print(json.dumps({"sampling": sampling_report}), flush=True)
    train_loader = loader(train_set, config, train=True, smoke=args.smoke, sampler=train_sampler)
    val_loader = loader(val_set, config, train=False, smoke=args.smoke)

    model = make_model(config, load_pressure_masks(config["cache_root"]))
    init_report: dict = {"initialization": config.get("initialization", "scratch")}
    if config.get("initialization", "scratch") == "opentouch":
        skip_prefixes = tuple(str(value) for value in config.get("initialization_skip_prefixes", ()))
        init_report.update(
            model.initialize_from_opentouch(
                config["checkpoint"],
                reset_temperature=True,
                skip_prefixes=skip_prefixes,
            )
        )
    elif config.get("initialization", "scratch") == "formal":
        payload = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
        source = payload["model"]
        target = model.state_dict()
        loaded, skipped = [], []
        for key, value in source.items():
            if key in target and target[key].shape == value.shape:
                target[key] = value
                loaded.append(key)
            elif key in target:
                skipped.append(f"{key}: source={tuple(value.shape)} target={tuple(target[key].shape)}")
        model.load_state_dict(target, strict=True)
        init_report.update({"checkpoint": config["checkpoint"], "loaded": loaded, "skipped": skipped})
    elif config.get("initialization", "scratch") != "scratch":
        raise ValueError(f"unknown initialization {config['initialization']!r}")
    (output_dir / "initialization_report.json").write_text(json.dumps(init_report, indent=2) + "\n")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config.get("weight_decay", 0.01))
    )
    max_steps = 2 if args.smoke else int(config["max_steps"])
    warmup = min(int(config.get("warmup_steps", 500)), max_steps // 4)

    def schedule(step: int) -> float:
        if warmup and step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(max_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    epochs = 1 if args.smoke else int(config["epochs"])
    metrics_path = output_dir / "metrics.jsonl"
    best_score = -math.inf
    global_step = 0
    started = time.time()
    shuffle_touch = bool(config.get("shuffle_touch", False))
    with metrics_path.open("w") as metrics_file:
        for epoch in range(1, epochs + 1):
            model.train()
            totals: dict[str, float] = {}
            seen = 0
            epoch_started = time.time()
            for raw in train_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    output = model(
                        batch["touch"], batch["pose"], batch["side"], shuffle_touch=shuffle_touch
                    )
                    loss, pieces = compute_loss(model, output, batch, config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                model.clamp_temperature()
                b = len(batch["side"])
                seen += b
                for key, value in pieces.items():
                    totals[key] = totals.get(key, 0.0) + value * b
                global_step += 1
                if global_step == 1 or global_step % int(config.get("log_every_steps", 100)) == 0:
                    print(
                        f"name={config['name']} epoch={epoch} step={global_step}/{max_steps} "
                        f"loss={pieces['total']:.6f} lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )
                if global_step >= max_steps:
                    break
            val_metrics = evaluate(model, val_loader, config, device)
            row = {
                "epoch": epoch,
                "step": global_step,
                "train": {key: value / max(seen, 1) for key, value in totals.items()},
                "val": val_metrics,
                "epoch_seconds": time.time() - epoch_started,
                "elapsed_seconds": time.time() - started,
            }
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            print(json.dumps(row), flush=True)
            current = score(val_metrics, config["task"])
            if current > best_score:
                best_score = current
                save(output_dir / "best.pt", model, optimizer, scheduler, epoch, global_step, config, row)
            save(output_dir / "last.pt", model, optimizer, scheduler, epoch, global_step, config, row)
            if global_step >= max_steps:
                break
    summary = {
        "name": config["name"],
        "best_validation_score": best_score,
        "steps": global_step,
        "elapsed_seconds": time.time() - started,
        "full_train_samples": len(train_set),
        "validation_samples": len(val_set),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
