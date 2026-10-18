"""Train and evaluate one native Tachin tactile editor for paper Table 1."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactile_editor.action_layout import FULL_ACTION_GROUPS
from tactile_editor.action_objectives import per_sample_action_loss
from tactile_editor.native_tachin import (NativeResidualEditor, NativeTachinEncoder,
                                          NativeTachinW3Editor)
from tactile_editor.rgb_world_editor import roi_weighted_world_loss
from tactile_editor.single_stage_world import masked_tactile_flow_loss
from tactile_editor.table1_experiment import summarize_physical
from tactile_editor.training import build_future_action_mask


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


class TachinEditorCache(Dataset):
    def __init__(self, root: Path, split: str, model: str, *, train_or_val: bool,
                 include_w3: bool):
        self.root, self.split, self.model = root, split, model
        shared = root / "shared" / split
        if not (shared / "complete.json").is_file():
            raise RuntimeError(f"Incomplete common cache: {shared}")
        names = ("target_action", "action_mask", "pressure", "pressure_mask",
                 "sample_valid", "future_pressure", "future_pressure_mask", "future_valid")
        self.common = {name: np.load(shared / f"{name}.npy", mmap_mode="r") for name in names}
        self.paired30 = (shared / "future_extra_pressure.npy").is_file()
        if self.paired30:
            for name in ("future_extra_pressure", "future_extra_pressure_mask", "future_extra_valid"):
                self.common[name] = np.load(shared / f"{name}.npy", mmap_mode="r")
        self.identity = dict(np.load(shared / "identity.npz", allow_pickle=False))
        count = len(self.common["target_action"])
        self.locations = None
        if model == "twla":
            self.shards = []
            locations = np.full((count, 2), -1, np.int64)
            for shard_index, path in enumerate(sorted((root / "bases/twla" / split).glob("shard_*"))):
                if not (path / "complete.json").is_file():
                    continue
                fields = ["base_action", "final_memory", "query_index"]
                if include_w3:
                    fields += ["image_tokens", "clean_latent", "rgb_valid", "future_roi"]
                payload = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in fields}
                ids = payload["query_index"]
                if np.any(locations[ids, 0] >= 0):
                    raise ValueError("Duplicate T-WLA query indices")
                locations[ids, 0] = len(self.shards)
                locations[ids, 1] = np.arange(len(ids))
                self.shards.append(payload)
            if not self.shards or np.any(locations < 0):
                raise RuntimeError(f"Incomplete T-WLA cache for {split}")
            self.locations = locations
        else:
            path = root / "bases" / model / split
            if not (path / "complete.json").is_file():
                raise RuntimeError(f"Incomplete {model} cache for {split}")
            self.base_action = np.load(path / "base_action.npy", mmap_mode="r")
            if self.base_action.shape != (count, 16, 192):
                raise ValueError("Frozen-base cache shape mismatch")

        latest_hand = self.common["sample_valid"][:, -1] & self.common["pressure_mask"][:, -1].any(-1)
        if self.paired30:
            future_latest = (self.common["future_valid"][:, :2]
                             & self.common["future_pressure_mask"][:, :2].any(-1))
            latest_hand = np.concatenate((latest_hand[:, None], future_latest), axis=1)
        else:
            latest_hand = latest_hand[:, None]
        action_hand = np.stack((self.common["action_mask"][:, :, :69].any((1, 2)),
                                self.common["action_mask"][:, :, 69:138].any((1, 2))), 1)
        self.latest_hand = np.asarray(latest_hand)
        eligible = (latest_hand & action_hand[:, None]).any(2)
        if self.paired30:
            for tick, committed in enumerate((0, 3, 6)):
                eligible[:, tick] &= self.common["action_mask"][:, committed:, :138].any((1, 2))
            self.rows = np.argwhere(eligible)
        else:
            indices = np.flatnonzero(eligible[:, 0]) if train_or_val else np.arange(count)
            self.rows = np.stack((indices, np.zeros_like(indices)), axis=1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        index, tick = map(int, self.rows[item])
        if self.model == "twla":
            shard, row = self.locations[index]
            base = self.shards[int(shard)]
            output = {name: torch.from_numpy(np.array(value[int(row)], copy=True))
                      for name, value in base.items() if name != "query_index"}
        else:
            output = {"base_action": torch.from_numpy(np.array(self.base_action[index], copy=True))}
        for name in ("target_action", "action_mask"):
            output[name] = torch.from_numpy(np.array(self.common[name][index], copy=True))
        if self.paired30:
            pressure = np.concatenate((self.common["pressure"][index],
                                       self.common["future_pressure"][index, :2]), axis=0)
            pressure_mask = np.concatenate((self.common["pressure_mask"][index],
                                            self.common["future_pressure_mask"][index, :2]), axis=0)
            sample_valid = np.concatenate((self.common["sample_valid"][index],
                                           self.common["future_valid"][index, :2]), axis=0)
            future_pressure = np.concatenate((self.common["future_pressure"][index],
                                              self.common["future_extra_pressure"][index]), axis=0)
            future_pressure_mask = np.concatenate((self.common["future_pressure_mask"][index],
                                                   self.common["future_extra_pressure_mask"][index]), axis=0)
            future_valid = np.concatenate((self.common["future_valid"][index],
                                           self.common["future_extra_valid"][index]), axis=0)
            output["pressure"] = torch.from_numpy(np.array(pressure[tick:tick + 8], copy=True))
            output["pressure_mask"] = torch.from_numpy(np.array(pressure_mask[tick:tick + 8], copy=True))
            output["sample_valid"] = torch.from_numpy(np.array(sample_valid[tick:tick + 8], copy=True))
            output["future_pressure"] = torch.from_numpy(np.array(future_pressure[tick + 2], copy=True))
            output["future_pressure_mask"] = torch.from_numpy(
                np.array(future_pressure_mask[tick + 2], copy=True))
            output["future_valid"] = torch.from_numpy(np.array(future_valid[tick + 2], copy=True))
        else:
            for name in ("pressure", "pressure_mask", "sample_valid"):
                output[name] = torch.from_numpy(np.array(self.common[name][index], copy=True))
            output["future_pressure"] = torch.from_numpy(
                np.array(self.common["future_pressure"][index, 2], copy=True))
            output["future_pressure_mask"] = torch.from_numpy(
                np.array(self.common["future_pressure_mask"][index, 2], copy=True))
            output["future_valid"] = torch.from_numpy(
                np.array(self.common["future_valid"][index, 2], copy=True))
        output["query_index"] = torch.tensor(index)
        output["decision_tick"] = torch.tensor(tick)
        output["decision_seconds"] = torch.tensor(tick / 30.0, dtype=torch.float32)
        return output


def edit_mask(batch):
    mask = build_future_action_mask(batch["action_mask"].bool(), batch["decision_seconds"])
    mask[..., 138:] = False
    hand_valid = batch["sample_valid"][:, -1].bool() & batch["pressure_mask"][:, -1].any(-1)
    mask[..., :69] &= hand_valid[:, 0, None, None]
    mask[..., 69:138] &= hand_valid[:, 1, None, None]
    return mask


def move(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def prepare_batch(raw, device, statistics):
    batch = move(raw, device)
    mean = statistics["action_mean"].to(device)
    scale = statistics["action_std"].to(device)
    batch["base_action"] = (batch["base_action"].float() - mean) / scale
    batch["target_action"] = (batch["target_action"].float() - mean) / scale
    return batch


def make_model(model_name, statistics, device):
    encoder = NativeTachinEncoder(statistics["pressure_mean"], statistics["pressure_scale"])
    if model_name == "twla":
        model = NativeTachinW3Editor(encoder, statistics["pressure_mean"],
                                     statistics["pressure_scale"])
    else:
        model = NativeResidualEditor(encoder)
    return model.to(device)


def forward(model, batch, *, auxiliary=False, rgb_head=None, generator=None,
            compute_rgb=True, compute_tactile=True):
    mask = edit_mask(batch)
    if isinstance(model, NativeResidualEditor):
        result = model(batch["base_action"], mask, batch["pressure"].float(),
                       batch["pressure_mask"], batch["sample_valid"])
        prediction, delta, allowed, tactile, hand_valid = result
        return prediction, delta, allowed, tactile, hand_valid, None
    kwargs = {}
    if auxiliary and compute_tactile:
        future_pressure = batch["future_pressure"].float()
        future_valid = batch["future_pressure_mask"].bool() & batch["future_valid"][:, :, None].bool()
        kwargs = dict(
            future_pressure=future_pressure,
            future_cell_valid=future_valid,
            tactile_tau=torch.rand(len(mask), device=mask.device, generator=generator),
            tactile_noise=torch.randn((len(mask), 2, 880), device=mask.device, generator=generator),
        )
    if auxiliary and compute_rgb:
        latent = batch["clean_latent"].float()
        kwargs.update(
            rgb_world_head=rgb_head,
            image_tokens=batch["image_tokens"].float(),
            clean_latent=latent,
            rgb_valid=batch["rgb_valid"].bool(),
            rgb_tau=torch.rand(latent.shape[:2], device=mask.device, generator=generator),
            rgb_noise=torch.randn(latent.shape, device=mask.device, generator=generator),
        )
    output = model(base_action=batch["base_action"], edit_mask=mask,
                   pressure=batch["pressure"].float(), cell_valid=batch["pressure_mask"],
                   sample_valid=batch["sample_valid"], final_memory=batch["final_memory"].float(),
                   compute_auxiliary=auxiliary, compute_rgb=compute_rgb,
                   compute_tactile=compute_tactile, **kwargs)
    return (output.final_action, output.delta_action, output.allowed_mask,
            output.tactile_features, output.hand_valid, output)


def objective(model, batch, *, rgb_head, generator, lambda_rgb, lambda_tactile):
    prediction, delta, allowed, _, _, output = forward(
        model, batch, auxiliary=isinstance(model, NativeTachinW3Editor),
        rgb_head=rgb_head, generator=generator,
        compute_rgb=lambda_rgb > 0, compute_tactile=lambda_tactile > 0)
    active = allowed.any((1, 2))
    action = per_sample_action_loss(prediction[active], batch["target_action"][active],
                                    allowed[active], groups=FULL_ACTION_GROUPS).mean()
    delta_l2 = delta.float().square().masked_fill(~allowed, 0).sum() / allowed.sum().clamp_min(1)
    tactile = action.new_zeros(())
    rgb = action.new_zeros(())
    if output is not None:
        if output.tactile_velocity is not None:
            tactile = masked_tactile_flow_loss(output.tactile_velocity, output.tactile_target,
                                               output.tactile_valid)
        if output.rgb is not None and output.rgb.valid.any():
            rgb = roi_weighted_world_loss(output.rgb, batch["future_roi"].bool(),
                                          background_weight=0.1)
    loss = action + 1e-4 * delta_l2 + lambda_rgb * rgb + lambda_tactile * tactile
    return loss, {"action": float(action.detach()), "rgb": float(rgb.detach()),
                  "tactile": float(tactile.detach()), "delta_l2": float(delta_l2.detach())}


@torch.no_grad()
def validate(model, dataset, statistics, batch_size, workers, device):
    model.eval()
    total = base_total = 0.0
    count = 0
    for raw in DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                          pin_memory=True, persistent_workers=workers > 0):
        batch = prepare_batch(raw, device, statistics)
        prediction, _, allowed, _, _, _ = forward(model, batch)
        active = allowed.any((1, 2))
        losses = per_sample_action_loss(prediction[active], batch["target_action"][active],
                                        allowed[active], groups=FULL_ACTION_GROUPS)
        bases = per_sample_action_loss(batch["base_action"][active], batch["target_action"][active],
                                      allowed[active], groups=FULL_ACTION_GROUPS)
        total += float(losses.sum())
        base_total += float(bases.sum())
        count += len(losses)
    return {"action_loss": total / max(count, 1),
            "base_action_loss": base_total / max(count, 1), "rows": count}


def base_actions(dataset):
    count = len(dataset.common["target_action"])
    if dataset.model != "twla":
        return np.asarray(dataset.base_action)
    result = np.empty((count, 16, 192), np.float32)
    for shard in dataset.shards:
        result[shard["query_index"]] = shard["base_action"]
    return result


@torch.no_grad()
def infer_condition(model, dataset, statistics, batch_size, workers, device, condition,
                    permutation=None):
    model.eval()
    base = base_actions(dataset)
    prediction = base.copy()
    mean = statistics["action_mean"]
    scale = statistics["action_std"]
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    row_offset = 0
    for raw in loader:
        indices = raw["query_index"].numpy()
        if condition == "shuffled_touch":
            donors = permutation[row_offset:row_offset + len(indices)]
            for name in ("pressure", "pressure_mask", "sample_valid"):
                raw[name] = torch.stack([dataset[int(donor)][name] for donor in donors])
        elif condition == "zero_touch":
            reference = statistics["pressure_mean"].view(1, 1, 2, 880)
            raw["pressure"] = reference.expand_as(raw["pressure"]).clone()
        batch = prepare_batch(raw, device, statistics)
        values, _, allowed, _, _, _ = forward(model, batch)
        physical = values.float().cpu() * scale + mean
        physical = physical.numpy()
        allowed = allowed.cpu().numpy()
        for local, index in enumerate(indices):
            prediction[index][allowed[local]] = physical[local][allowed[local]]
        row_offset += len(indices)
    return prediction


def final_test(model, dataset, statistics, args, device, output):
    count = len(dataset.common["target_action"])
    row_hands = dataset.latest_hand[dataset.rows[:, 0], dataset.rows[:, 1]]
    codes = row_hands[:, 0].astype(np.int8) + 2 * row_hands[:, 1].astype(np.int8)
    permutation = np.arange(len(dataset))
    rng = np.random.default_rng(args.seed + 700)
    for tick in np.unique(dataset.rows[:, 1]):
        for code in (0, 1, 2, 3):
            rows = np.flatnonzero((dataset.rows[:, 1] == tick) & (codes == code))
            donors = rows.copy()
            rng.shuffle(donors)
            permutation[rows] = donors
    base = base_actions(dataset)
    predictions = {"base": base}
    for condition in ("editor", "shuffled_touch", "zero_touch"):
        predictions[condition] = infer_condition(
            model, dataset, statistics, args.micro_batch, args.workers, device,
            condition, permutation)
        np.save(output / f"test_{condition}_action.npy", predictions[condition])
    target = np.asarray(dataset.common["target_action"])
    mask = np.asarray(dataset.common["action_mask"])
    episodes = dataset.identity["episode_id"].astype(str)
    environment = np.full(count, "Tachin")
    metrics = {
        name: summarize_physical(value, target, mask, episodes, environment,
                                 horizon=8, dataset="tachin")["overall"]
        for name, value in predictions.items()
    }
    metric_names = ("translation_cm", "rotation_deg", "joint_mpjpe_cm")
    reduction = {name: metrics["base"][name] - metrics["editor"][name] for name in metric_names}
    percentage = {name: 100 * reduction[name] / metrics["base"][name] for name in metric_names}
    cells = {name: f"{reduction[name]:.2f} / ({percentage[name]:.2f}%)" for name in metric_names}
    return {"horizon": "first 8 actions / 100 ms", "aggregation": "episode-equal",
            "editor_protocol": "paired-30" if dataset.paired30 else "single-decision",
            "decision_ticks": [0, 1, 2] if dataset.paired30 else [0],
            "metrics": metrics, "physical_error_reduction": reduction,
            "physical_error_reduction_percent": percentage, "table1_cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", choices=("smolvla", "vitra", "trex", "twla", "memoryvla"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-checkpoint", type=Path)
    parser.add_argument("--world-config", type=Path)
    parser.add_argument("--lambda-rgb", type=float, default=0.1,
                        help="T-WLA W3 RGB-world objective weight")
    parser.add_argument("--lambda-tactile", type=float, default=1e-4,
                        help="T-WLA W3 tactile-world objective weight")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum epochs; validation early stopping can finish sooner")
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5,
                        help="Stop after this many validation epochs without improvement")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Minimum absolute validation action-loss improvement")
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--effective-batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_epochs <= args.epochs:
        raise ValueError("min-epochs must be in [1, epochs]")
    if args.patience < 1 or args.min_delta < 0:
        raise ValueError("patience must be >= 1 and min-delta must be non-negative")
    if any(not math.isfinite(x) or x < 0 for x in (args.lambda_rgb, args.lambda_tactile)):
        raise ValueError("world objective weights must be finite and non-negative")
    if args.effective_batch % args.micro_batch:
        raise ValueError("effective batch must be divisible by micro batch")
    uses_rgb = args.model == "twla" and args.lambda_rgb > 0
    if uses_rgb and (args.world_checkpoint is None or args.world_config is None):
        parser.error("T-WLA W3 requires the RGB world checkpoint and config")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".train.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (args.output / "summary.json").exists():
        print(json.dumps({"model": args.model, "status": "already complete"}), flush=True)
        return
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    cache_audit = json.loads((args.cache / "audit/repartition.json").read_text())
    if cache_audit.get("status") != "PASS" or not cache_audit.get("recording_disjoint"):
        raise ValueError("Expected audited recording-disjoint Tachin cache")
    stats_np = dict(np.load(args.cache / "statistics.npz", allow_pickle=False))
    statistics = {name: torch.from_numpy(value).float() for name, value in stats_np.items()
                  if name in ("pressure_mean", "pressure_scale", "action_mean", "action_std")}
    train = TachinEditorCache(args.cache, "train", args.model, train_or_val=True,
                              include_w3=uses_rgb)
    validation = TachinEditorCache(args.cache, "val", args.model, train_or_val=True,
                                   include_w3=False)
    model = make_model(args.model, statistics, device)
    rgb_head = None
    if args.model == "twla":
        if uses_rgb:
            from tactile_editor.public_backbone import load_rgb_world_head
            rgb_head, _ = load_rgb_world_head(args.world_checkpoint, args.world_config, device)
        groups = model.optimizer_groups()
        lambda_rgb, lambda_tactile = args.lambda_rgb, args.lambda_tactile
    else:
        if args.lambda_rgb != 0.1 or args.lambda_tactile != 1e-4:
            parser.error("world objective weights apply only to --model twla")
        groups = [{"params": model.action.parameters(), "lr": 1e-4, "name": "action"},
                  {"params": model.encoder.parameters(), "lr": 1e-5, "name": "encoder"},
                  {"params": model.coefficients.parameters(), "lr": 1e-4, "name": "coefficients"}]
        lambda_rgb = lambda_tactile = 0.0
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / args.effective_batch)
    total_steps = args.epochs * steps_per_epoch
    warmup = min(100, max(total_steps - 1, 0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        (step + 1) / max(warmup, 1) if step < warmup else
        0.1 + 0.45 * (1 + math.cos(math.pi * min((step - warmup) / max(total_steps - warmup, 1), 1))))
    config = {**vars(args), "train_rows": len(train), "validation_rows": len(validation),
              "split": "revised recording-disjoint split", "gated": False,
              "normalization": "shared revised-train-only", "horizon": "8 actions / 100 ms",
              "editor_protocol": "paired-30" if train.paired30 else "single-decision",
              "vision_plan_hz": 10.0,
              "tactile_input_hz": 30.0 if train.paired30 else 10.0,
              "editor_decision_hz": 30.0 if train.paired30 else 10.0,
              "decision_ticks": [0, 1, 2] if train.paired30 else [0],
              "editor": "native 15-region 880-cell", "twla_variant": "W3" if args.model == "twla" else None,
              "lambda_rgb": lambda_rgb, "lambda_tactile": lambda_tactile}
    atomic_json(args.output / "config.json", config)
    best = float("inf")
    step = 0
    first_epoch = 0
    epochs_without_improvement = 0
    started = time.time()
    accumulation = args.effective_batch // args.micro_batch
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    if args.resume:
        resume_path = args.output / "last.pt"
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume requested but {resume_path} is missing")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        required = {"model", "optimizer", "scheduler", "epoch", "step", "best",
                    "epochs_without_improvement"}
        missing = required - set(checkpoint)
        if missing:
            raise ValueError(
                f"{resume_path} predates resumable checkpoints and is missing {sorted(missing)}; "
                "start a new output directory for a controlled early-stopping run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        first_epoch = int(checkpoint["epoch"])
        step = int(checkpoint["step"])
        best = float(checkpoint["best"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if checkpoint.get("aux_generator_state") is not None:
            # torch.load(map_location=device) also moves the RNG byte state,
            # while Generator.set_state requires a CPU ByteTensor.
            generator.set_state(checkpoint["aux_generator_state"].cpu())
        print(json.dumps({"model": args.model, "status": "resumed",
                          "next_epoch": first_epoch + 1, "step": step,
                          "best_validation_action_loss": best}), flush=True)
    stopped_early = False
    for epoch in range(first_epoch, args.epochs):
        model.train()
        loader = DataLoader(train, batch_size=args.micro_batch, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed + epoch),
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)
        optimizer.zero_grad(set_to_none=True)
        window = 0
        sums = np.zeros(4)
        seen = 0
        for batch_index, raw in enumerate(loader):
            batch = prepare_batch(raw, device, statistics)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = objective(model, batch, rgb_head=rgb_head, generator=generator,
                                        lambda_rgb=lambda_rgb, lambda_tactile=lambda_tactile)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            count = len(batch["base_action"])
            (loss * count / args.effective_batch).backward()
            window += count
            seen += count
            sums += np.asarray(list(parts.values())) * count
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                if window != args.effective_batch:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(args.effective_batch / window)
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                          error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                window = 0
                step += 1
                if step % 25 == 0 or args.smoke_steps:
                    print(json.dumps({"model": args.model, "step": step,
                                      "total_steps": total_steps, "loss": parts,
                                      "gradient": float(gradient)}), flush=True)
                if args.smoke_steps and step >= args.smoke_steps:
                    atomic_json(args.output / "smoke.json", {"steps": step, "loss": parts,
                                                              "gradient": float(gradient)})
                    return
        validation_metrics = validate(model, validation, statistics, args.micro_batch,
                                      args.workers, device)
        improved = validation_metrics["action_loss"] < best - args.min_delta
        if improved:
            best = validation_metrics["action_loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "step": step,
                      "validation": validation_metrics, "config": config,
                      "statistics": statistics, "best": best,
                      "epochs_without_improvement": epochs_without_improvement,
                      "aux_generator_state": generator.get_state()}
        torch.save(checkpoint, args.output / "last.pt")
        if improved:
            torch.save(checkpoint, args.output / "best.pt")
        record = {"epoch": epoch + 1, "step": step,
                  "train": dict(zip(("action", "rgb", "tactile", "delta_l2"),
                                    (sums / max(seen, 1)).tolist())),
                  "validation": validation_metrics, "improved": improved,
                  "epochs_without_improvement": epochs_without_improvement,
                  "seconds": time.time() - started}
        with (args.output / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({"model": args.model, **record}), flush=True)
        if epoch + 1 >= args.min_epochs and epochs_without_improvement >= args.patience:
            stopped_early = True
            print(json.dumps({"model": args.model, "status": "early_stop",
                              "epoch": epoch + 1, "patience": args.patience,
                              "best_validation_action_loss": best}), flush=True)
            break
    selected = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    test = TachinEditorCache(args.cache, "test", args.model, train_or_val=False,
                             include_w3=False)
    result = final_test(model, test, statistics, args, device, args.output)
    atomic_json(args.output / "summary.json", {
        "model": args.model, "selected_epoch": selected["epoch"],
        "selected_step": selected["step"], "validation": selected["validation"],
        "test": result, "seconds": time.time() - started, "config": config,
        "stopped_early": stopped_early})
    print(json.dumps({"model": args.model, "status": "complete",
                      "table1": result["table1_cells"]}), flush=True)


if __name__ == "__main__":
    main()
