"""Memory-mapped EgoTouch atomic-window dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FormalWindowDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        task: str,
        *,
        evaluation: bool = False,
        tactile_history: int | None = None,
        pose_history: int | None = None,
    ) -> None:
        self.cache_root = Path(root)
        self.root = self.cache_root / split
        manifest = json.loads((self.cache_root / "manifest.json").read_text())
        cache_history = int(manifest["history"])
        self.history = cache_history
        self.tactile_history = (
            cache_history if tactile_history is None else int(tactile_history)
        )
        self.pose_history = cache_history if pose_history is None else int(pose_history)
        if not 1 <= self.tactile_history <= cache_history:
            raise ValueError(
                "tactile history override must be in "
                f"[1,{cache_history}], got {self.tactile_history}"
            )
        if not 1 <= self.pose_history <= cache_history:
            raise ValueError(
                "pose history override must be in "
                f"[1,{cache_history}], got {self.pose_history}"
            )
        self.future_offset = int(manifest["future_offset"])
        self.target_hand_only = bool(manifest.get("target_hand_only", False))
        self.task = task
        self.touch = np.load(self.root / "touch.npy", mmap_mode="r")
        self.pose = np.load(self.root / "pose.npy", mmap_mode="r")
        self.windows = np.load(self.root / "windows.npy", mmap_mode="r")
        selected = np.arange(len(self.windows), dtype=np.int64)
        if task in {"p1", "p2", "p3"}:
            # P1 itself does not consume the future pose, but using the same
            # anchor population as P2/P3 makes the P1 -> P3 loss ablation a
            # controlled comparison.
            require = self.windows["history_pose_ok"] & self.windows["future_pose_ok"]
            selected = np.flatnonzero(require)
        if evaluation:
            fixed = np.load(self.root / "eval_indices.npy")
            allowed = np.zeros(len(self.windows), dtype=bool)
            allowed[selected] = True
            selected = fixed[allowed[fixed]]
        self.selected = selected

    def __len__(self) -> int:
        return len(self.selected)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = self.windows[int(self.selected[item])]
        anchor = int(row["frame"])
        side = int(row["side"])
        tactile_start = anchor - self.tactile_history + 1
        pose_start = anchor - self.pose_history + 1
        future = anchor + self.future_offset
        if self.target_hand_only:
            touch = self.touch[tactile_start : anchor + 1]
            pose = self.pose[pose_start : anchor + 1]
            future_touch = self.touch[future]
            future_pose = self.pose[future]
        else:
            touch = self.touch[tactile_start : anchor + 1, side]
            pose = self.pose[pose_start : anchor + 1, side]
            future_touch = self.touch[future, side]
            future_pose = self.pose[future, side]
        return {
            "touch": torch.from_numpy(np.array(touch, dtype=np.float32, copy=True)),
            "pose": torch.from_numpy(np.array(pose, dtype=np.float32, copy=True)),
            "future_touch": torch.from_numpy(np.array(future_touch, dtype=np.float32, copy=True)),
            "future_pose": torch.from_numpy(np.array(future_pose, dtype=np.float32, copy=True)),
            "side": torch.tensor(side, dtype=torch.long),
            "recording": torch.tensor(int(row["recording"]), dtype=torch.long),
            "anchor": torch.tensor(anchor, dtype=torch.long),
        }


def load_pressure_masks(root: str | Path) -> torch.Tensor:
    return torch.from_numpy(np.load(Path(root) / "pressure_masks.npy").astype(bool, copy=False))
