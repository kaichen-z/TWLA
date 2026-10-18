"""The trained model as a function: (image, state, field of view, instruction) -> actions.

`VLAPolicy` reads `configs/train.json` and `<repository root>/checkpoints/T-WLA-FD.pt` unless
other paths are given. The world head is not built, and the `world.*` tensors of a checkpoint
are skipped. Run this file as a script to predict from a single image.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); VLA_HAND_POSE = os.path.normpath(os.path.join(HERE, ".."))
if VLA_HAND_POSE not in sys.path:
    sys.path.insert(0, VLA_HAND_POSE)
from TWLA.config import TWLAConfig                 # noqa: E402
from TWLA.model import TWLAModel, Observation  # noqa: E402
from config_validation import REQUIRED, set_parameter       # noqa: E402

REPOSITORY = os.path.normpath(os.path.join(VLA_HAND_POSE, ".."))
DEFAULT_CHECKPOINT = os.path.join(REPOSITORY, "checkpoints", "T-WLA-FD.pt")
DEFAULT_CONFIG = os.path.join(VLA_HAND_POSE, "configs", "train.json")


def apply_overrides(cfg, overrides: dict) -> None:
    """Same as `train.apply_overrides`; copied so that inference does not import the training script."""
    for section, values in overrides.items():
        if not hasattr(cfg, section):
            raise KeyError(f"config has no section '{section}'")
        obj = getattr(cfg, section)
        if obj is None:
            setattr(cfg, section, values); continue
        for k, v in values.items():
            if not hasattr(obj, k):
                raise KeyError(f"'{section}' has no field '{k}'")
            cur = getattr(obj, k)
            if cur is REQUIRED:
                set_parameter(obj, k, v); continue
            setattr(obj, k, v if (cur is None or isinstance(v, (list, dict))) else type(cur)(v))


def to_canvas(rgb: np.ndarray, size: int = 512) -> np.ndarray:
    """Resize the long side to `size` and place the image in the bottom-right corner of a black
    (size, size, 3) canvas, as in training. `rgb` is (h, w, 3) uint8."""
    import cv2
    h, w = rgb.shape[:2]
    r = min(size / w, size / h)
    nh, nw = max(1, int(h * r)), max(1, int(w * r))
    small = rgb if (nh, nw) == (h, w) else cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), np.uint8); out[size - nh:, size - nw:] = small
    return out


def fov_from_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Horizontal and vertical field of view in radians: 2 * atan(cx / fx), 2 * atan(cy / fy)."""
    return np.array([2 * np.arctan(cx / fx), 2 * np.arctan(cy / fy)], np.float32)


class VLAPolicy:
    def __init__(self, config_path: str = DEFAULT_CONFIG, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cuda",
                 instruction: str | None = None, verbose: bool = True):
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"model weights not found at {checkpoint}. Download the released checkpoint to "
                                    f"{DEFAULT_CHECKPOINT}, or pass `checkpoint=<path>`.")
        C = json.load(open(config_path))
        if C.get("variant", "d") != "d":
            raise ValueError("this code is written for variant d")
        overrides = dict(C.get("config", {})); overrides.pop("world", None)      # no world head at inference
        cfg = TWLAConfig(); apply_overrides(cfg, overrides); cfg.validate()
        self.cfg = cfg; self.device = torch.device(device)
        t0 = time.time()
        self.model = TWLAModel(cfg, load_weights=False, dtype=torch.float32)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        weights = {k: v for k, v in ck["model"].items() if not k.startswith("world.")}
        self.model.load_state_dict(weights)               # strict; includes the normalisation statistics
        self.model.eval().to(self.device)
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.vlm.model_id)
        self.executed = int(cfg.action.executed_steps)
        self.instruction = None
        if instruction is not None:
            self.set_instruction(instruction)
        if verbose:
            print(f"[policy] {checkpoint} on {self.device} in {time.time()-t0:.1f}s; executes {self.executed} x "
                  f"{1000/cfg.action.action_hz:.1f} ms per call", flush=True)

    def set_instruction(self, instruction: str) -> None:
        """Set the instruction. Training format: `Left hand: <text or None>. Right hand: <text or None>.`"""
        enc = self._tokenizer([instruction], padding="max_length", truncation=True,
                              max_length=self.cfg.vlm.max_language_tokens, return_tensors="pt")
        self.instruction = instruction
        self.lang_ids = enc["input_ids"].to(self.device); self.lang_mask = enc["attention_mask"].bool().to(self.device)

    def reset(self):
        self.model._bank = None; self.model._episode_id = None

    @torch.no_grad()
    def act(self, rgb: np.ndarray, state: np.ndarray, mask: np.ndarray, fov: np.ndarray, episode_id: int,
            steps: int | None = None, return_mf: bool = False):
        """One control step: raw actions (executed_steps, 192), or the whole chunk with `steps=16`.

        `rgb` is the uint8 canvas from `to_canvas`, `state` the raw 212-d state, `mask` its bool
        validity mask, `fov` the output of `fov_from_intrinsics`. The memory bank is kept across calls
        and reset when `episode_id` changes. `return_mf=True` also returns the memory tokens [8, D]."""
        if self.instruction is None:
            raise RuntimeError("no instruction set: pass `instruction=` to VLAPolicy or call set_instruction()")
        S = int(self.cfg.vlm.image_size)
        if rgb.shape[:2] != (S, S):
            raise ValueError(f"image must be the {S}x{S} canvas (see to_canvas), got {rgb.shape[:2]}")
        img = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)[None]               # [1,S,S,3]
        img = img.permute(0, 3, 1, 2).float().div_(255).mul_(2).sub_(1)                     # [1,3,H,W]
        obs = Observation(images=img[:, None], lang_ids=self.lang_ids, lang_mask=self.lang_mask,
                          state=torch.from_numpy(np.asarray(state, np.float32))[None].to(self.device),
                          state_mask=torch.from_numpy(np.asarray(mask, bool))[None].to(self.device),
                          fov=torch.from_numpy(np.asarray(fov, np.float32))[None, None].to(self.device))
        a, m_f = self.model.select_action_context(obs, torch.tensor([int(episode_id)], device=self.device), steps=steps)
        a = a[0].float().cpu().numpy()
        return (a, m_f[0].float()) if return_mf else a


def main():
    import argparse
    import cv2
    ap = argparse.ArgumentParser(description="Run the model on one image.")
    ap.add_argument("--image", required=True, help="an RGB image of any size")
    ap.add_argument("--instruction", required=True, help='"Left hand: <text or None>. Right hand: <text or None>."')
    ap.add_argument("--intrinsics", type=float, nargs=4, required=True, metavar=("FX", "FY", "CX", "CY"),
                    help="intrinsics of the image, in pixels")
    ap.add_argument("--state", default=None, help=".npy, the raw 212-d state")
    ap.add_argument("--mask", default=None, help=".npy, the 212-d bool validity mask of the state")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT); ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None, help=".npy to write the actions to")
    a = ap.parse_args()
    bgr = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(a.image)
    pol = VLAPolicy(a.config, a.checkpoint, device=a.device, instruction=a.instruction)
    n = int(pol.cfg.action.state_dim)
    if a.state is None:
        print("[policy] no --state given: running with an all-masked state (the model was trained with the state present)")
        state, mask = np.zeros(n, np.float32), np.zeros(n, bool)
    else:
        state = np.load(a.state).astype(np.float32)
        mask = np.load(a.mask).astype(bool) if a.mask else np.ones(n, bool)
    actions = pol.act(to_canvas(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), state, mask, fov_from_intrinsics(*a.intrinsics), 0)
    print(f"[policy] actions {actions.shape}: left hand dims 0:69, right hand dims 69:138 "
          f"(3 wrist translation, 3 wrist rotation, 63 keypoints each)")
    if a.out:
        np.save(a.out, actions); print(f"[policy] saved to {a.out}")


if __name__ == "__main__":
    main()
