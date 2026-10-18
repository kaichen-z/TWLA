"""The SmolVLA baseline: `flow.py` wrapped for this repository's batches.

Takes the place of LeRobot's `SmolVLAPolicy`: it adapts the batch, normalises state and
actions with statistics kept as buffers, applies the masked loss of `group_loss.py` and
loads weights. The model has no memory: of a window of steps it reads only the last.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import Config, FlatConfig
from .flow import VLAFlowMatching
from .vla_utils import sample_noise, sample_time_beta

_BASELINES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASELINES not in sys.path:
    sys.path.insert(0, _BASELINES)
from group_loss import masked_group_loss_sq  # noqa: E402


@dataclass
class Observation:
    """One control step, as filled by the dataloader's `to_observations`."""

    images: torch.Tensor       # [B, N, 3, H, W]  float in [-1, 1], N cameras
    lang_ids: torch.Tensor     # [B, L]  int
    lang_mask: torch.Tensor    # [B, L]  bool
    state: torch.Tensor        # [B, 212]  raw; the model normalises it
    state_mask: torch.Tensor   # [B, 212]  bool
    fov: torch.Tensor          # [B, N, 2]  not used by this model


class SmolVLAOfficialModel(nn.Module):
    def __init__(self, cfg: Config, *, load_weights: bool = True,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        # load_weights=False builds the VLM from its config only, for when a checkpoint
        # is loaded right after.
        flat = FlatConfig(cfg)
        flat.load_vlm_weights = bool(load_weights)
        self.flow = VLAFlowMatching(flat)
        if dtype == torch.float32:
            # the LeRobot code loads the VLM in bfloat16
            self.flow = self.flow.to(torch.float32)

        d, s = cfg.action.dim, cfg.action.state_dim
        for name, n in (("action_mean", d), ("action_std", d),
                        ("state_mean", s), ("state_std", s)):
            self.register_buffer(name, torch.zeros(n) if name.endswith("mean")
                                 else torch.ones(n))
        self.register_buffer("stats_loaded", torch.zeros((), dtype=torch.bool))

        bad = [(n, p.dtype) for n, p in self.named_parameters() if p.dtype != dtype]
        if bad:
            raise RuntimeError(
                f"{len(bad)} parameters are not {dtype} after construction, e.g. "
                f"{bad[:3]}. A guard on the requested dtype is not enough here -- "
                f"upstream pins the VLM's dtype inside `from_pretrained`.")

    # ------------------------------------------------------------------ statistics

    def set_normalisation(self, *, action_mean, action_std, state_mean, state_std,
                          eps: float = 1e-6) -> None:
        """Install the dataset statistics as buffers.

        Standard deviations are clamped to `eps`: a constant slot, such as the keypoint at
        the palm origin, has a standard deviation of zero.
        """
        for name, v, n in (("action_mean", action_mean, self.cfg.action.dim),
                           ("action_std", action_std, self.cfg.action.dim),
                           ("state_mean", state_mean, self.cfg.action.state_dim),
                           ("state_std", state_std, self.cfg.action.state_dim)):
            t = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
            if t.numel() != n:
                raise ValueError(f"{name} must have {n} entries, got {t.numel()}")
            if not bool(torch.isfinite(t).all()):
                raise ValueError(
                    f"{name} has {int((~torch.isfinite(t)).sum())} non-finite entries. "
                    f"Nothing downstream checks for this: the first `(a - mean) / std` "
                    f"turns the loss into NaN several thousand steps from here.")
            if name.endswith("_std"):
                if bool((t < 0).any()):
                    raise ValueError(
                        f"{name} has negative entries (min {float(t.min()):.3e}); a "
                        f"standard deviation cannot be negative, so the file is corrupt.")
                t = t.clamp_min(eps)
            getattr(self, name).copy_(t.to(getattr(self, name).device))
        self.stats_loaded.fill_(True)

    def _require_stats(self) -> None:
        if not bool(self.stats_loaded):
            raise RuntimeError(
                "normalisation statistics were never installed. Call set_normalisation() "
                "with values measured on THIS dataset at the 80 Hz action convention.")

    def normalise_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_mean) / self.action_std

    def unnormalise_action(self, a: torch.Tensor) -> torch.Tensor:
        return a * self.action_std + self.action_mean

    # ------------------------------------------------------------------ parameters

    def trainable_params_setup(self) -> None:
        """Freeze parameters as the config says. With `train_expert_only` the whole VLM is
        frozen; the expert, `state_proj`, the action projections and the time MLP train."""
        self.flow.vlm_with_expert.set_requires_grad()
        self.flow.set_requires_grad()

    def param_groups(self) -> list[dict]:
        """Optimiser groups: one at `optim.lr`, or two when `optim.vlm_lr` is set and VLM
        parameters are trainable, the VLM then at `vlm_lr`."""
        o = self.cfg.optim
        trainable = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        vlm = [p for n, p in trainable if "vlm_with_expert.vlm" in n]
        rest = [p for n, p in trainable if "vlm_with_expert.vlm" not in n]
        if o.vlm_lr is None or not vlm:
            return [{"name": "all", "params": [p for _n, p in trainable], "lr": o.lr,
                     "weight_decay": o.weight_decay, "is_backbone": False}]
        return [
            {"name": "expert", "params": rest, "lr": o.lr,
             "weight_decay": o.weight_decay, "is_backbone": False},
            {"name": "vlm", "params": vlm, "lr": o.vlm_lr,
             "weight_decay": o.weight_decay, "is_backbone": True},
        ]

    # ------------------------------------------------------------------ adaptation

    def _inputs(self, obs: Observation):
        """`Observation` -> the arguments of `flow`: a list of per-camera images
        [B, 3, H, W], their masks, the language tokens and mask, the normalised state."""
        self._require_stats()
        imgs = obs.images
        images = [imgs[:, i] for i in range(imgs.shape[1])]
        ones = torch.ones(imgs.shape[0], dtype=torch.bool, device=imgs.device)
        img_masks = [ones for _ in images]
        state = (obs.state - self.state_mean) / self.state_std
        return images, img_masks, obs.lang_ids, obs.lang_mask, state

    # ------------------------------------------------------------------ training

    def forward(self, steps: list[Observation], step_valid: torch.Tensor,
                actions: torch.Tensor, action_mask: torch.Tensor):
        return self.loss(steps, step_valid, actions, action_mask)

    def loss(self, steps: list[Observation], step_valid: torch.Tensor,
             actions: torch.Tensor, action_mask: torch.Tensor,
             generator: torch.Generator | None = None):
        """Returns (scalar loss, dict of per-group values to log).

        `actions` are raw, [B, chunk, 192]. Only `steps[-1]` is read: the collate pads on
        the left, so it is the supervised step of every item. `step_valid` is ignored.
        `generator`, when given, draws the noise.
        """
        del step_valid
        obs = steps[-1]
        images, img_masks, lang_ids, lang_mask, state = self._inputs(obs)

        a = self.normalise_action(actions.to(state.dtype))
        f = self.cfg.flow
        noise = (sample_noise(a.shape, a.device) if generator is None else
                 torch.randn(a.shape, device=a.device, dtype=a.dtype, generator=generator))
        time = sample_time_beta(a.shape[0], a.device, alpha=f.time_alpha,
                                beta=f.time_beta, scale=f.time_scale, offset=f.time_offset)

        # per-element squared error, [B, chunk, 192]
        square = self.flow.forward(images, img_masks, lang_ids, lang_mask, state, a,
                                   noise=noise, time=time)
        return masked_group_loss_sq(square, action_mask, self.cfg.loss.groups)

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def predict_from_noise(self, steps: list[Observation], noise: torch.Tensor
                           ) -> torch.Tensor:
        """Integrate from the given `noise`, [B, chunk, 192]. Returns normalised actions of
        the same shape. Only `steps[-1]` is read."""
        obs = steps[-1]
        images, img_masks, lang_ids, lang_mask, state = self._inputs(obs)
        return self.flow.sample_actions(images, img_masks, lang_ids, lang_mask, state,
                                        noise=noise.to(state.dtype))

    # ------------------------------------------------------------------ weights

    def load_pretrained(self, path: str) -> dict[str, list[str]]:
        """Load weights from `path`. Returns {loaded, missing, unexpected, kind}.

        A checkpoint of this model is loaded whole and must match exactly. From a released
        SmolVLA policy only the VLM (`model.vlm_with_expert.vlm.*`) is taken: its expert
        and projections were trained for a 32-dimensional action and are left out.
        """
        from safetensors.torch import load_file

        raw = load_file(path) if path.endswith(".safetensors") else torch.load(
            path, map_location="cpu", weights_only=False)
        # a checkpoint of this model keeps the state dict under "model"
        if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
            raw = raw["model"]

        # Keys of this model (`flow.*`, the statistics buffers): load everything.
        own_keys = set(self.state_dict())
        if any(k in own_keys for k in raw):
            report = self.load_state_dict(raw, strict=False)
            if report.missing_keys or report.unexpected_keys:
                raise RuntimeError(
                    f"{path} is one of ours but does not match this model: "
                    f"{len(report.missing_keys)} missing, "
                    f"{len(report.unexpected_keys)} unexpected, e.g. "
                    f"{(report.missing_keys + report.unexpected_keys)[:3]}. Resuming "
                    f"across a config change would train a different model than the "
                    f"checkpoint describes.")
            return {"loaded": sorted(raw), "missing": [], "unexpected": [],
                    "kind": "resume"}

        # Otherwise a released policy: take its VLM.
        prefix = "model.vlm_with_expert.vlm."
        remapped = {"vlm_with_expert.vlm." + k[len(prefix):]: v
                    for k, v in raw.items() if k.startswith(prefix)}
        if not remapped:
            raise RuntimeError(
                f"{path} holds neither this model's own keys nor anything under "
                f"{prefix!r}; keys look like {sorted(raw)[:3]}. Loading nothing would "
                f"leave the VLM at its Hub initialisation and the baseline would be "
                f"measuring the wrong model.")

        own = dict(self.flow.named_parameters())
        own.update(dict(self.flow.named_buffers()))
        bad = [k for k, v in remapped.items()
               if k in own and tuple(own[k].shape) != tuple(v.shape)]
        if bad:
            raise RuntimeError(f"shape mismatch on {len(bad)} VLM tensors, e.g. {bad[:3]}")

        report = self.flow.load_state_dict(remapped, strict=False)
        missing_vlm = [k for k in report.missing_keys if k.startswith("vlm_with_expert.vlm.")]
        if missing_vlm:
            raise RuntimeError(
                f"{len(missing_vlm)} VLM parameters were not covered by the checkpoint, "
                f"e.g. {missing_vlm[:3]}. A partially loaded backbone trains without "
                f"complaint.")
        return {"loaded": sorted(remapped), "missing": list(report.missing_keys),
                "unexpected": list(report.unexpected_keys), "kind": "vlm-only"}


Model = SmolVLAOfficialModel
