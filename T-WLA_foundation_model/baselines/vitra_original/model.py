"""The VITRA baseline: PaliGemma2-3B -> cognition token z [B, 1, 2304] -> DiT, which also
takes the state -> epsilon -> DDIM -> action chunk.

The model has no memory. `loss` and `predict_from_noise` take a list of steps, like the
main model, and read only the last one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from . import diffusion as dfn
from .config import Config
from .dit import DiT
from .vlm import PaliGemmaVLM


@dataclass
class Observation:
    """One control step, as the loader's `to_observations` fills it."""

    images: torch.Tensor       # [B, N, 3, H, W]  float, already in [-1, 1]
    lang_ids: torch.Tensor     # [B, L]  int
    lang_mask: torch.Tensor    # [B, L]  bool
    state: torch.Tensor        # [B, 212]  raw; the model normalises it
    state_mask: torch.Tensor   # [B, 212]  bool
    fov: torch.Tensor          # [B, N, 2]


# Buffers that a VITRA checkpoint does not contain: the diffusion schedule and the
# normalisation statistics. `load_pretrained` does not report them as missing.
_OURS_NOT_THEIRS = ("alpha_bar", "action_mean", "action_std", "state_mean", "state_std",
                    "stats_loaded")

# Name segments added by FSDP, activation checkpointing and torch.compile. `param_groups`
# splits on the parameter name, so they are stripped first.
_WRAPPER_SEGMENTS = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod.")


def _strip_wrappers(name: str) -> str:
    for seg in _WRAPPER_SEGMENTS:
        name = name.replace(seg, "")
    # DDP adds a leading `module.`; a `module.` further inside is a real submodule.
    return name[len("module."):] if name.startswith("module.") else name


class VitraOriginalModel(nn.Module):
    def __init__(self, cfg: Config, *, load_weights: bool = True,
                 dtype: torch.dtype = torch.float32):
        """`load_weights=False` builds the backbone without reading its pretrained weights,
        for when a checkpoint supplies them. `dtype` must be float32: the parameters are
        fp32 and bf16 is used only inside the VLM forward."""
        super().__init__()
        if dtype is not torch.float32:
            raise ValueError(
                f"dtype={dtype}: this baseline holds fp32 parameters and uses bf16 only "
                f"inside the VLM forward, which is VITRA's configuration. Casting the "
                f"parameters would be a different recipe.")
        cfg.validate()
        self.cfg = cfg
        self.vlm = PaliGemmaVLM(cfg, load_weights=load_weights)
        self.dit = DiT(cfg)
        # A buffer, so that a checkpoint carries the schedule it was trained with.
        self.register_buffer("alpha_bar", dfn.alpha_bar_table(cfg), persistent=True)
        self._ddim_grid = dfn.ddim_grid(cfg)

        # The loader yields raw actions and state. The statistics are buffers of the
        # model, set by `set_normalisation` or loaded with a checkpoint.
        self.register_buffer("action_mean", torch.zeros(cfg.action.dim))
        self.register_buffer("action_std", torch.ones(cfg.action.dim))
        self.register_buffer("state_mean", torch.zeros(cfg.action.state_dim))
        self.register_buffer("state_std", torch.ones(cfg.action.state_dim))
        self.register_buffer("stats_loaded", torch.zeros((), dtype=torch.bool))

        # Which parameters are exempt from weight decay (biases and 1-D parameters).
        # Recorded by name here because FSDP flattens every parameter to 1-D afterwards.
        self._no_decay = {n: (p.ndim <= 1 or n.endswith(".bias"))
                          for n, p in self.named_parameters()}

        # Check the built model, not only the argument: the backbone's sub-configs carry
        # a dtype of their own.
        wrong = {}
        for n, prm in self.named_parameters():
            if prm.dtype is not torch.float32:
                wrong[str(prm.dtype)] = wrong.get(str(prm.dtype), 0) + prm.numel()
        if wrong:
            raise RuntimeError(
                f"parameters are not all fp32 after construction: {wrong}. VITRA holds "
                f"fp32 weights and uses bf16 only inside the VLM forward via autocast.")

    # ------------------------------------------------------------ normalisation

    def set_normalisation(self, *, action_mean, action_std, state_mean, state_std,
                          eps: float = 1e-6) -> None:
        """Install the dataset statistics: vectors of 192 (action) and 212 (state).

        A standard deviation of zero, as on the padding, is clamped to `eps`.
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
                "with values measured on THIS dataset at the 80 Hz action convention -- "
                "VITRA's shipped per-domain statistics are measured at the SOURCE rate "
                "and are off by 8/3 on the wrist channels, which raises nothing.")

    def normalise_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_mean) / self.action_std

    def unnormalise_action(self, a: torch.Tensor) -> torch.Tensor:
        return a * self.action_std + self.action_mean

    # ------------------------------------------------------------------ pieces

    def trainable_params_setup(self) -> None:
        """Set `requires_grad`, once. `llm_freeze_step` acts through the learning rate and
        does not change the trainable set."""
        self.vlm.trainable_params_setup()
        self.dit.requires_grad_(True)

    def param_groups(self) -> list[dict]:
        """Up to four optimiser groups: backbone or not, weight decay or not.

        The backbone is `vlm.pg` and uses `backbone_lr`. The FoV encoder, the cognition
        token and the DiT use `action_model_lr`. Biases and 1-D parameters are not
        decayed. `is_backbone` marks the groups a schedule holds at zero before
        `llm_freeze_step`.
        """
        o = self.cfg.optim
        wd_action = (o.weight_decay if o.action_model_weight_decay is None
                     else o.action_model_weight_decay)

        buckets: dict[tuple[bool, bool], list] = {(b, d): [] for b in (True, False)
                                                  for d in (True, False)}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            clean = _strip_wrappers(name)
            no_decay = self._no_decay[clean]
            if "cognition_token" in clean:
                no_decay = not o.cognition_token_weight_decay
            buckets[(clean.startswith("vlm.pg."), no_decay)].append(p)

        groups = [
            ("backbone_decay",      True,  False, o.backbone_lr,     o.weight_decay),
            ("backbone_no_decay",   True,  True,  o.backbone_lr,     0.0),
            ("action_decay",        False, False, o.action_model_lr, wd_action),
            ("action_no_decay",     False, True,  o.action_model_lr, 0.0),
        ]
        out = []
        for gname, is_backbone, no_decay, lr, wd in groups:
            params = buckets[(is_backbone, no_decay)]
            if not params:
                continue
            out.append({"params": params, "lr": lr, "weight_decay": wd,
                        "name": gname, "is_backbone": is_backbone})
        return out

    def encode(self, obs: Observation) -> torch.Tensor:
        """-> z [B, 1, hidden]."""
        return self.vlm(obs.images, obs.lang_ids, obs.lang_mask, obs.fov).unsqueeze(1)

    def _state_pair(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        """The normalised state [B, 1, 212] and its mask, as the DiT's state embedder
        takes them."""
        self._require_stats()
        z = (obs.state - self.state_mean) / self.state_std
        return z.unsqueeze(1), obs.state_mask.to(z.dtype).unsqueeze(1)

    def _eps(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor,
             mask_f: torch.Tensor, state: torch.Tensor, state_mask: torch.Tensor,
             generator: torch.Generator | None = None) -> torch.Tensor:
        """One network call. The action token is `[x_t * mask ; mask]`, so a masked slot
        is zero in both halves."""
        xin = torch.cat([x_t * mask_f, mask_f], dim=2)
        return self.dit(xin, t, z, state, state_mask, generator)

    # ------------------------------------------------------------------ training

    def forward(self, steps: list[Observation], step_valid: torch.Tensor,
                actions: torch.Tensor, action_mask: torch.Tensor):
        return self.loss(steps, step_valid, actions, action_mask)

    def loss(self, steps: list[Observation], step_valid: torch.Tensor,
             actions: torch.Tensor, action_mask: torch.Tensor,
             generator: torch.Generator | None = None):
        """-> (scalar loss, dict of per-group values for logging). Only `steps[-1]` is
        read; the collate pads on the left, so it is the supervised step of every item."""
        del step_valid
        obs = steps[-1]
        k = self.cfg.diffusion.repeated_diffusion_steps

        z = self.encode(obs)                                     # [B, 1, D]
        state, state_mask = self._state_pair(obs)

        # k independent (noise, t) draws share one VLM forward.
        if k > 1:
            z = z.repeat_interleave(k, dim=0)
            state = state.repeat_interleave(k, dim=0)
            state_mask = state_mask.repeat_interleave(k, dim=0)
            actions = actions.repeat_interleave(k, dim=0)
            action_mask = action_mask.repeat_interleave(k, dim=0)

        a = self.normalise_action(actions.to(z.dtype))
        mask_f = action_mask.to(z.dtype)
        noise = torch.randn(a.shape, device=a.device, dtype=a.dtype, generator=generator)
        t = dfn.sample_timesteps(a.shape[0], self.cfg, a.device, generator)
        x_t = dfn.q_sample(a, t, noise, self.alpha_bar)

        eps = self._eps(x_t, t, z, mask_f, state, state_mask, generator)
        return dfn.masked_group_loss(eps, noise, action_mask, self.cfg)

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def predict_from_noise(self, steps: list[Observation], noise: torch.Tensor
                           ) -> torch.Tensor:
        """-> [B, chunk, 192] normalised actions, sampled from the given `noise`.

        The noise is an argument so that a test can fix it per sample. With
        `cfg_scale > 1` every step evaluates the cognition token and the learned
        `uncondition` vector and combines the two epsilons (classifier-free guidance).
        """
        obs = steps[-1]
        b = obs.state.shape[0]
        cfg = self.cfg
        z = self.encode(obs)
        state, state_mask = self._state_pair(obs)

        mask = dfn.component_action_mask(b, cfg.action.chunk_size, cfg.action.dim,
                                         cfg, noise.device)
        mask_f = mask.to(z.dtype)
        x = noise.to(z.device, z.dtype).clone()
        # Masked slots start at zero, as in training.
        x = x * mask_f

        scale = float(cfg.diffusion.cfg_scale)
        if scale > 1.0:
            if cfg.dit.class_dropout_prob <= 0:
                raise ValueError("cfg_scale > 1 needs the learned `uncondition`, which "
                                 "only exists when class_dropout_prob > 0")
            uncond = self.dit.z_embedder.uncondition.to(z.dtype).expand(b, 1, -1)
            z_pair = torch.cat([z, uncond], dim=0)
            state_pair = torch.cat([state, state], dim=0)
            smask_pair = torch.cat([state_mask, state_mask], dim=0)
            mask_pair = torch.cat([mask_f, mask_f], dim=0)

            def eps_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                both = self._eps(torch.cat([x_t, x_t], 0), torch.cat([t, t], 0),
                                 z_pair, mask_pair, state_pair, smask_pair)
                cond_eps, uncond_eps = both.chunk(2, dim=0)
                return uncond_eps + scale * (cond_eps - uncond_eps)
        else:
            def eps_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return self._eps(x_t, t, z, mask_f, state, state_mask)

        out = dfn.ddim_sample(eps_fn, x, self.alpha_bar, self._ddim_grid,
                              cfg.diffusion.ddim_eta)
        return out * mask_f

    @torch.no_grad()
    def select_action(self, steps: list[Observation],
                      generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw the noise, then sample."""
        obs = steps[-1]
        b = obs.state.shape[0]
        shape = (b, self.cfg.action.chunk_size, self.cfg.action.dim)
        noise = torch.randn(shape, device=obs.state.device, dtype=obs.state.dtype,
                            generator=generator)
        return self.predict_from_noise(steps, noise)

    # ------------------------------------------------------------------ weights

    def load_pretrained(self, path: str) -> dict[str, list[str]]:
        """Load a VITRA checkpoint, renaming its keys with `init.key_remap`.

        Returns the missing and unexpected keys. Missing keys raise unless
        `init.allow_partial_load` is set.
        """
        try:
            raw = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:                                  # the file holds more than tensors
            raw = torch.load(path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        if isinstance(state, dict) and isinstance(state.get("model"), dict):   # the `{"model": state_dict}` format
            state = state["model"]

        remapped = {}
        for key, value in state.items():
            for old, new in self.cfg.init.key_remap:
                if key.startswith(old):
                    key = new + key[len(old):]
                    break
            remapped[key] = value

        incompatible = self.load_state_dict(remapped, strict=False)
        report = {"missing": [k for k in incompatible.missing_keys
                              if k not in _OURS_NOT_THEIRS],
                  "unexpected": list(incompatible.unexpected_keys)}
        if report["missing"] and not self.cfg.init.allow_partial_load:
            head = "\n  ".join(report["missing"][:20])
            raise RuntimeError(
                f"{len(report['missing'])} parameters were NOT found in {path} and keep "
                f"their random initialisation. First few:\n  {head}\n"
                f"Fix `init.key_remap`, or set `init.allow_partial_load` if this is "
                f"deliberate (say so in the run's log).")
        return report
