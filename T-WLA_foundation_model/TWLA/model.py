"""The full model: backbone, memory, action expert and normalisation statistics.

Training calls `loss(steps, ...)`, inference calls `select_action(obs, episode_id)`.
A training sample is L control steps: steps 0..L-2 run under no_grad and fill the bank,
step L-1 is supervised. Batches are left-padded, so the supervised step is the last index.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from . import flow
from .action_expert import ActionExpert
from .config import TWLAConfig
from .memory_bank import MemoryBank
from .memory_module import MemoryModule
from .vlm import SmolVLM2Backbone


@dataclass
class Observation:
    """The inputs of one control step. `state` is in raw units; the model normalises it."""

    images: torch.Tensor       # [B, N, 3, H, W]  already resized and in [-1, 1]
    lang_ids: torch.Tensor     # [B, max_language_tokens]
    lang_mask: torch.Tensor    # [B, max_language_tokens]  bool
    state: torch.Tensor        # [B, state_dim]     raw units
    state_mask: torch.Tensor   # [B, state_dim]     bool, which dims carry real data
    fov: torch.Tensor          # [B, N, 2]

    def to(self, device, dtype=None) -> "Observation":
        f = lambda t, d=None: t.to(device=device, dtype=d) if d else t.to(device)  # noqa: E731
        return Observation(f(self.images, dtype), f(self.lang_ids), f(self.lang_mask),
                           f(self.state, dtype), f(self.state_mask), f(self.fov, dtype))

    def index_select(self, rows: torch.Tensor) -> "Observation":
        """Select the batch rows `rows` [K]. Rows do not interact in the model, so a
        subset encodes to the same values as in the full batch."""
        return Observation(self.images[rows], self.lang_ids[rows], self.lang_mask[rows],
                           self.state[rows], self.state_mask[rows], self.fov[rows])


class TWLAModel(nn.Module):
    """SmolVLM2 backbone + memory + flow-matching action expert."""

    def __init__(self, cfg: TWLAConfig, *, load_weights: bool = True,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.vlm = SmolVLM2Backbone(cfg, load_weights=load_weights, dtype=dtype)
        self.memory = MemoryModule(cfg.memory, cfg.num_age_values).to(dtype)
        self.expert = ActionExpert(
            cfg.expert, cfg.action.chunk_size, cfg.action.dim, cfg.memory.num_slots,
            state_dim_with_mask=(2 * cfg.action.state_dim
                                 if cfg.action.direct_state_to_expert else None),
        ).to(dtype)

        # The statistics are buffers, so they are saved in the checkpoint.
        self.register_buffer("action_mean", torch.zeros(cfg.action.dim))
        self.register_buffer("action_std", torch.ones(cfg.action.dim))
        self.register_buffer("state_mean", torch.zeros(cfg.action.state_dim))
        self.register_buffer("state_std", torch.ones(cfg.action.state_dim))
        self.register_buffer("stats_loaded", torch.zeros((), dtype=torch.bool))

        # The world head is built only when configured; without it the state_dict has no
        # `world.*` keys. Imported here to avoid a circular import. The DC-AE that encodes
        # the target frames is not a submodule: the training loop passes the targets in.
        if cfg.world is not None:
            from world_model.head import WorldHead
            self.world = WorldHead(cfg.world, dtype=dtype,
                                   inherit=load_weights)

        # Inference state. Training builds its own bank for every batch.
        self._bank: MemoryBank | None = None
        self._episode_id: torch.Tensor | None = None

    @property
    def has_world_head(self) -> bool:
        """True when the world head was built."""
        return getattr(self, "world", None) is not None

    # ------------------------------------------------------------- normalisation

    def set_normalisation(self, *, action_mean, action_std, state_mean, state_std,
                          eps: float = 1e-6) -> None:
        """Install the dataset statistics. Call before training or inference."""
        for name, v, n in (("action_mean", action_mean, self.cfg.action.dim),
                           ("action_std", action_std, self.cfg.action.dim),
                           ("state_mean", state_mean, self.cfg.action.state_dim),
                           ("state_std", state_std, self.cfg.action.state_dim)):
            t = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
            if t.numel() != n:
                raise ValueError(f"{name} must have {n} entries, got {t.numel()}")

            # Reject corrupt statistics files.
            if not bool(torch.isfinite(t).all()):
                bad = int((~torch.isfinite(t)).sum())
                raise ValueError(
                    f"{name} has {bad} non-finite entries; normalising with it would give NaN.")
            if name.endswith("_std") and bool((t < 0).any()):
                raise ValueError(
                    f"{name} has negative entries (min {float(t.min()):.3e}); the file is corrupt.")

            if name.endswith("_std") and (t.abs() < eps).any():
                # A zero std is normal (a constant dim, or padding): clamp it so that the
                # normalisation stays finite.
                t = t.clamp_min(eps)
            getattr(self, name).copy_(t.to(getattr(self, name).device))
        self.stats_loaded.fill_(True)

    def _require_stats(self) -> None:
        if not bool(self.stats_loaded):
            raise RuntimeError(
                "normalisation statistics were never installed. Call set_normalisation() "
                "with values computed on this dataset at the configured action rate.")

    def normalise_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_mean) / self.action_std

    def unnormalise_action(self, a: torch.Tensor) -> torch.Tensor:
        return a * self.action_std + self.action_mean

    def state_with_mask(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """state [B, 212], mask [B, 212] -> [B, 424]: the normalised state, zeroed where
        masked, followed by the mask."""
        if self.cfg.action.hide_hand_state:                     # hide the keypoints of both hands
            mask = mask.clone(); mask[:, 6:69] = False; mask[:, 75:138] = False
        z = (state - self.state_mean) / self.state_std
        m = mask.to(z.dtype)
        return torch.cat([z * m, m], dim=-1)

    # ------------------------------------------------------------- one control step

    def encode(self, obs: Observation, bank: MemoryBank, *,
               rows: torch.Tensor | None = None,
               drop_memory: torch.Tensor | None = None,
               return_image_tokens: bool = False):
        """Read the bank, run the VLM and return `m_f` [B, S, D]. Does not write the bank.

        rows: restricts the bank read to these batch rows; `obs` must be the same subset.
        drop_memory: [B] bool; those items read the bank as empty (memory dropout).
        return_image_tokens: also return the image tokens, for the world head.
        """
        read = bank.read(rows)
        if drop_memory is not None and bool(drop_memory.any()):
            keep = (~drop_memory)[:, None]
            read.valid = read.valid & keep
        m_t = self.memory(read)
        return self.vlm(obs.images, obs.lang_ids, obs.lang_mask,
                        self.state_with_mask(obs.state, obs.state_mask), obs.fov, m_t,
                        return_image_tokens=return_image_tokens)

    # ------------------------------------------------------------------- training

    def loss(self, steps: list[Observation], step_valid: torch.Tensor,
             actions: torch.Tensor, action_mask: torch.Tensor, *,
             generator: torch.Generator | None = None,
             world_targets: torch.Tensor | None = None,
             world_valid: torch.Tensor | None = None,
             lam_world: float | torch.Tensor = 0.0,
             probe_mf_grad: bool = False,
             probe_store: dict | None = None,
             sample_source: list[str] | None = None,
             domain_probe: dict | None = None,
             ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Training loss for a batch of samples. Returns (loss, parts for logging).

        steps: L_max observations, left-padded; `steps[-1]` is the supervised step.
        step_valid: [B, L_max] bool; padded steps are False and never touch the bank.
        actions: [B, chunk_size, action_dim] raw ground truth of the supervised step.
        action_mask: [B, chunk_size, action_dim] bool, which targets are real.
        world_targets, world_valid: targets of the world head; None skips the world loss.
        lam_world: weight of the world loss, a float or a per-sample tensor [B].
        probe_mf_grad, probe_store, domain_probe, sample_source: optional gradient
            diagnostics, described where they are computed.
        """
        self._require_stats()
        if step_valid.shape[1] != len(steps):
            raise ValueError(f"{len(steps)} steps but step_valid has {step_valid.shape[1]}")
        if not bool(step_valid[:, -1].all()):
            raise ValueError("the last step is the supervised one; it may not be padding")

        b = actions.shape[0]
        cfg = self.cfg
        bank = MemoryBank(b, cfg.memory.bank_capacity, cfg.memory.num_slots,
                          cfg.memory.dim, device=actions.device, dtype=self.vlm.dtype)

        # Fill the bank without gradient. At each time index only the rows that are real
        # there are encoded.
        with torch.no_grad():
            for t, obs in enumerate(steps[:-1]):
                rows = torch.nonzero(step_valid[:, t], as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                sub = obs if rows.numel() == b else obs.index_select(rows)
                bank.append_rows(rows, self.encode(sub, bank, rows=rows))

        drop_s, drop_m = self._draw_dropout(b, actions.device, generator)
        obs = steps[-1]
        if drop_s is not None:
            obs = Observation(obs.images, obs.lang_ids, obs.lang_mask,
                              obs.state * (~drop_s)[:, None],
                              obs.state_mask & (~drop_s)[:, None], obs.fov)
        want_world = self.has_world_head and world_targets is not None
        if want_world:
            m_f, image_tokens = self.encode(obs, bank, drop_memory=drop_m,
                                            return_image_tokens=True)
        else:
            m_f = self.encode(obs, bank, drop_memory=drop_m)
        # Built from `obs` after state dropout, so the direct path is dropped together
        # with the prefix path.
        state_condition = (self.state_with_mask(obs.state, obs.state_mask)
                           if cfg.action.direct_state_to_expert else None)

        a = self.normalise_action(actions.to(self.vlm.dtype))
        k = max(1, cfg.flow.draws_per_condition)
        if k > 1:
            m_f = flow.repeat_for_draws(m_f, k)
            a = flow.repeat_for_draws(a, k)
            action_mask = flow.repeat_for_draws(action_mask, k)
            if state_condition is not None:
                state_condition = flow.repeat_for_draws(state_condition, k)

        tau = flow.sample_time_beta(a.shape[0], a.device, cfg.flow, generator)
        noise = torch.randn(a.shape, device=a.device, dtype=a.dtype, generator=generator)
        x_t, u = flow.make_flow_target(a, noise, tau.to(a.dtype))
        pred = self.expert(m_f, x_t, action_mask, tau.to(a.dtype), state_condition)
        action_loss, parts = flow.masked_component_loss(pred, u, action_mask, cfg.flow)

        if not want_world:
            return action_loss, parts

        # `m_f` is passed without detach: the world loss sends gradient into the memory.
        if k > 1:
            raise NotImplementedError(
                "flow.draws_per_condition > 1 repeats m_f for the action loss; the world "
                "head needs one row per sample and the two conventions have not been "
                "reconciled. Set draws_per_condition to 1 for world-head runs.")

        # Per-sample lambda: rows are weighted by lam_b / mean(lam) inside the world loss,
        # and the scalar mean(lam) is used from here on.
        if torch.is_tensor(lam_world):
            lam_vec = lam_world.to(m_f.device, torch.float32).reshape(-1)
            if lam_vec.numel() != b:
                raise ValueError(f"per-sample lam_world must have {b} entries, got {lam_vec.numel()}")
            lam_s = float(lam_vec.mean())
            sw = lam_vec / lam_s if lam_s > 0.0 else None
        else:
            lam_s, sw = float(lam_world), None
        lam_world = lam_s
        world_loss, wparts = self.world.loss(
            image_tokens, m_f, world_targets, world_valid, generator=generator,
            sample_weight=sw)
        per_sample_w = wparts.pop("per_sample")
        per_sample_raw = wparts.pop("per_sample_raw")
        parts.update({f"world_{n}": v for n, v in wparts.items()})
        if domain_probe is not None and sample_source is not None and m_f.requires_grad:
            # Per-domain gradients on the memory parameters, for the domains named in
            # `sample_source`: the action loss restricted to the domain's rows, and the
            # mean unweighted world loss of those rows. Stored as (g_action, g_world, rows).
            mem_params = [q for q in self.memory.parameters() if q.requires_grad]
            if mem_params:
                src = list(sample_source)
                if len(src) != b:
                    raise ValueError(f"sample_source has {len(src)} names for {b} rows")
                flat = lambda ts: torch.cat([  # noqa: E731
                    (t if t is not None else torch.zeros_like(q)).float().reshape(-1)
                    for t, q in zip(ts, mem_params)])
                for name in sorted(set(src)):
                    rows = torch.tensor([i for i, n_ in enumerate(src) if n_ == name],
                                        device=m_f.device)
                    rm = torch.zeros(b, dtype=torch.bool, device=m_f.device)
                    rm[rows] = True
                    la_d, _ = flow.masked_component_loss(pred, u, action_mask & rm[:, None, None],
                                                         cfg.flow)
                    lw_d = per_sample_raw[rows].mean()
                    if not (la_d.requires_grad and lw_d.requires_grad):
                        continue
                    ta = torch.autograd.grad(la_d, mem_params, retain_graph=True, allow_unused=True)
                    tw = torch.autograd.grad(lw_d, mem_params, retain_graph=True, allow_unused=True)
                    domain_probe[name] = (flat(ta).detach(), flat(tw).detach(), int(rows.numel()))
        if probe_store is not None:
            # Hands out the live tensors for offline analysis; not set in training.
            probe_store.update(action_loss=action_loss, world_loss=world_loss, m_f=m_f,
                               pred=pred, u=u, action_mask=action_mask)
        parts["action"] = action_loss.detach()
        parts["world"] = world_loss.detach()
        parts["lam_world"] = torch.as_tensor(float(lam_world))

        if probe_mf_grad and lam_world > 0.0 and m_f.requires_grad:
            # Gradient norms of both terms at `m_f`, their ratio and cosine, over the whole
            # batch. retain_graph: the training backward still runs over this graph.
            g_a = torch.autograd.grad(action_loss, m_f, retain_graph=True)[0]
            g_w = torch.autograd.grad(world_loss, m_f, retain_graph=True)[0]
            na, nw = g_a.float().norm(), lam_world * g_w.float().norm()
            parts["mf_gA"] = na.detach()
            parts["mf_gW"] = nw.detach()
            parts["mf_r"] = (nw / na.clamp_min(1e-12)).detach()
            parts["mf_cos"] = torch.nn.functional.cosine_similarity(
                g_a.float().reshape(-1), g_w.float().reshape(-1), dim=0).detach()
            # The same on the memory parameters. autograd.grad leaves `.grad` untouched.
            mem_params = [q for q in self.memory.parameters() if q.requires_grad]
            if mem_params:
                ta = torch.autograd.grad(action_loss, mem_params, retain_graph=True,
                                         allow_unused=True)
                tw = torch.autograd.grad(world_loss, mem_params, retain_graph=True,
                                         allow_unused=True)
                flat = lambda ts: torch.cat([  # noqa: E731
                    (t if t is not None else torch.zeros_like(q)).float().reshape(-1)
                    for t, q in zip(ts, mem_params)])
                va, vw = flat(ta), lam_world * flat(tw)
                pa, pw = va.norm(), vw.norm()
                parts["mem_gA"] = pa.detach()
                parts["mem_gW"] = pw.detach()
                parts["mem_r"] = (pw / pa.clamp_min(1e-12)).detach()
                parts["mem_cos"] = torch.nn.functional.cosine_similarity(va, vw, dim=0).detach()

        # If the action path is frozen (world stage 1), the action loss has no gradient
        # and is left out of the total.
        if not action_loss.requires_grad:
            total = world_loss
        else:
            total = action_loss + lam_world * world_loss
        return total, parts

    def forward(self, steps: list[Observation], step_valid: torch.Tensor,
                actions: torch.Tensor, action_mask: torch.Tensor, *,
                generator: torch.Generator | None = None,
                world_targets: torch.Tensor | None = None,
                world_valid: torch.Tensor | None = None,
                lam_world: float | torch.Tensor = 0.0,
                probe_mf_grad: bool = False,
                sample_source: list[str] | None = None,
                domain_probe: dict | None = None):
        """Same as `loss`. Under DDP call the wrapper (`ddp(...)`): `ddp.module.loss(...)`
        would skip the gradient synchronisation between ranks."""
        return self.loss(steps, step_valid, actions, action_mask, generator=generator,
                         world_targets=world_targets, world_valid=world_valid,
                         lam_world=lam_world, probe_mf_grad=probe_mf_grad,
                         sample_source=sample_source, domain_probe=domain_probe)

    def _draw_dropout(self, b: int, device, generator):
        """Per-sample state and memory dropout masks; None when the rate is 0 or in eval mode."""
        tr = self.cfg.train
        out = []
        for rate in (tr.state_dropout, tr.memory_dropout):
            if rate <= 0 or not self.training:
                out.append(None)
            else:
                out.append(torch.rand(b, device=device, generator=generator) < rate)
        return tuple(out)

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def select_action(self, obs: Observation, episode_id: torch.Tensor) -> torch.Tensor:
        """Advance one control step. Returns the actions to execute, in raw units.

        The bank persists across calls and is reset for the batch items whose
        `episode_id` differs from the previous call.
        """
        return self.select_action_context(obs, episode_id)[0]

    def select_action_context(self, obs: Observation, episode_id: torch.Tensor, steps: int | None = None
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        """`select_action` that also returns `m_f` [B, S, D]. `steps` is the number of
        actions returned from the chunk (default `executed_steps`)."""
        self._require_stats()
        cfg = self.cfg
        b = obs.images.shape[0]
        eid = episode_id.to(obs.images.device).reshape(-1)
        if eid.numel() != b:
            raise ValueError(f"episode_id must have {b} entries, got {eid.numel()}")

        if self._bank is None or self._bank.batch_size != b:
            self.reset(b, obs.images.device)
        elif self._episode_id is not None:
            self._bank.reset_where(eid != self._episode_id)
        # `_episode_id is None` with a bank means reset() was just called: already empty.
        self._episode_id = eid.clone()

        m_f = self.encode(obs, self._bank)
        self._bank.append(m_f)

        state_condition = (self.state_with_mask(obs.state, obs.state_mask)
                           if cfg.action.direct_state_to_expert else None)
        ctx = self.expert.prepare(m_f, state_condition)
        # The mask used in training: component groups on, padding off.
        mask = flow.component_action_mask(b, cfg.action.chunk_size, cfg.action.dim,
                                          cfg.flow, m_f.device)
        noise = torch.randn(b, cfg.action.chunk_size, cfg.action.dim,
                            device=m_f.device, dtype=m_f.dtype)
        # Masked dims start at zero rather than noise: their predicted velocity stays
        # zero, so noise there would be returned unchanged.
        noise.masked_fill_(~mask, 0)
        a = flow.euler_integrate(
            lambda x, tau: self.expert.velocity(ctx, x, mask, tau),
            noise, cfg.flow.num_inference_steps)
        n = cfg.action.executed_steps if steps is None else int(steps)
        return self.unnormalise_action(a.float())[:, :n], m_f

    # ------------------------------------------------------------------ optimiser

    def param_groups(self) -> list[dict]:
        """AdamW parameter groups, with weight decay on weight matrices only.

        Parameters with `ndim <= no_decay_ndim_max`, or a name containing one of
        `no_decay_name_parts`, are not decayed. `expert.*` and `vlm.*` get their own
        groups when `lr_expert` / `lr_vlm` are set, and the world head adds its own.
        """
        o = self.cfg.optim
        world_ids = ({id(p) for p in self.world.parameters()}
                     if self.has_world_head else set())

        split_expert = o.lr_expert is not None
        split_vlm = o.lr_vlm is not None
        decay, no_decay, e_decay, e_no_decay, v_decay, v_no_decay = [], [], [], [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in world_ids:
                continue
            skip = (p.ndim <= o.no_decay_ndim_max
                    or any(part in name for part in o.no_decay_name_parts))
            if split_expert and name.startswith("expert."):
                (e_no_decay if skip else e_decay).append(p)
            elif split_vlm and name.startswith("vlm."):
                (v_no_decay if skip else v_decay).append(p)
            else:
                (no_decay if skip else decay).append(p)

        # Every group carries `base_lr`, which the schedule multiplies (`lr_scale_at`).
        # Empty groups are dropped.
        groups = [g for g in
                  ({"params": decay, "weight_decay": o.weight_decay, "base_lr": o.lr,
                    "name": "decay"},
                   {"params": no_decay, "weight_decay": 0.0, "base_lr": o.lr,
                    "name": "no_decay"},
                   {"params": e_decay, "weight_decay": o.weight_decay,
                    "base_lr": o.lr_expert, "name": "expert.decay", "expert": True},
                   {"params": e_no_decay, "weight_decay": 0.0,
                    "base_lr": o.lr_expert, "name": "expert.no_decay", "expert": True},
                   {"params": v_decay, "weight_decay": o.weight_decay,
                    "base_lr": o.lr_vlm, "name": "vlm.decay"},
                   {"params": v_no_decay, "weight_decay": 0.0,
                    "base_lr": o.lr_vlm, "name": "vlm.no_decay"})
                  if g["params"]]

        if self.has_world_head:
            # The world head defines its own groups and rates. Frozen parameters are left out.
            for g in self.world.param_groups():
                ps = [p for p in g["params"] if p.requires_grad]
                if not ps:
                    continue
                groups.append({"params": ps, "weight_decay": 0.0,
                               "base_lr": g["lr"], "name": g["name"]})
        return groups

    def make_optimizer(self) -> torch.optim.AdamW:
        o = self.cfg.optim
        return torch.optim.AdamW(self.param_groups(), lr=o.lr, betas=o.betas, eps=o.eps)

    def lr_scale_at(self, step: int) -> float:
        """The schedule as a multiplier on each group's `base_lr`: `lr_at(step) / lr`.
        The training loop scales `base_lr`, so that the groups keep their own rates."""
        return self.lr_at(step) / self.cfg.optim.lr

    def expert_warmup_scale_at(self, step: int) -> float:
        """Extra multiplier for the `expert.*` groups: a linear warmup over
        `expert_warmup_steps` from `schedule_offset`, then 1."""
        o = self.cfg.optim
        if o.expert_warmup_steps <= 0:
            return 1.0
        s = step - o.schedule_offset
        return min(1.0, (max(0, s) + 1) / o.expert_warmup_steps)

    def freeze_expert(self) -> int:
        """Freeze the action expert. Gradient still flows through it to the memory.
        Returns the number of parameters frozen."""
        self.expert.requires_grad_(False)
        return sum(p.numel() for p in self.expert.parameters())

    # --------------------------------------------------------------- world head stages
    def freeze_for_world_stage1(self) -> dict:
        """World stage 1: freeze the backbone, the memory and the expert, and train the
        world head alone. Returns the frozen and trainable parameter counts."""
        if not self.has_world_head:
            raise RuntimeError("no world head to train; TWLAConfig.world is None")
        for m in (self.vlm, self.memory, self.expert):
            m.requires_grad_(False)
            m.eval()
        n_frozen = sum(p.numel() for m in (self.vlm, self.memory, self.expert)
                       for p in m.parameters())
        n_train = sum(p.numel() for p in self.world.parameters() if p.requires_grad)
        return {"frozen": n_frozen, "trainable": n_train}

    def freeze_world_head(self) -> int:
        """Freeze the world head. Its loss still has a gradient with respect to `m_f`.
        Returns the number of parameters frozen."""
        if not self.has_world_head:
            raise RuntimeError("no world head to freeze; TWLAConfig.world is None")
        self.world.requires_grad_(False)
        self.world.eval()
        return sum(p.numel() for p in self.world.parameters())

    def load_world_head(self, state: dict) -> int:
        """Load the `world.*` tensors of a full-model state dict into the world head
        (strict). The rest of `state` is ignored. Returns the number of tensors loaded."""
        if not self.has_world_head:
            raise RuntimeError("no world head to load into; TWLAConfig.world is None")
        head = {k[len("world."):]: v for k, v in state.items() if k.startswith("world.")}
        if not head:
            raise RuntimeError("checkpoint has no `world.*` tensors")
        self.world.load_state_dict(head, strict=True)   # a missing or renamed tensor raises
        return len(head)

    def lr_at(self, step: int) -> float:
        """Learning rate at `step`: linear warmup, then cosine decay to `final_lr`, or
        constant with `constant_after_warmup`. Steps are counted from `schedule_offset`."""
        import math
        o = self.cfg.optim
        s = step - o.schedule_offset
        if s < o.warmup_steps:
            return o.lr * (max(0, s) + 1) / max(1, o.warmup_steps)
        if o.constant_after_warmup:
            return o.lr
        t = min(1.0, (s - o.warmup_steps) / max(1, o.decay_steps - o.warmup_steps))
        return o.final_lr + 0.5 * (o.lr - o.final_lr) * (1.0 + math.cos(math.pi * t))

    def reset(self, batch_size: int, device: torch.device) -> None:
        """Create an empty inference bank. Episode boundaries are handled by
        `select_action`; call this to start over or to change the batch size."""
        cfg = self.cfg
        self._bank = MemoryBank(batch_size, cfg.memory.bank_capacity, cfg.memory.num_slots,
                                cfg.memory.dim, device=device, dtype=self.vlm.dtype)
        self._episode_id = None
