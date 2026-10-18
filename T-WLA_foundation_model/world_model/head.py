"""`WorldHead`: the adapter and the small SANA as one module, held by `TWLAModel.world`.

The DC-AE is not part of it: the training loop encodes the target frames with
`dc_ae.encode_targets` and passes the latents in. The head never sees or returns actions.
"""

from __future__ import annotations

import torch
from torch import nn

from .adapter import WorldAdapter
from .config import WorldConfig
from .loss import world_loss


class WorldHead(nn.Module):
    """Image tokens E_t and memory tokens m_f -> the scalar `L_world`."""

    def __init__(self, cfg: WorldConfig, *, dtype: torch.dtype | None = None,
                 inherit: bool = True):
        super().__init__()
        from .sana import build as build_sana, load_inherited
        from .horizon import install as horizon_install

        self.cfg = cfg
        self.adapter = WorldAdapter(cfg.adapter)
        self.sana = build_sana(cfg.sana)

        # Where the pretrained weights came from. A plain dict, not in the state dict.
        self.provenance = load_inherited(self.sana, cfg.sana) if inherit else None

        # Installed after `load_inherited`, which expects the stock key names: the
        # wrapper moves `time_embed.emb.*` to `time_embed.emb.inner.*`. The returned
        # parameter is not stored on `self`, or the state dict would hold it twice.
        if cfg.horizon_inject == "modulation" and cfg.loss.num_horizons > 1:
            horizon_install(self.sana, cfg.loss.num_horizons)

        if dtype is not None:
            self.to(dtype)

    def loss(self, image_tokens: torch.Tensor, final_memory: torch.Tensor,
             z0: torch.Tensor, valid: torch.Tensor, *,
             generator: torch.Generator | None = None,
             sample_weight: torch.Tensor | None = None):
        """Returns (scalar loss, dict of log values); see `world_loss` for the arguments.
        Pass `final_memory` without `detach()` when the loss should reach the memory."""
        return world_loss(self.sana, self.adapter, image_tokens, final_memory,
                          z0, valid, self.cfg.loss,
                          generator=generator,
                          timestep_scale=self.cfg.sana.timestep_scale,
                          sample_weight=sample_weight)

    def param_groups(self) -> list[dict]:
        """Optimiser groups: the adapter at `lr_adapter`, the SANA blocks at `lr_sana`.

        The horizon embedding sits inside SANA but is trained from scratch, so it gets
        `lr_adapter`.
        """
        hp = self.horizon_param
        hp_ids = {id(hp)} if hp is not None else set()
        sana_params = [p for p in self.sana.parameters() if id(p) not in hp_ids]

        groups = [
            {"params": list(self.adapter.parameters()), "lr": self.cfg.lr_adapter,
             "name": "world.adapter"},
            {"params": sana_params, "lr": self.cfg.lr_sana, "name": "world.sana"},
        ]
        if hp is not None:
            groups.append({"params": [hp], "lr": self.cfg.lr_adapter,
                           "name": "world.horizon"})
        return groups

    @property
    def horizon_param(self):
        """The per-horizon embedding inside SANA, or None if it is not installed."""
        from .horizon import is_installed
        return (self.sana.time_embed.emb.horizon_emb
                if is_installed(self.sana) else None)

    def freeze(self) -> None:
        """Freeze SANA. The adapter stays trainable, since it is the only path from the
        memory to SANA."""
        self.sana.requires_grad_(False)
        self.sana.eval()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def scale_report(self, image_tokens: torch.Tensor,
                     final_memory: torch.Tensor) -> dict[str, float]:
        """See `WorldAdapter.scale_report`."""
        return self.adapter.scale_report(image_tokens, final_memory)
