"""The stable bridge from two EgoTouch glove streams to two tactile tokens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class TouchFeatures:
    tokens: torch.Tensor  # [B, 2, 64], left then right
    valid: torch.Tensor  # [B, 2]


def _selected_encoder_api():
    try:
        from selected_encoder import (  # type: ignore
            DEFAULT_MODEL_ID,
            DEFAULT_SEED,
            SelectedTactileEncoder,
            canonicalize_raw_touch,
        )
    except ImportError as exc:  # pragma: no cover - exercised by packaging smoke
        raise ImportError(
            "Selected EgoTouch encoder is unavailable. Install the tactile_editor "
            "package, which includes selected_encoder and egotouch_formal."
        ) from exc
    return (
        DEFAULT_MODEL_ID,
        DEFAULT_SEED,
        SelectedTactileEncoder,
        canonicalize_raw_touch,
    )


def load_huggingface_encoder(
    *,
    model_id: str = "MIT-Media-Lab/egotouch-tactile-encoder-p1-v1",
    seed: int = 20260831,
    cache_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    history_frames: int = 8,
) -> nn.Module:
    """Download only the standalone tactile weights and masks, then load strictly."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional network path
        raise ImportError(
            "huggingface_hub is required for remote encoder loading"
        ) from exc
    _, _, selected_cls, _ = _selected_encoder_api()
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    root = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=[
            "pressure_masks.npy",
            f"seed-{seed}/tactile_encoder.safetensors",
        ],
    )
    return selected_cls.from_pretrained_dir(
        root, seed=seed, device=device, history_frames=history_frames
    )


def load_formal_t2p_encoder(
    checkpoint: str | Path,
    *,
    history_frames: int,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load only the tactile tower from a formal P1 T2P checkpoint.

    The pose tower and contrastive temperature are training-only components.
    The returned module has the same normalized 64-D runtime contract as the
    published selected encoder.
    """

    _, _, selected_cls, _ = _selected_encoder_api()
    from egotouch_formal.model import TactileEncoder

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    if config.get("task") != "p1":
        raise ValueError(f"formal checkpoint must be P1, got {config.get('task')!r}")
    trained_touch_history = int(config.get("tactile_history_frames", 8))
    if trained_touch_history != int(history_frames):
        raise ValueError(
            f"checkpoint history is {trained_touch_history}, requested {history_frames}"
        )
    source = payload.get("model")
    if not isinstance(source, dict):
        raise ValueError("formal checkpoint has no model state")
    tactile_state = {
        key.removeprefix("tactile."): value
        for key, value in source.items()
        if key.startswith("tactile.")
    }
    encoder = TactileEncoder(embed_dim=int(config.get("embed_dim", 64)))
    encoder.load_state_dict(tactile_state, strict=True)
    pressure_masks = source.get("pressure_masks")
    if pressure_masks is None:
        raise ValueError("formal checkpoint does not contain pressure_masks")
    return selected_cls(
        encoder,
        pressure_masks,
        history_frames=history_frames,
    ).to(device)


def initialize_random_t2p_encoder(
    pressure_masks: torch.Tensor,
    *,
    history_frames: int,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Construct the P1 tactile architecture without pretrained weights."""

    _, _, selected_cls, _ = _selected_encoder_api()
    from egotouch_formal.model import TactileEncoder

    return selected_cls(
        TactileEncoder(embed_dim=64),
        pressure_masks,
        history_frames=history_frames,
    ).to(device)


class DualHandTactileEncoder(nn.Module):
    """Run one shared encoder over left/right causal histories.

    Raw input keeps the online cache contract ``[B,8,2,21,21]``. The wrapped
    checkpoint sees the latest ``history_frames`` as ``[B*2,T,21,21]`` plus a
    side id. Left pressure is flipped exactly once when ``canonicalized=False``.
    """

    def __init__(self, encoder: nn.Module, *, freeze: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        self.history_frames = int(getattr(encoder, "history_frames", 8))
        if not 1 <= self.history_frames <= 8:
            raise ValueError("editor bridge supports history_frames in [1,8]")
        self.set_frozen(freeze)

    def set_frozen(self, frozen: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not frozen
        if frozen:
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def forward(
        self,
        pressure: torch.Tensor,
        temporal_valid: torch.Tensor | None = None,
        *,
        canonicalized: bool = False,
    ) -> TouchFeatures:
        expected = (8, 2, 21, 21)
        if pressure.ndim != 5 or tuple(pressure.shape[1:]) != expected:
            raise ValueError(
                f"pressure must be [B,8,2,21,21], got {tuple(pressure.shape)}"
            )
        b = pressure.shape[0]
        if temporal_valid is None:
            temporal_valid = torch.ones(
                b, 8, 2, dtype=torch.bool, device=pressure.device
            )
        if tuple(temporal_valid.shape) != (b, 8, 2):
            raise ValueError(
                f"temporal_valid must be {(b, 8, 2)}, got {tuple(temporal_valid.shape)}"
            )

        # Keep the most recent causal frames. [B,T,H,W] for each hand becomes
        # [B*2,T,H,W], ordered L0,R0,L1,R1,...
        pressure = pressure[:, -self.history_frames :]
        temporal_valid = temporal_valid[:, -self.history_frames :]
        flat = pressure.permute(0, 2, 1, 3, 4).reshape(
            b * 2, self.history_frames, 21, 21
        )
        valid_flat = temporal_valid.permute(0, 2, 1).reshape(
            b * 2, self.history_frames
        )
        flat = flat * valid_flat[:, :, None, None].to(flat.dtype)
        side = torch.arange(2, device=pressure.device).repeat(b)
        if not canonicalized:
            _, _, _, canonicalize = _selected_encoder_api()
            flat = canonicalize(flat, side)
        pressure_masks = getattr(self.encoder, "pressure_masks", None)
        if pressure_masks is not None:
            spatial_valid = pressure_masks[side.to(pressure_masks.device)]
            flat = flat * spatial_valid[:, None].to(
                device=flat.device, dtype=flat.dtype
            )

        tokens = self.encoder(flat, side).reshape(b, 2, -1)
        hand_valid = valid_flat.any(dim=1).reshape(b, 2)
        tokens = tokens * hand_valid.unsqueeze(-1).to(tokens.dtype)
        return TouchFeatures(tokens=tokens, valid=hand_valid)
