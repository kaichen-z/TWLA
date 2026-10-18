"""World head: a flow-matching image predictor that sits beside the action expert.

Conditioned on the current image tokens and the final memory slots, it predicts the DC-AE
latents of the frames the action chunk reaches. It is trained for the gradient it sends
back into memory; the frozen DC-AE that encodes the targets lives outside the model.
"""
from .config import (AdapterConfig, SanaConfig, DCAEConfig, WorldLossConfig,
                     WorldConfig)
from .adapter import WorldAdapter
from .sana import remap_keys, provenance     # these two do not need diffusers
from .loss import world_loss, sample_tau, lam_at
from .head import WorldHead
from .sample import sample_latent, decode, to_uint8, predict_frames
from .horizon import HorizonTimestepEmbedding, horizon_rows

__all__ = ["AdapterConfig", "SanaConfig", "DCAEConfig", "WorldLossConfig", "WorldConfig",
           "WorldAdapter", "WorldHead", "HorizonTimestepEmbedding", "horizon_rows",
           "sample_latent", "decode", "to_uint8", "predict_frames",
           "remap_keys", "provenance",
           "world_loss", "sample_tau", "lam_at"]
