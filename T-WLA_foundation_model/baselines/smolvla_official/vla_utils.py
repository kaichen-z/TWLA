"""Helper functions of SmolVLA: noise and timestep sampling, the timestep embedding,
attention masks, padding.

Adapted from LeRobot (https://github.com/huggingface/lerobot), Apache-2.0
(`policies/common/flow_matching.py`, `policies/common/vla_utils.py`, `utils/device_utils.py`).
"""


from __future__ import annotations


import logging
import math


import torch
import torch.nn.functional as F  # noqa: N812

from torch import Tensor



# ---- flow_matching.py -------------------------------------------------------

def sample_beta(alpha: float, beta: float, bsize: int, device) -> Tensor:  # see openpi (exact copy)
    # sampled on the CPU: Beta sampling is not implemented for MPS
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)

def sample_noise(shape, device) -> Tensor:
    """Standard-normal float32 noise."""
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )

def sample_time_beta(bsize: int, device, *, alpha: float, beta: float, scale: float, offset: float) -> Tensor:
    """Beta-distributed flow-matching timesteps: ``Beta(alpha, beta) * scale + offset`` (openpi convention)."""
    time_beta = sample_beta(alpha, beta, bsize, device)
    time = time_beta * scale + offset
    return time.to(dtype=torch.float32, device=device)


# ---- utils/device_utils.py ---------------------------------------------------

def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """float32 in place of float64 on devices without float64 (mps, some xpu)."""
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    if device == "xpu" and dtype == torch.float64:
        if hasattr(torch.xpu, "get_device_capability"):
            device_capability = torch.xpu.get_device_capability()
            # some Intel XPU devices have no float64
            if not device_capability.get("has_fp64", False):
                logging.warning(f"Device {device} does not support float64, using float32 instead.")
                return torch.float32
        else:
            logging.warning(
                f"Device {device} capability check failed. Assuming no support for float64, using float32 instead."
            )
            return torch.float32
        return dtype
    else:
        return dtype

# ---- vla_utils.py -----------------------------------------------------------

def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Sine-cosine embedding [B, dimension] of scalar positions `time` [B]."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # outer product of the frequencies and the positions
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:  # see openpi (exact copy)
    """2D attention mask [B, N, N] from a padding mask and block markers (from big_vision).

    pad_masks: bool [B, N], true for real tokens. att_masks: [B, N], 1 where a token starts
    a new block that earlier tokens cannot attend to, 0 where it shares the previous
    token's block. A token attends to the real tokens whose cumulative `att_masks` is not
    larger than its own:
      [1 1 1 1 1 1]  causal attention
      [0 0 0 1 1 1]  the first three attend to each other, the last three causally
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks

def pad_vector(vector: Tensor, new_dim: int, *, truncate: bool = False) -> Tensor:
    """Pad the last dimension with zeros to `new_dim`. A vector that is already longer is
    returned unchanged, or cut to `new_dim` when `truncate` is set."""
    if vector.shape[-1] == new_dim:
        return vector
    if not truncate:
        if vector.shape[-1] >= new_dim:
            return vector
        return F.pad(vector, (0, new_dim - vector.shape[-1]))
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = vector.new_zeros(*shape)
    length = min(current_dim, new_dim)
    new_vector[..., :length] = vector[..., :length]
    return new_vector
