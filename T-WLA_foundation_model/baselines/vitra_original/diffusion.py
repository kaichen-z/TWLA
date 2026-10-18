"""DDPM training and DDIM sampling as VITRA configures them: epsilon prediction, fixed
variance, MSE on the noise, 100 training steps and 10 DDIM steps without clipping.

VITRA's diffusion code re-expressed for the transformers version used here; only the
path its config reaches is kept.
"""

from __future__ import annotations

import math

import torch

from .config import Config


# --------------------------------------------------------------------------- schedule

def betas_for_alpha_bar(num_steps: int, max_beta: float = 0.999) -> torch.Tensor:
    """The `squaredcos_cap_v2` schedule.

    `alpha_bar(u) = cos^2((u + 0.008) / 1.008 * pi/2)` at `i/T` and `(i+1)/T`; each
    beta is `1 - abar(t2)/abar(t1)`, capped at `max_beta`. The cap acts on the last
    step, where the cosine reaches zero.
    """
    def alpha_bar(u: float) -> float:
        return math.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_steps):
        t1, t2 = i / num_steps, (i + 1) / num_steps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float64)


def alpha_bar_table(cfg: Config) -> torch.Tensor:
    """Cumulative product of `1 - beta`, length `num_train_steps`, float32."""
    d = cfg.diffusion
    if d.schedule != "squaredcos_cap_v2":
        raise ValueError(f"schedule {d.schedule!r}: only 'squaredcos_cap_v2' is "
                         f"implemented, which is the one VITRA uses")
    betas = betas_for_alpha_bar(d.num_train_steps)
    return torch.cumprod(1.0 - betas, dim=0).float()


def ddim_grid(cfg: Config) -> list[int]:
    """The timesteps DDIM visits, from high to low: `range(0, T, T // N)` reversed.

    For T=100, N=10 that is 90, 80, ..., 0. The grid starts at `T - T // N`, not at
    `T - 1`.
    """
    T, n = cfg.diffusion.num_train_steps, cfg.diffusion.num_inference_steps
    for stride in range(1, T + 1):
        if len(range(0, T, stride)) == n:
            return list(range(0, T, stride))[::-1]
    raise ValueError(f"cannot space {T} steps into {n} for DDIM")


# --------------------------------------------------------------------------- masks

def component_action_mask(batch: int, chunk: int, dim: int, cfg: Config,
                          device: torch.device) -> torch.Tensor:
    """Action mask for inference, [batch, chunk, dim] bool: True on the channels of the
    loss groups (both hands), False on the padding.

    The mask is part of the DiT's input and the padding is masked in every training
    sample, so it has to be masked here too.
    """
    mask = torch.zeros(batch, chunk, dim, dtype=torch.bool, device=device)
    for _, lo, hi, _ in cfg.diffusion.loss_groups:
        mask[..., lo:hi] = True
    return mask


# --------------------------------------------------------------------------- training

def sample_timesteps(batch: int, cfg: Config, device: torch.device,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniform over the `num_train_steps` training steps."""
    return torch.randint(0, cfg.diffusion.num_train_steps, (batch,), device=device,
                         generator=generator)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
             alpha_bar: torch.Tensor) -> torch.Tensor:
    ab = alpha_bar.to(x0.device)[t].view(-1, *([1] * (x0.dim() - 1)))
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def masked_group_loss(pred: torch.Tensor, target: torch.Tensor,
                      action_mask: torch.Tensor, cfg: Config,
                      ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The masked group loss of `group_loss.py` over `cfg.diffusion.loss_groups`."""
    import sys, os
    _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from group_loss import masked_group_loss as _shared

    return _shared(pred, target, action_mask, cfg.diffusion.loss_groups)


# --------------------------------------------------------------------------- sampling

@torch.no_grad()
def ddim_sample(eps_fn, noise: torch.Tensor, alpha_bar: torch.Tensor,
                grid: list[int], eta: float = 0.0) -> torch.Tensor:
    """DDIM from `noise` back to data, without clipping. `eta = 0` is deterministic.

    `eps_fn(x_t, t) -> eps` is called once per timestep of `grid`, which comes from
    `ddim_grid`. x0 is estimated by dividing by `sqrt(alpha_bar[t])`, so the first
    timestep of the grid sets how much the error in eps is amplified. The last step
    returns the estimated x0.
    """
    ab = alpha_bar.to(noise.device)
    x = noise
    for i, t in enumerate(grid):
        a_t = ab[t].view(1, 1, 1)
        eps = eps_fn(x, torch.full((x.shape[0],), t, device=x.device, dtype=torch.long))
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-8)
        if i + 1 == len(grid):
            return x0
        a_prev = ab[grid[i + 1]].view(1, 1, 1)
        sigma = (eta * ((1 - a_prev) / (1 - a_t)).clamp_min(0).sqrt()
                 * (1 - a_t / a_prev).clamp_min(0).sqrt())
        x = a_prev.sqrt() * x0 + (1 - a_prev - sigma ** 2).clamp_min(0).sqrt() * eps
        if eta > 0:
            x = x + sigma * torch.randn_like(x)
    raise AssertionError("unreachable: the loop returns on its last iteration")
