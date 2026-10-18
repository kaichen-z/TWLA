"""MemoryVLA on the hand datasets: an adapter around the unmodified code in `upstream/`.

The VLM, perceptual compression, memory banks and DiT action head are upstream's. Differences:
  - 192 action channels instead of 7: the DiT weights of that shape are not loaded from CogACT-Large.
  - The memory is fed by walking a window of consecutive control steps. The earlier steps are written
    to the banks without gradient; only the last is supervised or predicted. Timesteps count from
    the window start.
  - The noise-prediction loss is reduced with the action mask over channel groups.
  - Actions use a per-channel mean/std normalisation; upstream's clipping to [-1, 1] and gripper
    binarisation are not applied.
  - A checkpoint holds only the trained parts; the frozen VLM is rebuilt from CogACT-Large.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")
if UPSTREAM not in sys.path:
    sys.path.insert(0, UPSTREAM)

from group_loss import masked_group_loss_sq  # noqa: E402

VLM_PREFIX = "mvla.vlm."          # frozen VLM: not saved, rebuilt from CogACT-Large


def _expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


# --------------------------------------------------------------------------- observation

@dataclass
class Observation:
    """One control step, as `dataloader.to_observations` builds it.

    `state`, `state_mask` and `fov` are filled by the loader but not used: the model takes image
    and language only.
    """

    images: torch.Tensor      # [B, 1, 3, 224, 224] in [-1, 1]
    lang_ids: torch.Tensor    # [B, T] Llama ids of the prompt, right-padded with <PAD>
    lang_mask: torch.Tensor   # [B, T] bool
    state: torch.Tensor
    state_mask: torch.Tensor
    fov: torch.Tensor


# --------------------------------------------------------------------------- tokenizer

def _patch_llama_path(cfg) -> str:
    """Make Prismatic read the Llama-2 config and tokenizer from the local directory instead of
    the hub id `meta-llama/Llama-2-7b-hf`. Returns that directory."""
    from prismatic.models.backbones.llm import llama2
    local = _expand(cfg.vlm.llama_local_dir)
    if not os.path.isfile(os.path.join(local, "config.json")):
        raise FileNotFoundError(f"{local}/config.json -- see config.VLMConfig.llama_local_dir")
    llama2.LLAMA2_MODELS[cfg.vlm.llm_backbone_id]["hf_hub_path"] = local
    return local


class PromptTokenizer:
    """Tokenizer for `make_collate`: upstream's training prompt, the Llama tokenizer and <PAD>.

    Each instruction becomes "In: What action should the robot take to {x}?\\nOut: </s>", tokenised
    with BOS and right-padded with <PAD> (added as in upstream's `llama2.py`). A prompt longer than
    `max_length` raises: the cognitive token is read at the last real token, which must be the EOS.
    """

    def __init__(self, cfg):
        from transformers import AutoTokenizer
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        local = _patch_llama_path(cfg)
        self.tok = AutoTokenizer.from_pretrained(local, model_max_length=cfg.vlm.llm_max_length,
                                                 padding_side="right")
        self.tok.add_special_tokens({"pad_token": "<PAD>"})
        self.builder_cls = PurePromptBuilder
        self.eos_id = self.tok.eos_token_id

    def prompt(self, instruction: str) -> str:
        b = self.builder_cls("openvla")
        b.add_turn("human", f"What action should the robot take to {str(instruction).lower()}?")
        b.add_turn("gpt", "")
        return b.get_prompt()

    def __call__(self, texts, padding="max_length", truncation=True, max_length=64,
                 return_tensors="pt"):
        ids = [self.tok(self.prompt(t), add_special_tokens=True).input_ids for t in texts]
        longest = max(len(x) for x in ids)
        if longest > max_length:
            raise ValueError(f"prompt of {longest} tokens > vlm.max_language_tokens "
                             f"{max_length}; truncating would drop the EOS the cognitive "
                             f"token is read at. Raise the limit.")
        out = torch.full((len(ids), max_length), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(ids), max_length), dtype=torch.long)
        for i, x in enumerate(ids):
            if x[-1] != self.eos_id:
                raise AssertionError(f"prompt does not end with EOS: {x[-4:]}")
            out[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        return {"input_ids": out, "attention_mask": mask}


def build_tokenizer(cfg) -> PromptTokenizer:
    return PromptTokenizer(cfg)


# --------------------------------------------------------------------------- model

class MemoryVLAModel(nn.Module):
    """Upstream `MemoryVLA` with window-fed memory, a masked loss and mean/std normalisation."""

    def __init__(self, cfg, *, load_weights: bool = True, vlm_dtype=torch.bfloat16):
        super().__init__()
        from prismatic.models.materialize import (get_llm_backbone_and_tokenizer,
                                                  get_vision_backbone_and_transform)
        from prismatic.models.vlms.prismatic import PrismaticVLM
        from memory_vla import MemoryVLA

        self.cfg = cfg
        v, m, d = cfg.vlm, cfg.memory, cfg.diffusion
        _patch_llama_path(cfg)

        vision, self.image_transform = get_vision_backbone_and_transform(
            v.vision_backbone_id, v.image_resize_strategy)
        # inference_mode=True builds the LLM from its config, without weights; they come from CogACT-Large.
        llm, _ = get_llm_backbone_and_tokenizer(v.llm_backbone_id,
                                                llm_max_length=v.llm_max_length,
                                                inference_mode=True)
        vlm = PrismaticVLM(v.model_id, vision, llm, enable_mixed_precision_training=True,
                           arch_specifier=v.arch_specifier)

        state = None
        if load_weights:
            state = torch.load(_expand(v.pretrained_checkpoint), map_location="cpu")["model"]
            vlm.projector.load_state_dict(state["projector"])
            vlm.llm_backbone.load_state_dict(state["llm_backbone"])
            if "vision_backbone" in state:
                vlm.vision_backbone.load_state_dict(state["vision_backbone"])
        vlm.to(vlm_dtype)
        if v.freeze_vlm:
            vlm.requires_grad_(False)
            vlm.eval()

        self.mvla = MemoryVLA(
            vlm,
            action_model_type=d.action_model_type,
            token_size=vlm.llm_backbone.llm.lm_head.in_features,
            action_dim=cfg.action.dim,
            future_action_window_size=cfg.action.chunk_size - 1,
            use_ema=False,
            norm_stats=None,
            # `CogMemBank.__init__` asserts 'stream' or 'group' (`memory_vla.py:171`); replaced below.
            dataloader_type="stream",
            per_token_size=m.per_token_size,
            mem_length=m.mem_length,
            retrieval_layers=m.retrieval_layers,
            use_timestep_pe=m.use_timestep_pe,
            fusion_type=m.fusion_type,
            consolidate_type=m.consolidate_type,
            update_fused=m.update_fused,
        )
        # `dataloader_type` only selects how `process_batch` clears the banks (`memory_vla.py:267-288`).
        # A third value turns that off; `_burn_in_and_query` resets the banks itself.
        for obj in (self.mvla, self.mvla.cog_mem_bank, self.mvla.per_mem_bank):
            obj.dataloader_type = m.dataloader_type
        self.load_report = {"dropped": [], "missing": [], "unexpected": []}
        if state is not None and "action_model" in state:
            self.load_report = self._load_action_model(state["action_model"])
        del state

        # Normalisation statistics, saved with the checkpoint.
        A = cfg.action.dim
        self.register_buffer("action_mean", torch.zeros(A))
        self.register_buffer("action_std", torch.ones(A))
        self.register_buffer("stats_loaded", torch.tensor(False))

    # ------------------------------------------------------------------ weights

    def _load_action_model(self, sd: dict) -> dict:
        """Load CogACT's DiT-L weights into the 192-channel DiT and return a report.

        Expected differences, anything else raises: the action input and output weights are dropped
        for their shape (7 channels in CogACT); `per_token_embedder`, `blocks.*.per_attn` and
        `blocks.*.norm3` do not exist in CogACT and keep their initialisation; CogACT's
        `history_embedder` is ignored.
        """
        import re
        own = self.mvla.action_model.state_dict()
        dropped = sorted(k for k, t in sd.items() if k in own and own[k].shape != t.shape)
        keep = {k: t for k, t in sd.items() if k not in dropped}
        res = self.mvla.action_model.load_state_dict(keep, strict=False)
        io_ok = {"net.x_embedder.linear.weight", "net.x_embedder.linear.bias",
                 "net.final_layer.linear.weight", "net.final_layer.linear.bias"}
        new_ok = re.compile(r"^net\.(per_token_embedder\.|blocks\.\d+\.(per_attn|norm3)\.)")
        old_ok = re.compile(r"^net\.history_embedder\.")
        missing = [k for k in res.missing_keys if k not in dropped]
        bad = ([k for k in dropped if k not in io_ok]
               + [k for k in missing if not new_ok.match(k)]
               + [k for k in res.unexpected_keys if not old_ok.match(k)])
        if bad:
            raise RuntimeError(f"CogACT DiT load: unexpected differences {bad}")
        n_new = sum(own[k].numel() for k in missing)
        return {"dropped": dropped, "new_in_memoryvla": len(missing),
                "new_params_M": round(n_new / 1e6, 2),
                "ignored_from_cogact": list(res.unexpected_keys)}

    def trainable_params_setup(self) -> None:
        """Everything outside the VLM trains; the VLM follows `cfg.vlm.freeze_vlm`."""
        self.mvla.vlm.requires_grad_(not self.cfg.vlm.freeze_vlm)
        for mod in (self.mvla.per_compr, self.mvla.cog_mem_bank, self.mvla.per_mem_bank,
                    self.mvla.action_model):
            mod.requires_grad_(True)

    def param_groups(self) -> list[dict]:
        o = self.cfg.optim
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params, "lr": o.lr, "weight_decay": o.weight_decay,
                 "name": "memory+action", "is_backbone": False}]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.vlm.freeze_vlm:
            self.mvla.vlm.eval()          # the frozen VLM stays in eval mode
        return self

    def state_dict(self, *args, **kwargs):
        """The state dict without the VLM weights."""
        sd = super().state_dict(*args, **kwargs)
        pre = kwargs.get("prefix", "")
        return type(sd)((k, t) for k, t in sd.items() if not k.startswith(pre + VLM_PREFIX))

    def load_state_dict(self, sd, strict: bool = True):
        """Load a checkpoint of this baseline. It has no VLM keys; with `strict`, every other key
        must match."""
        res = super().load_state_dict(sd, strict=False)
        missing = [k for k in res.missing_keys if not k.startswith(VLM_PREFIX)]
        if strict and (missing or res.unexpected_keys):
            raise RuntimeError(f"missing {missing}, unexpected {res.unexpected_keys}")
        return res

    def load_pretrained(self, path: str) -> dict:
        ck = torch.load(_expand(path), map_location="cpu", weights_only=False)
        res = self.load_state_dict(ck.get("model", ck), strict=not self.cfg.init.allow_partial_load)
        return {"missing": [k for k in res.missing_keys if not k.startswith(VLM_PREFIX)],
                "unexpected": list(res.unexpected_keys)}

    # ------------------------------------------------------------------ normalisation

    def set_normalisation(self, *, action_mean, action_std, state_mean=None, state_std=None,
                          eps: float = 1e-6) -> None:
        """Install the per-channel action mean and std ([192] each); the state statistics are ignored."""
        for name, val in (("action_mean", action_mean), ("action_std", action_std)):
            t = torch.as_tensor(val, dtype=torch.float32).reshape(-1)
            if t.numel() != self.cfg.action.dim:
                raise ValueError(f"{name} must have {self.cfg.action.dim} entries")
            if not bool(torch.isfinite(t).all()):
                raise ValueError(f"{name} has non-finite entries")
            if name == "action_std":
                if bool((t < 0).any()):
                    raise ValueError("action_std has negative entries")
                t = t.clamp_min(eps)
            getattr(self, name).copy_(t.to(getattr(self, name).device))
        self.stats_loaded.fill_(True)

    def _require_stats(self) -> None:
        if not bool(self.stats_loaded):
            raise RuntimeError("normalisation statistics were never installed")

    def normalise_action(self, a):
        return (a - self.action_mean) / self.action_std

    def unnormalise_action(self, a):
        return a * self.action_std + self.action_mean

    # ------------------------------------------------------------------ pieces

    @property
    def _vlm(self):
        return self.mvla.vlm

    def _pixels(self, images: torch.Tensor) -> dict:
        """[b, 1, 3, 224, 224] in [-1, 1] -> the batched output of upstream's `image_transform`, which
        takes uint8 PIL images and applies the DINOv2 and SigLIP normalisations."""
        from PIL import Image
        u8 = ((images[:, 0].float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        u8 = u8.permute(0, 2, 3, 1).cpu().numpy()
        outs = [self.image_transform(Image.fromarray(x)) for x in u8]
        dev = images.device
        dtype = next(self._vlm.parameters()).dtype
        return {k: torch.stack([o[k] for o in outs]).to(dev, dtype) for k in outs[0]}

    def _tokens(self, obs: Observation, idx: torch.Tensor, grad_per: bool):
        """Rows `idx` of one step -> (cognitive token [b, 1, 4096], perceptual tokens [b, Np, 256]).

        Follows upstream's `MemoryVLA.forward` up to the memory banks (`memory_vla.py:500-534`).
        """
        ids = obs.lang_ids[idx]
        am = obs.lang_mask[idx].long()
        px = self._pixels(obs.images[idx])
        with torch.no_grad():
            # `inference_mode=True` enabled the LLM's key-value cache; it is not needed here.
            out = self._vlm(input_ids=ids, attention_mask=am, pixel_values=px,
                            output_hidden_states=True, return_dict=True, use_cache=False)
            vb = self._vlm.vision_backbone
            if vb.featurizer is not None:
                num_patch = vb.featurizer.patch_embed.num_patches
            else:
                num_patch = vb.siglip_featurizer.patch_embed.num_patches
            last = out.hidden_states[-1][:, num_patch:]
            cs = am.cumsum(dim=1)
            last_true = (cs == cs.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
            gather = last_true.unsqueeze(-1).expand(-1, last.size(-1))
            cog = last.gather(1, gather.unsqueeze(1))                       # [b, 1, D]
            feats = self._vlm.vision_feats
        # The VLM (bfloat16 by default) and the trainable modules may differ in dtype: cast at the hand-off.
        mem_dt = next(self.mvla.cog_mem_bank.parameters()).dtype
        per_dt = next(self.mvla.per_compr.parameters()).dtype
        # `grad_per` can only switch gradients off; an outer `torch.no_grad()` is left in force.
        with (contextlib.nullcontext() if grad_per else torch.no_grad()):
            per = self.mvla.per_compr(feats.to(per_dt))
        return cog.to(mem_dt), per

    def _reset_memory(self) -> None:
        self.mvla.cog_mem_bank.reset()
        self.mvla.per_mem_bank.reset()

    def _burn_in_and_query(self, steps, step_valid, *, grad: bool):
        """Walk the window -> the fused tokens at its last step (cog [B, 1, 4096], per [B, Np, 256]).

        Steps 0..L-2 write each valid row into its own bank entry (episode id = row index,
        timestep = position among the row's valid steps). Step L-1 is valid for every row, because
        the collate pads on the left, and reads the banks.
        """
        B, L = step_valid.shape
        if not bool(step_valid[:, -1].all()):
            raise AssertionError("step L-1 must be valid for every row (the collate left-pads)")
        first = (~step_valid).sum(dim=1)                  # index of the first valid step per row
        self._reset_memory()
        eids = list(range(B))
        with torch.no_grad():
            for t in range(L - 1):
                rows = step_valid[:, t].nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                cog, per = self._tokens(steps[t], rows, grad_per=False)
                r = rows.tolist()
                ts = [int(t - first[i]) for i in r]
                self.mvla.cog_mem_bank.process_batch(cog, [eids[i] for i in r], ts)
                self.mvla.per_mem_bank.process_batch(per, [eids[i] for i in r], ts)
        allr = torch.arange(B, device=step_valid.device)
        cog, per = self._tokens(steps[L - 1], allr, grad_per=grad)
        ts = [int(L - 1 - first[i]) for i in range(B)]
        with (contextlib.nullcontext() if grad else torch.no_grad()):
            cog = self.mvla.cog_mem_bank.process_batch(cog, eids, ts)
            per = self.mvla.per_mem_bank.process_batch(per, eids, ts)
        self._reset_memory()
        return cog, per

    # ------------------------------------------------------------------ training

    def forward(self, steps, step_valid, actions, action_mask, generator=None):
        return self.loss(steps, step_valid, actions, action_mask, generator=generator)

    def loss(self, steps, step_valid, actions, action_mask, generator=None):
        """-> (scalar, {group: value}): the masked noise-prediction loss at the last step of the window."""
        self._require_stats()
        cog, per = self._burn_in_and_query(steps, step_valid, grad=torch.is_grad_enabled())
        return self._eps_loss(cog, per, actions, action_mask, generator)

    def _eps_loss(self, cog, per, actions, action_mask, generator=None):
        am = self.mvla.action_model
        net_dtype = next(am.net.parameters()).dtype
        a = self.normalise_action(actions.float())                       # [B, 16, 192]
        r = self.cfg.diffusion.repeated_diffusion_steps
        a_r = a.repeat(r, 1, 1).to(net_dtype)
        m_r = action_mask.repeat(r, 1, 1)
        cog_r = cog.repeat(r, 1, 1).to(net_dtype)
        per_r = per.repeat(r, 1, 1).to(net_dtype)
        # As upstream's `ActionModel.loss` (`action_model.py:57-74`) up to the squared error; the
        # reduction is the masked group loss.
        if generator is None:
            noise = torch.randn_like(a_r)
            tstep = torch.randint(0, am.diffusion.num_timesteps, (a_r.size(0),), device=a.device)
        else:
            noise = torch.randn(a_r.shape, device=a.device, dtype=a_r.dtype, generator=generator)
            tstep = torch.randint(0, am.diffusion.num_timesteps, (a_r.size(0),),
                                  device=a.device, generator=generator)
        x_t = am.diffusion.q_sample(a_r, tstep, noise)
        eps = am.net(x_t, tstep, cog_r, per_token=per_r)
        if eps.shape != noise.shape:
            raise AssertionError(f"eps {tuple(eps.shape)} != noise {tuple(noise.shape)}")
        square = (eps.float() - noise.float()) ** 2
        return masked_group_loss_sq(square, m_r, self.cfg.loss.groups)

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def predict_from_noise(self, steps, noise, step_valid=None):
        """-> [B, 16, 192] normalised actions, sampled from the given noise [B, 16, 192].

        The sampler is upstream's (`memory_vla.py:787-832`): classifier-free guidance with the DiT's
        learned unconditional token, then DDIM with `eta=0` and `clip_denoised=False`.
        """
        self._require_stats()
        B = noise.shape[0]
        if step_valid is None:
            step_valid = torch.ones(B, len(steps), dtype=torch.bool, device=noise.device)
        cog, per = self._burn_in_and_query(steps, step_valid, grad=False)
        return self._sample(cog, per, noise)

    @torch.no_grad()
    def _sample(self, cog, per, noise):
        B = noise.shape[0]
        am = self.mvla.action_model
        d = self.cfg.diffusion
        dt = next(am.net.parameters()).dtype
        cog, per, x = cog.to(dt), per.to(dt), noise.to(dt)
        if d.cfg_scale > 1.0:
            x = torch.cat([x, x], 0)
            unc = am.net.z_embedder.uncondition.unsqueeze(0).expand(B, *am.net.z_embedder.uncondition.shape)
            kw = dict(z=torch.cat([cog, unc.to(dt)], 0), cfg_scale=d.cfg_scale,
                      per_token=per.repeat(2, 1, 1))
            fn = am.net.forward_with_cfg
        else:
            kw = dict(z=cog, per_token=per)
            fn = am.net.forward
        if d.use_ddim:
            if am.ddim_diffusion is None:
                am.create_ddim(ddim_step=d.num_ddim_steps)
            out = am.ddim_diffusion.ddim_sample_loop(fn, x.shape, x, clip_denoised=False,
                                                     model_kwargs=kw, progress=False,
                                                     device=x.device, eta=0.0)
        else:
            out = am.diffusion.p_sample_loop(fn, x.shape, x, clip_denoised=False,
                                             model_kwargs=kw, progress=False, device=x.device)
        if d.cfg_scale > 1.0:
            out, _ = out.chunk(2, dim=0)
        return out.float()

    @torch.no_grad()
    def val_step(self, steps, step_valid, actions, action_mask, noise, generator=None):
        """Loss and sample from one walk of the window.
        -> ((scalar, parts), prediction [B, 16, 192] normalised)."""
        self._require_stats()
        cog, per = self._burn_in_and_query(steps, step_valid, grad=False)
        return (self._eps_loss(cog, per, actions, action_mask, generator),
                self._sample(cog, per, noise))


Model = MemoryVLAModel
