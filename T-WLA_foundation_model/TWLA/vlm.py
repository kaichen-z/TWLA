"""The vision-language backbone: one observation -> `m_f`.

    [ image 64N | FoV N | language 32 | state 1 | m_t 8 ]  ->  SmolLM2 x16  ->  m_f (last 8)

`m_f` conditions the action expert and is written to the memory bank. As in SmolVLA, image
and language embeddings are scaled by sqrt(hidden_size); state, FoV and `m_t` are not.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import TWLAConfig
from .masks import additive_mask, build_prefix_att_masks, make_att_2d_masks


class SmolVLM2Backbone(nn.Module):
    """SmolVLM2 with the decoder cut to its first `num_layers` layers, plus the state and
    FoV projections. The prefix length is fixed at construction (`cfg.prefix_len`)."""

    def __init__(self, cfg: TWLAConfig, *, load_weights: bool = True,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import (AutoConfig, AutoModelForImageTextToText,
                                  SmolVLMForConditionalGeneration)

        self.cfg = cfg
        v = cfg.vlm

        if load_weights:
            outer = AutoModelForImageTextToText.from_pretrained(
                v.model_id, dtype=dtype, low_cpu_mem_usage=True)
        else:                              # random weights; for tests and benchmarks
            outer = SmolVLMForConditionalGeneration(AutoConfig.from_pretrained(v.model_id))
            outer = outer.to(dtype)

        # Keep the inner model; the vocabulary head is not used.
        self.vlm = outer.model
        del outer

        text = self.vlm.text_model
        if v.num_layers > len(text.layers):
            raise ValueError(
                f"asked for {v.num_layers} layers, checkpoint has {len(text.layers)}")
        text.layers = text.layers[: v.num_layers]

        if text.config.hidden_size != v.hidden_size:
            raise ValueError(f"checkpoint hidden_size {text.config.hidden_size} != "
                             f"config {v.hidden_size}")

        # State and FoV projections. Their outputs are not scaled by sqrt(d).
        self.state_proj = nn.Linear(2 * cfg.action.state_dim, v.hidden_size)
        self.fov_proj = nn.Linear(2, v.hidden_size)

        # Scale for the image and language embeddings, as in SmolVLA.
        self.embed_scale = math.sqrt(v.hidden_size)

        self.register_buffer("att_marks", build_prefix_att_masks(
            n_image=v.tokens_per_frame * v.num_cameras,
            n_fov=v.num_cameras,
            n_language=v.max_language_tokens,
            n_state=1,
            n_memory=cfg.memory.num_slots,
        ), persistent=False)

        self.to(dtype)
        self.set_requires_grad()

    # ------------------------------------------------------------------ freezing

    def set_requires_grad(self) -> None:
        """Set `requires_grad` from the freeze fields of `VLMConfig`."""
        v = self.cfg.vlm
        text = self.vlm.text_model

        for p in self.vlm.parameters():
            p.requires_grad = False

        if not v.freeze_vision:
            for p in self.vlm.vision_model.parameters():
                p.requires_grad = True
        if not v.freeze_connector:
            for p in self.vlm.connector.parameters():
                p.requires_grad = True
        if not v.freeze_embeddings:
            for p in text.get_input_embeddings().parameters():
                p.requires_grad = True

        # The top k of the kept layers, the ones closest to `m_f`.
        k = v.train_top_k_layers
        if k:
            if k > len(text.layers):
                raise ValueError(f"train_top_k_layers={k} > {len(text.layers)} kept layers")
            for layer in text.layers[len(text.layers) - k:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in text.norm.parameters():         # the final norm produces m_f
                p.requires_grad = True

    def train(self, mode: bool = True):     # noqa: D102
        super().train(mode)
        if self.cfg.vlm.freeze_vision:
            self.vlm.vision_model.eval()
        return self

    # ------------------------------------------------------------------ embedding

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, N, 3, H, W] in [-1, 1] -> [B, N * tokens_per_frame, D]."""
        b, n = pixel_values.shape[:2]
        if n != self.cfg.vlm.num_cameras:
            raise ValueError(f"expected {self.cfg.vlm.num_cameras} cameras, got {n}")
        flat = pixel_values.flatten(0, 1).to(self.dtype)
        h = self.vlm.vision_model(pixel_values=flat).last_hidden_state
        tok = self.vlm.connector(h)                                  # [B*N, 64, D]
        return tok.reshape(b, -1, tok.shape[-1]) * self.embed_scale

    def embed_language(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, L] token ids -> [B, L, D]."""
        return self.vlm.text_model.get_input_embeddings()(ids) * self.embed_scale

    # ------------------------------------------------------------------ prefix

    def build_prefix(self, images: torch.Tensor, lang_ids: torch.Tensor,
                     lang_mask: torch.Tensor, state_with_mask: torch.Tensor,
                     fov: torch.Tensor, m_t: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble the prefix. Returns (embeds, pad_masks, att_marks, image_tokens).

        Memory comes last, so that it attends to the current frame before it is banked.
        `image_tokens` [B, 64*N, D] are the image embeddings, used by the world head.
        """
        b = images.shape[0]
        v = self.cfg.vlm
        dev = images.device

        img = self.embed_image(images)                                        # [B, 64N, D]
        fov_emb = self.fov_proj(fov.to(self.dtype))                           # [B, N, D]
        lang = self.embed_language(lang_ids)                                  # [B, 32, D]
        state = self.state_proj(state_with_mask.to(self.dtype))[:, None]      # [B, 1, D]
        mem = m_t.to(self.dtype)                                              # [B, S, D]

        embs = torch.cat([img, fov_emb, lang, state, mem], dim=1)
        if embs.shape[1] != self.cfg.prefix_len:
            raise ValueError(f"prefix is {embs.shape[1]}, config says {self.cfg.prefix_len}")

        ones = lambda n: torch.ones(b, n, dtype=torch.bool, device=dev)  # noqa: E731
        pad = torch.cat([ones(img.shape[1]), ones(v.num_cameras), lang_mask.to(dev).bool(),
                         ones(1), ones(self.cfg.memory.num_slots)], dim=1)

        return embs, pad, self.att_marks[None].expand(b, -1), img

    # ------------------------------------------------------------------ forward

    def forward(self, images: torch.Tensor, lang_ids: torch.Tensor,
                lang_mask: torch.Tensor, state_with_mask: torch.Tensor,
                fov: torch.Tensor, m_t: torch.Tensor, *,
                return_image_tokens: bool = False):
        """One VLM pass. Returns `m_f` [B, num_slots, D], or `(m_f, image_tokens)` with
        `return_image_tokens`."""
        embs, pad, marks, img = self.build_prefix(
            images, lang_ids, lang_mask, state_with_mask, fov, m_t)

        att2d = make_att_2d_masks(pad, marks)
        att4d = additive_mask(att2d, embs.dtype)[:, None]                # [B, 1, N, N]

        # Positions count real tokens only, so padding does not shift the positions of
        # the state and memory tokens (as in SmolVLA).
        pos = torch.cumsum(pad.to(torch.int64), dim=1) - 1

        out = self.vlm.text_model(inputs_embeds=embs, attention_mask=att4d,
                                  position_ids=pos).last_hidden_state
        m_f = out[:, -self.cfg.memory.num_slots:]
        return (m_f, img) if return_image_tokens else m_f

    # ------------------------------------------------------------------ misc

    @property
    def dtype(self) -> torch.dtype:
        return self.state_proj.weight.dtype


def resize_with_pad(img: torch.Tensor, height: int, width: int,
                    pad_value: float = 0.0) -> torch.Tensor:
    """Aspect-preserving resize onto a (height, width) canvas, values in [0, 1].

    Padding goes on the left and top, so the image sits in the bottom-right corner, as in
    SmolVLA. Convert to [-1, 1] afterwards with `img * 2 - 1`.
    """
    if img.ndim != 4:
        raise ValueError(f"expected [B, 3, H, W], got {tuple(img.shape)}")
    _, _, h, w = img.shape
    r = min(width / w, height / h)
    nh, nw = max(1, int(h * r)), max(1, int(w * r))
    small = nn.functional.interpolate(img, size=(nh, nw), mode="bilinear",
                                      align_corners=False)
    out = img.new_full((img.shape[0], img.shape[1], height, width), pad_value)
    out[..., height - nh:, width - nw:] = small
    return out
