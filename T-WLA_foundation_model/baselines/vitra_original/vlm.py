"""PaliGemma2-3B wrapper: images + instruction + FoV -> one cognition vector. VITRA's VLM
path re-expressed for the transformers version used here.

    prefix   [ <image> x256 | <bos> | instruction | \\n | fov | cognition ]  <- pad right
    output   the hidden state at the last valid position, the cognition token
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config


def resolve_model_dir(model_id: str) -> str:
    """The local snapshot directory of a Hub repo id when it is cached, else the id
    unchanged. `from_pretrained` on a directory needs no network access."""
    import os

    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return model_id


class MLPProjector(nn.Module):
    """VITRA's `MLPProjector`: Linear -> GELU -> Linear. As upstream, the biases are
    initialised from N(0, 0.02) like the weights."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )
        for layer in (self.projector[0], self.projector[2]):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.bias, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


def compact_left(embeds: torch.Tensor, mask: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Move every valid token to the front, order preserved; padding to the back.

    The FoV and cognition embeddings are appended after a right-padded instruction.
    After this step the cognition token is the last valid position of every row.
    """
    # stable sort on `~mask`: valid tokens first, in their original order
    order = torch.argsort((~mask).to(torch.uint8), dim=1, stable=True)
    gathered = embeds.gather(1, order.unsqueeze(-1).expand(-1, -1, embeds.shape[-1]))
    return gathered, mask.gather(1, order)


class PaliGemmaVLM(nn.Module):
    """Wraps `PaliGemmaForConditionalGeneration` and returns the cognition vector."""

    def __init__(self, cfg: Config, *, load_weights: bool = True):
        super().__init__()
        from transformers import (AutoConfig, PaliGemmaForConditionalGeneration)

        self.cfg = cfg
        v = cfg.vlm
        local = resolve_model_dir(v.model_id)
        if load_weights:
            self.pg = PaliGemmaForConditionalGeneration.from_pretrained(
                local, dtype=torch.float32)
        else:
            # Architecture only; a checkpoint supplies every weight. The text and vision
            # sub-configs carry a bf16 dtype of their own, hence the cast to fp32.
            pg_cfg = AutoConfig.from_pretrained(local)
            self.pg = PaliGemmaForConditionalGeneration(pg_cfg).to(torch.float32)

        text_cfg = self.pg.config.get_text_config()
        if text_cfg.hidden_size != v.hidden_size:
            raise ValueError(
                f"config.vlm.hidden_size is {v.hidden_size} but {v.model_id} has "
                f"{text_cfg.hidden_size}. This number is also the DiT's `token_size`; a "
                f"mismatch would only surface as a shape error deep in the action head.")

        self.image_token_id = self.pg.config.image_token_id

        if v.use_fov:
            self.fov_encoder = MLPProjector(2, v.hidden_size)

        # The cognition token is a parameter of its own, initialised from a row of the
        # word embedding.
        if not v.untied_cognition_token:
            raise NotImplementedError(
                "untied_cognition_token=False shares the word-embedding row and needs the "
                "embedding matrix trainable; VITRA's configs all set True.")
        init_id = v.cognition_token_init_id
        init_id = v.cognition_token_id if init_id is None else init_id
        with torch.no_grad():
            row = self.pg.get_input_embeddings().weight[init_id].detach().clone()
        self.cognition_token = nn.Parameter(row)

    # ------------------------------------------------------------------ freezing

    def trainable_params_setup(self) -> None:
        """Apply `freeze_option`. The FoV encoder and the cognition token always train."""
        v = self.cfg.vlm
        if v.freeze_option == "full_finetune":
            self.pg.requires_grad_(True)
        elif v.freeze_option == "freeze_vision_encoder":
            self.pg.requires_grad_(True)
            self.pg.model.vision_tower.requires_grad_(False)
        elif v.freeze_option == "only_head_and_token":
            self.pg.requires_grad_(False)
        else:
            raise ValueError(f"unknown freeze_option {v.freeze_option!r}")

        if v.use_fov:
            self.fov_encoder.requires_grad_(True)
        self.cognition_token.requires_grad_(True)

    # ------------------------------------------------------------------ forward

    def _autocast(self):
        """bf16 autocast, used for the vision tower and the language model only. The
        action head and the loss run in fp32, as in VITRA."""
        return torch.autocast("cuda", dtype=torch.bfloat16,
                              enabled=self.cfg.vlm.use_bf16 and torch.cuda.is_available())

    def _image_features(self, images: torch.Tensor) -> torch.Tensor:
        """`images` [B, N, 3, H, W] in [-1, 1] -> [B*N, image_tokens, hidden]."""
        b, n = images.shape[:2]
        with self._autocast():
            out = self.pg.model.get_image_features(images.flatten(0, 1))
        feats = out[0] if isinstance(out, tuple) else out.pooler_output
        expect = (b * n, self.cfg.vlm.image_tokens, self.cfg.vlm.hidden_size)
        if tuple(feats.shape) != expect:
            raise ValueError(f"image features {tuple(feats.shape)} != expected {expect}")
        return feats

    def forward(self, images: torch.Tensor, lang_ids: torch.Tensor,
                lang_mask: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
        """Returns the cognition vector, [B, hidden].

        `images`   [B, N, 3, H, W]  already in [-1, 1]
        `lang_ids` [B, L]           image placeholders + <bos> + text + \\n, right-padded
        `lang_mask`[B, L] bool
        `fov`      [B, N, 2]
        """
        b = lang_ids.shape[0]
        dev = lang_ids.device
        emb = self.pg.get_input_embeddings()

        # Image placeholders are looked up as id 0; their embeddings are replaced below.
        is_image = lang_ids == self.image_token_id
        safe_ids = lang_ids.masked_fill(is_image, 0)
        embeds = emb(safe_ids)

        feats = self._image_features(images).to(embeds.dtype)
        n_slots = int(is_image.sum())
        if n_slots * embeds.shape[-1] != feats.numel():
            raise ValueError(
                f"{n_slots} <image> placeholders in the prompt but {feats.numel() // embeds.shape[-1]} "
                f"image feature rows. The tokenizer shim and vlm.image_tokens disagree.")
        embeds = embeds.masked_scatter(is_image.unsqueeze(-1), feats)

        extra, extra_mask = [], []
        if self.cfg.vlm.use_fov:
            # one FoV token, from the first camera
            extra.append(self.fov_encoder(fov[:, 0]).unsqueeze(1))
            extra_mask.append(torch.ones(b, 1, dtype=torch.bool, device=dev))
        extra.append(self.cognition_token.to(embeds.dtype).expand(b, 1, -1))
        extra_mask.append(torch.ones(b, 1, dtype=torch.bool, device=dev))

        embeds = torch.cat([embeds, *extra], dim=1)
        mask = torch.cat([lang_mask, *extra_mask], dim=1)
        embeds, mask = compact_left(embeds, mask)

        # No `token_type_ids` are passed, so attention is causal over the whole sequence
        # instead of bidirectional over the prefix. `attention_mask` is the padding mask.
        with self._autocast():
            out = self.pg.model(
                inputs_embeds=embeds,
                attention_mask=mask.long(),
                use_cache=False,
            )
        hidden = out.last_hidden_state
        # back to fp32 for the action head
        return self.extract_cognition(hidden, mask).float()

    @staticmethod
    def extract_cognition(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The hidden state at the last valid position, [B, D]. `mask` must be compact
        (see `compact_left`)."""
        last = mask.to(torch.int32).cumsum(1).argmax(1)          # index of the last True
        idx = last[:, None, None].expand(-1, 1, hidden.shape[-1])
        return hidden.gather(1, idx).squeeze(1)
