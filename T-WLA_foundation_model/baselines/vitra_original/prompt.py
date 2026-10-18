"""Builds PaliGemma's prompt behind the tokenizer call that the loader's `make_collate`
makes.

A row is `<image>` x 256, <bos>, the instruction, "\\n", then padding. The ids are
assembled directly, so truncation shortens only the instruction and the padding always
follows the newline.
"""

from __future__ import annotations

import torch

from .config import Config
from .vlm import resolve_model_dir


class PaliGemmaPromptTokenizer:
    """Callable with the part of the HF tokenizer signature that `make_collate` uses;
    returns `input_ids` and `attention_mask`, both [B, max_length]."""

    def __init__(self, cfg: Config):
        from transformers import AutoProcessor

        self.cfg = cfg
        proc = AutoProcessor.from_pretrained(resolve_model_dir(cfg.vlm.model_id))
        self.tokenizer = proc.tokenizer
        self.image_token_id = proc.image_token_id
        self.image_seq_len = int(proc.image_seq_length)

        if self.image_seq_len != cfg.vlm.image_tokens:
            raise ValueError(
                f"{cfg.vlm.model_id} uses {self.image_seq_len} image tokens but "
                f"config.vlm.image_tokens is {cfg.vlm.image_tokens}. These must agree: "
                f"the first is what the prompt reserves, the second is what the vision "
                f"tower produces, and `masked_scatter` requires them equal.")

        self.bos_id = self.tokenizer.bos_token_id
        if self.bos_id is None:
            raise ValueError(f"{cfg.vlm.model_id}'s tokenizer has no bos_token_id")
        nl = self.tokenizer("\n", add_special_tokens=False)["input_ids"]
        if len(nl) != 1:
            raise ValueError(f"'\\n' tokenizes to {len(nl)} ids, expected 1")
        self.newline_id = nl[0]
        self.pad_id = (self.tokenizer.pad_token_id
                       if self.tokenizer.pad_token_id is not None else 0)

        # image block + <bos> + "\n"; at least one text token has to fit as well
        self.frame = self.image_seq_len + 2
        if cfg.vlm.max_language_tokens <= self.frame:
            raise ValueError(
                f"max_language_tokens {cfg.vlm.max_language_tokens} leaves no room for "
                f"text: {self.image_seq_len} image tokens + <bos> + '\\n' already need "
                f"{self.frame}.")
        self.max_text = cfg.vlm.max_language_tokens - self.frame

    def __call__(self, texts, padding: str = "max_length", truncation: bool = True,
                 max_length: int | None = None, return_tensors: str = "pt"):
        if padding != "max_length" or not truncation:
            raise ValueError("this shim implements padding='max_length' with truncation, "
                             "which is what make_collate asks for")
        total = max_length or self.cfg.vlm.max_language_tokens
        max_text = total - self.frame
        if max_text < 1:
            raise ValueError(f"max_length {total} leaves no room for text")

        if isinstance(texts, str):
            texts = [texts]

        ids = torch.full((len(texts), total), self.pad_id, dtype=torch.long)
        mask = torch.zeros(len(texts), total, dtype=torch.long)
        ids[:, :self.image_seq_len] = self.image_token_id
        ids[:, self.image_seq_len] = self.bos_id

        encoded = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        for i, text_ids in enumerate(encoded):
            text_ids = text_ids[:max_text]
            n = len(text_ids)
            lo = self.image_seq_len + 1
            if n:
                ids[i, lo:lo + n] = torch.tensor(text_ids, dtype=torch.long)
            ids[i, lo + n] = self.newline_id
            mask[i, :lo + n + 1] = 1

        if return_tensors != "pt":
            raise ValueError("only return_tensors='pt' is implemented")
        return {"input_ids": ids, "attention_mask": mask}
