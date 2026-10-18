"""The VITRA baseline: PaliGemma2-3B with a DiT action head.

Exports `Config`, `Model`, `Observation` and the prompt tokenizer.
"""

from .config import Config, from_json
from .model import Observation, VitraOriginalModel
from .prompt import PaliGemmaPromptTokenizer

Model = VitraOriginalModel


def build_tokenizer(cfg):
    """The prompt builder. A plain Gemma tokenizer would not emit the `<image>`
    placeholders that the image features are scattered into."""
    return PaliGemmaPromptTokenizer(cfg)


# Aliases for loaders that look a model up as `V4{KEY}Config` / `V4{KEY}Model`, key "vo".
V4VOConfig = Config
V4VOModel = VitraOriginalModel

__all__ = ["Config", "Model", "Observation", "VitraOriginalModel",
           "V4VOConfig", "V4VOModel",
           "PaliGemmaPromptTokenizer", "build_tokenizer", "from_json"]
