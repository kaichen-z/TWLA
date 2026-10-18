"""The SmolVLA baseline: LeRobot's policy fine-tuned on the hand datasets.

    smolvlm_with_expert.py   the VLM and the action expert (from LeRobot)
    flow.py                  flow matching and Euler sampling (from LeRobot)
    vla_utils.py             helper functions (from LeRobot)
    config.py, model.py      settings; batch adaptation, normalisation, loss, weight loading
"""

from .config import Config, FlatConfig, apply_overrides, from_json
from .model import Model, Observation, SmolVLAOfficialModel


def build_tokenizer(cfg):
    """The VLM's tokenizer. SmolVLA embeds the images separately and concatenates them with
    the language tokens, so the text carries no image placeholders."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.vlm.model_id)


__all__ = ["Config", "FlatConfig", "Model", "Observation", "SmolVLAOfficialModel",
           "apply_overrides", "build_tokenizer", "from_json"]
