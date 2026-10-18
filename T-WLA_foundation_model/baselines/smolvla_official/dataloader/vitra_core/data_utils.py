"""Vendored from VITRA (`vitra/utils/data_utils.py`), pruned to the statistics reader and the
Gaussian normaliser."""
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import PIL.Image as Image
from scipy.spatial.transform import Rotation as R
import json
import os
from huggingface_hub import hf_hub_download

def read_dataset_statistics(statistics_path: str, default_value=1e-4) -> dict:
    """Read the per-hand mean and std of state and action from a statistics JSON. If the left
    hand is missing, its statistics are filled with `default_value`."""
    with open(statistics_path, 'r') as file:
        dataset_stats = json.load(file)
        
        assert 'state_right' in dataset_stats, "Right hand statistics must exist"
        
        state_right_mean = np.array(dataset_stats['state_right']['mean'])
        state_right_std = np.array(dataset_stats['state_right']['std'])
        action_right_mean = np.array(dataset_stats['action_right']['mean'])
        action_right_std = np.array(dataset_stats['action_right']['std'])
        
        if 'state_left' in dataset_stats:
            state_left_mean = np.array(dataset_stats['state_left']['mean'])
            state_left_std = np.array(dataset_stats['state_left']['std'])
            action_left_mean = np.array(dataset_stats['action_left']['mean'])
            action_left_std = np.array(dataset_stats['action_left']['std'])
        else:
            state_left_mean = np.full_like(state_right_mean, default_value)
            state_left_std = np.full_like(state_right_std, default_value)
            action_left_mean = np.full_like(action_right_mean, default_value)
            action_left_std = np.full_like(action_right_std, default_value)
        
        data_statistics = {
            'state_right_mean': state_right_mean,
            'state_right_std': state_right_std,
            'action_right_mean': action_right_mean,
            'action_right_std': action_right_std,
            'state_left_mean': state_left_mean,
            'state_left_std': state_left_std,
            'action_left_mean': action_left_mean,
            'action_left_std': action_left_std,
        }
    return data_statistics

class GaussianNormalizer:
    """Normalise and denormalise state and action arrays laid out as [left, right]."""
    def __init__(self, data_statistics: dict):
        """`data_statistics` is the dict returned by `read_dataset_statistics`."""
        self.state_mean = np.concatenate([data_statistics['state_left_mean'], data_statistics['state_right_mean']])
        self.state_std = np.concatenate([data_statistics['state_left_std'], data_statistics['state_right_std']])
        self.action_mean = np.concatenate([data_statistics['action_left_mean'], data_statistics['action_right_mean']])
        self.action_std = np.concatenate([data_statistics['action_left_std'], data_statistics['action_right_std']])

    def normalize_state(self, state: np.ndarray, epsilon=1e-7) -> np.ndarray:
        return (state - self.state_mean) / (self.state_std + epsilon)

    def unnormalize_state(self, norm_state: np.ndarray, epsilon=1e-7) -> np.ndarray:
        return norm_state * (self.state_std + epsilon) + self.state_mean

    def normalize_action(self, action: np.ndarray, epsilon=1e-7) -> np.ndarray:
        return (action - self.action_mean) / (self.action_std + epsilon)

    def unnormalize_action(self, norm_action: np.ndarray, epsilon=1e-7) -> np.ndarray:
        return norm_action * (self.action_std + epsilon) + self.action_mean
