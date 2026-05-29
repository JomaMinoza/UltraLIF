# -*- coding: utf-8 -*-
"""
Dataset loading and spike encoding utilities.

Functions:
    get_dataset:   Load train/test DataLoaders for any supported dataset.
    rate_encode:   Convert pixel tensors to Poisson spike trains.
    set_seed:      Set random seeds for reproducibility.
"""

from .loader import get_dataset
from .encoding import rate_encode, SpikeEncoder
from .utils import set_seed, get_device

__all__ = [
    "get_dataset",
    "rate_encode",
    "SpikeEncoder",
    "set_seed",
    "get_device",
]
