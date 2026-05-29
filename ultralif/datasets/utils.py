# -*- coding: utf-8 -*-
"""Dataset and training utilities."""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for full reproducibility.

    Sets seeds for Python random, NumPy, PyTorch CPU, and all CUDA devices.
    Also enables CUDNN deterministic mode.

    Args:
        seed: Integer seed value.

    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Return the best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        torch.device: Selected device.

    Example:
        >>> device = get_device()
        >>> print(device)  # cuda / mps / cpu
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
