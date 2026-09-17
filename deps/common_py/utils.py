"""Reproducibility helpers."""

import random

import numpy as np
import torch


def setupSeed(seed):
    """Seed PyTorch, NumPy and Python's random number generator."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
