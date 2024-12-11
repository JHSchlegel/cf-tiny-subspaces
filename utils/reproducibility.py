"""
This module provides utility functions to ensure reproducibility in PyTorch by
setting seed values for random number generators in Python, NumPy, and PyTorch.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import numpy as np
import random
import torch


# =========================================================================== #
#                            Random Seeding Utility                           #
# =========================================================================== #
def set_all_seeds(seed: int = 42) -> None:
    """
    Set seed for random number generators in Python, NumPy, PyTorch (CPU and GPU),
    and ensure deterministic behavior in CuDNN.

    Args:
        seed (int, optional): Seed value. Defaults to 42.
    """
    # random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # Ensure deterministic behavior in PyTorch's CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
