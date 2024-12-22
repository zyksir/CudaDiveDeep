# Copyright (c) 2024 Aliyun Inc. All Rights Reserved.

r"""Random related classes."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def use_deterministic_algorithms(seed: int = 0) -> None:
    """Use deterministic algorithms.

    Args:
        seed: Fixed seed for random operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # See https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # See https://pytorch.org/docs/stable/notes/randomness.html
    torch.use_deterministic_algorithms(True, warn_only=True)