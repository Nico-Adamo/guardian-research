"""Deterministic seeding.

Reproducibility is a first-class requirement: every run records its seed, and
``seed_everything`` makes a best effort at bit-for-bit determinism on CPU. Full
determinism on GPU also requires deterministic algorithms, which we enable but
do not hard-fail on (some ops have no deterministic kernel).
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed python, numpy and torch. Returns the seed for convenience/logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Best effort; some kernels lack deterministic implementations.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            # Required for deterministic CUDA matmuls.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        # torch optional at import time; seeding still applies to numpy/random.
        pass

    return seed
