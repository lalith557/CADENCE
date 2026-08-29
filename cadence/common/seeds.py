"""Deterministic seeding across numpy, python-random, and (if installed) torch.

Every experiment run must call `set_global_seed(seed)` before anything stochastic,
and log the seed to MLflow (see `cadence.common.tracking.start_run`).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedReport:
    seed: int
    torch_available: bool
    cuda_available: bool
    deterministic_algorithms: bool


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> SeedReport:
    """Seed every RNG we depend on. Returns a report to log to MLflow.

    `deterministic_torch=True` sets `torch.use_deterministic_algorithms(True)`
    and the CuBLAS workspace env var — this is the mode we use for reported
    numbers. Set False when profiling for speed.
    """
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"seed must fit in uint32, got {seed}")

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dep, but keep the shim robust
        pass

    torch_available = False
    cuda_available = False
    try:
        import torch

        torch_available = True
        cuda_available = torch.cuda.is_available()
        torch.manual_seed(seed)
        if cuda_available:
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    os.environ["PYTHONHASHSEED"] = str(seed)

    return SeedReport(
        seed=seed,
        torch_available=torch_available,
        cuda_available=cuda_available,
        deterministic_algorithms=deterministic_torch,
    )
