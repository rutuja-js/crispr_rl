"""Deterministic seed management across numpy and random."""

import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Set seed for all random number generators used by the package."""
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Create an isolated numpy RNG from a seed."""
    return np.random.default_rng(seed)
