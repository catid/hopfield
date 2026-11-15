"""Board-shared uniform derangements, with a fresh independent board latent."""
import numpy as np


def shared_draw(rng):
    uniform = float(rng.random())
    attempts = 0
    while True:
        permutation = rng.permutation(9)
        attempts += 1
        if np.all(permutation != np.arange(9)):
            return uniform, permutation.astype(np.int8), attempts


def apply_shared(clean, uniform, permutation):
    assert np.array_equal(np.sort(permutation),np.arange(9))
    assert np.all(permutation != np.arange(9))
    return permutation[clean] if uniform < .05 else clean.copy()
