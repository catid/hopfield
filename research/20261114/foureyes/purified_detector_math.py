"""Privileged false-positive removal and per-row matched true-location sampling."""
import numpy as np


def checked(detector_mask, true_mask):
    assert detector_mask.dtype == true_mask.dtype == bool
    assert detector_mask.ndim == 2 and detector_mask.shape == true_mask.shape
    assert (detector_mask.sum(1) == 9).all() and (true_mask.sum(1) == 9).all()


def masks(detector_mask, true_mask, seed):
    checked(detector_mask, true_mask)
    assert type(seed) is int and seed >= 0
    purified = detector_mask & true_mask
    random = np.zeros_like(purified)
    rng = np.random.Generator(np.random.PCG64(seed))
    for row in range(len(purified)):
        cells = np.flatnonzero(true_mask[row])
        # Calling choice on every row, including k=0 and k=9, is frozen.
        random[row, rng.choice(cells, int(purified[row].sum()), replace=False)] = True
    return purified, random


def independent_masks(detector_mask, true_mask, seed):
    checked(detector_mask, true_mask)
    rows, n = true_mask.shape
    purified = np.array([[bool(detector_mask[r, i] and true_mask[r, i])
                          for i in range(n)] for r in range(rows)], bool)
    rng = np.random.default_rng(seed)
    sampled = []
    for r in range(rows):
        true_cells = [i for i in range(n) if true_mask[r, i]]
        k = sum(bool(detector_mask[r, i]) for i in true_cells)
        sampled.append(set(map(int, rng.choice(true_cells, k, replace=False))))
    random = np.array([[i in sampled[r] for i in range(n)] for r in range(rows)], bool)
    return purified, random
