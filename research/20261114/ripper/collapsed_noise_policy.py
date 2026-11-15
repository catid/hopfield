"""One board-shared collapse event and output digit, sampled after mask choice."""
import numpy as np


def collapse_draw(rng):
    return float(rng.random()),int(rng.integers(0,9))


def apply_collapse(clean,uniform,digit):
    assert 0<=digit<9
    return np.full_like(clean,digit) if uniform<9/160 else clean.copy()
