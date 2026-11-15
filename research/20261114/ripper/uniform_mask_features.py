"""Uniform masks depend only on their declared RNG seed, never on observations."""
import numpy as np


def masks(seed,rows=900):
    rng=np.random.default_rng(seed);given=np.zeros((rows,81),bool)
    for row in range(rows):given[row,rng.permutation(81)[:25]]=True
    return given
