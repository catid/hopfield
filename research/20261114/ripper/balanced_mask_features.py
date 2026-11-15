"""All masks are determined by past exposure counts and fixed tie permutations."""
import numpy as np
from noisy_policy import choose_mask


def masks(seed,rows=900):
    rng=np.random.default_rng(seed);given=np.zeros((rows,81),bool);joint=np.zeros((81,81),np.int32);excluded=np.zeros((81,81),bool)
    for row in range(rows):
        given[row]=choose_mask('exponential_e',joint,excluded,25,rng.permutation(81))
        cells=np.flatnonzero(given[row]);joint[np.ix_(cells,cells)]+=1
    assert not excluded.any()
    return given
