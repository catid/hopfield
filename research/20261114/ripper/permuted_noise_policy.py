"""Board-shared position permutation, drawn after the current mask is fixed."""
import numpy as np

def permutation_draw(rng):
    return float(rng.random()),rng.permutation(81).astype(np.uint8)

def apply_permutation(clean,uniform,permutation):
    assert clean.shape==(81,) and np.array_equal(np.sort(permutation),np.arange(81))
    return clean[permutation] if uniform<9/160 else clean.copy()
