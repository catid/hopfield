"""Candidate geometry-only routing and explicit source-preserving scatter."""
import numpy as np
from bp_geometry_checks import simple_path


def routes(tokens, bp_boards):
    t, b = np.asarray(tokens), np.asarray(bp_boards)
    if t.ndim != 3 or t.shape != b.shape or b.dtype != np.bool_:
        raise ValueError('Expected matching batched tokens and boolean BP boards')
    valid = np.array([simple_path(x, y) for x, y in zip(t, b)], dtype=bool)
    return valid, np.flatnonzero(~valid)


def scatter(bp, lp_subset, indices):
    n = len(bp['prediction'])
    ix = np.asarray(indices)
    if ix.ndim != 1 or ix.dtype.kind not in 'iu' or np.any(ix < 0) or np.any(ix >= n):
        raise ValueError('Invalid subset indices')
    if not np.array_equal(ix, np.unique(ix)):
        raise ValueError('Indices must be unique and ascending')
    names = ['prediction', 'iterations', 'stopped', 'finite', 'own_margin', 'native_cap', 'block_cap']
    result = {}
    for name in names:
        original = np.asarray(bp[name])
        replacement = np.asarray(lp_subset[name])
        if original.shape[0] != n or replacement.shape != (len(ix), *original.shape[1:]) or replacement.dtype != original.dtype:
            raise ValueError('Subset shape or dtype mismatch: ' + name)
        result[name] = original.copy()
        result[name][ix] = replacement
    result['chosen_model'] = np.zeros(n, dtype=np.int8)
    result['chosen_model'][ix] = 1
    return result
