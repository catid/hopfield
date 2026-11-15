"""Oracle-only postfit interventions; geometry is deliberately exposed here."""
import numpy as np


def peer_mask():
    i, j = np.triu_indices(81, 1)
    return (i//9 == j//9) | (i%9 == j%9) | ((i//27 == j//27) & (i%9//3 == j%9//3))


def interventions(base):
    assert base.shape == (3240,) and base.dtype == np.uint8
    assert set(np.unique(base)) <= {0, 1, 16}
    peer = peer_mask()
    assert np.all(base[peer] > 0)
    repair = base.copy()
    repair[peer & (base == 1)] = 16
    soften = base.copy()
    soften[~peer & (base == 16)] = 1
    delete = base.copy()
    removed = ~peer & (base == 1)
    delete[removed] = 0
    assert not np.any(removed & peer) and np.array_equal(delete[peer], base[peer])
    assert np.array_equal(repair > 0, base > 0) and np.array_equal(soften > 0, base > 0)
    return {'base': base.copy(), 'repair_true': repair,
            'soften_strong_extra': soften, 'delete_weak_extra': delete}
