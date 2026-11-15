"""H4at: retain a pair observation only if both locations were uncorrupted."""
import numpy as np


def mask_from_indices(cells, n=81):
    assert cells.dtype == np.int64 and cells.ndim == 2 and cells.shape[1] == 9
    assert ((cells >= 0) & (cells < n)).all()
    assert all(len(set(map(int, row))) == 9 for row in cells)
    mask = np.zeros((len(cells), n), bool)
    mask[np.arange(len(cells))[:, None], cells] = True
    return mask


def checked(noisy, mask):
    assert noisy.ndim == 2 and mask.shape == noisy.shape and mask.dtype == bool
    assert np.issubdtype(noisy.dtype, np.integer) and ((noisy >= 0) & (noisy <= 8)).all()
    return noisy.shape


def dense_score(noisy, mask):
    _, n = checked(noisy, mask)
    ii, jj = np.triu_indices(n, 1)
    retained = ~(mask[:, ii] | mask[:, jj])
    exposures = retained.sum(0, dtype=np.int64)
    counts = (retained & (noisy[:, ii] == noisy[:, jj])).sum(0, dtype=np.int64)
    return dict(exposure=exposures, cost=counts)


def blocked_score(noisy, mask):
    rows, n = checked(noisy, mask)
    ii, jj = np.triu_indices(n, 1)
    exposure, cost = np.zeros(len(ii), np.int64), np.zeros(len(ii), np.int64)
    for start in range(0, rows, 64):
        x, m = noisy[start:start+64], mask[start:start+64]
        retained = ~(m[:, ii] | m[:, jj])
        exposure += retained.sum(0, dtype=np.int64)
        cost += (retained & (x[:, ii] == x[:, jj])).sum(0, dtype=np.int64)
    return dict(exposure=exposure, cost=cost)


def independent_score(noisy, mask):
    """Direct per-pair Boolean row selection, independent of matrix batching."""
    rows, n = checked(noisy, mask)
    exposure, cost = [], []
    for i in range(n):
        for j in range(i+1, n):
            row_indices = np.flatnonzero(~mask[:, i] & ~mask[:, j])
            exposure.append(len(row_indices))
            cost.append(int(np.count_nonzero(noisy[row_indices, i] == noisy[row_indices, j])))
    return dict(exposure=np.array(exposure, np.int64), cost=np.array(cost, np.int64))


def literal_score(noisy, mask):
    rows, n = checked(noisy, mask)
    exposure, cost = [], []
    for i in range(n):
        for j in range(i+1, n):
            available = [r for r in range(rows) if not mask[r, i] and not mask[r, j]]
            exposure.append(len(available))
            cost.append(sum(int(noisy[r, i]) == int(noisy[r, j]) for r in available))
    return dict(exposure=np.array(exposure, np.int64), cost=np.array(cost, np.int64))


def score(noisy, mask):
    return dense_score(noisy, mask)
