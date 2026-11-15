"""Independent denominator-256 integer checks for dyadic Sudoku energies."""
import numpy as np


def check_state(search, weight256):
    assert weight256.shape == (3240,) and weight256.dtype == np.uint16
    assert set(np.unique(weight256)) <= {0, 1, 4, 16, 256}
    assert search.theta == .45 and search.ebar == search.tau == 0 and search.tol == .001
    positive = weight256[weight256 > 0]
    assert len(positive) and int(positive.min()) / 256 > search.tau + search.tol
    ii, jj = np.triu_indices(81, 1)
    matrix = np.zeros((81, 81), np.int64)
    matrix[ii, jj] = weight256; matrix[jj, ii] = weight256
    y = search.y.cpu().numpy().astype(np.int64)
    free = search.fr.cpu().numpy()
    if len(y) == 0:
        return 0
    assert np.all((0 <= y) & (y < 9))
    fields = np.stack([(y == value).astype(np.int64) @ matrix.T for value in range(9)], axis=2)
    energy = ((y[:, ii] == y[:, jj]) * weight256.astype(np.int64)).sum(1)
    assert np.array_equal(search.G.cpu().numpy() * 256, fields)
    assert np.array_equal(search.P.cpu().numpy() * 256, energy)
    own = np.take_along_axis(fields, y[:, :, None], axis=2).squeeze(2)
    # ceil(256 * 9/20) = 116, independently of floating implementation.
    assert np.array_equal(search.flag.cpu().numpy(), ((own >= 116) & free).astype(np.int32))
    clues = search.x.cpu().numpy() - 1
    assert np.array_equal(y[~free], clues[~free])
    found = search.found.cpu().numpy() >= 0
    cert = search.ycert.cpu().numpy()[found]
    assert not (((cert[:, ii] == cert[:, jj]) * weight256.astype(np.int64)).sum(1)).any()
    cert_free = free[found]
    assert np.array_equal(cert[~cert_free], clues[found][~cert_free])
    return len(y)
