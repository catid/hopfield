"""Pure fixed repair: only supplied positives/model/classes; no file or pool access."""
import itertools
import numpy as np


def views(board):
    square = np.asarray(board, dtype=bool).reshape(13, 13)
    # Eight deterministic correlated views; deduplicate identical whole boards.
    return list({np.ascontiguousarray(np.rot90(np.fliplr(square) if reflect else square, turn)).tobytes():
                 np.ascontiguousarray(np.rot90(np.fliplr(square) if reflect else square, turn)).reshape(-1)
                 for reflect in (False, True) for turn in range(4)}.values())


def offsets(boards):
    seen = set()
    for board in boards:
        for a, b in itertools.combinations(np.flatnonzero(board), 2):
            seen.add((int(b)//13-int(a)//13, int(b)%13-int(a)%13))
    return seen


def repair(boards, inherited_F, finite_offsets, augment):
    """No target labels, population object, scoring function or I/O is accepted."""
    assert len(boards) == 32 and np.asarray(boards).shape == (32, 169)
    transformed = [view for board in boards for view in views(board)] if augment else list(boards)
    unique = list({np.asarray(board, dtype=bool).tobytes():board for board in transformed}.values())
    witnessed = offsets(unique) & {tuple(offset) for offset in finite_offsets}
    F = inherited_F.copy()
    for a, b in itertools.combinations(range(225), 2):
        if (b//15-a//15, b%15-a%15) in witnessed:
            F[a, b] = F[b, a] = 0
    return F, dict(raw_positive_labels=32, attempted_views=256 if augment else 32,
                   unique_views=len(unique), witnessed_finite_classes=sorted(witnessed))
