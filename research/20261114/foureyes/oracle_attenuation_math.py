"""H4as privileged-mask score only; no clean digits, offsets or graph input."""
from fractions import Fraction
import numpy as np
from normalized_deficit import half_even


def mask_from_indices(cells, n=81):
    assert cells.dtype == np.int64 and cells.ndim == 2 and cells.shape[1] == 9
    assert ((cells >= 0) & (cells < n)).all()
    assert all(len(set(map(int, row))) == 9 for row in cells)
    mask = np.zeros((len(cells), n), bool)
    mask[np.arange(len(cells))[:, None], cells] = True
    assert (mask.sum(1) == 9).all()
    return mask


def score(counts, mask):
    assert mask.dtype == bool and mask.ndim == 2
    rows, n = mask.shape
    ii, jj = np.triu_indices(n, 1)
    assert counts.dtype == np.int64 and counts.shape == ii.shape
    assert ((counts >= 0) & (counts <= rows)).all()
    channel = 8-9*mask.astype(np.int64)
    # Integer accumulation also handles the all-clean maximum 64*600=38400.
    weight = (channel.T @ channel)[ii, jj]
    numerators = 64000000*(9*counts-rows)
    nonpositive = np.flatnonzero(weight <= 0).astype(np.int64)
    costs = (np.array([half_even(int(a), int(b)) for a, b in zip(numerators, weight)], np.int64)
             if not len(nonpositive) else np.empty(0, np.int64))
    return dict(attenuation=weight, numerators=numerators, nonpositive_pairs=nonpositive, cost=costs)


def reference_score(counts, mask):
    rows, n = mask.shape
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    weights = [sum((8 if not mask[r, i] else -1)*(8 if not mask[r, j] else -1)
                   for r in range(rows)) for i, j in edges]
    numerators = [64000000*(9*int(c)-rows) for c in counts]
    bad = [i for i, w in enumerate(weights) if w <= 0]
    costs = [] if bad else [round(Fraction(a, b)) for a, b in zip(numerators, weights)]
    return {k:np.array(v, np.int64) for k,v in
            [('attenuation', weights), ('numerators', numerators), ('nonpositive_pairs', bad), ('cost', costs)]}


def independent_score(counts, mask):
    """Exact expanded identity using Boolean intersections, not channel products."""
    rows, n = mask.shape
    margins = [int(np.count_nonzero(mask[:, i])) for i in range(n)]
    weights = []
    for i in range(n):
        joints = np.count_nonzero(mask[:, i, None] & mask[:, i+1:], axis=0)
        weights.extend(64*rows - 72*(margins[i]+margins[j]) + 81*int(joint)
                       for j, joint in zip(range(i+1, n), joints))
    numerators = [64000000*(9*int(c)-rows) for c in counts]
    bad = [i for i, w in enumerate(weights) if w <= 0]
    costs = [] if bad else [round(Fraction(a, b)) for a, b in zip(numerators, weights)]
    return {k: np.array(v, np.int64) for k, v in
            [('attenuation', weights), ('numerators', numerators), ('nonpositive_pairs', bad), ('cost', costs)]}


def channel_gate():
    """Exhaustive rational conditional identity; no study masks or labels."""
    cases = outcomes = 0
    for left in range(9):
        for right in range(9):
            for m_left in [0, 1]:
                for m_right in [0, 1]:
                    left_offsets = list(range(1, 9)) if m_left else [0]
                    right_offsets = list(range(1, 9)) if m_right else [0]
                    equal = sum((left+a) % 9 == (right+b) % 9
                                for a in left_offsets for b in right_offsets)
                    total = len(left_offsets)*len(right_offsets)
                    actual = 9*Fraction(equal, total)-1
                    expected = Fraction((8-9*m_left)*(8-9*m_right), 64)*(9*int(left == right)-1)
                    assert actual == expected
                    cases += 1
                    outcomes += total
    all_clean = np.zeros((600, 3), bool)
    zero_counts = np.zeros(3, np.int64)
    high = score(zero_counts, all_clean)
    assert high['attenuation'].tolist() == [38400]*3
    assert high['cost'].tolist() == [-1000000]*3
    always_one_bad = all_clean.copy()
    always_one_bad[:, 0] = True
    bad = score(zero_counts, always_one_bad)
    assert bad['nonpositive_pairs'].tolist() == [0, 1] and not len(bad['cost'])
    # Eight all-clean and 64 one-corrupted rows yield a zero denominator.
    zero = np.zeros((72, 3), bool)
    zero[8:, 0] = True
    assert score(zero_counts, zero)['attenuation'].tolist() == [0, 0, 4608]
    rng = np.random.default_rng(982020)
    indices = np.stack([rng.choice(81, 9, replace=False) for _ in range(12)]).astype(np.int64)
    mask = mask_from_indices(indices)
    count = rng.integers(0, 13, 3240, dtype=np.int64)
    producer, reference = score(count, mask), reference_score(count, mask)
    assert all(np.array_equal(producer[k], reference[k]) for k in producer)
    for broken in [indices.astype(np.int32), np.full((12, 9), 81, np.int64), np.zeros((12, 9), np.int64)]:
        try:
            mask_from_indices(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid corrupted-cell indices accepted')
    return dict(status='passed', conditional_channel_cases=cases, offset_outcomes_enumerated=outcomes,
                maximum_W_over_int16_checked=True, nonpositive_pairs_preserved=True,
                independent_integer_pair_sums=True, study_masks_or_scores_read=False)


if __name__ == '__main__':
    import json
    print(json.dumps(channel_gate()))
