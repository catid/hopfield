"""Independent source-only enumeration, incidence and all-subset certificate."""
import hashlib
import itertools
import math
from pathlib import Path
import numpy as np

def verify(result, certificate_path, source_path):
    n = 12
    source = np.load(source_path)
    data = np.load(certificate_path)
    offsets = {(r, c): 0 for r in range(n) for c in range(1-n, n) if r or c > 0}
    for x in source:
        cells = np.where(x)[0]
        for i in range(len(cells)):
            for j in range(i+1, len(cells)):
                r1, c1 = divmod(int(cells[i]), n)
                r2, c2 = divmod(int(cells[j]), n)
                offsets[r2-r1, c2-c1] += 1
    normals = []
    for a in range(3):
        for b in range(-2, 3):
            if (a == 0 and b <= 0) or math.gcd(a, abs(b)) != 1:
                continue
            support = [count for (r, c), count in offsets.items() if a*r == -b*c]
            if len(support) >= 3 and all(v == 0 for v in support):
                normals.append((a, b))
    assert normals == [tuple(row) for row in result['inferred_lines']]
    assert (0, 1) in normals and (1, 0) in normals
    # Independent bit-mask propagation visits every permitted row-column choice.
    attacks = {}
    for row in range(n):
        for col in range(n):
            attacks[row, col] = [sum(1 << other for other in range(n) if any(a*(future-row)+b*(other-col) == 0 for a, b in normals)) for future in range(n)]
    full = (1 << n)-1
    solutions = []
    def expand(row, prohibited, cols):
        if row == n:
            solutions.append(cols)
            return
        remaining = full & ~prohibited[row]
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            col = bit.bit_length()-1
            next_prohibited = prohibited[:]
            for future in range(row+1, n):
                next_prohibited[future] |= attacks[row, col][future]
            expand(row+1, next_prohibited, cols+[col])
    expand(0, [0]*n, [])
    independent = np.asarray(solutions, np.uint8)
    assert np.array_equal(independent, data['universe_columns'])
    source_keys = {tuple(map(int, row.reshape(n, n).argmax(1))) for row in source}
    flags = np.array([tuple(map(int, row)) in source_keys for row in independent])
    assert np.array_equal(flags, data['is_positive']) and flags.sum() == len(source_keys) == 176
    off_line_zeros = {o for o, count in offsets.items() if count == 0 and not any(a*o[0]+b*o[1] == 0 for a, b in normals)}
    generators = sorted({(min(abs(r), abs(c)), max(abs(r), abs(c))) for r, c in off_line_zeros})
    assert np.array_equal(generators, data['generators'])
    rebuilt = np.zeros(data['incidence'].shape, bool)
    for row_index, cols in enumerate(independent.astype(int)):
        pair_orbits = {(min(j-i, abs(int(cols[j]-cols[i]))), max(j-i, abs(int(cols[j]-cols[i])))) for i, j in itertools.combinations(range(n), 2)}
        for orbit_index, generator in enumerate(generators):
            rebuilt[row_index, orbit_index] = generator in pair_orbits
    assert np.array_equal(rebuilt, data['incidence']) and not rebuilt[flags].any()
    all_covers = []
    for expected in result['subsets']:
        selected = expected['orbit_indices']
        assert sum(1 << i for i in selected) == expected['mask'] and len(selected) == expected['cardinality']
        uncovered = [int(i) for i in np.flatnonzero(~flags) if not any(rebuilt[i, j] for j in selected)]
        assert expected['rejected_negatives'] == int((~flags).sum())-len(uncovered)
        assert expected['covers_all_negatives'] == (len(uncovered) == 0)
        assert expected['first_uncovered_universe_index'] == (uncovered[0] if uncovered else None)
        assert expected['first_uncovered_columns'] == (independent[uncovered[0]].astype(int).tolist() if uncovered else None)
        if not uncovered:
            all_covers.append((len(selected), expected['mask']))
    assert len(result['subsets']) == 1 << len(generators)
    assert sorted(s['mask'] for s in result['subsets']) == list(range(1 << len(generators)))
    minimum = min((card for card, mask in all_covers), default=None)
    minima = [mask for card, mask in all_covers if card == minimum]
    assert minimum == result['minimum_cardinality'] and minima == result['all_minimum_masks']
    return dict(status='passed', independent_enumeration='precomputed pair-attack bit-mask propagation',
                independent_incidence='literal row-pair orbit sets', all_source_boards_equal=True,
                all_positive_memberships_equal=True, all_pair_incidence_equal=True,
                all_subsets_checked=len(result['subsets']), all_subset_witnesses_verified=True,
                all_minima_verified=minima, source_universe_boards=len(independent),
                no_target_data_access=True, verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
