"""Synthetic exact-gap and endpoint gates; no study outcomes read."""
from pathlib import Path
import time
import numpy as np
import consensus_gap as C
from audit_objective_gap import certify


def main():
    start = time.perf_counter()
    x = np.array([2, 0, 0, 0, 0, 2], np.int64)
    y = np.ones(4, np.int64)
    z = np.zeros(6, np.int64)
    truth = np.array([False, True, False, False, True, False])
    cost = np.array([1, 4, 3, 3, 4, 1], np.int64)
    out = C.evaluate(4, 1, cost, x, y, z, truth)
    assert out['gap_twice'] == 12 and out['truth_objective_status'] == 'strictly_suboptimal'
    assert certify(4, 1, cost, x, y, z) == 4
    matches = [[0, 5], [1, 4], [2, 3]]
    assert min(sum(int(cost[e]) for e in m) for m in matches) == 2
    tie = np.ones(6, np.int64)
    assert C.evaluate(4, 1, tie, x, y, z, truth)['gap_twice'] == 0
    # Values beyond exact binary64 integer resolution stay exact in the gap path.
    large = np.int64(2**54)
    result = C.evaluate(4, 1, cost+large, x, y+large, z, truth)
    assert result['gap_twice'] == 12
    for changed in [np.zeros(6, bool), np.ones(6, bool)]:
        try:
            C.evaluate(4, 1, cost, x, y, z, changed)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid truth degree accepted')
    rejected = 0
    for k in range(3):
        bad = [x.copy(), y.copy(), z.copy()]
        bad[k][0] += 1
        try:
            C.evaluate(4, 1, cost, *bad, truth)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError('Invalid proof accepted')
    assert C.decision(24) == 'falsified' and C.decision(25) == 'supported'
    # Independent loop and vector vote reductions agree on synthetic half-votes.
    toy = np.random.default_rng(0).integers(0, 3, (32, 3240), dtype=np.int64)
    before = time.perf_counter()
    vector = toy.sum(0)
    vector_seconds = time.perf_counter()-before
    before = time.perf_counter()
    lists = toy.tolist()
    literal = [sum(v[e] for v in lists) for e in range(3240)]
    loop_seconds = time.perf_counter()-before
    assert np.array_equal(vector, literal)
    sources = [Path(__file__), C.HERE/'consensus_gap.py', C.HERE/'audit_consensus_gap.py',
               C.HERE/'objective_gap.py', C.HERE/'audit_objective_gap.py',
               C.HERE.parent/'ripper/audit_degree_prior.py']
    C.save(C.HERE/'consensus-gap-preflight.json', dict(status='passed',
        study_outcomes_read=False, altered_proofs_rejected=rejected,
        gate_seconds=time.perf_counter()-start, synthetic_vote_vector_seconds=vector_seconds,
        synthetic_vote_literal_seconds=loop_seconds,
        optimization='Each NPZ member is decompressed once before independent vote summation; no study pilot.',
        source_sha256={str(p.relative_to(C.ROOT)): C.sha(p) for p in sources}))
    print('PREFLIGHT_PASSED')


if __name__ == '__main__':
    main()
