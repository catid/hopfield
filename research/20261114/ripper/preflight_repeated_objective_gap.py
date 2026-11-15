"""Synthetic H1bx strict/cooptimal proofs, endpoint bounds and member loading."""
import json, platform, time, statistics
import numpy as np
from repeated_objective_gap import ROOT, HERE, PUBLIC, PROTOCOL, sha, save, evaluate, aggregate, truth_graph
from audit_repeated_objective_gap import independent_truth, certify


def main():
    assert platform.node() == 'ripper'
    assert not PROTOCOL.exists() and not PUBLIC.exists() and not (HERE/'repeated-objective-gap-preflight.json').exists()
    assert np.array_equal(truth_graph(), independent_truth())
    z4, z6 = np.zeros(4, np.int64), np.zeros(6, np.int64)
    truth4 = np.array([1, 0, 0, 0, 0, 1], bool)
    alternative = np.array([0, 0, 2, 2, 0, 0], np.int64)
    zero_cost = np.array([0, 1, 0, 0, 1, 0], np.int64)
    strict_cost = np.array([1, 1, 0, 0, 1, 1], np.int64)
    fixtures = [
        (4, zero_cost, [2*truth4.astype(np.int64), z4, z6], truth4, 0, True, True),
        (4, zero_cost, [alternative, z4, z6], truth4, 0, False, True),
        (4, strict_cost, [alternative, z4, z6], truth4, 4, False, True),
    ]
    edges6 = [(i, j) for i in range(6) for j in range(i+1, 6)]
    within = np.array([i//3 == j//3 for i, j in edges6], bool)
    truth6 = np.array([(i, j) in [(0, 3), (1, 4), (2, 5)] for i, j in edges6], bool)
    fixtures.append((6, (~within).astype(np.int64), [within.astype(np.int64), np.zeros(6, np.int64), np.zeros(15, np.int64)], truth6, 6, False, False))
    receipts = []
    mutations = 0
    for n, cost, proof, truth, gap, exact, integral in fixtures:
        result = evaluate(n, 1, cost, proof, truth)
        opt = certify(n, 1, cost, *proof)
        assert result['optimum_twice'] == opt and result['gap_twice'] == gap and result['exact'] == exact and result['integral'] == integral
        assert result['truth_cost_twice'] == 2*sum(int(cost[i]) for i in range(len(cost)) if truth[i])
        for k in range(3):
            altered = [v.copy() for v in proof]
            altered[k][0] += 1
            try:
                certify(n, 1, cost, *altered)
            except AssertionError:
                mutations += 1
            else:
                raise AssertionError('Synthetic altered proof accepted')
        receipts.append(result)
    assert mutations == 12
    endpoints = []
    for strict in [0, 9, 10, 14]:
        rows = []
        for seed in range(64):
            for ai, arm in enumerate(['single270', 'triple90']):
                fail = arm == 'triple90' and seed < 14
                rows.append(dict(index=2*seed+ai, seed=seed, arm=arm, exact=not fail, integral=True,
                                 gap_twice=int(fail and seed < strict)))
        result = aggregate(rows)
        assert result['strict_failures'] == strict and result['cooptimal_failures'] == 14-strict
        assert result['decision'] == ('supported' if strict >= 10 else 'falsified')
        endpoints.append(result)
    # Unrelated compressed arrays emulate archive member structure, never parent study data.
    folder = ROOT/'artifacts/research/20261114/ripper/repeated_objective_gap_preflight'
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(987001)
    arrays = {f'd{d:02d}_{name}':rng.integers(-100, 101, shape, dtype=np.int64)
              for d in range(1, 41) for name, shape in [('primal_twice', 3240), ('node_dual_twice', 81), ('upper_dual_twice', 3240)]}
    path = folder/'unrelated-members.npz'
    np.savez_compressed(path, **arrays)
    selected_keys = ['d20_'+key for key in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
    timings = {}
    for mode in ['selected_members', 'full_archive']:
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            with np.load(path, allow_pickle=False) as z:
                keys = selected_keys if mode == 'selected_members' else z.files
                loaded = {key:z[key] for key in keys}
            samples.append(time.perf_counter()-start)
            assert all(np.array_equal(loaded[key], arrays[key]) for key in selected_keys)
        timings[mode] = dict(samples_wall_seconds=samples, median_wall_seconds=statistics.median(samples))
    dependencies = json.loads((HERE/'repeat-query-tradeoff-preflight.json').read_text())['source_sha256']
    files = {ROOT/rel for rel in dependencies}
    files |= {HERE/name for name in ['repeated_objective_gap.py', 'audit_repeated_objective_gap.py', 'preflight_repeated_objective_gap.py']}
    files.add(HERE.parent/'foureyes/audit_objective_gap.py')
    result = dict(status='passed', study_gaps_read=False, synthetic_proof_fixtures=receipts, synthetic_mutations_rejected=mutations,
                  exact_truth_reconstruction=True, endpoint_cases=endpoints, archive_loading_timings=timings,
                  timing_scope='Unrelated synthetic compressed arrays,3repetitions per mode; selected members equal full-load values. Not full-study speedup.',
                  source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sorted(files)})
    save(HERE/'repeated-objective-gap-preflight.json', result)
    print(json.dumps(dict(status='passed', source_files=len(files), archive_loading_timings=timings)))


if __name__ == '__main__':
    main()
