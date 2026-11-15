"""Independent H1bx clique truth, rational costs, exact proofs and endpoints."""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
from repeated_objective_gap import ROOT, HERE, PARENT, PUBLIC, PROTOCOL, sha, read, verify
from audit_laplace_rate import independent_rate_costs
sys.path.insert(0, str(HERE.parent/'foureyes'))
from audit_objective_gap import certify


def independent_truth():
    graph = np.zeros((81, 81), bool)
    groups = [[9*r+c for c in range(9)] for r in range(9)]
    groups += [[9*r+c for r in range(9)] for c in range(9)]
    groups += [[9*(3*br+r)+3*bc+c for r in range(3) for c in range(3)] for br in range(3) for bc in range(3)]
    for group in groups:
        for i in group:
            for j in group:
                if i != j:
                    graph[i, j] = True
    assert not np.diag(graph).any() and np.array_equal(graph, graph.T) and (graph.sum(0) == 20).all()
    truth = np.array([graph[i, j] for i in range(81) for j in range(i+1, 81)], bool)
    assert truth.sum() == 810
    return truth


def main(initial=False):
    start = time.perf_counter()
    verify()
    prior = read(PARENT/'summary.json')['rows']
    rows = read(PUBLIC/'rows.json')
    summary = read(PUBLIC/'summary.json')
    assert summary['protocol_sha256'] == sha(PROTOCOL) and summary['rows_sha256'] == sha(PUBLIC/'rows.json')
    assert len(rows) == len(prior) == 128
    truth = independent_truth()
    groups = {arm:dict(arm=arm, total=0, exact=0, strict=0, cooptimal_nonexact=0, zero_gap=0, selected_fractional=0) for arm in ['single270', 'triple90']}
    strict_failures = failed = mutations = 0
    for index, row in enumerate(rows):
        seed, ai = divmod(index, 2)
        arm = ['single270', 'triple90'][ai]
        old = prior[index]
        assert old['selected_degree'] == 20 and old['index'] == index and old['seed'] == seed and old['arm'] == arm
        receipt = read(PARENT/f's{seed}/{arm}-receipt.json')
        assert receipt['index'] == index and receipt['seed'] == seed and receipt['arm'] == arm
        assert receipt['models_sha256'] == sha(PARENT/f's{seed}/{arm}-models.npz')
        assert receipt['protocol_sha256'] == sha(HERE/'protocol-repeat-query-tradeoff.json')
        with np.load(PARENT/f's{seed}/{arm}-models.npz', allow_pickle=False) as z:
            cost = independent_rate_costs(z['joint_count'], z['equal_count'])
            assert np.array_equal(cost, z['cost'])
            proof = [z['d20_'+key] for key in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
            assert np.array_equal(2*z['d20_primal_values'], proof[0])
            assert np.max(abs(2*z['d20_node_dual_values']-proof[1])) <= 1e-6
        optimum = certify(81, 20, cost, *proof)
        assert old['objective_twice'][20] == optimum
        record = receipt['records'][19]
        assert record['degree'] == 20 and record['certificate']['certified']
        assert record['certificate']['primal_objective_twice'] == record['certificate']['dual_objective_twice'] == optimum
        assert record['certificate']['primal_rounding_used'] is False
        assert record['solver']['solver_optimal'] and record['solver']['value_valid'] and record['solver']['dual_valid']
        assert record['solver']['solver_objective'] == optimum/2
        for k in range(3):
            altered = [v.copy() for v in proof]
            altered[k][0] += 1
            try:
                certify(81, 20, cost, *altered)
            except AssertionError:
                mutations += 1
            else:
                raise AssertionError('Altered proof accepted')
        actual = sum(2*int(cost[e]) for e in range(3240) if truth[e])
        gap = actual-optimum
        assert gap >= 0
        exact = all(int(proof[0][e]) == 2*int(truth[e]) for e in range(3240))
        integral = all(int(x) in [0, 2] for x in proof[0])
        assert record['certificate']['integral'] == integral
        kind = 'success' if exact else 'resolved_failure_nonexact' if integral else 'resolved_failure_fractional'
        assert old['classification'] == kind
        if exact:
            assert gap == 0
        expected = dict(index=index, seed=seed, arm=arm, degree=20, parent_classification=kind,
                        optimum_twice=optimum, truth_cost_twice=actual, gap_twice=gap, exact=exact, integral=integral,
                        truth_objective_status='cooptimal' if gap == 0 else 'strictly_suboptimal')
        assert row == expected
        group = groups[arm]
        group['total'] += 1
        group['exact'] += exact
        group['strict'] += gap > 0
        group['cooptimal_nonexact'] += gap == 0 and not exact
        group['zero_gap'] += gap == 0
        group['selected_fractional'] += not integral
        if arm == 'triple90' and not exact:
            failed += 1
            strict_failures += gap > 0
    assert failed == 14 and mutations == 384
    assert [groups[arm]['exact'] for arm in ['single270', 'triple90']] == [64, 50]
    expected = dict(denominator=14, required_strict=10, strict_failures=strict_failures, cooptimal_failures=14-strict_failures,
                    decision='supported' if strict_failures >= 10 else 'falsified', grouped=[groups[arm] for arm in ['single270', 'triple90']],
                    all128_retained=True, selected_certificates_rechecked=128, new_LPs=0, new_labels=0,
                    parent_physical_noisy_union=2073600, official_noisy_label_sequence=14687319)
    for key, value in expected.items():
        assert summary[key] == value, key
    if initial:
        assert summary['status'] == 'pending_independent_audit'
    else:
        assert summary['status'] == 'complete' and summary['aggregate_cpu_seconds'] < 60 and summary['wall_seconds'] < 90
        assert summary['independent_audit_sha256'] == sha(PUBLIC/'independent-audit.json')
    print(json.dumps(dict(status='passed', **expected, mutations_rejected=mutations, wall_seconds=time.perf_counter()-start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial', action='store_true')
    args = parser.parse_args()
    main(args.initial)
