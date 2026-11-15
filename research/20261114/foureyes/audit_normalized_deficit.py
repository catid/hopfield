"""Independent rational scores, exact LP proofs and H4ar paired outcomes."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
import time
import numpy as np
from normalized_deficit import ROOT, HERE, OUT, PUBLIC, PARENT, PROTOCOL, FAMILIES, sha, read, verify
from audit_degree_prior import check_certificate
from audit_objective_gap import reconstruct


def reference_scores(n, degree, counts, rows=600):
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    assert len(counts) == len(edges) and counts.dtype == np.int64
    deficits = [[] for _ in range(n)]
    for (i, j), count in zip(edges, counts):
        assert 0 <= int(count) <= rows
        value = max(rows-9*int(count), 0)
        deficits[i].append(value)
        deficits[j].append(value)
    scales = [max(degree, sum(sorted(row)[-degree:])) for row in deficits]
    numerators = [10**6*degree*degree*(9*int(c)-rows) for c in counts]
    denominators = [scales[i]*scales[j] for i, j in edges]
    costs = [round(Fraction(num, den)) for num, den in zip(numerators, denominators)]
    return {key: np.array(values, np.int64) for key, values in
            [('scales', scales), ('numerators', numerators), ('denominators', denominators), ('cost', costs)]}


def aggregate(rows):
    assert len(rows) == 256 and [r['index'] for r in rows] == list(range(256))
    assert sum(r['baseline_exact'] for r in rows) == 129
    good = sum(r['classification'] == 'success' for r in rows)
    unknown = sum(r['classification'] == 'unresolved' for r in rows)
    lost = sum(r['baseline_exact'] and r['classification'].startswith('resolved_failure') for r in rows)
    unknown_old = sum(r['baseline_exact'] and r['classification'] == 'unresolved' for r in rows)
    primary = dict(recovery=dict(lower=good, upper=good+unknown, required=145,
                   decision='supported' if good >= 145 else 'falsified' if good+unknown < 145 else 'inconclusive'),
                   paired_loss=dict(lower=lost, upper=lost+unknown_old, maximum=8,
                   decision='supported' if lost+unknown_old <= 8 else 'falsified' if lost > 8 else 'inconclusive'))
    groups = []
    for family in FAMILIES:
        part = [r for r in rows if r['family'] == family]
        assert len(part) == 64
        groups.append(dict(family=family, baseline_exact=sum(r['baseline_exact'] for r in part),
                      **{k: sum(r['classification'] == k for r in part) for k in
                         ['success', 'resolved_failure_nonexact', 'resolved_failure_fractional', 'unresolved']},
                      gains=sum(not r['baseline_exact'] and r['classification'] == 'success' for r in part),
                      losses=sum(r['baseline_exact'] and r['classification'].startswith('resolved_failure') for r in part)))
    return dict(primary=primary, grouped=groups, certificates_checked=sum(r['certificates_checked'] for r in rows),
                mutations_rejected=sum(r['mutations_rejected'] for r in rows), all256_banks_retained=True)


def audit_bank(index, p, public=False):
    directory = (PUBLIC if public else OUT) / f'b{index:03d}'
    job = p['jobs'][index]
    group, fi = divmod(index, 4)
    family = FAMILIES[fi]
    pi = 12*group+3*fi+1
    assert job['index'] == index and job['parent_prefix_index'] == pi
    assert job['family'] == family and job['group'] == group
    degree = job['inferred_degree']
    with np.load(PARENT/f'j{pi:03d}/models.npz', allow_pickle=False) as z:
        counts = z['cost']
        baseline = [z[f'd{degree:02d}_'+k] for k in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
    with np.load(directory/'models.npz', allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    assert np.array_equal(counts, a['counts'])
    reference = reference_scores(81, degree, counts)
    for key, value in reference.items():
        assert a[key].dtype == np.int64 and np.array_equal(a[key], value), key
    name = 'varying_degree' if family in ['cycle81', 'cliques9'] else 'cycle_power'
    gi = 2*group+int(family in ['cliques9', 'power3'])
    with np.load(ROOT/f'results/research/foureyes/{name}/controls/graphs.npz') as z:
        graph = reconstruct(z['orders'][gi], family)
        ii, jj = np.triu_indices(81, 1)
        truth = graph[ii, jj]
        assert np.array_equal(truth, z['truths'][gi]) and (graph.sum(0) == degree).all()
    assert check_certificate(81, degree, ii, jj, counts, *baseline)
    baseline_exact = bool(np.array_equal(baseline[0], truth.astype(np.int64)*2))
    assert baseline_exact == job['baseline_exact']
    receipt = read(directory/'receipt.json')
    assert receipt['index'] == index and receipt['degree'] == degree
    assert receipt['protocol_sha256'] == sha(PROTOCOL) and receipt['models_sha256'] == sha(directory/'models.npz')
    if not public:
        assert receipt['log_sha256'] == sha(directory/'solver.log')
    metadata = receipt['solver']
    for key, value in p['solver'].items():
        if key == 'highspy':
            assert metadata['highs_version'] == value
        elif key == 'numpy':
            assert metadata['numpy'] == value
        else:
            option = 'random_seed' if key == 'random_seed_base' else key
            assert metadata['solver_options'][option] == (value+index if key == 'random_seed_base' else value)
    cost = a['cost']
    proof = [a[k] for k in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
    valid = check_certificate(81, degree, ii, jj, cost, *proof)
    assert valid == receipt['certificate']['certified']
    classification = 'unresolved'
    missing = extra = None
    witnesses = [(counts, baseline)]
    if valid:
        witnesses.append((cost, proof))
        assert metadata['solver_optimal'] and metadata['value_valid'] and metadata['dual_valid']
        assert np.array_equal(2*a['primal_values'], proof[0]) and np.isin(a['primal_values'], [0., .5, 1.]).all()
        assert np.max(abs(2*a['node_dual_values']-proof[1])) <= 1e-6
        obj = sum(int(c)*int(v) for c, v in zip(cost, proof[0]))
        assert obj == receipt['certificate']['primal_objective_twice'] == receipt['certificate']['dual_objective_twice']
        assert metadata['solver_objective'] == obj/2 and receipt['certificate']['primal_rounding_used'] is False
        integral = bool(np.isin(proof[0], [0, 2]).all())
        assert integral == receipt['certificate']['integral']
        if integral:
            missing = int((truth & (proof[0] != 2)).sum())
            extra = int((~truth & (proof[0] == 2)).sum())
            classification = 'success' if missing == extra == 0 else 'resolved_failure_nonexact'
        else:
            classification = 'resolved_failure_fractional'
    mutations = 0
    for c, witness in witnesses:
        for k in range(3):
            bad = [v.copy() for v in witness]
            bad[k][0] += 1
            assert not check_certificate(81, degree, ii, jj, c, *bad)
            mutations += 1
    rounding_ties = sum(2*(abs(int(n)) % int(d)) == int(d) for n, d in zip(a['numerators'], a['denominators']))
    row = dict(index=index, group=group, family=family, degree=degree, baseline_exact=baseline_exact,
               classification=classification, missing_true=missing, extra=extra,
               certificates_checked=len(witnesses), mutations_rejected=mutations,
               scale_min=int(a['scales'].min()), scale_max=int(a['scales'].max()),
               cost_min=int(cost.min()), cost_max=int(cost.max()), rounding_ties=rounding_ties)
    return dict(row=row, protocol_sha256=sha(PROTOCOL), auditor_sha256=sha(HERE/'audit_normalized_deficit.py'),
                artifact_sha256={name: sha(directory/name) for name in ['models.npz', 'receipt.json']})


def public_job(index):
    return audit_bank(index, read(PROTOCOL), public=True)


def main():
    start = time.perf_counter()
    protocol = read(PROTOCOL)
    verify(protocol)
    summary = read(PUBLIC/'summary.json')
    assert summary['protocol_sha256'] == sha(PROTOCOL)
    for name, digest in summary['artifact_sha256'].items():
        assert sha(PUBLIC/name) == digest
    with ProcessPoolExecutor(max_workers=8) as pool:
        audits = list(pool.map(public_job, range(256)))
    for i, audit in enumerate(audits):
        assert audit == read(PUBLIC/f'b{i:03d}/audit.json')
    rows = [a['row'] for a in audits]
    assert rows == summary['rows']
    for key, value in aggregate(rows).items():
        assert summary[key] == value
    runtime = read(PUBLIC/'runtime.json')
    assert runtime['status'] == 'complete' and runtime['error'] is None and runtime['returncodes'] == [0]*8
    assert runtime['wall_seconds'] < 90 and runtime['aggregate_cpu_seconds'] < 300
    assert runtime['samples'][-1]['audited_banks'] == 256 and runtime['samples'][-1]['active_pids'] == []
    print(json.dumps(dict(status='passed', **aggregate(rows), wall_seconds=time.perf_counter()-start)))


if __name__ == '__main__':
    main()
