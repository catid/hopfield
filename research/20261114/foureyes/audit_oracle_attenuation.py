"""Independent rational scores, exact LP proofs and H4as paired outcomes."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
import time
import numpy as np
from oracle_attenuation import ROOT, HERE, OUT, PUBLIC, PARENT, PROTOCOL, FAMILIES, sha, read, verify
from audit_degree_prior import check_certificate
from audit_objective_gap import reconstruct


def selection_stream(hot_seed, noise_seed):
    """Replay the entire original selection stream before slicing its prefix."""
    hot = sorted(map(int, np.random.default_rng(hot_seed).choice(81, 18, replace=False)))
    uniform = 1 - np.random.default_rng(noise_seed).random((2400, 81))
    weights = [32 if i in hot else 1 for i in range(81)]
    priority = -np.log(uniform) / np.array(weights)
    # Python's tuple sort independently implements stable increasing cell ties.
    full = np.array([[i for _, i in sorted((float(v), i) for i, v in enumerate(row))[:9]]
                     for row in priority], np.int64)
    return full[:600]


def regenerate_cells(job):
    archive = job['archive']
    protocol = read(HERE/(archive['parent']+'.json'))
    index = archive['parent_index']
    return selection_stream(protocol['heatmap_seed_base']+index, protocol['noise_rng_base']+index)


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
    assert job['archive'] == read(HERE/'prefix_budget.json')['jobs'][pi]
    degree = job['inferred_degree']
    with np.load(PARENT/f'j{pi:03d}/models.npz', allow_pickle=False) as z:
        counts = z['cost']
        baseline = [z[f'd{degree:02d}_'+k] for k in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
    with np.load(directory/'models.npz', allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    assert np.array_equal(counts, a['counts'])
    from oracle_attenuation_math import independent_score
    cells = regenerate_cells(job)
    assert cells.shape == (600, 9) and a['cells'].dtype == np.int64
    assert np.array_equal(a['cells'], cells)
    mask = np.array([[i in row for i in range(81)] for row in map(set, cells.tolist())], bool)
    assert np.array_equal(a['mask'], mask) and a['mask'].dtype == bool
    reference = independent_score(counts, mask)
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
    skipped = bool(len(a['nonpositive_pairs']))
    if skipped:
        assert metadata == dict(skipped=True, reason='nonpositive_attenuation')
        assert receipt['certificate'] == dict(certified=False, reason='nonpositive_attenuation')
        assert a['primal_values'].size == a['node_dual_values'].size == 0
    for key, value in ([] if skipped else p['solver'].items()):
        if key == 'highspy':
            assert metadata['highs_version'] == value
        elif key == 'numpy':
            assert metadata['numpy'] == value
        else:
            option = 'random_seed' if key == 'random_seed_base' else key
            assert metadata['solver_options'][option] == (value+index if key == 'random_seed_base' else value)
    cost = a['cost']
    proof = [a[k] for k in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
    valid = False if skipped else check_certificate(81, degree, ii, jj, cost, *proof)
    if skipped:
        assert all(w.size == 0 for w in proof)
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
    rounding_ties = None if skipped else sum(2*(abs(int(n)) % int(d)) == int(d) for n, d in zip(a['numerators'], a['attenuation']))
    row = dict(index=index, group=group, family=family, degree=degree, baseline_exact=baseline_exact,
               classification=classification, missing_true=missing, extra=extra,
               certificates_checked=len(witnesses), mutations_rejected=mutations,
               attenuation_min=int(a['attenuation'].min()), attenuation_max=int(a['attenuation'].max()),
               nonpositive_pairs=len(a['nonpositive_pairs']), LP_attempted=not skipped,
               cost_min=int(cost.min()) if len(cost) else None, cost_max=int(cost.max()) if len(cost) else None, rounding_ties=rounding_ties)
    return dict(row=row, protocol_sha256=sha(PROTOCOL), auditor_sha256=sha(HERE/'audit_oracle_attenuation.py'),
                artifact_sha256={name: sha(directory/name) for name in ['models.npz', 'receipt.json']})


def public_job(index):
    return audit_bank(index, read(PROTOCOL), public=True)


def main():
    start = time.perf_counter()
    protocol = read(PROTOCOL)
    unavailable = verify(protocol, public=True)
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
    print(json.dumps(dict(status='passed', **aggregate(rows), wall_seconds=time.perf_counter()-start,
                         unavailable_raw_archives=len(unavailable),
                         count_provenance='Inherited hash-bound H4an count audits; raw count generation not repeated.',
                         masks_independently_regenerated_from_full2400_streams=True)))


if __name__ == '__main__':
    main()
