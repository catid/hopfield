"""Independent H1by count/cost, selected parent proof, full new grid and endpoints."""
from concurrent.futures import ProcessPoolExecutor
import json, time
import numpy as np
from raw_count_quantization import ROOT, HERE, OUT, PUBLIC, PARENT, PROTOCOL, ARMS, EXPOSURE, read, sha, npz, verify
from audit_degree_prior import check_certificate
from audit_laplace_rate import independent_rate_costs
from audit_partial_degree import grid
from audit_extra_coverage import statistics


def aggregate(rows):
    assert [row['index'] for row in rows] == list(range(128))
    groups, paired = [], {}
    for arm in ARMS:
        part = [row for row in rows if row['arm'] == arm]
        assert len(part) == 64
        old_good = sum(row['old_exact'] for row in part)
        assert old_good == (64 if arm == 'single270' else 50)
        success = sum(row['classification'] == 'success' for row in part)
        unknown = sum(row['classification'] == 'unresolved' for row in part)
        losses = [row['seed'] for row in part if row['old_exact'] and row['classification'].startswith('resolved_failure')]
        unknown_old = [row['seed'] for row in part if row['old_exact'] and row['classification'] == 'unresolved']
        gains = [row['seed'] for row in part if not row['old_exact'] and row['classification'] == 'success']
        groups.append(dict(arm=arm, old_exact=old_good, success=success, unresolved=unknown,
                           resolved_failure_nonexact=sum(row['classification'] == 'resolved_failure_nonexact' for row in part),
                           resolved_failure_fractional=sum(row['classification'] == 'resolved_failure_fractional' for row in part),
                           selected_degree20=sum(row['selected_degree'] == 20 for row in part),
                           fixed20_exact=sum(row['fixed_degree20']['classification'] == 'success' for row in part),
                           top810_exact=sum(row['top810']['classification'] == 'success' for row in part)))
        paired[arm] = dict(gains=gains, losses=losses, unresolved_old_success=unknown_old,
                           both_exact=[row['seed'] for row in part if row['old_exact'] and row['classification'] == 'success'],
                           both_resolved_failure=[row['seed'] for row in part if not row['old_exact'] and row['classification'].startswith('resolved_failure')],
                           any_unresolved=[row['seed'] for row in part if row['classification'] == 'unresolved'])
    triple = groups[1]
    lower, upper = triple['success']-50, triple['success']+triple['unresolved']-50
    lowloss = len(paired['triple90']['losses'])
    highloss = lowloss+len(paired['triple90']['unresolved_old_success'])
    endpoints = dict(net_triple_gain=dict(lower=lower, upper=upper, required=3, total=64,
                     decision='supported' if lower >= 3 else 'falsified' if upper < 3 else 'inconclusive'),
                     triple_regression=dict(lower=lowloss, upper=highloss, maximum=1, total=50,
                     decision='supported' if highloss <= 1 else 'falsified' if lowloss > 1 else 'inconclusive'))
    return dict(grouped=groups, paired_changes=paired, endpoints=endpoints)


def audit_bank(seed, public=False, protocol=None):
    p = verify() if protocol is None else protocol
    directory = (PUBLIC if public else OUT)/f's{seed}'
    parent_rows = read(PARENT/'summary.json')['rows']
    ii, jj = np.triu_indices(81, 1)
    truth = (ii//9 == jj//9) | (ii % 9 == jj % 9) | ((ii//27 == jj//27) & (ii % 9//3 == jj % 9//3))
    rows = []
    for ai, arm in enumerate(ARMS):
        index = 2*seed+ai
        source = PARENT/f's{seed}/{arm}-models.npz'
        with np.load(source, allow_pickle=False) as z:
            joint, equal, old_cost = z['joint_count'], z['equal_count'], z['cost']
            old_proof = [z['d20_'+key] for key in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
        old_receipt = read(PARENT/f's{seed}/{arm}-receipt.json')
        assert old_receipt['models_sha256'] == sha(source) and old_receipt['index'] == index
        assert np.array_equal(independent_rate_costs(joint, equal), old_cost)
        assert check_certificate(81, 20, ii, jj, old_cost, *old_proof)
        old_exact = bool(np.array_equal(old_proof[0], truth.astype(np.int64)*2))
        assert old_exact == (parent_rows[index]['classification'] == 'success')
        for k in range(3):
            bad = [v.copy() for v in old_proof]
            bad[k][0] += 1
            assert not check_certificate(81, 20, ii, jj, old_cost, *bad)
        assert joint.dtype == equal.dtype == np.int64 and joint.shape == equal.shape == (3240,)
        reference = []
        for exposed, count in zip(joint, equal, strict=True):
            assert int(exposed) == EXPOSURE[arm] and 0 <= int(count) <= int(exposed)
            reference.append(int(count))
        cost = np.array(reference, np.int64)
        arrays = npz(directory/f'{arm}-models.npz')
        assert np.array_equal(arrays['cost'], cost) and np.array_equal(arrays['joint_count'], joint) and np.array_equal(arrays['equal_count'], equal)
        receipt = read(directory/f'{arm}-receipt.json')
        assert receipt['index'] == index and receipt['seed'] == seed and receipt['arm'] == arm
        assert receipt['protocol_sha256'] == sha(PROTOCOL) and receipt['parent_models_sha256'] == sha(source)
        assert receipt['models_sha256'] == sha(directory/f'{arm}-models.npz')
        assert len(receipt['records']) == 40
        if not public:
            for record in receipt['records']:
                assert record['log_sha256'] == sha(directory/f'{arm}-d{record["degree"]:02d}.log')
        checked = grid(arrays, receipt['records'], cost, p['solver'], index)
        degree = checked['selected_degree']
        primal = np.empty(0, np.int64) if degree is None else arrays[f'd{degree:02d}_primal_twice']
        result = statistics(primal, truth)
        fixed = statistics(arrays['d20_primal_twice'], truth) if receipt['records'][19]['certificate']['certified'] else statistics(np.empty(0, np.int64), truth)
        top = np.zeros(3240, bool)
        top[[e for _, e in sorted((int(cost[e]), e) for e in range(3240))[:810]]] = True
        assert np.array_equal(top, arrays['top810_support'])
        gap = None
        if degree == 20:
            gap = 2*sum(int(cost[e]) for e in range(3240) if truth[e])-checked['objective_twice'][20]
            assert gap >= 0
        rows.append(dict(index=index, seed=seed, arm=arm, old_exact=old_exact, **checked, **result,
                         fixed_degree20=fixed, top810=statistics(2*top.astype(np.int64), truth),
                         true_graph_gap_twice_at_selected_degree20=gap))
    return dict(status='validated', seed=seed, protocol_sha256=sha(PROTOCOL), auditor_sha256=sha(__file__), rows=rows,
                artifact_sha256={f'{arm}-{suffix}':sha(directory/f'{arm}-{suffix}') for arm in ARMS for suffix in ['models.npz', 'receipt.json']})


def public_job(seed):
    return audit_bank(seed, True)


def main():
    started = time.perf_counter()
    p = verify()
    saved = read(PUBLIC/'summary.json')
    assert saved['protocol_sha256'] == sha(PROTOCOL)
    for rel, digest in saved['artifact_sha256'].items():
        assert sha(PUBLIC/rel) == digest
    with ProcessPoolExecutor(max_workers=8) as pool:
        audits = list(pool.map(public_job, range(64)))
    for seed, audited in enumerate(audits):
        assert audited == read(PUBLIC/f's{seed}/audit.json')
    rows = [row for audit in audits for row in audit['rows']]
    assert rows == saved['rows']
    outcomes = aggregate(rows)
    for key, value in outcomes.items():
        assert saved[key] == value
    assert saved['new_certified_LPs'] == sum(row['certified_grid_LPs'] for row in rows)
    assert saved['mutations_rejected'] == sum(row['mutations_rejected']+3 for row in rows)
    assert saved['archived_selected_certificates'] == saved['analytic_zero_proofs'] == 128
    assert saved['query_accounting'] == p['query_accounting']
    print(json.dumps(dict(status='passed', **outcomes, certified_LPs=saved['new_certified_LPs'],
                         archived_selected_certificates=128, mutations_rejected=saved['mutations_rejected'],
                         all128_retained=True, wall_seconds=time.perf_counter()-started)))


if __name__ == '__main__':
    main()
