"""Independent H1bt shuffle law, counts, exact LP proofs, rational p-values and endpoints."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib, json, time
import numpy as np
from signal_detection_study import ROOT, HERE, OUT, PUBLIC, PARENT, RAW_PARENT, PROTOCOL, ARMS, B, sha, save, npz, verify
from audit_marginal_shuffle import reference_masks, reference_histograms, reference_shuffle, reference_observe, reference_concentration, label_gate
from audit_partial_degree import grid
from audit_laplace_rate import independent_rate_costs
from audit_extra_coverage import statistics


def reference_test(observed, shuffles):
    assert len(shuffles) == 31
    if observed is None or None in shuffles:
        return dict(numerator=None, denominator=32, greater_or_equal=None, signal=None)
    value = Fraction(observed['numerator'], observed['denominator'])
    count = len([other for other in shuffles if Fraction(other['numerator'], other['denominator']) >= value])
    probability = Fraction(1+count, 32)
    return dict(numerator=1+count, denominator=32, greater_or_equal=count, signal=probability <= Fraction(1, 32))


def joint_decision(pair):
    assert len(pair) == 2
    return 'no_signal' if False in pair else 'signal' if pair == [True, True] else 'unresolved'


def aggregate(rows):
    assert [(r['case'], r['arm']) for r in rows] == [(case, arm) for case in range(64) for arm in ARMS]
    grouped = []
    banks = {}
    endpoints = {}
    for cohort, begin in [('real', 0), ('null', 32)]:
        bank_results = dict(signal=[], no_signal=[], unresolved=[])
        for case in range(begin, begin+32):
            pair = [rows[2*case+ai]['test']['signal'] for ai in range(2)]
            bank_results[joint_decision(pair)].append(case % 32)
        banks[cohort] = bank_results
        lower = len(bank_results['signal'])
        upper = lower+len(bank_results['unresolved'])
        if cohort == 'real':
            endpoint = dict(lower=lower, upper=upper, required=29, total=32,
                            decision='supported' if lower >= 29 else 'falsified' if upper < 29 else 'inconclusive')
        else:
            endpoint = dict(lower=lower, upper=upper, maximum=2, total=32,
                            decision='supported' if upper <= 2 else 'falsified' if lower > 2 else 'inconclusive')
        endpoints[cohort] = endpoint
        for arm in ARMS:
            part = [r for r in rows if r['cohort'] == cohort and r['arm'] == arm]
            assert len(part) == 32
            grouped.append(dict(cohort=cohort, arm=arm, signals=sum(r['test']['signal'] is True for r in part),
                                no_signal=sum(r['test']['signal'] is False for r in part), unresolved=sum(r['test']['signal'] is None for r in part),
                                observed_exact=sum(r['observed_outcome']['classification'] == 'success' for r in part),
                                signals_on_nonexact=sum(r['test']['signal'] is True and r['observed_outcome']['classification'] in ['resolved_failure_nonexact', 'resolved_failure_fractional'] for r in part)))
    return dict(grouped=grouped, joint_banks=banks, endpoints=endpoints)


def audit_case(case, public=False):
    from signal_detection_study import concentration, randomization_test
    p = verify(not public)
    bank = case % 32
    directory = (PUBLIC if public else OUT)/f'c{case:02d}'
    masks = npz(directory/'masks.npz')
    assert sha(directory/'masks.npz') == sha(PARENT/f's{bank}/masks.npz')
    reference = reference_masks(bank)
    for arm in ARMS:
        assert np.array_equal(masks[arm], reference[arm]) and (masks[arm].sum(1) == 72).all()
    base = npz(directory/'base.npz')
    if case >= 32:
        expected_board = np.random.default_rng(p['null_seed_base']+bank).integers(0, 9, (180, 81), dtype=np.int8)
        assert np.array_equal(base['null_board'], expected_board)
    ii, jj = np.triu_indices(81, 1)
    truth = (ii//9 == jj//9) | (ii % 9 == jj % 9) | ((ii//27 == jj//27) & (ii % 9//3 == jj % 9//3))
    parent = json.loads((PARENT/'summary.json').read_text())
    rows = []
    provenance = []
    gate_labels = {}
    gate_expected = {}
    files = ['masks.npz', 'base.npz']
    for ai, arm in enumerate(ARMS):
        hist = base[arm+'_histogram']
        assert hist.dtype == np.int64 and hist.shape == (81, 9) and (hist >= 0).all() and np.array_equal(hist.sum(1), masks[arm].sum(0))
        if case < 32:
            original_path = RAW_PARENT/f's{bank}/{arm}-observations.npz'
            original_hash = p['raw_input_sha256'][str(original_path.relative_to(ROOT))]
            if not public:
                original = npz(original_path)
                assert np.array_equal(original['given'], masks[arm]) and np.array_equal(reference_histograms(masks[arm], original['values']), hist)
                original_counts = reference_observe(masks[arm], original['values'])
                for key, value in original_counts.items():
                    assert np.array_equal(value, original[key])
            provenance.append(dict(arm=arm, raw_observations_sha256=original_hash, histogram_bytes_sha256=hashlib.sha256(hist.tobytes()).hexdigest(),
                                   raw_histogram_status='inherited_raw_audit' if public else 'independently_validated'))
        else:
            assert np.array_equal(reference_histograms(masks[arm], np.where(masks[arm], base['null_board'], -1)), hist)
        checked_grids = []
        ratios = []
        observed_outcome = None
        for draw in range(32):
            if case < 32 and draw == 0:
                model_path = PARENT/f's{bank}/{arm}-models.npz'
                receipt_path = PARENT/f's{bank}/{arm}-receipt.json'
                arrays = npz(model_path)
                receipt = json.loads(receipt_path.read_text())
                index = 2*bank+ai
                assert receipt['index'] == index and receipt['seed'] == bank and receipt['arm'] == arm and receipt['protocol_sha256'] == p['parent_protocol_sha256']
                assert receipt['models_sha256'] == sha(model_path) and receipt['observations_sha256'] == original_hash
                if not public:
                    for key in ['joint_count', 'equal_count']:
                        assert np.array_equal(arrays[key], original_counts[key])
                config = p['parent_solver']
            else:
                location = directory/arm/f'b{draw:02d}'
                model_path = location/'models.npz'
                receipt_path = location/'receipt.json'
                arrays = npz(model_path)
                receipt = json.loads(receipt_path.read_text())
                index = 64*case+32*ai+draw
                assert receipt['case'] == case and receipt['bank'] == bank and receipt['arm'] == arm and receipt['draw'] == draw and receipt['index'] == index
                assert receipt['cohort'] == ('real' if case < 32 else 'null') and receipt['protocol_sha256'] == sha(PROTOCOL) and receipt['models_sha256'] == sha(model_path)
                if draw == 0:
                    labels = np.where(masks[arm], base['null_board'], -1).astype(np.int8)
                else:
                    seed = p['shuffle_seed_base']+case*62+ai*31+(draw-1)
                    labels = reference_shuffle(masks[arm], hist, seed)
                    assert np.array_equal(reference_histograms(masks[arm], labels), hist)
                expected = reference_observe(masks[arm], labels)
                assert receipt['labels_bytes_sha256'] == hashlib.sha256(expected['values'].tobytes()).hexdigest()
                for key in ['joint_count', 'equal_count']:
                    assert np.array_equal(arrays[key], expected[key])
                if draw == 1:
                    gate_labels[arm] = labels
                    gate_expected[arm] = expected
                if not public:
                    for record in receipt['records']:
                        assert record['log_sha256'] == sha(location/f'd{record["degree"]:02d}.log')
                files += [str(path.relative_to(directory)) for path in [model_path, receipt_path]]
                config = p['solver']
            assert len(receipt['records']) == 40
            cost = independent_rate_costs(arrays['joint_count'], arrays['equal_count'])
            assert np.array_equal(cost, arrays['cost'])
            checked = grid(arrays, receipt['records'], cost, config, index)
            ratio = reference_concentration(checked['objective_twice'])
            assert ratio == concentration(checked['objective_twice'])
            checked_grids.append(checked)
            ratios.append(ratio)
            if draw == 0:
                degree = checked['selected_degree']
                observed_outcome = statistics(np.empty(0, np.int64) if degree is None else arrays[f'd{degree:02d}_primal_twice'], truth)
                if case < 32:
                    prior = parent['rows'][2*bank+ai]
                    for key, value in (checked | observed_outcome).items():
                        assert prior[key] == value
        test = reference_test(ratios[0], ratios[1:])
        assert test == randomization_test(ratios[0], ratios[1:])
        rows.append(dict(case=case, bank=bank, cohort='real' if case < 32 else 'null', arm=arm, test=test,
                         observed_grid=checked_grids[0], shuffle_grids=checked_grids[1:], observed_ratio=ratios[0], shuffle_ratios=ratios[1:], observed_outcome=observed_outcome))
    gates = label_gate(masks, gate_labels, gate_expected)
    union = int((masks['uniform'] | masks['balanced']).sum())
    accounting = dict(shuffled_assignments=2*31*180*72, new_IID_generated_cells=180*81 if case >= 32 else 0,
                      new_IID_observed_union=union if case >= 32 else 0, historical_real_union_reused=union if case < 32 else 0)
    return dict(status='validated', case=case, protocol_sha256=sha(PROTOCOL), auditor_sha256=sha(__file__), rows=rows,
                complete_label_gates=gates, histogram_provenance=provenance, query_accounting=accounting,
                artifact_sha256={name: sha(directory/name) for name in files})


def public_job(case):
    return audit_case(case, True)


def public_audit():
    start = time.monotonic()
    p = verify()
    saved = json.loads((PUBLIC/'summary.json').read_text())
    assert saved['protocol_sha256'] == sha(PROTOCOL) and saved['query_accounting'] == p['query_accounting']
    for name, digest in saved['artifact_sha256'].items():
        assert sha(PUBLIC/name) == digest
    with ProcessPoolExecutor(max_workers=8) as pool:
        audited = list(pool.map(public_job, range(64)))
    rows = [row for case in audited for row in case['rows']]
    assert rows == saved['rows']
    outcomes = aggregate(rows)
    for key, value in outcomes.items():
        assert saved[key] == value
    all_grids = [g for row in rows for g in [row['observed_grid']]+row['shuffle_grids']]
    assert len(all_grids) == saved['analytic_zero_witnesses_checked'] == 4096
    assert saved['certified_LPs'] == sum(g['certified_grid_LPs'] for g in all_grids)
    assert saved['mutations_rejected'] == sum(g['mutations_rejected'] for g in all_grids)
    totals = {key: sum(a['query_accounting'][key] for a in audited) for key in audited[0]['query_accounting']}
    assert totals == dict(shuffled_assignments=51425280, new_IID_generated_cells=466560, new_IID_observed_union=460721, historical_real_union_reused=460721)
    assert totals == saved['literal_synthetic_accounting']
    for actual, original in zip(audited, saved['raw_audits'], strict=True):
        for field in ['case', 'rows', 'query_accounting', 'artifact_sha256', 'complete_label_gates']:
            assert actual[field] == original[field]
        for current, raw in zip(actual['histogram_provenance'], original['histogram_provenance'], strict=True):
            assert raw['raw_histogram_status'] == 'independently_validated'
            assert {k: v for k, v in current.items() if k != 'raw_histogram_status'} == {k: v for k, v in raw.items() if k != 'raw_histogram_status'}
    assert sum(len(a['complete_label_gates']) for a in audited) == 128
    print(json.dumps(dict(status='passed', **outcomes, certified_LPs=saved['certified_LPs'], mutations_rejected=saved['mutations_rejected'],
          wall_seconds=time.monotonic()-start, scope='Every new null board and all 3968 conditional shuffles, counts, rational p-values and 4096 grid proofs. Original real-label histogram provenance inherited from charged raw audit.')))


if __name__ == '__main__':
    public_audit()
