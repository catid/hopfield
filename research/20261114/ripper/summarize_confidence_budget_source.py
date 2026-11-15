"""Frozen finite-bank structural hypotheses for half-budget confidence weighting."""
import csv
import hashlib
import json
import numpy as np
from run import ROOT, HERE, sha, save, now
from confidence_budget_common import paths, load_protocol


def main():
    out, pp = paths()
    p = load_protocol()
    source = json.loads((out/'input_audit.json').read_text())
    manifest = json.loads((out/'input_manifest.json').read_text())
    assert source['status'] == 'validated' and source['protocol_sha256'] == sha(pp)
    assert source['wrapper_sha256'] == sha(HERE/'audit_confidence_budget_inputs.py')
    assert source['calibration_training_overlap'] == source['within_bank_overlap'] == source['prior_bank_overlap'] == 0
    i, j = np.triu_indices(81, 1)
    peer = (i//9 == j//9) | (i%9 == j%9) | ((i//27 == j//27) & (i%9//3 == j%9//3))
    rows = []
    timing = []
    fits = []
    hashes = {}
    for seed in p['seeds']:
        d = out/f's{seed}'
        receipt = json.loads((d/'receipt.json').read_text())
        audit = json.loads((d/'audit.json').read_text())
        assert receipt['protocol_sha256'] == audit['protocol_sha256'] == sha(pp)
        assert receipt['wrapper_sha256'] == sha(HERE/'confidence_budget_acquire.py')
        assert audit['wrapper_sha256'] == sha(HERE/'audit_confidence_budget_acquisition.py')
        assert audit['status'] == 'validated' and audit['receipt_sha256'] == sha(d/'receipt.json')
        assert audit['independent_masks_replayed'] == audit['noise_rows_verified'] == 1110
        assert audit['log_lr_values_verified'] == 3596400 and audit['decimal_threshold_checks'] == 1111
        assert audit['first_crossing_paths_verified'] == 3240 and audit['weight_models_verified'] == 3
        assert audit['noise_cached_readonly_and_unchanged']
        for name, expected in receipt['artifact_sha256'].items():
            assert sha(d/name) == expected
        fit = json.loads((d/'calibration.json').read_text())['fit']
        assert fit == audit['calibration_fit']
        fits.append(dict(seed=seed, **fit))
        base = np.load(d/'fit.npz')
        support = base['support']
        assert np.array_equal(support, base['first_crossing'] < 0)
        weights = {}
        for model in receipt['models']:
            path = ROOT/model['path']
            assert sha(path) == model['sha256']
            arrays = np.load(path)
            weight = arrays['weight16']
            assert np.array_equal(arrays['support'], support) and np.array_equal(weight > 0, support)
            assert model['weight16_bytes_sha256'] == hashlib.sha256(weight.tobytes()).hexdigest()
            weights[model['arm']] = weight
            rows.append({'seed': seed, 'arm': model['arm'],
                         'missing_true_edges': int((peer & ~support).sum()),
                         'extra_edges': int((~peer & support).sum()),
                         'weak_true_edges': int((peer & (weight == 1)).sum()),
                         'weak_extra_edges': int((~peer & (weight == 1)).sum()),
                         'strong_true_edges': int((peer & (weight == 16)).sum()),
                         'strong_extra_edges': int((~peer & (weight == 16)).sum()),
                         'never_queried_pairs': int((base['joint_count'] == 0).sum()),
                         'weight16_bytes_sha256': model['weight16_bytes_sha256']})
        assert set(weights) == {'uniform', 'evidence', 'shuffled'}
        assert np.array_equal(np.sort(weights['evidence']), np.sort(weights['shuffled']))
        timing.append({'seed': seed, 'acquisition_s': receipt['elapsed_s'], 'replay_s': audit['elapsed_s']})
        hashes[str((d/'receipt.json').relative_to(ROOT))] = sha(d/'receipt.json')
        hashes[str((d/'audit.json').relative_to(ROOT))] = sha(d/'audit.json')
    totals = []
    for arm in p['arms']:
        part = [r for r in rows if r['arm'] == arm]
        total = {'arm': arm}
        for key in ('missing_true_edges', 'extra_edges', 'weak_true_edges', 'weak_extra_edges',
                    'strong_true_edges', 'strong_extra_edges', 'never_queried_pairs'):
            total[key] = sum(r[key] for r in part)
        totals.append(total)
    evidence = next(r for r in totals if r['arm'] == 'evidence')
    endpoints = {'all_true_edges_retained': evidence['missing_true_edges'] == 0,
                 'no_true_edges_weak': evidence['weak_true_edges'] == 0,
                 'at_least_95_percent_retained_extras_weak':
                 100*evidence['weak_extra_edges'] >= 95*evidence['extra_edges']}
    result = {'status': 'validated', 'utc': now(), 'study': p['id'],
              'decision': 'supported_finite_bank_conjunction' if all(endpoints.values()) else 'falsified_finite_bank_conjunction',
              'endpoints': endpoints, 'totals': totals, 'rows': rows, 'calibration_results': fits,
              'calibration_coverage_banks': sum(f['postfit_covers_actual_eta'] for f in fits),
              'calibration_fallback_banks': sum(f['no_pruning_fallback'] for f in fits),
              'retained_extra_edges_weak_pct': 100*evidence['weak_extra_edges']/evidence['extra_edges'] if evidence['extra_edges'] else None,
              'source_only': True, 'heldout_inputs_accessed': 0, 'gpu_seconds': 0,
              'bank_count': 32, 'queries_per_fit': 29750, 'input_audit': source,
              'input_manifest_sha256': sha(out/'input_manifest.json'),
              'all_source_manifest_sha256': sha(out/'all_source_manifest.json'),
              'future_excluded_source_union': manifest['excluded_unique_source_rows'] + source['independent_csv_answers_verified'],
              'masks_replayed': 35520, 'log_lr_values_verified': 115084800,
              'first_crossing_paths_verified': 103680, 'weight_models_verified': 96,
              'timings': timing, 'artifact_sha256': hashes, 'protocol_sha256': sha(pp),
              'summarizer_sha256': sha(__file__), 'limitations': p['limitations']}
    save(out/'source_audit.json', result)
    public = ROOT/'results/research/ripper'
    save(public/'confidence_budget_source.json', result)
    with (public/'confidence_budget_source_structure.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# H1ai: half-budget confidence weighting, source-only stage', '',
             f'The frozen finite-bank structural conjunction is **{result["decision"]}**.', '',
             'This stage uses no held-out puzzles, GPU search or accuracy outcomes. It tests the unchanged H1ah rule at actual 5% noise, M1000/N1110: 29,750 queries per shared fit.', '',
             '| Method | Missing true edges | Weak true edges | Retained extra edges | Weak extra edges | Strong extra edges |',
             '|---|---:|---:|---:|---:|---:|',
             '| BP AdamW: outside this study; not run | — | — | — | — | — |']
    for r in totals:
        lines.append(f'| {r["arm"]} | {r["missing_true_edges"]} | {r["weak_true_edges"]} | {r["extra_edges"]} | {r["weak_extra_edges"]} | {r["strong_extra_edges"]} |')
    lines += ['', 'Counts are pooled pair-bank observations across 32 fresh banks, not independent edge samples. Sudoku geometry enters only this postfit descriptive audit and never acquisition or weight placement.', '',
              f'Evidence weakens {result["retained_extra_edges_weak_pct"]:.4f}% of retained extra edges; the frozen target is at least 95%. The other frozen targets require zero missing true edges and zero weak true edges. No universal or simultaneous statistical guarantee is claimed.', '',
              f'Calibration coverage is {result["calibration_coverage_banks"]}/32, with {result["calibration_fallback_banks"]} safe fallbacks. All banks remain included irrespective of calibration coverage or structure.', '',
              f'Source audit verified {source["independent_csv_answers_verified"]:,} distinct complete boards, with calibration and training roles globally disjoint and all 1,729,093 previous source rows excluded. Independent replay checked 35,520 masks, 115,084,800 likelihood-ratio values, 103,680 first-crossing paths and 96 exact weight assignments.', '',
              'Strong weight 1, weak weight 1/16, exact rational final LR threshold 1, no mean normalization and the independent shuffled multiset control remain unchanged. No thresholds, source banks or future search settings are selected from these outcomes.', '',
              'A future held-out stage requires its own frozen cohort and execution registration. Its prospective targets remain at least 70% evidence canonical accuracy, at least 10 percentage-point gains against both uniform and shuffled weights, and zero invalid certified stops. Correct capped returns, certificates, canonical compatibility and given–given infeasibility must be reported separately. Source structure alone does not establish any of these accuracy targets.', '',
              '[Protocol](protocol-confidence-budget-source.json) · [Source results](../../../results/research/ripper/confidence_budget_source.json)', '']
    (HERE/'REPORT-confidence-budget-source.md').write_text('\n'.join(lines))
    print(json.dumps({k: result[k] for k in ('decision', 'endpoints', 'totals', 'calibration_coverage_banks', 'calibration_fallback_banks', 'retained_extra_edges_weak_pct', 'future_excluded_source_union')}), flush=True)


if __name__ == '__main__':
    main()
