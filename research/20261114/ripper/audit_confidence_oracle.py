"""Full saved-board audit and three separate frozen oracle mechanism endpoints."""
import csv
import datetime
import hashlib
import json
import numpy as np
import torch
from run import ROOT, HERE, sha, save, now
from audit_clues import validity
from bootstrap_two_way import resample_means
from confidence_oracle_common import OUT, PROTOCOL, load_protocol


def main():
    p = load_protocol()
    manifest = json.loads((OUT/'models.json').read_text())
    model_audit = json.loads((OUT/'model_audit.json').read_text())
    mapping = json.loads((OUT/'search_mapping.json').read_text())
    queue = json.loads((OUT/'queue.json').read_text())
    assert all(job['status'] == 'complete' for job in queue)
    assert len(queue) == mapping['new_unique_searches'] and len(mapping['rows']) == 128
    assert mapping['protocol_sha256'] == sha(PROTOCOL) and mapping['wrapper_sha256'] == sha(HERE/'confidence_oracle_search.py')
    assert mapping['test_data_sha256'] == p['test_data_sha256']
    validation = json.loads((ROOT/'artifacts/research/20261114/ripper/bootstrap_validation.json').read_text())
    assert validation['status'] == 'validated' and validation['helper_sha256'] == sha(HERE/'bootstrap_two_way.py')
    prior = ROOT/'artifacts/research/20261114/ripper/confidence_budget'
    previous = json.loads((prior/'audit.json').read_text())
    assert previous['status'] == 'validated'
    models = {}
    for entry in manifest['models']:
        path = ROOT/entry['path']
        assert sha(path) == entry['sha256']
        z = np.load(path)
        weight = z['weight16']
        assert np.array_equal(z['support'], weight > 0)
        assert hashlib.sha256(weight.tobytes()).hexdigest() == entry['weight16_bytes_sha256']
        models[entry['arm'], entry['seed']] = weight
    datafile = ROOT/'upstream/data/full-test/full-all.npz'
    assert sha(datafile) == p['test_data_sha256']
    data = np.load(datafile)
    lo, hi = p['rows_half_open']
    target = data['solutions'][lo:hi].astype(np.int8)-1
    free = ~data['clue_mask'][lo:hi].astype(bool)
    assert len(target) == 5000
    i, j = np.triu_indices(81, 1)
    true = (i//9 == j//9) | (i%9 == j%9) | ((i//27 == j//27) & (i%9//3 == j%9//3))
    seen = {}
    actual = []
    hashes = {}
    for entry in mapping['rows']:
        directory = entry['canonical_directory']
        key = entry['arm'], entry['seed']
        weight = models[key]
        assert hashlib.sha256(weight.tobytes()).hexdigest() == entry['weight16_bytes_sha256']
        assert np.all(weight[true] > 0)
        if directory in seen:
            assert np.array_equal(weight, seen[directory]['weight']) and entry['seed'] == seen[directory]['seed']
            continue
        d = ROOT/directory
        receipt = json.loads((d/'receipt.json').read_text())
        is_new = entry['canonical_kind'] == 'new'
        assert receipt['status'] == 'complete' and receipt['seed'] == entry['seed']
        assert receipt['protocol_sha256'] == (sha(PROTOCOL) if is_new else p['prior_inference_protocol_sha256'])
        expected_wrapper = 'confidence_oracle_search.py' if is_new else 'confidence_budget_search.py'
        assert receipt['wrapper_sha256'] == sha(HERE/expected_wrapper)
        assert receipt['backend_sha256'] == p['backend_sha256'] and receipt['capture_sha256'] == p['capture_sha256']
        assert receipt['integer_audit_sha256'] == p['integer_audit_sha256']
        assert receipt['rows_half_open'] == [400000,405000] and receipt['test_rows'] == 5000
        assert receipt['move_cap'] == 262144 and receipt['used_unit'] == 1 and receipt['tau'] == receipt['ebar'] == 0
        assert receipt['preflight_state_comparisons'] == 150 and receipt['hidden_target_state_comparisons'] == 132 and receipt['integer_field_checks'] == 6
        expected = dict(temp=.4, theta=.45, focused=True, C=4, tau=0., tol=.001, ebar=0.,
                        cap=4e11, check=4096, seed=entry['seed'], anneal=[.4,.31,524288])
        assert receipt['search_kwargs'] == expected
        assert receipt['weight16_bytes_sha256'] == hashlib.sha256(weight.tobytes()).hexdigest()
        for name, digest in receipt['output_sha256'].items():
            assert sha(d/name) == digest
        if not is_new:
            relative = str((d/'receipt.json').relative_to(ROOT))
            assert sha(d/'receipt.json') == previous['artifact_sha256'][relative]
        searched = torch.load(d/'searched_model.pt', weights_only=True, map_location='cpu')
        assert torch.equal(searched['W'], torch.from_numpy(weight.astype(np.float32)/16)[:,None,None]*torch.eye(9))
        assert not searched['B'].any() and np.array_equal(searched['weight16'].numpy(), weight)
        z = np.load(d/'boards.npz')
        boards, stopped, moves = z['boards'], z['stopped'], z['moves']
        assert np.array_equal(z['solutions'], target) and np.array_equal(z['free'], free)
        assert np.array_equal(z['puzzle'], np.arange(lo, hi)) and np.array_equal(boards[~free], target[~free])
        exact = (boards == target).all(1)
        valid = validity(boards, target, free)
        assert np.array_equal(exact, z['exact']) and np.array_equal(valid, z['valid'])
        assert int(exact.sum()) == receipt['solved'] and int(valid.sum()) == receipt['valid']
        assert int((~stopped).sum()) == receipt['budget_failures'] and int((stopped & ~valid).sum()) == receipt['wrong_stops']
        assert np.all((moves > 0) & (moves <= 262144))
        per = json.loads((d/'result.json').read_text())['per_puzzle']
        assert np.array_equal(per['solved'], exact) and np.array_equal(per['stopped'], stopped)
        assert np.array_equal(per['flops'], moves*receipt['fl_move']*4)
        support = weight > 0
        ii, jj, ww = i[support], j[support], weight[support].astype(np.int64)
        target_conflicts = target[:,ii] == target[:,jj]
        target_energy = (target_conflicts*ww).sum(1)
        returned_energy = ((boards[:,ii] == boards[:,jj])*ww).sum(1)
        given_conflicts = (target_conflicts & ~free[:,ii] & ~free[:,jj]).sum(1)
        assert np.all(~stopped | (returned_energy == 0))
        assert not (stopped & (given_conflicts > 0)).any()
        assert not (stopped & ~valid).any(), 'All true edges are positive, so a zero-energy invalid certificate is an implementation failure'
        compatible = target_energy == 0
        stats = {'canonical_compatible': int(compatible.sum()), 'given_given_infeasible': int((given_conflicts > 0).sum()),
                 'certified_returns': int(stopped.sum()), 'correct_certified': int((stopped & exact).sum()),
                 'correct_capped': int((~stopped & exact).sum()),
                 'incompatible_correct_capped': int((~compatible & ~stopped & exact).sum()),
                 'compatible_target_misses': int((compatible & ~exact).sum()),
                 'wrong_stops': int((stopped & ~valid).sum()), 'invalid_capped': int((~stopped & ~valid).sum()),
                 'noncanonical_valid_returns': int((valid & ~exact).sum()),
                 'noncanonical_valid_certificates': int((stopped & valid & ~exact).sum()),
                 'mean_canonical_energy': float(target_energy.mean()/16)}
        seen[directory] = {'weight': weight, 'seed': entry['seed'], 'exact': exact, 'stats': stats}
        actual.append({'directory': directory, 'new_search': is_new, 'arm': receipt['arm'],
                       'seed': receipt['seed'], 'solved': int(exact.sum()), 'wall_s': receipt['wall_s'], **stats})
        hashes[str((d/'receipt.json').relative_to(ROOT))] = sha(d/'receipt.json')
    arrays = {}
    diagnostics = []
    for entry in mapping['rows']:
        observed = seen[entry['canonical_directory']]
        arrays[entry['arm'], entry['seed']] = observed['exact']
        diagnostics.append({'arm': entry['arm'], 'seed': entry['seed'], 'reused': entry['reused'], **observed['stats']})
    grouped = []
    for arm in p['arms']:
        solved = [int(arrays[arm, seed].sum()) for seed in p['seeds']]
        part = [row for row in diagnostics if row['arm'] == arm]
        group = {'arm': arm, 'mean_accuracy_pct': float(np.mean(solved)/50),
                 'solved_per_seed': solved, 'banks': 32, 'n_per_bank': 5000}
        for key in part[0]:
            if key in ('arm', 'seed', 'reused'):
                continue
            if key == 'mean_canonical_energy':
                group[key] = float(np.mean([row[key] for row in part]))
            else:
                group[key] = sum(row[key] for row in part)
                group[key+'_pct'] = group[key]/1600
        grouped.append(group)
    endpoints = []
    for arm in p['arms'][1:]:
        diff = np.array([arrays[arm, seed].astype(float)-arrays['base', seed] for seed in p['seeds']])
        boot = resample_means(diff, p['bootstrap_seed'], 5000)
        lower, upper = map(float, np.quantile(boot, [.025,.975]))
        if arm == 'repair_true':
            decision = 'supported_5pp_equivalence' if lower >= -5 and upper <= 5 else 'falsified_5pp_equivalence' if upper < -5 or lower > 5 else 'inconclusive'
            criterion = '95% interval entirely within [-5,+5] pp; falsified if entirely outside; otherwise inconclusive'
        else:
            decision = 'supported_10pp_gain' if lower >= 10 else 'falsified_10pp_gain' if upper < 10 else 'inconclusive'
            criterion = '95% lower endpoint >=10 pp; falsified if upper <10 pp; otherwise inconclusive'
        endpoints.append({'arm': arm, 'reference': 'base', 'effect_pp': float(100*diff.mean()),
                          'paired_two_way_ci95_pp': [lower, upper], 'decision': decision, 'criterion': criterion})
    stamp = datetime.datetime.fromisoformat
    span = (max(stamp(job['completed_utc']) for job in queue)-min(stamp(job['started_utc']) for job in queue)).total_seconds()
    gpu_seconds = 4*span
    assert gpu_seconds < 1800
    wrong = sum(row['wrong_stops'] for row in actual)
    certificates = sum(row['certified_returns'] for row in actual)
    report = {'status': 'validated', 'utc': now(), 'endpoints': endpoints, 'grouped': grouped,
              'diagnostics': diagnostics, 'actual_searches': actual, 'models_audit': model_audit,
              'search_mapping': mapping, 'requested_cells': 128, 'new_unique_searches': len(queue),
              'archived_reuse_cells': mapping['archived_reuse_cells'],
              'unique_archived_searches': sum(not row['new_search'] for row in actual),
              'saved_boards_verified': len(actual)*5000, 'new_saved_boards_verified': len(queue)*5000,
              'wrong_certified_stops': wrong, 'certified_returns_in_unique_searches': certificates,
              'zero_wrong_stop_gate': wrong == 0, 'zero_wrong_stop_gate_vacuous': certificates == 0,
              'total_allocated_gpu_s': gpu_seconds, 'new_queries': 0, 'new_test_inputs': 0,
              'distinct_reused_test_inputs': 5000, 'bootstrap_draws': 5000, 'bootstrap_seed': p['bootstrap_seed'],
              'interval_scope': 'Three separate marginal 95% intervals; no simultaneous or selection-adjusted claim.',
              'source_test_overlap': previous['source_test_completeboard_overlap'],
              'artifact_sha256': hashes, 'protocol_sha256': sha(PROTOCOL), 'audit_sha256': sha(__file__),
              'limitations': p['limitations']}
    save(OUT/'audit.json', report)
    public = ROOT/'results/research/ripper'
    save(public/'confidence_oracle.json', report)
    for name, entries in [('confidence_oracle_per_seed.csv', diagnostics), ('confidence_oracle_models.csv', model_audit['rows'])]:
        with (public/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
    lines = ['# H1aj: oracle diagnosis of the half-budget boundary', '',
             'This is a separately frozen post-hoc mechanism experiment on the same already-seen H1ai banks and 5,000 puzzles. Its interventions use known Sudoku geometry. It is not a deployable learner or a fresh confirmation.', '',
             'The three frozen endpoints are evaluated separately; no conjunction or joint dominance claim is made.', '']
    for endpoint in endpoints:
        lower, upper = endpoint['paired_two_way_ci95_pp']
        lines.append(f'- {endpoint["arm"]}: **{endpoint["decision"]}**, gain {endpoint["effect_pp"]:.4f} pp, marginal paired two-way 95% interval [{lower:.4f}, {upper:.4f}] pp.')
    lines += ['', '| Oracle intervention | Canonical accuracy (%) | Correct certified (%) | Correct capped (%) | Canonical compatibility (%) | Given–given infeasibility (%) | Invalid certificates |',
              '|---|---:|---:|---:|---:|---:|---:|', '| BP AdamW: outside this study; not run | — | — | — | — | — | — |']
    for group in grouped:
        lines.append(f'| {group["arm"]} | {group["mean_accuracy_pct"]:.4f} | {group["correct_certified_pct"]:.4f} | {group["correct_capped_pct"]:.4f} | {group["canonical_compatible_pct"]:.4f} | {group["given_given_infeasible_pct"]:.4f} | {group["wrong_stops"]} |')
    lines += ['', 'The repair arm restores exactly 82 evidence-weak true edge-bank pairs from 1/16 to 1. The softening arm changes exactly 503 strong extra edge-bank pairs from 1 to 1/16. The deletion arm removes exactly 19,445 weak extra edge-bank pairs. An independent row/column/box construction exhaustively checked all 414,720 model weights and verified zero true-edge deletions.', '',
              'Repair and softening preserve the positive support and zero set. Deleting weak extras expands the zero set and may increase canonical compatibility. All 810 true Sudoku edges per bank remain positive in every arm, so exact zero-energy returns remain sound. These claims do not bound positive-energy capped-return accuracy or establish solution uniqueness.', '',
              f'Observed invalid certificates: {wrong}; unique-search certificates: {certificates}. The zero-invalid gate is {"vacuous because no certificates occurred" if certificates == 0 else "nonvacuous"}.', '',
              'All arms retain the same unit1, temperature .4→.31, theta .45, tau0/ebar0, four chains, 262,144-move cap and paired search seeds. Archived baseline trajectories and any identical complete weight/input/settings cells are reused exactly. No new labels, source banks or held-out cohort were consumed.', '',
              f'Of 128 requested cells, {len(queue)} require new searches; {mapping["archived_reuse_cells"]} cells reuse archived baseline results. All {len(actual)*5000:,} unique returned boards were independently audited, including {len(queue)*5000:,} new boards. New GPU allocation is {gpu_seconds/60:.4f} GPU-min. Every new search passed 150 general/sparse, 132 hidden-target and six independent integer field checks, including 127 unrefreshed moves.', '',
              'All three intervals use 5,000 paired two-way resamples over 32 banks and the same 5,000 puzzle indices. They are marginal, approximate, and not adjusted for choosing this mechanism study after H1ai. Earlier study results are not pooled as new evidence.', '',
              '[Protocol](protocol-confidence-oracle.json) · [Oracle policy](confidence_oracle_policy.py) · [Model audit](audit_confidence_oracle_models.py) · [Results](../../../results/research/ripper/confidence_oracle.json)', '']
    (HERE/'REPORT-confidence-oracle.md').write_text('\n'.join(lines))
    print(json.dumps({'endpoints': endpoints, 'grouped': grouped, 'new_unique_searches': len(queue),
                      'archived_reuse_cells': mapping['archived_reuse_cells'], 'gpu_seconds': gpu_seconds,
                      'certified_returns': certificates, 'wrong_stops': wrong}), flush=True)


if __name__ == '__main__':
    main()
