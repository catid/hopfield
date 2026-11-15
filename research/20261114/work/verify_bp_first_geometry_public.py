"""Portable H2bn final-state/source and complete-cost checker."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from audit_connectivity_screen import path_valid
from verify_hard_negative_feasibility import valid_shortest
from verify_training_feasibility import flip, LD


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory, source):
    started = time.perf_counter()
    p = json.loads((directory / 'protocol.json').read_text())
    audit = json.loads((directory / 'audit.json').read_text())
    sealed = json.loads((directory / 'sealed-replays.json').read_text())
    ph = sha(directory / 'protocol.json')
    assert audit['status'] == 'passed' and sealed['status'] == 'complete'
    assert audit['protocol_sha256'] == sealed['protocol_sha256'] == ph
    assert audit['sealed_replays_sha256'] == sha(directory / 'sealed-replays.json')
    assert not sealed['labels_or_shortestness_outcomes_joined']
    for file, digest in sealed['cohort_and_worker_receipt_sha256'].items():
        assert sha(directory / file) == digest
    with (directory / 'per-model-input.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 98304 and sha(directory / 'per-model-input.csv') == audit['rows_sha256']
    bycase = {(int(r['pair']), r['mode'], int(r['input_index']), int(r['model_seed'])): r for r in rows}
    assert len(bycase) == 98304
    source_manifest = json.loads((source / 'manifest.json').read_text())
    source_audit = json.loads((source / 'source-audit.json').read_text())
    assert sha(source / 'source-audit.json') == source_manifest['source_audit_sha256']
    source_jobs = {r['job']['id']: r for r in source_audit['rows']}
    with (source.parent / 'per-model-input.csv').open() as stream:
        source_scalars = {(int(r['input_index']), int(r['model_seed'])): r for r in csv.DictReader(stream)}
    h2bl_audit = json.loads((source.parent / 'audit.json').read_text())
    assert sha(source.parent / 'per-model-input.csv') == h2bl_audit['rows_sha256']
    for row in rows:
        i, seed = int(row['input_index']), int(row['model_seed'])
        original = source_scalars[i, seed]
        lp_selected = row['mode'] == 'LP96' or original['graph_valid_path'] == '0'
        assert int(row['chosen_model']) == int(lp_selected)
        assert int(row['routed']) == int(row['mode'] != 'LP96' and lp_selected)
        source_job = source_jobs[f'shard{i // 256:02d}-s{seed}']
        assert int(row['success']) == source_job['success'][7 if lp_selected else 0][i % 256]
        assert int(row['BP_source_success']) == source_job['success'][0][i % 256]
        assert int(row['success']) == (int(row['graph_valid_shortest']) & int(row['numerically_accepted']))
    for pair in range(4):
        control = []
        candidate = []
        for i in range(4096):
            control.append(all(int(bycase[pair, 'LP96', i, seed]['success']) for seed in range(3)))
            candidate.append(all(int(bycase[pair, 'BP_geometry_LP96', i, seed]['success']) for seed in range(3)))
        r = audit['source_preservation']['pairs'][pair]
        assert sum(control) == r['LP96_joint'] == 4095
        assert sum(candidate) == r['candidate_joint']
        assert sum(a and not b for a, b in zip(control, candidate)) == r['losses']
        assert sum(b and not a for a, b in zip(control, candidate)) == r['gains']
        assert r['decision'] == ('supported' if r['losses'] == 0 else 'falsified')
    costs = []
    barrier_checks = 0
    marker_bytes = json.loads((directory / 'barrier-markers.json').read_text())['markers']
    for pair in range(4):
        sums = {}
        for mode in ['LP96', 'BP_geometry_LP96']:
            cohort = json.loads((directory / 'runs' / f'pair{pair}-{mode}' / 'result.json').read_text())
            assert cohort['model_inputs'] == 12288 and len(cohort['rows']) == 384
            ready = cohort['model_readiness']
            assert {(r['seed'], r['model']) for r in ready} == {(s, m) for s in range(3) for m in (['lp'] if mode == 'LP96' else ['bp', 'lp'])}
            assert abs(sum(r['seconds'] for r in ready) + cohort['readiness_orchestration_seconds'] - cohort['model_readiness_seconds']) < 1e-8
            assert cohort['readiness_orchestration_seconds'] >= 0
            barrier = cohort['host_barrier']
            base = Path(barrier['directory']).relative_to('artifacts/research/20261114/work/bp_first_geometry_cascade')
            event_map = {}
            for e in barrier['events']:
                times = []
                for side in ['own', 'peer']:
                    raw = marker_bytes[str(base / e[side + '_marker'])]
                    assert hashlib.sha256(raw.encode()).hexdigest() == e[side + '_sha256']
                    v = json.loads(raw)
                    assert v['protocol_sha256'] == ph and v['stage'] == e['stage']
                    times.append(v['arrived_monotonic'])
                event_map[e['stage']] = times
                assert 0 <= e['wait_seconds'] <= 60.01
                barrier_checks += 1
            assert cohort['readiness_started_monotonic'] >= max(event_map['readiness-begin'])
            assert cohort['readiness_finished_monotonic'] <= event_map['readiness-done'][0]
            for index, r in enumerate(cohort['rows']):
                t = r['timing']
                c = t['components']
                required = {'input_transfer', 'hidden_readout_and_flags', 'LP96_complete_prefix', 'LP96_answer_return', 'LP96_original_margin_and_return', 'final_numerical_acceptance'}
                required |= {'model_provenance'} if mode == 'LP96' else {'BP_native_solve', 'BP_cap_flags', 'BP_answer_return', 'BP_original_margin_and_return', 'geometry_screen_and_indices', 'LP_subset_gather', 'subset_scatter_and_provenance'}
                assert required <= c.keys() and all(np.isfinite(v) and v >= 0 for v in c.values())
                assert abs(sum(c.values()) + t['unattributed_in_segment_seconds'] - t['answer_seconds']) < 1e-8
                assert t['unattributed_in_segment_seconds'] >= 0 and r['archive_tasks_at_answer_start'] == 0
                assert t['answer_started_monotonic'] >= max(event_map[f'batch{index:04d}-begin'])
                assert t['answer_finished_monotonic'] <= event_map[f'batch{index:04d}-answer-done'][0]
                assert t['diagnostic_started_monotonic'] >= max(event_map[f'batch{index:04d}-answer-done'])
            total = cohort['model_readiness_seconds'] + sum(r['timing']['answer_seconds'] for r in cohort['rows'])
            assert abs(total - cohort['complete_answer_seconds']) < 1e-8
            sums[mode] = total
        costs.append(Fraction(1) - Fraction(sums['BP_geometry_LP96']) / Fraction(sums['LP96']))
    ordered = sorted(costs)
    median = (ordered[1] + ordered[2]) / 2
    assert str(median) == audit['cost_endpoint']['exact_median_fraction']
    assert audit['cost_endpoint']['decision'] == ('supported' if median >= Fraction(1, 10) else 'falsified')
    d = directory / 'state-fixtures'
    m = json.loads((d / 'manifest.json').read_text())
    original = json.loads((source / 'manifest.json').read_text())
    assert sha(source / 'manifest.json') == m['source_fixture_manifest_sha256']
    original_cases = {(r['input_index'], r['model_seed']): r for r in original['fixed_cases'] + original['negative_or_retention_loss_cases']}
    fixed = {(shard * 256, seed) for shard in range(16) for seed in range(3)}
    routed = {(int(r['input_index']), int(r['model_seed'])) for r in rows if r['mode'] == 'BP_geometry_LP96' and r['routed'] == '1'}
    failures = {(int(r['input_index']), int(r['model_seed'])) for r in rows if r['mode'] == 'BP_geometry_LP96' and r['success'] == '0'}
    selected = {(r['input_index'], r['model_seed']) for r in m['records']}
    assert selected == fixed | routed | failures
    assert len(routed) == m['complete_routed_cases'] and len(failures) == m['complete_candidate_failure_cases']
    checked = 0
    max_margin_error = 0.
    graphs = {}
    labels = set()
    for record in m['records']:
        key = (record['input_index'], record['model_seed'])
        old = original_cases[key]
        assert old['file'] == record['original_fixture'] and old['sha256'] == record['original_fixture_sha256']
        file = source / old['file']
        assert sha(file) == old['sha256']
        with np.load(file, allow_pickle=False) as z:
            frozen = {k: z[k] for k in ['tokens', 'board', 'labels', 'couplings', 'theta', 'mu_offsets', 'aoffs']}
        t = frozen['tokens']
        token_hash = hashlib.sha256(t.tobytes()).hexdigest()
        if token_hash not in labels:
            assert valid_shortest(t[0], frozen['labels'][0])
            labels.add(token_hash)
        bp_geometry = path_valid(t[0], frozen['board'][0])
        assert record['actually_routed'] == (not bp_geometry)
        file = d / record['file']
        assert sha(file) == record['sha256']
        with np.load(file, allow_pickle=False) as z:
            actual = {k: z[k] for k in z.files}
        for pair in range(4):
            for mode, prefix in [('LP96', 'control'), ('BP_geometry_LP96', 'candidate')]:
                row = bycase[pair, mode, *key]
                board = actual[prefix + '_answer_prediction'][pair]
                model = int(actual[prefix + '_answer_chosen_model'][pair])
                assert model == (1 if mode == 'LP96' else int(not bp_geometry)) == int(row['chosen_model'])
                if model == 0:
                    assert np.array_equal(board, frozen['board'][0])
                elif mode != 'LP96':
                    assert np.array_equal(board, actual['control_answer_prediction'][pair])
                delta = np.where(t == 1, flip(t, board[None], frozen['couplings'].astype(LD),
                    frozen['theta'][model].astype(LD), frozen['mu_offsets'], frozen['aoffs']), np.inf)
                margin = float(delta.min())
                err = abs(margin - actual[prefix + '_answer_own_margin'][pair])
                assert err < 1e-9
                max_margin_error = max(max_margin_error, err)
                clamps = bool(np.all((t[0] == 1) | (board == (t[0] >= 2))))
                accepted = bool(actual[prefix + '_answer_hidden_good'][pair] and actual[prefix + '_answer_finite'][pair]
                    and actual[prefix + '_answer_stopped'][pair] and not actual[prefix + '_answer_native_cap'][pair]
                    and not actual[prefix + '_answer_block_cap'][pair] and clamps and np.isfinite(margin) and margin >= -1e-6)
                assert accepted == bool(actual[prefix + '_answer_accepted'][pair]) == bool(int(row['numerically_accepted']))
                graph_key = (token_hash, np.packbits(board).tobytes())
                if graph_key not in graphs:
                    graphs[graph_key] = valid_shortest(t[0], board)
                assert graphs[graph_key] == bool(int(row['graph_valid_shortest']))
                assert (accepted and graphs[graph_key]) == bool(int(row['success']))
                checked += 1
    return {'status': 'passed', 'scalar_model_input_arm_copies': 98304,
            'source_preservation': audit['source_preservation'], 'cost_endpoint': audit['cost_endpoint'],
            'complete_routed_fixture_cases': len(routed), 'candidate_failure_fixture_cases': len(failures),
            'fixed_fixture_cases': len(fixed), 'selected_fixture_cases': len(selected),
            'actual_final_state_checks': checked, 'maximum_original_margin_error': max_margin_error,
            'host_barrier_events_verified': barrier_checks, 'seconds': time.perf_counter() - started,
            'source_sha256': sha(Path(__file__)), 'protocol_sha256': ph,
            'scope': 'Complete public scalar/source comparisons, synchronized complete answer accounting and all routed/failure plus fixed final-state fixtures. Full internal solver trajectory audit is inherited from Work and bound by receipts; not reconstructed here. Zero fresh accuracy or population evidence.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.source), indent=2))
