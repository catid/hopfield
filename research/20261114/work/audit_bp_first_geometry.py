"""Candidate independent all-output H2bn audit; no production decoder imports."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
from audit_connectivity_screen import path_valid
from verify_hard_negative_feasibility import valid_shortest
from verify_training_feasibility import flip, LD


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace_by_input(payload, arrays, owners, retain):
    """Independent per-owner transcript; no candidate normalizer import."""
    transcript = {int(owner): [] for owner in owners if int(owner) in retain}
    for step_index, step in enumerate(payload['trace']):
        decisions = set(step['chosen'])
        for local, owner in enumerate(owners):
            owner = int(owner)
            if owner not in retain:
                continue
            geometry = next((g for g in step['geometry'] if g['local_input'] == local), None)
            if geometry is None:
                continue
            costs = [(tuple(c['cells']), float(c['delta']), j in decisions)
                     for j, c in enumerate(step['candidates']) if c['local_input'] == local]
            native = next((r for r in step.get('native_results', []) if r['local_input'] == local), None)
            transcript[owner].append((step['round'], {k: v for k, v in geometry.items() if k != 'local_input'},
                costs, {k: v for k, v in native.items() if k != 'local_input'} if native else None,
                np.packbits(arrays['states_before'][step_index, local]).tobytes()))
    return transcript


def terminal_by_input(records, owners, retain):
    return {int(owners[r['local_input']]): {k: v for k, v in r.items() if k != 'local_input'}
            for r in records if int(owners[r['local_input']]) in retain}


def check_traces(actual_directory, row, source_directory, source_hashes, subset_indices, size):
    selected = set(map(int, subset_indices))
    expected_arms = ['connectivity_veto', 'original_lexical16', 'unified16', 'unified48', 'unified64', 'unified96']
    if not len(subset_indices):
        assert not row['trace_files']
        return 0, 0, 0.
    assert len(row['trace_files']) == 6
    candidates = active_states = 0
    maximum = 0.
    for arm, filename in zip(expected_arms, row['trace_files']):
        actual_path = actual_directory / filename
        assert sha(actual_path) == row['trace_sha256'][filename]
        source_path = source_directory / f'batch{row["batch"]:02d}-{arm}-trace.npz'
        assert sha(source_path) == source_hashes[source_path.name]
        with np.load(actual_path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        with np.load(source_path, allow_pickle=False) as z:
            b = {k: z[k] for k in z.files}
        ap, bp = json.loads(a['trace_utf8'].tobytes()), json.loads(b['trace_utf8'].tobytes())
        assert ap['arm'] == bp['arm'] == arm
        actual_owners = subset_indices[a['valid_prefix_indices']]
        original_owners = b['valid_prefix_indices']
        assert np.array_equal(actual_owners, original_owners[np.isin(original_owners, subset_indices)])
        assert np.array_equal(a['initial_terminal'], b['initial_terminal'][subset_indices])
        for field, ao, bo in [('initial_terminal_checks', subset_indices, np.arange(size)),
                              ('final_terminal_checks', actual_owners, original_owners)]:
            assert terminal_by_input(ap[field], ao, selected) == terminal_by_input(bp[field], bo, selected)
        aa = trace_by_input(ap, a, actual_owners, selected)
        bb = trace_by_input(bp, b, original_owners, selected)
        assert aa.keys() == bb.keys()
        for owner in aa:
            assert len(aa[owner]) == len(bb[owner])
            for x, y in zip(aa[owner], bb[owner]):
                assert x[0:2] == y[0:2] and x[3:] == y[3:]
                assert len(x[2]) == len(y[2])
                for c, d in zip(x[2], y[2]):
                    assert c[0] == d[0] and c[2] == d[2]
                    error = abs(c[1] - d[1])
                    assert np.isfinite(error) and error < 1e-8
                    maximum = max(maximum, error)
                    candidates += 1
                active_states += 1
        positions = np.flatnonzero(np.isin(original_owners, subset_indices))
        for field in ['pre_terminal_stopped', 'pre_terminal_block_cap']:
            expected = b[field][positions] if len(b[field]) else np.zeros(0, bool)
            assert np.array_equal(a[field], expected)
    return candidates, active_states, maximum


def one_shard(args):
    root, directory, p, shard, cohorts = args
    source = root / 'artifacts/research/20261114/work/loop80_confirmation'
    dataset = p['datasets'][shard]
    assert sha(root / dataset['path']) == dataset['sha256']
    with np.load(root / dataset['path'], allow_pickle=False) as z:
        tokens, labels, indices = z['tokens'], z['labels'], z['indices']
    assert np.array_equal(indices, np.arange(shard * 256, (shard + 1) * 256))
    assert all(valid_shortest(t, y) for t, y in zip(tokens, labels))
    shortest_cells = labels.sum((1, 2))
    total_cases = candidate_checks = trace_states = 0
    maximum_margin = maximum_score = 0.
    output_rows = []
    graph_cache = {}
    for seed in range(3):
        job = next(j for j in p['source_jobs'] if j['shard'] == shard and j['seed'] == seed)
        source_job = source / 'runs' / job['id']
        parameters = []
        for arm in ['pl', 'old_binary']:
            with np.load(source / f'model-seed{seed}-{arm}.npz', allow_pickle=False) as z:
                parameters.append((z['theta'].astype(LD), z['mu_offsets'], z['aoffs']))
        for bi in range(8):
            original = source_job / f'batch{bi:02d}.npz'
            assert sha(original) == job['batch_sha256'][original.name]
            with np.load(original, allow_pickle=False) as z:
                old = {k: z[k] for k in z.files}
            tt = tokens[bi * 32:(bi + 1) * 32]
            n = len(tt)
            for arm, (theta, mu, ao) in zip([0, 7], parameters):
                literal = np.where(tt == 1, flip(tt, old['predictions'][arm], old['couplings'].astype(LD), theta, mu, ao), np.inf).reshape(n, -1).min(1)
                error = float(np.max(np.abs(literal - old['minimum_flip_margin64'][arm])))
                assert error < 1e-9
                maximum_margin = max(maximum_margin, error)
            geometry = np.array([path_valid(t, b) for t, b in zip(tt, old['predictions'][0])])
            source_graph = []
            for arm in [0, 7]:
                values = []
                for i, (t, b) in enumerate(zip(tt, old['predictions'][arm])):
                    key = (bi * 32 + i, np.packbits(b).tobytes())
                    if key not in graph_cache:
                        graph_cache[key] = path_valid(t, b) and int(b.sum()) == int(shortest_cells[bi * 32 + i])
                    values.append(graph_cache[key])
                source_graph.append(np.array(values))
            for cohort in cohorts:
                rows = [r for r in cohort['rows'] if r['seed'] == seed and r['shard'] == shard and r['batch'] == bi]
                if not rows:
                    continue
                assert len(rows) == 1
                row = rows[0]
                actual_dir = root / cohort['directory']
                file = actual_dir / row['file']
                assert sha(file) == row['file_sha256']
                with np.load(file, allow_pickle=False) as z:
                    actual = {k: z[k] for k in z.files}
                mode = cohort['mode']
                ix = np.arange(n) if mode == 'LP96' else np.flatnonzero(~geometry)
                chosen = np.ones(n, np.int8) if mode == 'LP96' else (~geometry).astype(np.int8)
                assert np.array_equal(actual['answer_chosen_model'], chosen)
                assert np.array_equal(actual['answer_hidden_good'], old['hidden_good'])
                assert np.array_equal(actual['hidden_steps'], old['hidden_steps'])
                recorded_parity = row['source_parity']
                assert recorded_parity['status'] == 'passed'
                assert recorded_parity['source_batch_sha256'] == job['batch_sha256'][original.name]
                typed_hash = hashlib.sha256()
                typed_hash.update(old['couplings'].dtype.str.encode())
                typed_hash.update(json.dumps(list(old['couplings'].shape)).encode())
                typed_hash.update(np.ascontiguousarray(old['couplings']).tobytes())
                assert recorded_parity['actual_couplings_sha256'] == typed_hash.hexdigest()
                assert recorded_parity['full_model_inputs'] == n
                assert recorded_parity['actual_subset_inputs'] == (len(ix) if mode != 'LP96' else 0)
                if mode != 'LP96':
                    assert np.array_equal(actual['geometry_valid'], geometry)
                    assert np.array_equal(actual['subset_indices'], ix)
                    for name, oldname in [('prediction', 'predictions'), ('iterations', 'iterations'), ('stopped', 'stopped'), ('finite', 'finite')]:
                        assert np.array_equal(actual['bp_' + name], old[oldname][0])
                    assert np.array_equal(actual['bp_final_logits'], old['native_final_logits'][0])
                if len(ix):
                    for name, oldname in [('prediction', 'predictions'), ('iterations', 'iterations'), ('stopped', 'stopped'), ('finite', 'finite')]:
                        assert np.array_equal(actual['LP_' + name], old[oldname][1:8][:, ix])
                    for name, oldname in [('block_moves', 'block_moves'), ('post_block_sweeps', 'post_block_sweeps'), ('native_cap', 'block_native_cap'), ('block_cap', 'block_cap'), ('initial_simple_path', 'initial_simple_path')]:
                        assert np.array_equal(actual['LP_' + name], old[oldname][:, ix])
                    assert np.array_equal(actual['LP_binary_flip_count'], old['binary_flip_count'][0, ix])
                    assert np.max(np.abs(actual['LP_binary_cache_margin'] - old['binary_cache_margin'][0, ix])) < 1e-9
                arms = np.where(chosen == 1, 7, 0)
                for name, oldname in [('prediction', 'predictions'), ('iterations', 'iterations'), ('stopped', 'stopped'), ('finite', 'finite')]:
                    assert np.array_equal(actual['answer_' + name], old[oldname][arms, np.arange(n)])
                own_margin = old['minimum_flip_margin64'][arms, np.arange(n)]
                assert np.max(np.abs(actual['answer_own_margin'] - own_margin)) < 1e-9
                native_cap = np.where(chosen == 1, old['block_native_cap'][5], ~old['stopped'][0] & old['finite'][0])
                block_cap = np.where(chosen == 1, old['block_cap'][5], False)
                assert np.array_equal(actual['answer_native_cap'], native_cap)
                assert np.array_equal(actual['answer_block_cap'], block_cap)
                clamps = ((tt == 1) | (actual['answer_prediction'] == (tt >= 2))).all((1, 2))
                accepted = (old['hidden_good'] & actual['answer_stopped'] & actual['answer_finite']
                            & ~native_cap & ~block_cap & clamps & np.isfinite(own_margin) & (own_margin >= -1e-6))
                assert np.array_equal(actual['answer_accepted'], accepted)
                c, states, score = check_traces(actual_dir, row, source_job, job['batch_sha256'], ix, n)
                candidate_checks += c
                trace_states += states
                maximum_score = max(maximum_score, score)
                total_cases += n
                if cohort['kind'] == 'timing':
                    graph = np.where(chosen == 1, source_graph[1], source_graph[0])
                    bp_accept = old['hidden_good'] & old['finite'][0] & old['stopped'][0] & (old['minimum_flip_margin64'][0] >= -1e-6)
                    for i in range(n):
                        output_rows.append({'pair': cohort['pair'], 'mode': mode,
                            'input_index': shard * 256 + bi * 32 + i, 'model_seed': seed,
                            'chosen_model': int(chosen[i]), 'routed': int(mode != 'LP96' and not geometry[i]),
                            'graph_valid_shortest': int(graph[i]), 'numerically_accepted': int(accepted[i]),
                            'success': int(graph[i] and accepted[i]),
                            'BP_source_success': int(source_graph[0][i] and bp_accept[i])})
    return {'shard': shard, 'shortest_labels_checked': 256, 'source_final_margin_checks': 1536,
            'model_pipeline_copies_checked': total_cases, 'candidate_score_comparisons': candidate_checks,
            'active_trace_states': trace_states, 'maximum_margin_error': maximum_margin,
            'maximum_candidate_score_error': maximum_score, 'rows': output_rows}


def main(root, directory, workers):
    started = time.perf_counter()
    p = json.loads((directory / 'protocol.json').read_text())
    ph = sha(directory / 'protocol.json')
    sealed = json.loads((directory / 'sealed-replays.json').read_text())
    assert sealed['status'] == 'complete' and sealed['protocol_sha256'] == ph
    assert sealed['labels_or_shortestness_outcomes_joined'] is False
    assert len(sealed['cohort_and_worker_receipt_sha256']) == 26
    for name, digest in sealed['cohort_and_worker_receipt_sha256'].items():
        assert sha(directory / name) == digest, name
    assert workers == 16
    for name, digest in {**p['source_sha256'], **p['source_evidence_sha256']}.items():
        assert sha(root / name) == digest
    cohorts = []
    for gpu in [0, 1]:
        for kind in ['gate', 'timing']:
            w = json.loads((directory / f'{kind}-worker{gpu}.json').read_text())
            assert w['status'] == 'complete' and w['protocol_sha256'] == ph
            assert w['worker_wall_seconds'] <= (300 if kind == 'gate' else 1500)
            assert w['peak_cpu_rss_kib'] <= 8 * 1024**2
        for mode in ['LP96', 'BP_geometry_LP96']:
            for kind, folder in [('gate', 'source-gates'), ('warmup', 'warmup')]:
                path = directory / folder / f'gpu{gpu}-{mode}'
                r = json.loads((path / 'result.json').read_text())
                r.update(kind=kind, directory=str(path.relative_to(root)))
                cohorts.append(r)
    for pair in range(4):
        for mode in ['LP96', 'BP_geometry_LP96']:
            path = directory / 'runs' / f'pair{pair}-{mode}'
            r = json.loads((path / 'result.json').read_text())
            r.update(kind='timing', pair=pair, directory=str(path.relative_to(root)))
            cohorts.append(r)
    for r in cohorts:
        assert r['status'] == 'complete' and r['protocol_sha256'] == ph
        assert r['all_writes_closed_hashed_and_joined'] and r['archive_tasks_during_answer_segments'] == 0
        assert r['archive_writer_metrics']['maximum_pending'] <= 2 and r['archive_writer_metrics']['maximum_rss_kib'] <= 8 * 1024**2
        expected = {(seed, shard, bi) for seed in range(3) for shard in r['shards'] for bi in ([0] if r['only_first_batch'] else range(8))}
        assert {(x['seed'], x['shard'], x['batch']) for x in r['rows']} == expected and len(r['rows']) == len(expected)
        ready = {(x['seed'], x['model']) for x in r['model_readiness']}
        assert ready == {(seed, model) for seed in range(3) for model in (['lp'] if r['mode'] == 'LP96' else ['bp', 'lp'])}
        assert len(ready) == len(r['model_readiness'])
        assert all(np.isfinite(x['seconds']) and x['seconds'] > 0 for x in r['model_readiness'])
        assert abs(sum(x['seconds'] for x in r['model_readiness']) + r['readiness_orchestration_seconds'] - r['model_readiness_seconds']) < 1e-8
        assert r['readiness_orchestration_seconds'] >= 0
        barrier = r['host_barrier']
        events = barrier['events']
        required_stages = ['readiness-begin', 'readiness-done']
        for index in range(len(r['rows'])):
            required_stages += [f'batch{index:04d}-begin', f'batch{index:04d}-answer-done']
        required_stages += ['archives-done']
        assert [event['stage'] for event in events] == required_stages
        marker_times = {}
        command = 'gate' if r['kind'] == 'gate' else 'timing'
        for event in events:
            times = []
            for side, gpu in [('own', barrier['gpu']), ('peer', 1 - barrier['gpu'])]:
                file = root / barrier['directory'] / event[side + '_marker']
                assert sha(file) == event[side + '_sha256']
                marker = json.loads(file.read_text())
                assert marker['stage'] == event['stage'] and marker['gpu'] == gpu and marker['protocol_sha256'] == ph
                worker = json.loads((directory / f'{command}-worker{gpu}.json').read_text())
                assert marker['pid'] == worker['pid']
                assert np.isfinite(marker['arrived_monotonic'])
                times.append(marker['arrived_monotonic'])
            assert 0 <= event['wait_seconds'] <= 60.01
            marker_times[event['stage']] = times
        assert abs(sum(e['wait_seconds'] for e in events) - barrier['wait_seconds']) < 1e-8
        assert r['readiness_started_monotonic'] >= max(marker_times['readiness-begin'])
        assert r['readiness_finished_monotonic'] <= marker_times['readiness-done'][0]
        assert abs(r['readiness_finished_monotonic'] - r['readiness_started_monotonic'] - r['model_readiness_seconds']) < 1e-8
        costs = []
        for index, row in enumerate(r['rows']):
            t = row['timing']
            c = t['components']
            required = {'input_transfer', 'hidden_readout_and_flags', 'LP96_complete_prefix', 'LP96_answer_return', 'LP96_original_margin_and_return', 'final_numerical_acceptance'}
            required |= ({'BP_native_solve', 'BP_cap_flags', 'BP_answer_return', 'BP_original_margin_and_return', 'geometry_screen_and_indices', 'LP_subset_gather', 'subset_scatter_and_provenance'} if r['mode'] != 'LP96' else {'model_provenance'})
            assert required <= c.keys() and all(np.isfinite(v) and v >= 0 for v in c.values())
            assert t['n'] == 32 and 0 <= t['actual_subset_inputs'] <= 32
            if r['mode'] == 'BP_geometry_LP96':
                for key in ['LP96_complete_prefix', 'LP96_answer_return', 'LP96_original_margin_and_return']:
                    assert (c[key] == 0) == (t['actual_subset_inputs'] == 0)
            assert t['answer_seconds'] > 0 and t['unattributed_in_segment_seconds'] >= 0
            assert abs(sum(c.values()) + t['unattributed_in_segment_seconds'] - t['answer_seconds']) < 1e-8
            assert row['archive_tasks_at_answer_start'] == 0 and row['archive_drain_seconds_before_answer'] >= 0
            before = marker_times[f'batch{index:04d}-begin']
            after = marker_times[f'batch{index:04d}-answer-done']
            assert t['answer_started_monotonic'] >= max(before)
            assert t['answer_finished_monotonic'] <= after[0]
            assert t['diagnostic_started_monotonic'] >= max(after)
            assert abs(t['answer_finished_monotonic'] - t['answer_started_monotonic'] - t['answer_seconds']) < 1e-8
            assert abs(t['post_answer_barrier_wait_seconds'] - events[3 + 2 * index]['wait_seconds']) < 1e-8
            costs.append(t['answer_seconds'])
        assert abs(sum(costs) - r['batch_answer_seconds']) < 1e-8
        assert abs(sum(costs) + r['model_readiness_seconds'] - r['complete_answer_seconds']) < 1e-8
    shard_results = []
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        for r in pool.map(one_shard, [(root, directory, p, shard, cohorts) for shard in range(16)]):
            shard_results.append(r)
            partials = directory / 'audit-partials'
            partials.mkdir(exist_ok=True)
            (partials / f'shard{r["shard"]:02d}.json').write_text(json.dumps(r) + '\n')
            print(json.dumps({'event': 'cascade_shard_audited', 'shard': r['shard'], 'complete': len(shard_results)}), flush=True)
    except BaseException:
        # Only terminate processes owned by this exact executor, preserving all
        # unrelated host workloads and already-written partial audit receipts.
        for process in list(pool._processes.values()):
            if process.is_alive():
                process.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    rows = sorted([row for shard in shard_results for row in shard['rows']], key=lambda r: (r['pair'], r['mode'], r['input_index'], r['model_seed']))
    assert len(rows) == 4 * 2 * 12288
    comparisons = []
    pair_costs = []
    for pair in range(4):
        joint = {}
        model_counts = {}
        for mode in ['LP96', 'BP_geometry_LP96']:
            selected = [r for r in rows if r['pair'] == pair and r['mode'] == mode]
            byinput = {(r['input_index'], r['model_seed']): r for r in selected}
            assert len(byinput) == 12288
            joint[mode] = [all(byinput[i, seed]['success'] for seed in range(3)) for i in range(4096)]
            model_counts[mode] = sum(r['success'] for r in selected)
        assert sum(joint['LP96']) == 4095 and model_counts['LP96'] == 12285
        gains = sum(b and not a for a, b in zip(joint['LP96'], joint['BP_geometry_LP96']))
        losses = sum(a and not b for a, b in zip(joint['LP96'], joint['BP_geometry_LP96']))
        candidate_rows = [r for r in rows if r['pair'] == pair and r['mode'] == 'BP_geometry_LP96']
        bp_rows = {(r['input_index'], r['model_seed']): r for r in candidate_rows}
        bp_joint = sum(all(bp_rows[i, seed]['BP_source_success'] for seed in range(3)) for i in range(4096))
        bp_model = sum(r['BP_source_success'] for r in candidate_rows)
        assert bp_joint == 3917 and bp_model == 11778
        comparisons.append({'pair': pair, 'BP_executed_prefix_joint': bp_joint, 'BP_executed_prefix_model_successes': bp_model,
            'LP96_joint': 4095, 'candidate_joint': sum(joint['BP_geometry_LP96']),
            'model_successes': model_counts, 'gains': gains, 'losses': losses,
            'decision': 'supported' if losses == 0 else 'falsified'})
        a, b = [next(c for c in cohorts if c['kind'] == 'timing' and c['pair'] == pair and c['mode'] == mode)['complete_answer_seconds'] for mode in ['LP96', 'BP_geometry_LP96']]
        pair_costs.append({'pair': pair, 'LP96_seconds': a, 'candidate_seconds': b, 'saving_fraction': Fraction(1) - Fraction(b) / Fraction(a)})
    ordered = sorted(r['saving_fraction'] for r in pair_costs)
    median = (ordered[1] + ordered[2]) / 2
    for r in pair_costs:
        r['saving'] = float(r.pop('saving_fraction'))
    with (directory / 'per-model-input.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    elapsed = time.perf_counter() - started
    assert elapsed <= 600
    result = {'status': 'passed', 'hypothesis': 'H2bn', 'protocol_sha256': ph,
              'sealed_replays_sha256': sha(directory / 'sealed-replays.json'),
              'source_preservation': {'pairs': comparisons, 'decision': 'supported' if all(r['losses'] == 0 for r in comparisons) else 'falsified'},
              'cost_endpoint': {'pairs': pair_costs, 'median_saving': float(median), 'exact_median_fraction': str(median),
                  'minimum_saving': .10, 'decision': 'supported' if median >= Fraction(1, 10) else 'falsified'},
              'shortest_labels_checked': 4096, 'original_final_margin_checks': sum(r['source_final_margin_checks'] for r in shard_results),
              'model_pipeline_copies_checked': sum(r['model_pipeline_copies_checked'] for r in shard_results),
              'candidate_score_comparisons': sum(r['candidate_score_comparisons'] for r in shard_results),
              'active_trace_states': sum(r['active_trace_states'] for r in shard_results),
              'maximum_margin_error': max(r['maximum_margin_error'] for r in shard_results),
              'maximum_candidate_score_error': max(r['maximum_candidate_score_error'] for r in shard_results),
              'rows_sha256': sha(directory / 'per-model-input.csv'), 'seconds': elapsed, 'workers': workers,
              'source_sha256': sha(Path(__file__)), 'new_scientific_evaluations': 0,
              'scope': 'Complete source replays, original-factor margins, source-normalized actual subset trajectories and full answer-cost accounting. Source success preservation and finite paired timing endpoints only; no fresh accuracy/generalization claim.'}
    (directory / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    progress = args.directory / 'audit-worker.json'
    if progress.exists() or (args.directory / 'audit.json').exists() or (args.directory / 'per-model-input.csv').exists():
        raise FileExistsError('Refusing an existing independent audit attempt')
    began = time.perf_counter()
    state = {'status': 'running', 'pid': os.getpid(), 'wall_budget_seconds': 600, 'workers': args.workers}
    progress.write_text(json.dumps(state, indent=2) + '\n')
    def watchdog(signum, frame):
        raise TimeoutError('Frozen 600-second independent audit wall budget exhausted')
    signal.signal(signal.SIGALRM, watchdog)
    signal.setitimer(signal.ITIMER_REAL, 600)
    try:
        main(args.root, args.directory, args.workers)
    except BaseException as error:
        state.update(status='incomplete', error=repr(error), traceback=traceback.format_exc(),
                     wall_seconds=time.perf_counter() - began, partial_receipts_retained=True)
        progress.write_text(json.dumps(state, indent=2) + '\n')
        raise
    else:
        state.update(status='complete', wall_seconds=time.perf_counter() - began)
        progress.write_text(json.dumps(state, indent=2) + '\n')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
