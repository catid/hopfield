"""Independent fresh full-cohort repetition, actual subset and graph outcomes."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from audit_connectivity_screen import path_valid
from verify_hard_negative_feasibility import valid_shortest
from audit_bp_first_fresh_states import numeric_error
from verify_bp_first_fresh_replay import check_trace_pair


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    a = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(a.dtype.str.encode())
    digest.update(json.dumps(list(a.shape)).encode())
    digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def one_shard(args):
    root, directory, p, datasets, shard, cohorts = args
    started = time.perf_counter()
    ds = datasets[shard]
    assert sha(root / ds['path']) == ds['sha256']
    with np.load(root / ds['path'], allow_pickle=False) as z:
        tokens, labels, indices = z['tokens'], z['labels'], z['indices']
    assert np.array_equal(indices, np.arange(shard * 256, (shard + 1) * 256))
    assert all(valid_shortest(t, b) for t, b in zip(tokens, labels))
    shortest_cells = labels.sum((1, 2))
    records = {(r['pair'], r['mode']): r for r in cohorts if r['kind'] == 'timing'}
    outputs = []
    graph_cache = {}
    candidate_scores = trace_states = repeated_cases = 0
    maximum_score_error = 0.
    for seed in range(3):
        job_id = f'shard{shard:02d}-s{seed}'
        for bi in range(8):
            tt = tokens[bi * 32:(bi + 1) * 32]
            source = {}
            reference_rows = {}
            for mode in ['LP96', 'BP_geometry_LP96']:
                receipt = records[0, mode]
                row = next(r for r in receipt['rows'] if r['job_id'] == job_id and r['batch'] == bi)
                file = root / receipt['directory'] / row['file']
                assert sha(file) == row['file_sha256']
                with np.load(file, allow_pickle=False) as z:
                    source[mode] = {k: z[k] for k in z.files}
                reference_rows[mode] = row
            lp, bp = source['LP96'], source['BP_geometry_LP96']
            assert np.array_equal(lp['couplings'], bp['couplings'], equal_nan=True)
            assert np.array_equal(lp['hidden_steps'], bp['hidden_steps'])
            assert np.array_equal(lp['answer_hidden_good'], bp['answer_hidden_good'])
            geometry = np.array([path_valid(t, b) for t, b in zip(tt, bp['bp_prediction'])])
            bp_graph, lp_graph = [], []
            for i in range(32):
                for board, target in [(bp['bp_prediction'][i], bp_graph), (lp['answer_prediction'][i], lp_graph)]:
                    key = (bi * 32 + i, np.packbits(board).tobytes())
                    if key not in graph_cache:
                        graph_cache[key] = path_valid(tt[i], board) and int(board.sum()) == int(shortest_cells[bi * 32 + i])
                    target.append(graph_cache[key])
            bp_graph, lp_graph = np.array(bp_graph), np.array(lp_graph)
            bp_clamps = ((tt == 1) | (bp['bp_prediction'] == (tt >= 2))).all((1, 2))
            bp_accept = (bp['answer_hidden_good'] & bp['bp_finite'] & bp['bp_stopped'] & bp_clamps
                         & ~bp['bp_native_cap'] & ~bp['bp_block_cap']
                         & np.isfinite(bp['bp_own_margin']) & (bp['bp_own_margin'] >= -1e-6))
            for pair in range(4):
                for mode in ['LP96', 'BP_geometry_LP96']:
                    cohort = records[pair, mode]
                    row = next(r for r in cohort['rows'] if r['job_id'] == job_id and r['batch'] == bi)
                    folder = root / cohort['directory']
                    file = folder / row['file']
                    assert sha(file) == row['file_sha256']
                    with np.load(file, allow_pickle=False) as z:
                        actual = {k: z[k] for k in z.files}
                    reference = source[mode]
                    assert set(actual) - {'couplings'} == set(reference) - {'couplings'}
                    for name, value in actual.items():
                        if name == 'couplings':
                            continue
                        assert np.array_equal(value, reference[name], equal_nan=True), (pair, mode, name)
                    array_receipt = row['fresh_array_receipt']
                    assert row['source_parity'] is None
                    assert array_receipt['status'] == 'fresh_arrays_sealed_without_outcome_join'
                    assert array_receipt['labels_distances_source_outcomes_read'] is False
                    assert array_receipt['actual_couplings_sha256'] == array_sha(lp['couplings'])
                    assert array_receipt['tokens_sha256'] == array_sha(tt)
                    assert array_receipt['full_model_inputs'] == 32
                    ix = np.arange(32) if mode == 'LP96' else np.flatnonzero(~geometry)
                    chosen = np.ones(32, np.int8) if mode == 'LP96' else (~geometry).astype(np.int8)
                    assert np.array_equal(actual['answer_chosen_model'], chosen)
                    if mode != 'LP96':
                        assert np.array_equal(actual['subset_indices'], ix)
                        assert np.array_equal(actual['geometry_valid'], geometry)
                    assert array_receipt['actual_subset_inputs'] == (len(ix) if mode != 'LP96' else 0)
                    if len(ix):
                        for name in ['prediction', 'iterations', 'stopped', 'finite', 'block_moves', 'post_block_sweeps', 'native_cap', 'block_cap', 'initial_simple_path']:
                            assert np.array_equal(actual['LP_' + name], lp['LP_' + name][:, ix])
                        assert np.array_equal(actual['LP_binary_flip_count'], lp['LP_binary_flip_count'][ix])
                        numeric_error(actual['LP_binary_cache_margin'], lp['LP_binary_cache_margin'][ix], 1e-9)
                        numeric_error(actual['LP_minimum_flip_margin64'], lp['LP_minimum_flip_margin64'][:, ix], 1e-9)
                        numeric_error(actual['LP_energy64'], lp['LP_energy64'][:, ix], 1e-6)
                    c, states, error = check_trace_pair(folder, row, root / records[0, 'LP96']['directory'],
                        reference_rows['LP96'], ix, 32)
                    candidate_scores += c
                    trace_states += states
                    maximum_score_error = max(maximum_score_error, error)
                    for name in ['prediction', 'iterations', 'stopped', 'finite', 'own_margin', 'native_cap', 'block_cap']:
                        expected = lp['answer_' + name].copy() if mode == 'LP96' else bp['bp_' + name].copy()
                        if mode != 'LP96':
                            expected[ix] = lp['answer_' + name][ix]
                        if name == 'own_margin':
                            numeric_error(actual['answer_' + name], expected, 1e-9)
                        else:
                            assert np.array_equal(actual['answer_' + name], expected)
                    clamps = ((tt == 1) | (actual['answer_prediction'] == (tt >= 2))).all((1, 2))
                    accepted = (lp['answer_hidden_good'] & actual['answer_finite'] & actual['answer_stopped']
                        & ~actual['answer_native_cap'] & ~actual['answer_block_cap'] & clamps
                        & np.isfinite(actual['answer_own_margin']) & (actual['answer_own_margin'] >= -1e-6))
                    assert np.array_equal(actual['answer_accepted'], accepted)
                    assert np.array_equal(actual['answer_clamps_ok'], clamps)
                    repeated_cases += 32
                    if pair == 0:
                        graph = np.where(chosen == 1, lp_graph, bp_graph)
                        for i in range(32):
                            outputs.append({'input_index': shard * 256 + bi * 32 + i, 'model_seed': seed,
                                'arm': 'LP96' if mode == 'LP96' else 'cascade',
                                'success': int(accepted[i] and graph[i]), 'graph_valid_shortest': int(graph[i]),
                                'numerically_accepted': int(accepted[i]), 'chosen_model': int(chosen[i]),
                                'routed': int(mode != 'LP96' and not geometry[i]),
                                'hidden_good': int(lp['answer_hidden_good'][i]),
                                'finite': int(actual['answer_finite'][i]), 'stopped': int(actual['answer_stopped'][i]),
                                'native_cap': int(actual['answer_native_cap'][i]), 'block_cap': int(actual['answer_block_cap'][i]),
                                'BP_geometry_valid_longer': int(geometry[i] and not bp_graph[i])})
                            if mode == 'BP_geometry_LP96':
                                outputs.append({'input_index': shard * 256 + bi * 32 + i, 'model_seed': seed,
                                    'arm': 'BP_AdamW', 'success': int(bp_accept[i] and bp_graph[i]),
                                    'graph_valid_shortest': int(bp_graph[i]), 'numerically_accepted': int(bp_accept[i]),
                                    'chosen_model': 0, 'routed': 0, 'hidden_good': int(lp['answer_hidden_good'][i]),
                                    'finite': int(bp['bp_finite'][i]), 'stopped': int(bp['bp_stopped'][i]),
                                    'native_cap': int(bp['bp_native_cap'][i]), 'block_cap': int(bp['bp_block_cap'][i]),
                                    'BP_geometry_valid_longer': int(geometry[i] and not bp_graph[i])})
    assert len(outputs) == 256 * 3 * 3
    return {'shard': shard, 'status': 'passed', 'shortest_labels_checked': 256,
            'measured_model_pipeline_copies': repeated_cases, 'candidate_score_comparisons': candidate_scores,
            'active_trace_states': trace_states, 'maximum_candidate_score_error': maximum_score_error,
            'rows': outputs, 'seconds': time.perf_counter() - started}
