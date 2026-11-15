"""Candidate complete H2bn source parity, including normalized LP subset traces."""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_connectivity_screen import path_valid

ARMS = ('pl', 'old_binary', 'connectivity_veto', 'original_lexical16',
        'unified16', 'unified48', 'unified64', 'unified96')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    array = np.asarray(value)
    h = hashlib.sha256()
    h.update(array.dtype.str.encode())
    h.update(json.dumps(array.shape).encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def terminal_records(records, owners, allowed):
    result = {}
    for row in records:
        owner = int(owners[row['local_input']])
        if owner in allowed:
            result[owner] = {k: v for k, v in row.items() if k != 'local_input'}
    return result


def trace_rows(trace, states, owners, allowed):
    result = {}
    for round_index, step in enumerate(trace):
        selected = set(step['chosen'])
        native = {r['local_input']: {k: v for k, v in r.items() if k != 'local_input'}
                  for r in step.get('native_results', [])}
        for geometry in step['geometry']:
            local = geometry['local_input']
            owner = int(owners[local])
            if owner not in allowed:
                continue
            candidates = []
            for j, row in enumerate(step['candidates']):
                if row['local_input'] == local:
                    candidates.append({'cells': row['cells'], 'delta': row['delta'], 'chosen': j in selected})
            result.setdefault(owner, []).append({
                'round': step['round'],
                'geometry': {k: v for k, v in geometry.items() if k != 'local_input'},
                'candidates': candidates, 'native': native.get(local),
                'state_sha256': array_sha(states[round_index, local])})
    return result


def compare_trace(actual, expected):
    assert actual.keys() == expected.keys()
    maximum = 0.
    candidate_count = state_count = 0
    for owner in actual:
        assert len(actual[owner]) == len(expected[owner])
        for a, b in zip(actual[owner], expected[owner]):
            for key in ['round', 'geometry', 'native', 'state_sha256']:
                assert a[key] == b[key], (owner, key)
            assert len(a['candidates']) == len(b['candidates'])
            for x, y in zip(a['candidates'], b['candidates']):
                assert x['cells'] == y['cells'] and x['chosen'] == y['chosen']
                error = abs(x['delta'] - y['delta'])
                assert np.isfinite(error) and error < 1e-8
                maximum = max(maximum, error)
                candidate_count += 1
            state_count += 1
    return {'candidate_comparisons': candidate_count, 'active_trace_states': state_count,
            'maximum_candidate_score_error': maximum}


def verify(result, host_tokens, source_job, batch_number, bound_hashes):
    source_job = Path(source_job)
    source_file = source_job / f'batch{batch_number:02d}.npz'
    assert sha(source_file) == bound_hashes[source_file.name]
    names = ['couplings', 'hidden_steps', 'hidden_good', 'predictions', 'iterations', 'stopped', 'finite',
             'native_final_logits', 'minimum_flip_margin64', 'binary_flip_count', 'binary_cache_margin',
             'block_moves', 'post_block_sweeps', 'block_native_cap', 'block_cap', 'initial_simple_path']
    with np.load(source_file, allow_pickle=False) as archive:
        old = {k: archive[k] for k in names}
    audit = result['audit']
    assert np.array_equal(audit['couplings'], old['couplings'])
    assert np.array_equal(audit['hidden_steps'], old['hidden_steps'])
    assert np.array_equal(result['answer']['hidden_good'], old['hidden_good'])
    mode = result['timing']['mode']
    if mode == 'BP_geometry_LP96':
        bp = result['bp_prefix']
        for key, source_key in [('prediction', 'predictions'), ('iterations', 'iterations'),
                                ('finite', 'finite'), ('stopped', 'stopped')]:
            assert np.array_equal(bp[key], old[source_key][0]), key
        assert np.array_equal(audit['bp_final_logits'], old['native_final_logits'][0])
        assert np.max(np.abs(bp['own_margin'] - old['minimum_flip_margin64'][0])) < 1e-9
        geometry = np.array([path_valid(t, b) for t, b in zip(host_tokens, bp['prediction'])])
        assert np.array_equal(geometry, result['geometry_valid'])
        assert np.array_equal(result['subset_indices'], np.flatnonzero(~geometry))
        indices = result['subset_indices']
    else:
        indices = np.arange(len(host_tokens))
    totals = {'candidate_comparisons': 0, 'active_trace_states': 0, 'maximum_candidate_score_error': 0.}
    core = result['core']
    if len(indices):
        assert core is not None
        for key, source_key in [('prediction', 'predictions'), ('iterations', 'iterations'),
                                ('finite', 'finite'), ('stopped', 'stopped')]:
            assert np.array_equal(audit['LP_' + key], old[source_key][1:8][:, indices]), key
        for key, source_key in [('block_moves', 'block_moves'), ('post_block_sweeps', 'post_block_sweeps'),
                                ('native_cap', 'block_native_cap'), ('block_cap', 'block_cap'),
                                ('initial_simple_path', 'initial_simple_path')]:
            assert np.array_equal(audit['LP_' + key], old[source_key][:, indices]), key
        assert np.array_equal(audit['LP_binary_flip_count'], old['binary_flip_count'][0, indices])
        assert np.max(np.abs(audit['LP_binary_cache_margin'] - old['binary_cache_margin'][0, indices])) < 1e-9
        final = result['answer'] if mode == 'LP96' else result['lp_subset']
        assert np.max(np.abs(final['own_margin'] - old['minimum_flip_margin64'][7, indices])) < 1e-9
        allowed = set(map(int, indices))
        for v in range(2, 8):
            file = source_job / f'batch{batch_number:02d}-{ARMS[v]}-trace.npz'
            assert sha(file) == bound_hashes[file.name]
            with np.load(file, allow_pickle=False) as z:
                source = {k: z[k] for k in z.files}
            payload = json.loads(source['trace_utf8'].tobytes())
            block = core['blocks'][v]
            owners = indices[block['valid_prefix_indices']]
            original_owners = source['valid_prefix_indices']
            assert np.array_equal(owners, original_owners[np.isin(original_owners, indices)])
            assert np.array_equal(block['initial_terminal'], source['initial_terminal'][indices])
            assert terminal_records(block['initial_terminal_records'], indices, allowed) == terminal_records(
                payload['initial_terminal_checks'], np.arange(len(host_tokens)), allowed)
            raw = block['raw']
            trace = raw['trace'] if raw else []
            states = raw['states_before'] if raw else np.zeros((0, 0, *host_tokens.shape[1:]), bool)
            comparison = compare_trace(trace_rows(trace, states, owners, allowed),
                trace_rows(payload['trace'], source['states_before'], original_owners, allowed))
            for key in ['candidate_comparisons', 'active_trace_states']:
                totals[key] += comparison[key]
            totals['maximum_candidate_score_error'] = max(totals['maximum_candidate_score_error'], comparison['maximum_candidate_score_error'])
            assert terminal_records(raw.get('terminal_check_records', []) if raw else [], owners, allowed) == terminal_records(
                payload['final_terminal_checks'], original_owners, allowed)
            selected = np.flatnonzero(np.isin(original_owners, indices))
            for name in ['pre_terminal_stopped', 'pre_terminal_block_cap']:
                actual = raw.get(name, np.zeros(0, bool)) if raw else np.zeros(0, bool)
                expected = source[name][selected] if len(source[name]) else np.zeros(0, bool)
                assert np.array_equal(actual, expected), name
    else:
        assert core is None
    selected_model = np.ones(len(host_tokens), np.int8) if mode == 'LP96' else (~result['geometry_valid']).astype(np.int8)
    assert np.array_equal(result['answer']['chosen_model'], selected_model)
    original_arm = np.where(selected_model == 1, 7, 0)
    for key, source_key in [('prediction', 'predictions'), ('iterations', 'iterations'),
                            ('finite', 'finite'), ('stopped', 'stopped')]:
        assert np.array_equal(result['answer'][key], old[source_key][original_arm, np.arange(len(host_tokens))])
    answer = result['answer']
    expected_margin = old['minimum_flip_margin64'][original_arm, np.arange(len(host_tokens))]
    assert np.max(np.abs(answer['own_margin'] - expected_margin)) < 1e-9
    clamps = ((host_tokens == 1) | (answer['prediction'] == (host_tokens >= 2))).all((1, 2))
    expected_acceptance = (old['hidden_good'] & answer['finite'] & answer['stopped'] & clamps
                           & ~answer['native_cap'] & ~answer['block_cap']
                           & np.isfinite(expected_margin) & (expected_margin >= -1e-6))
    assert np.array_equal(answer['clamps_ok'], clamps)
    assert np.array_equal(answer['accepted'], expected_acceptance)
    return {'status': 'passed', 'mode': mode, 'full_model_inputs': len(host_tokens),
            'actual_subset_inputs': len(indices) if mode != 'LP96' else 0,
            'source_batch_sha256': bound_hashes[source_file.name],
            'actual_couplings_sha256': array_sha(audit['couplings']), **totals}
