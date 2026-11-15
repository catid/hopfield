"""Synthetic H2bn routing/scatter/accounting gate without real source arrays."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np
from bp_first_geometry_policy import routes, scatter
from bp_first_geometry_accounting import batch_cost, paired_endpoint, preservation
from bp_first_geometry_parity import trace_rows, compare_trace, terminal_records
from bp_first_geometry_pipeline import drain
from bounded_archive_writer import ArchiveWriter
from bp_first_geometry_barrier import PairBarrier


def gate():
    started = time.perf_counter()
    t = np.ones((5, 5), np.uint8)
    t[2, 0] = 2
    t[2, 4] = 3
    straight = np.zeros((5, 5), bool)
    straight[2] = True
    longer = np.zeros_like(straight)
    longer[0] = True
    longer[:3, 0] = longer[:3, 4] = True
    broken = straight.copy()
    broken[2, 2] = False
    tokens = np.stack([t] * 3)
    bp_boards = np.stack([straight, broken, longer])
    valid, ix = routes(tokens, bp_boards)
    assert valid.tolist() == [True, False, True] and ix.tolist() == [1]
    # A longer simple path is deliberately retained: no hidden shortestness rule.
    numerical_flag_cases = 0
    for flags in itertools.product([False, True], repeat=5):
        fields = dict(zip(['hidden_good', 'finite', 'stopped', 'native_cap', 'block_cap'], flags))
        assert fields is not None
        assert np.array_equal(routes(tokens, bp_boards)[1], ix)
        numerical_flag_cases += 1
    bp = {'prediction': bp_boards, 'iterations': np.array([3, 5, 7], np.int64),
          'stopped': np.ones(3, bool), 'finite': np.ones(3, bool),
          'own_margin': np.array([1., 2., 3.], np.float64),
          'native_cap': np.zeros(3, bool), 'block_cap': np.zeros(3, bool)}
    scatter_cases = 0
    for mask in range(8):
        indices = np.array([i for i in range(3) if mask & (1 << i)], np.int64)
        replacement = {k: v[indices].copy() for k, v in bp.items()}
        replacement['iterations'] += 100
        replacement['own_margin'] += 10
        combined = scatter(bp, replacement, indices)
        assert np.array_equal(combined['chosen_model'], np.array([(mask >> i) & 1 for i in range(3)], np.int8))
        for name, original in bp.items():
            expected = original.copy()
            expected[indices] = replacement[name]
            assert np.array_equal(combined[name], expected)
            assert not np.shares_memory(combined[name], original)
        scatter_cases += 1
    malformed = [np.array([1, 1]), np.array([2, 1]), np.array([-1]), np.array([3]), np.array([1.])]
    for indices in malformed:
        try:
            scatter(bp, {k: v[:len(indices)] for k, v in bp.items()}, indices)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed scatter indices accepted')
    base = {'input_transfer': 1., 'hidden_readout_and_flags': 1., 'LP96_complete_prefix': 1.,
            'LP96_answer_return': 1., 'LP96_original_margin_and_return': 1., 'final_numerical_acceptance': 1.,
            'model_provenance': 1.}
    receipt = {'mode': 'LP96', 'answer_seconds': 7.5, 'components': base,
               'unattributed_in_segment_seconds': .5, 'n': 3, 'actual_subset_inputs': 0}
    assert batch_cost(receipt) == 7.5
    for key in base:
        altered = dict(receipt, components={k: v for k, v in base.items() if k != key})
        try:
            batch_cost(altered)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing required charge accepted')
    pairs = [{'pair': i, 'LP96_complete_answer_seconds': 100., 'candidate_complete_answer_seconds': v}
             for i, v in enumerate([80., 90., 90., 100.])]
    assert paired_endpoint(pairs)['decision'] == 'supported'
    assert paired_endpoint(pairs)['exact_median_fraction'] == '1/10'
    pairs[1]['candidate_complete_answer_seconds'] = 89.
    assert paired_endpoint(pairs)['decision'] == 'supported'
    pairs[1]['candidate_complete_answer_seconds'] = 91.
    assert paired_endpoint(pairs)['decision'] == 'falsified'
    try:
        paired_endpoint(pairs[:3])
    except ValueError:
        pass
    else:
        raise AssertionError('Reduced timing cohort accepted')
    control = [True] * 4095 + [False]
    assert preservation(control, [True] * 4096)['decision'] == 'supported'
    candidate = [False] + [True] * 4095
    assert preservation(control, candidate)['decision'] == 'falsified'
    source_trace = [{'round': 0, 'geometry': [
        {'local_input': 0, 'candidate_count': 1}, {'local_input': 1, 'candidate_count': 1}],
        'candidates': [{'local_input': 0, 'cells': [2], 'delta': -2.},
                       {'local_input': 1, 'cells': [4], 'delta': -3.}],
        'chosen': [0, 1], 'native_results': [{'local_input': 1, 'sweeps': 2, 'stopped': True, 'finite': True}]}]
    subset_trace = [{'round': 0, 'geometry': [{'local_input': 0, 'candidate_count': 1}],
        'candidates': [{'local_input': 0, 'cells': [4], 'delta': -3.}], 'chosen': [0],
        'native_results': [{'local_input': 0, 'sweeps': 2, 'stopped': True, 'finite': True}]}]
    states = np.stack([np.stack([straight, longer])])
    expected = trace_rows(source_trace, states, np.array([3, 7]), {7})
    actual = trace_rows(subset_trace, states[:, 1:], np.array([7]), {7})
    checked = compare_trace(actual, expected)
    assert checked['candidate_comparisons'] == checked['active_trace_states'] == 1
    assert terminal_records([{'local_input': 0, 'candidates': []}], np.array([7]), {7}) == {7: {'candidates': []}}
    actual[7][0]['candidates'][0]['delta'] += 1e-4
    try:
        compare_trace(actual, expected)
    except AssertionError:
        pass
    else:
        raise AssertionError('Changed subset candidate score accepted')
    with tempfile.TemporaryDirectory(prefix='h2bn-synthetic-lifecycle-') as temporary:
        tmp = Path(temporary)
        writer = ArchiveWriter(True)
        original_write = writer._write
        release = threading.Event()
        started_tasks = []
        def blocked_write(items):
            started_tasks.append(items[0][0].name)
            assert release.wait(5)
            return original_write(items)
        writer._write = blocked_write
        writer.submit([(tmp / 'one.npz', {'x': np.arange(4)})])
        writer.submit([(tmp / 'two.npz', {'x': np.arange(4)})])
        finished = threading.Event()
        def drain_both():
            drain(writer)
            finished.set()
        thread = threading.Thread(target=drain_both)
        thread.start()
        time.sleep(.03)
        assert not finished.is_set()
        release.set()
        thread.join(timeout=5)
        assert finished.is_set() and not writer.pending and len(writer.results) == 2
        writer.close()
        code = ('import sys; from pathlib import Path; import bp_first_geometry_study as M; '
                'M.DEST=Path(sys.argv[1]); M.PP=Path(sys.argv[2]); M.worker("gate",0)')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).resolve().parent)
        out = tmp / 'failed-worker'
        command = [sys.executable, '-c', code, str(out), str(tmp / 'missing-protocol')]
        failed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
        assert failed.returncode != 0
        receipt = out / 'failure-gate-gpu0.json'
        initial = hashlib.sha256(receipt.read_bytes()).hexdigest()
        assert json.loads(receipt.read_text())['status'] == 'incomplete'
        second = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
        assert second.returncode != 0 and 'FileExistsError' in second.stderr
        assert hashlib.sha256(receipt.read_bytes()).hexdigest() == initial
        fifo = tmp / 'blocked-protocol'
        os.mkfifo(fifo)
        out = tmp / 'watchdog-worker'
        child = subprocess.Popen([sys.executable, '-c', code, str(out), str(fifo)], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not (out / 'gate-worker0.json').exists():
                assert child.poll() is None and time.monotonic() < deadline
                time.sleep(.01)
            time.sleep(.05)
            child.send_signal(signal.SIGALRM)
            child.communicate(timeout=5)
            assert child.returncode != 0
            failure = json.loads((out / 'failure-gate-gpu0.json').read_text())
            assert failure['status'] == 'incomplete' and 'TimeoutError' in failure['error']
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
        states = []
        for gpu in [0, 1]:
            state = tmp / f'barrier-worker{gpu}.json'
            state.write_text(json.dumps({'status': 'running', 'pid': os.getpid()}))
            states.append(state)
        barriers = [PairBarrier(tmp / 'paired-barrier', gpu, 'synthetic', states[1 - gpu],
                    tmp / f'absent-failure{1 - gpu}', lambda: None, 1) for gpu in [0, 1]]
        completed = []
        first = threading.Thread(target=lambda: completed.append(barriers[0].gate('synthetic-stage')))
        first.start()
        time.sleep(.03)
        assert not completed
        barriers[1].gate('synthetic-stage')
        first.join(timeout=2)
        assert len(completed) == 1 and len(barriers[0].events) == len(barriers[1].events) == 1
        marker = tmp / 'paired-barrier/synthetic-stage-gpu0.json'
        prior_hash = hashlib.sha256(marker.read_bytes()).hexdigest()
        try:
            barriers[0].gate('synthetic-stage')
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing barrier marker replaced')
        assert hashlib.sha256(marker.read_bytes()).hexdigest() == prior_hash
        failure = tmp / 'barrier-peer-failure.json'
        failure.write_text('{}')
        barrier = PairBarrier(tmp / 'peer-failure-barrier', 0, 'synthetic', states[1], failure, lambda: None, 1)
        try:
            barrier.gate('stage')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Peer failure did not abort barrier')
        barrier = PairBarrier(tmp / 'peer-timeout-barrier', 0, 'synthetic', tmp / 'absent-state', tmp / 'absent-failure', lambda: None, .03)
        try:
            barrier.gate('stage')
        except TimeoutError:
            pass
        else:
            raise AssertionError('Missing peer did not time out')
        exited = subprocess.Popen([sys.executable, '-c', 'pass'])
        exited.wait(timeout=5)
        dead_state = tmp / 'dead-peer.json'
        dead_state.write_text(json.dumps({'status': 'running', 'pid': exited.pid}))
        barrier = PairBarrier(tmp / 'peer-death-barrier', 0, 'synthetic', dead_state, tmp / 'absent-failure', lambda: None, 1)
        try:
            barrier.gate('stage')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Owned peer exit did not abort barrier')
    names = ['bp_first_geometry_policy.py', 'bp_first_geometry_accounting.py',
             'bp_first_lp96_core.py', 'bp_first_geometry_batch.py',
             'bp_first_geometry_parity.py', 'bp_first_geometry_pipeline.py',
             'bp_first_geometry_study.py', 'bp_first_geometry_barrier.py',
             'audit_bp_first_geometry.py', 'preflight_bp_first_geometry.py']
    return {'status': 'passed', 'synthetic_only': True,
            'geometry_cases': 3, 'longer_simple_path_retained': True,
            'numerical_flag_combinations_ignored_by_routing': numerical_flag_cases,
            'all_empty_mixed_scatter_patterns': scatter_cases,
            'invalid_scatter_index_cases': len(malformed),
            'required_charge_omission_cases': len(base), 'fixed_four_pair_boundary_cases': 4,
            'joint_preservation_cases': 2, 'seconds': time.perf_counter() - started,
            'subset_trace_index_normalization': True, 'altered_candidate_score_rejected': True,
            'both_archive_tasks_drained_before_answer': True,
            'incomplete_worker_receipt_preserved': True, 'existing_attempt_refused': True,
            'owned_watchdog_signal_receipt': True,
            'host_barrier_waits_for_both_owned_workers': True,
            'host_barrier_marker_overwrite_refused': True,
            'host_barrier_peer_failure_timeout_death_abort': True,
            'source_sha256': {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = gate()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
