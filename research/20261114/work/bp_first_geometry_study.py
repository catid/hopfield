"""Candidate H2bn owned workers; no invocation before protocol freeze/review."""
import time
PROCESS_STARTED = time.perf_counter()
import argparse
import json
import os
import resource
import signal
import threading
import traceback
import numpy as np
import study as S
import selective_fallback as F
from bp_first_geometry_pipeline import cohort
from bp_first_geometry_barrier import PairBarrier

PP = S.HERE / 'bp-first-geometry-cascade-protocol.json'
DEST = S.OUT / 'bp_first_geometry_cascade'


def protocol():
    p = json.loads(PP.read_text())
    assert p['status'] == 'frozen_before_complete_source_gate_and_paired_replays'
    for key in ['source_sha256', 'source_evidence_sha256']:
        for name, digest in p[key].items():
            assert S.sha(S.ROOT / name) == digest, name
    for group in p['models'].values():
        for model in group.values():
            assert S.sha(S.ROOT / model['path']) == model['sha256']
    return p


def worker(command, gpu):
    started = PROCESS_STARTED
    cpu_started = time.process_time()
    destination = DEST / f'{command}-worker{gpu}.json'
    failure = DEST / f'failure-{command}-gpu{gpu}.json'
    if destination.exists() or failure.exists():
        raise FileExistsError('Refusing to replace an existing worker attempt')
    DEST.mkdir(exist_ok=True, parents=True)
    limit = 300 if command == 'gate' else 1500
    state = {'status': 'starting', 'command': command, 'gpu': gpu, 'pid': os.getpid(),
             'started_utc': S.utc(), 'wall_budget_seconds': limit, 'watchdog_checks': 0,
             'completed_cohorts': [], 'phase': 'source_bindings'}
    S.save(destination, state)
    stopped = threading.Event()
    def alarm(signum, frame):
        raise TimeoutError('Frozen worker wall budget exhausted')
    previous_handler = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, limit - (time.perf_counter() - started)))
    def hard_guard():
        # This guard may terminate only its own process. No system-wide kill,
        # GPU reset or unrelated process action is available here.
        while not stopped.wait(.25):
            elapsed = time.perf_counter() - started
            if elapsed > limit + 5:
                S.save(DEST / f'hard-timeout-{command}-gpu{gpu}.json',
                    {'status': 'incomplete', 'pid': os.getpid(), 'worker_wall_seconds': elapsed,
                     'wall_budget_seconds': limit, 'reason': 'Owned worker did not exit within five seconds of its wall watchdog',
                     'partial_outputs_retained': True})
                os._exit(124)
    guard = threading.Thread(target=hard_guard, name='owned-wall-watchdog', daemon=True)
    guard.start()
    def watchdog():
        state['watchdog_checks'] += 1
        elapsed = time.perf_counter() - started
        if elapsed > limit:
            raise TimeoutError('Frozen worker wall budget exhausted')
        # RUSAGE_SELF includes both owned archive threads in this process.
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 8 * 1024**2:
            raise MemoryError('Owned worker plus archive threads exceeded 8 GiB RSS')
        if (DEST / f'failure-{command}-gpu{1 - gpu}.json').exists():
            raise RuntimeError('Peer worker failed; preserving this incomplete attempt')
        if (DEST / f'hard-timeout-{command}-gpu{1 - gpu}.json').exists():
            raise RuntimeError('Owned peer hit its hard wall watchdog')
    def progress(**fields):
        state.update(fields, worker_wall_seconds=time.perf_counter() - started,
                     process_cpu_seconds=time.process_time() - cpu_started,
                     peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     updated_utc=S.utc())
        S.save(destination, state)
    lock = None
    try:
        p = protocol()
        ph = S.sha(PP)
        assert p['budgets']['source_gate_wall_seconds_per_GPU'] == 300
        assert p['budgets']['timed_worker_wall_seconds_including_warmups'] == 1500
        assert p['budgets']['host_barrier_maximum_wait_seconds'] == 60
        if command == 'timing':
            gates = [json.loads((DEST / f'gate-gpu{g}.json').read_text()) for g in [0, 1]]
            assert all(r['status'] == 'passed' and r['protocol_sha256'] == ph for r in gates)
            assert sum(r['source_model_inputs_per_arm'] for r in gates) == 12288
            assert sum(r['actual_subset_inputs'] for r in gates) == 510
        lock, inventory = F.acquire(gpu)
        torch, _, _ = S.imports()
        torch.cuda.init()
        state.update(status='running', inventory=inventory, protocol_sha256=ph)
        shards = list(range(gpu, 16, 2)) if command == 'gate' else list(range(16))
        tokens = {}
        preload_started = time.perf_counter()
        for shard in shards:
            dataset = p['datasets'][shard]
            assert S.sha(S.ROOT / dataset['path']) == dataset['sha256']
            with np.load(S.ROOT / dataset['path'], allow_pickle=False) as archive:
                tokens[shard] = archive['tokens']
            assert tokens[shard].shape == (256, 180, 180)
        state['host_token_preload_seconds'] = time.perf_counter() - preload_started
        progress(phase='ready')
        watchdog()
        def make_barrier(phase):
            return PairBarrier(DEST / 'barriers' / command / phase, gpu, ph,
                DEST / f'{command}-worker{1 - gpu}.json', DEST / f'failure-{command}-gpu{1 - gpu}.json',
                watchdog, maximum_wait_seconds=60)
        if command == 'gate':
            rows = {}
            for mode in ['LP96', 'BP_geometry_LP96']:
                progress(phase='source-gate-' + mode)
                rows[mode] = cohort(p, ph, mode, tokens, shards,
                    DEST / 'source-gates' / f'gpu{gpu}-{mode}', watchdog, progress,
                    make_barrier('source-' + mode))
                state['completed_cohorts'].append(mode)
                progress(phase='source-gate-complete-' + mode)
            assert all(r['model_inputs'] == 6144 for r in rows.values())
            receipt = {'status': 'passed', 'gpu': gpu, 'protocol_sha256': ph,
                       'source_model_inputs_per_arm': 6144,
                       'actual_subset_inputs': rows['BP_geometry_LP96']['actual_subset_inputs'],
                       'cohorts': {mode: {'result_path': str((DEST / 'source-gates' / f'gpu{gpu}-{mode}' / 'result.json').relative_to(S.ROOT)),
                          'complete_answer_seconds': r['complete_answer_seconds'], 'model_inputs': r['model_inputs']}
                          for mode, r in rows.items()}}
            S.save(DEST / f'gate-gpu{gpu}.json', receipt)
        else:
            for mode in ['LP96', 'BP_geometry_LP96']:
                progress(phase='warmup-' + mode)
                result = cohort(p, ph, mode, tokens, [0], DEST / 'warmup' / f'gpu{gpu}-{mode}',
                                watchdog, progress, make_barrier('warmup-' + mode), only_first_batch=True)
                assert result['model_inputs'] == 96
                state['completed_cohorts'].append('warmup-' + mode)
            pairs = []
            for local_pair, pair in enumerate(p['paired_schedule'][f'GPU{gpu}']):
                record = {'pair': pair['pair'], 'gpu': gpu, 'order': pair['order'], 'runs': {}}
                for ordinal, mode in enumerate(pair['order']):
                    progress(phase=f'pair{pair["pair"]}-' + mode)
                    directory = DEST / 'runs' / f'pair{pair["pair"]}-{mode}'
                    result = cohort(p, ph, mode, tokens, shards, directory, watchdog, progress,
                                    make_barrier(f'paired-phase{2 * local_pair + ordinal}'))
                    assert result['model_inputs'] == 12288
                    record['runs'][mode] = {'result_path': str((directory / 'result.json').relative_to(S.ROOT)),
                        'result_sha256': S.sha(directory / 'result.json'),
                        'complete_answer_seconds': result['complete_answer_seconds'],
                        'actual_subset_inputs': result['actual_subset_inputs']}
                    state['completed_cohorts'].append(f'pair{pair["pair"]}-' + mode)
                    progress(phase='paired-arm-complete')
                    watchdog()
                pairs.append(record)
                S.save(DEST / f'pairs-gpu{gpu}.json', {'protocol_sha256': ph, 'rows': pairs})
            S.save(DEST / f'timing-gpu{gpu}.json', {'status': 'passed', 'protocol_sha256': ph,
                    'gpu': gpu, 'pairs': pairs, 'fresh_scientific_evaluations': 0})
        watchdog()
        progress(status='complete', phase='finished')
    except BaseException as error:
        progress(status='incomplete', error=repr(error))
        S.save(failure, {'status': 'incomplete', 'command': command, 'gpu': gpu, 'pid': os.getpid(),
                        'error': repr(error), 'traceback': traceback.format_exc(),
                        'worker_wall_seconds': time.perf_counter() - started,
                        'partial_outputs_retained': True, 'new_scientific_evaluations': 0})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stopped.set()
        guard.join(timeout=1)
        signal.signal(signal.SIGALRM, previous_handler)
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['gate', 'timing'])
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    worker(args.command, args.gpu)
