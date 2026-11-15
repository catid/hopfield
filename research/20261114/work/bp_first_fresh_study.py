"""H2bo bounded source gates and label-blind fresh paired workers."""
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
from bp_first_fresh_pipeline import cohort
from bp_first_geometry_barrier import PairBarrier

PP = S.HERE / 'bp-first-fresh-confirmation-protocol.json'
DEST = S.OUT / 'bp_first_fresh_confirmation'


def protocol():
    p = json.loads(PP.read_text())
    assert p['status'] == 'frozen_before_source_gates_and_fresh_draws'
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
    limit = 180 if command == 'gate' else 1500
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
        assert p['budgets']['source_gate_wall_seconds_per_GPU'] == 180
        assert p['budgets']['timed_worker_wall_seconds_including_warmup'] == 1500
        assert p['budgets']['host_barrier_maximum_wait_seconds'] == 60
        if command == 'timing':
            gates = [json.loads((DEST / f'gate-gpu{g}.json').read_text()) for g in [0, 1]]
            assert all(r['status'] == 'passed' and r['protocol_sha256'] == ph for r in gates)
            assert all(r['source_model_inputs_per_arm'] == 864 for r in gates)
            inputs = json.loads((DEST / 'independent-input-audit.json').read_text())
            assert inputs['status'] == 'passed' and inputs['protocol_sha256'] == ph
            assert inputs['inputs_checked'] == inputs['unique'] == 4096
        lock, inventory = F.acquire(gpu)
        torch, _, _ = S.imports()
        torch.cuda.init()
        state.update(status='running', inventory=inventory, protocol_sha256=ph)
        shards = [0, 15] if command == 'gate' else list(range(16))
        tokens = {}
        preload_started = time.perf_counter()
        datasets = (p['source_datasets'] if command == 'gate'
                    else json.loads((DEST / 'datasets.json').read_text())['datasets'])
        if command == 'timing':
            assert datasets == inputs['datasets']
        for shard in shards:
            dataset = datasets[shard]
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
                    make_barrier('source-' + mode), batch_selection={0: list(range(8)), 15: [7]},
                    source_gate=True)
                state['completed_cohorts'].append(mode)
                progress(phase='source-gate-complete-' + mode)
            assert all(r['model_inputs'] == 864 for r in rows.values())
            receipt = {'status': 'passed', 'gpu': gpu, 'protocol_sha256': ph,
                       'source_model_inputs_per_arm': 864,
                       'actual_subset_inputs': rows['BP_geometry_LP96']['actual_subset_inputs'],
                       'cohorts': {mode: {'result_path': str((DEST / 'source-gates' / f'gpu{gpu}-{mode}' / 'result.json').relative_to(S.ROOT)),
                          'complete_answer_seconds': r['complete_answer_seconds'], 'model_inputs': r['model_inputs']}
                          for mode, r in rows.items()}}
            S.save(DEST / f'gate-gpu{gpu}.json', receipt)
        else:
            source_dataset = p['source_datasets'][0]
            assert S.sha(S.ROOT / source_dataset['path']) == source_dataset['sha256']
            with np.load(S.ROOT / source_dataset['path'], allow_pickle=False) as archive:
                warm_tokens = {0: archive['tokens']}
            for mode in ['LP96', 'BP_geometry_LP96']:
                progress(phase='warmup-' + mode)
                result = cohort(p, ph, mode, warm_tokens, [0], DEST / 'warmup' / f'gpu{gpu}-{mode}',
                                watchdog, progress, make_barrier('warmup-' + mode),
                                batch_selection={0: [0]}, source_gate=True)
                assert result['model_inputs'] == 96
                state['completed_cohorts'].append('warmup-' + mode)
            pairs = []
            for local_pair, pair in enumerate(p['paired_schedule'][f'GPU{gpu}']):
                record = {'pair': pair['pair'], 'gpu': gpu, 'order': pair['order'], 'runs': {}}
                for ordinal, mode in enumerate(pair['order']):
                    progress(phase=f'pair{pair["pair"]}-' + mode)
                    directory = DEST / 'runs' / f'pair{pair["pair"]}-{mode}'
                    result = cohort(p, ph, mode, tokens, shards, directory, watchdog, progress,
                                    make_barrier(f'paired-phase{2 * local_pair + ordinal}'),
                                    pair=pair['pair'], retain_couplings=pair['pair'] == 0)
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
                    'gpu': gpu, 'pairs': pairs,
                    'accuracy_copy': 'Pair0 predetermined; other pairs are integrity/timing copies, not extra scientific trials.'})
        watchdog()
        progress(status='complete', phase='finished')
    except BaseException as error:
        progress(status='incomplete', error=repr(error))
        S.save(failure, {'status': 'incomplete', 'command': command, 'gpu': gpu, 'pid': os.getpid(),
                        'error': repr(error), 'traceback': traceback.format_exc(),
                        'worker_wall_seconds': time.perf_counter() - started,
                        'partial_outputs_retained': True,
                        'scientific_accounting': 'Incomplete attempt; retained outputs do not establish a complete fresh endpoint.'})
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
