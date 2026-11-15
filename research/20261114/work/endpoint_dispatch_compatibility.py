"""H2br bounded immutable-source compatibility gates; no fresh draws or study timing."""
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
from endpoint_dispatch_pipeline import cohort
from bp_first_geometry_barrier import PairBarrier

PP = S.OUT / 'bp_first_endpoint_dispatch/source-compatibility-packet.json'
DEST = S.OUT / 'bp_first_endpoint_dispatch/source-compatibility-attempt1'


def protocol():
    p = json.loads(PP.read_text())
    assert p['status'] == 'ready_for_parent_source_compatibility_review'
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
    assert command == 'gate'
    limit = 240
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
        assert p['wall_budget_seconds_per_gpu'] == 240 and p['fresh_inputs_drawn'] == 0
        lock, inventory = F.acquire(gpu)
        torch, _, _ = S.imports()
        torch.cuda.init()
        state.update(status='running', inventory=inventory, protocol_sha256=ph)
        def make_barrier(phase):
            return PairBarrier(DEST / 'barriers' / phase, gpu, ph,
                DEST / f'gate-worker{1-gpu}.json', DEST / f'failure-gate-gpu{1-gpu}.json',
                watchdog, maximum_wait_seconds=60)
        receipts=[]
        for source in p['sources']:
            tokens={}
            for shard in [0,15]:
                dataset=source['datasets'][shard]
                assert S.sha(S.ROOT / dataset['path']) == dataset['sha256']
                with np.load(S.ROOT / dataset['path'],allow_pickle=False) as z:tokens[shard]=z['tokens']
                assert tokens[shard].shape==(256,180,180)
            for mode in ['LP96','BP_geometry_LP96','endpoint_dispatch']:
                label=source['name']+'-'+mode
                progress(phase=label)
                result=cohort(p,ph,mode,tokens,[0,15],DEST/'source-gates'/f'gpu{gpu}-{label}',
                    watchdog,progress,make_barrier(label.lower()),batch_selection={0:[0],15:[0]},
                    source_gate=True,retain_couplings=True,source_reference=source)
                assert result['model_inputs']==192
                receipts.append({'source':source['name'],'mode':mode,'model_inputs':result['model_inputs'],
                    'result_path':str((DEST/'source-gates'/f'gpu{gpu}-{label}'/'result.json').relative_to(S.ROOT)),
                    'result_sha256':S.sha(DEST/'source-gates'/f'gpu{gpu}-{label}'/'result.json')})
                state['completed_cohorts'].append(label)
                progress(phase='completed-'+label)
        assert len(receipts)==6
        assert all(sum(r['model_inputs'] for r in receipts if r['mode']==mode)==384 for mode in ['LP96','BP_geometry_LP96','endpoint_dispatch'])
        S.save(DEST/f'gate-gpu{gpu}.json',{'status':'passed','gpu':gpu,'protocol_sha256':ph,
            'approved_proposal_sha256':p['approved_proposal_sha256'],'source_model_inputs_per_arm':384,
            'receipts':receipts,'fresh_inputs_drawn':0,'scientific_timing_pairs':0})
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
    parser.add_argument('command', choices=['gate'])
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    worker(args.command, args.gpu)
