"""Prospective H2bt bounded original-order and packed dispatcher workers."""
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
from packed_controls_pipeline import cohort
from audit_packed_controls_cohort import verify as verify_cohort
from bp_first_geometry_barrier import PairBarrier

PP = S.HERE / 'bp-first-packed-controls-protocol.json'
GATE_PP = S.OUT / 'bp_first_packed_controls/source-packet.json'
DEST = S.OUT / 'bp_first_packed_controls'


def protocol(path):
    p = json.loads(path.read_text())
    assert p['status'] in ['frozen_before_changed_source_gates','frozen_before_packed_timing']
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
    assert command in ['gate','timing']
    limit = 240 if command=='gate' else 3000
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
        path=GATE_PP if command=='gate' else PP
        p=protocol(path);ph=S.sha(path)
        assert p['budgets']['source_gate_wall_seconds_per_gpu']==240 and p['budgets']['timing_wall_seconds_per_gpu_including_warmups']==3000 and p['budgets']['host_barrier_timeout_seconds']==60
        prior=p['packing_source'];previous=S.ROOT/prior['directory']
        assert S.sha(previous/'audit.json')==prior['audit_sha256'] and json.loads((previous/'audit.json').read_text())['status']=='passed'
        assert S.sha(previous/'sealed-replays.json')==prior['seal_sha256']
        source=p['source'];base=S.ROOT/source['directory']
        assert S.sha(base/'audit.json')==source['audit_sha256'] and json.loads((base/'audit.json').read_text())['status']=='passed'
        assert S.sha(base/'sealed-replays.json')==source['seal_sha256']
        if command=='timing':
            for g in [0,1]:
                file=DEST/f'gate-gpu{g}.json';assert S.sha(file)==p['gate_receipts_sha256'][str(g)]
                assert json.loads(file.read_text())['status']=='passed'
        lock,inventory=F.acquire(gpu);torch,_,_=S.imports();torch.cuda.init()
        state.update(status='running',inventory=inventory,protocol_sha256=ph)
        tokens={};preload_started=time.perf_counter()
        for shard,ds in enumerate(source['datasets']):
            assert ds['shard']==shard and S.sha(S.ROOT/ds['path'])==ds['sha256']
            with np.load(S.ROOT/ds['path'],allow_pickle=False) as archive:tokens[shard]=archive['tokens']
            assert tokens[shard].shape==(256,180,180)
        state['host_token_preload_seconds']=time.perf_counter()-preload_started;progress(phase='ready')
        def make_barrier(phase):return PairBarrier(DEST/'barriers'/command/phase,gpu,ph,DEST/f'{command}-worker{1-gpu}.json',DEST/f'failure-{command}-gpu{1-gpu}.json',watchdog,maximum_wait_seconds=60)
        if command=='gate':
            indices=np.r_[np.arange(64),np.arange(15*256,15*256+64)];checks=[]
            for mode in ['LP96','BP_geometry_LP96','packed_dispatch']:
                label=f'gpu{gpu}-'+mode;folder=DEST/'source-gates'/label;progress(phase=label)
                result=cohort(p,ph,mode,tokens,indices,folder,watchdog,progress,make_barrier(mode),source_gate=True)
                assert result['model_inputs']==384
                checks.append(verify_cohort(S.ROOT,folder,p,ph,indices,DEST,command));watchdog();state['completed_cohorts'].append(label)
            S.save(DEST/f'gate-gpu{gpu}.json',{'status':'passed','protocol_sha256':ph,'gpu':gpu,'source_model_inputs_per_arm':384,'checks':checks})
        else:
            for mode in ['LP96','BP_geometry_LP96','packed_dispatch']:
                label=f'gpu{gpu}-'+mode;folder=DEST/'warmup'/label;progress(phase='warmup-'+label)
                result=cohort(p,ph,mode,tokens,np.arange(32),folder,watchdog,progress,make_barrier('warmup-'+mode),source_gate=True)
                assert result['model_inputs']==96
                check=verify_cohort(S.ROOT,folder,p,ph,np.arange(32),DEST,command);S.save(folder/'source-check.json',check)
                watchdog();state['completed_cohorts'].append('warmup-'+label)
            blocks=[]
            schedule=[item for item in p['timing_schedule'] if item['gpu']==gpu]
            assert [item['block'] for item in schedule]==([0,1,2] if gpu==0 else [3,4,5])
            for local_block,item in enumerate(schedule):
                record={'block':item['block'],'gpu':gpu,'order':item['order'],'runs':{}}
                for ordinal,mode in enumerate(item['order']):
                    label=f'block{item["block"]}-'+mode;progress(phase=label);folder=DEST/'runs'/label
                    result=cohort(p,ph,mode,tokens,np.arange(4096),folder,watchdog,progress,make_barrier(f'paired-phase{3*local_block+ordinal}'),block=item['block'])
                    assert result['model_inputs']==12288
                    record['runs'][mode]={'result_path':str((folder/'result.json').relative_to(S.ROOT)),'result_sha256':S.sha(folder/'result.json'),'complete_answer_seconds':result['complete_answer_seconds']}
                    state['completed_cohorts'].append(label);progress(phase='completed-'+label);watchdog()
                blocks.append(record);S.save(DEST/f'blocks-gpu{gpu}.json',{'protocol_sha256':ph,'rows':blocks})
            S.save(DEST/f'timing-gpu{gpu}.json',{'status':'passed','protocol_sha256':ph,'gpu':gpu,'blocks':blocks,'accounting':'Performance-only source copies; zero new scientific observations.'})
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
    parser.add_argument('command', choices=['gate','timing'])
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    worker(args.command, args.gpu)
