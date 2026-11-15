"""H2br bounded three-arm fresh mixture workers; token-only archive reads."""
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

PP = S.HERE / 'bp-first-endpoint-dispatch-protocol.json'
DEST = S.OUT / 'bp_first_endpoint_dispatch'


def protocol():
    p = json.loads(PP.read_text())
    assert p['status'] == 'frozen_before_fresh_mixture_draws_and_timing'
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
    assert command == 'timing'
    limit = 3000
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
        assert p['budgets']['timing_wall_seconds_per_gpu_including_warmups']==3000
        assert p['budgets']['host_barrier_timeout_seconds']==60
        packet_hash=S.sha(DEST/'source-compatibility-packet.json')
        assert p['compatibility_packet_sha256']==packet_hash
        for g in [0,1]:
            gate=json.loads((DEST/'source-compatibility-attempt1'/f'gate-gpu{g}.json').read_text())
            assert gate['status']=='passed' and gate['protocol_sha256']==packet_hash and gate['source_model_inputs_per_arm']==384
        inputs=json.loads((DEST/'independent-input-audit.json').read_text())
        datasets=json.loads((DEST/'datasets.json').read_text())
        assert inputs['status']=='passed' and inputs['protocol_sha256']==datasets['protocol_sha256']==ph
        assert inputs['inputs_checked']==inputs['unique']==inputs['mixture_coin_replays']==4096
        assert inputs['datasets']==datasets['datasets']
        lock, inventory = F.acquire(gpu)
        torch, _, _ = S.imports()
        torch.cuda.init()
        state.update(status='running', inventory=inventory, protocol_sha256=ph)
        tokens={};preload_started=time.perf_counter()
        for shard,ds in enumerate(datasets['datasets']):
            assert ds['shard']==shard and S.sha(S.ROOT/ds['path'])==ds['sha256']
            with np.load(S.ROOT/ds['path'],allow_pickle=False) as archive:tokens[shard]=archive['tokens']
            assert tokens[shard].shape==(256,180,180)
        state['host_token_preload_seconds']=time.perf_counter()-preload_started
        progress(phase='ready')
        def make_barrier(phase):
            return PairBarrier(DEST/'barriers'/'timing'/phase,gpu,ph,
                DEST/f'timing-worker{1-gpu}.json',DEST/f'failure-timing-gpu{1-gpu}.json',watchdog,maximum_wait_seconds=60)
        for source in p['sources']:
            ds=source['datasets'][0];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
            with np.load(S.ROOT/ds['path'],allow_pickle=False) as archive:warm_tokens={0:archive['tokens']}
            for mode in ['LP96','BP_geometry_LP96','endpoint_dispatch']:
                label=source['name']+'-'+mode
                progress(phase='warmup-'+label)
                result=cohort(p,ph,mode,warm_tokens,[0],DEST/'warmup'/f'gpu{gpu}-{label}',watchdog,progress,
                    make_barrier('warmup-'+label.lower()),batch_selection={0:[0]},source_gate=True,
                    retain_couplings=True,source_reference=source)
                assert result['model_inputs']==96
                state['completed_cohorts'].append('warmup-'+label)
        blocks=[]
        schedule=[item for item in p['paired_schedule'] if item['gpu']==gpu]
        assert [item['block'] for item in schedule]==([0,1,2] if gpu==0 else [3,4,5])
        for local_block,block in enumerate(schedule):
            record={'block':block['block'],'gpu':gpu,'order':block['order'],'runs':{}}
            for ordinal,mode in enumerate(block['order']):
                label=f'block{block["block"]}-'+mode;progress(phase=label)
                directory=DEST/'runs'/label
                result=cohort(p,ph,mode,tokens,list(range(16)),directory,watchdog,progress,
                    make_barrier(f'paired-phase{3*local_block+ordinal}'),pair=block['block'],retain_couplings=block['block']==0)
                assert result['model_inputs']==12288
                record['runs'][mode]={'result_path':str((directory/'result.json').relative_to(S.ROOT)),
                    'result_sha256':S.sha(directory/'result.json'),'complete_answer_seconds':result['complete_answer_seconds'],
                    'actual_subset_inputs':result['actual_subset_inputs']}
                state['completed_cohorts'].append(label);progress(phase='completed-'+label);watchdog()
            blocks.append(record);S.save(DEST/f'blocks-gpu{gpu}.json',{'protocol_sha256':ph,'rows':blocks})
        S.save(DEST/f'timing-gpu{gpu}.json',{'status':'passed','protocol_sha256':ph,'gpu':gpu,'blocks':blocks,
            'accuracy_copy':'Block0 predetermined; other blocks are timing/integrity copies only.'})
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
    parser.add_argument('command', choices=['timing'])
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    worker(args.command, args.gpu)
