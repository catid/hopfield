"""Draw the fixed H2bq cohort only after freeze and both source gates pass."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
import signal
import time
import traceback
import numpy as np
import study as S
import fast_npz as Z
from far_placement_study import protocol, PP, DEST
from far_placement_generator import one as generate

BASE = 2026111531


def one(index):
    return generate(index, BASE)


def main():
    started = time.perf_counter()
    p = protocol()
    assert p['budgets']['fresh_generation_wall_seconds'] == 300
    assert p['budgets']['fresh_generation_processes'] == 8
    assert p['cohort']['base_seed'] == BASE and p['cohort']['n'] == 4096
    ph = S.sha(PP)
    gates = [json.loads((DEST / f'gate-gpu{gpu}.json').read_text()) for gpu in [0, 1]]
    assert all(g['status'] == 'passed' and g['protocol_sha256'] == ph and g['source_model_inputs_per_arm'] == 864 for g in gates)
    target = DEST / 'generation-worker.json'
    if target.exists() or (DEST / 'datasets.json').exists() or list(DEST.glob('shard*.npz')):
        raise FileExistsError('Refusing an existing fresh generation attempt')
    state = {'status': 'running', 'pid': os.getpid(), 'protocol_sha256': ph,
             'base': BASE, 'completed_shards': 0, 'inputs_drawn': 0, 'wall_budget_seconds': 300}
    S.save(target, state)
    def alarm(signum, frame):
        raise TimeoutError('Frozen 300-second generation wall budget exhausted')
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 300 - (time.perf_counter() - started)))
    pool = ProcessPoolExecutor(max_workers=8)
    datasets = []
    try:
        for shard in range(16):
            indices = np.arange(shard * 256, (shard + 1) * 256)
            items = list(pool.map(one, indices, chunksize=8))
            file = DEST / f'shard{shard:02d}.npz'
            Z.save(file, tokens=np.stack([x[0] for x in items]), labels=np.stack([x[1] for x in items]),
                   shortest_edges=np.array([x[2] for x in items]), indices=indices,
                   opened_loops=np.array([x[3] for x in items]), eligible_walls=np.array([x[4] for x in items]),
                   original_endpoints=np.stack([x[5] for x in items]),
                   sampled_endpoints=np.stack([x[6] for x in items]),
                   endpoint_orientation=np.array([x[7] for x in items],np.int8))
            datasets.append({'shard': shard, 'path': str(file.relative_to(S.ROOT)), 'sha256': S.sha(file), 'n': 256})
            state.update(completed_shards=shard + 1, inputs_drawn=(shard + 1) * 256,
                         wall_seconds=time.perf_counter() - started)
            S.save(target, state)
            print(json.dumps(state), flush=True)
        pool.shutdown(wait=True)
        state.update(status='complete', wall_seconds=time.perf_counter() - started)
        assert state['wall_seconds'] <= 300
        S.save(DEST / 'datasets.json', {'protocol_sha256': ph, 'datasets': datasets,
            'base_seed': BASE, 'source_gate_sha256': {str(g): S.sha(DEST / f'gate-gpu{g}.json') for g in [0, 1]},
            'seconds': state['wall_seconds']})
        S.save(target, state)
    except BaseException as error:
        for child in list((getattr(pool, '_processes', None) or {}).values()):
            if child.is_alive():
                child.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        state.update(status='incomplete', error=repr(error), traceback=traceback.format_exc(),
                     wall_seconds=time.perf_counter() - started, partial_datasets_retained=True)
        S.save(target, state)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
