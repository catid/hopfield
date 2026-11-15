"""H2bo all-label and complete prior-cohort exclusion audit before inference."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
import signal
import time
import traceback
import numpy as np
import study as S
from audit_larger_relaxation import check_dataset
from audit_contrastive_inputs import digest
from bp_first_fresh_study import PP, DEST


def main():
    started = time.perf_counter()
    target = DEST / 'input-audit-worker.json'
    if target.exists() or (DEST / 'independent-input-audit.json').exists():
        raise FileExistsError('Refusing an existing input-audit attempt')
    state = {'status': 'running', 'pid': os.getpid(), 'wall_budget_seconds': 120}
    S.save(target, state)
    def watchdog(signum, frame):
        raise TimeoutError('Frozen 120-second input-audit budget exhausted')
    signal.signal(signal.SIGALRM, watchdog)
    signal.setitimer(signal.ITIMER_REAL, 120)
    pool = None
    try:
        p = json.loads(PP.read_text())
        assert p['budgets']['independent_input_audit_wall_seconds'] == 120
        assert p['budgets']['independent_input_audit_processes'] == 8
        manifest = json.loads((DEST / 'datasets.json').read_text())
        assert manifest['protocol_sha256'] == S.sha(PP)
        assert [d['shard'] for d in manifest['datasets']] == list(range(16))
        fresh = set()
        for ds in manifest['datasets']:
            file = S.ROOT / ds['path']
            assert S.sha(file) == ds['sha256']
            with np.load(file, allow_pickle=False) as z:
                assert z['tokens'].shape == (256, 180, 180)
                assert np.array_equal(z['indices'], np.arange(ds['shard'] * 256, (ds['shard'] + 1) * 256))
                fresh.update(digest(t) for t in z['tokens'])
        assert len(fresh) == 4096
        pool = ProcessPoolExecutor(max_workers=8)
        labels = list(pool.map(check_dataset, [S.ROOT / ds['path'] for ds in manifest['datasets']]))
        pool.shutdown(wait=True)
        pool = None
        prior, seen = [], {}
        for registered in p['cohort']['prior_registry']:
            file = S.ROOT / registered['path']
            assert S.sha(file) == registered['sha256']
            old = json.loads(file.read_text())['datasets']
            datasets = old.values() if isinstance(old, dict) else old
            count = overlap = new_files = 0
            for ds in datasets:
                file = S.ROOT / ds['path']
                if file in seen:
                    assert seen[file] == ds['sha256']
                    continue
                assert S.sha(file) == ds['sha256']
                seen[file] = ds['sha256']
                new_files += 1
                with np.load(file, allow_pickle=False) as z:
                    for t in z['tokens']:
                        count += 1
                        overlap += digest(t) in fresh
            assert overlap == 0
            prior.append({**registered, 'newly_checked_files': new_files, 'inputs_checked': count, 'overlap': overlap})
        result = {'status': 'passed', 'protocol_sha256': S.sha(PP), 'datasets': manifest['datasets'],
                  'distances': labels, 'inputs_checked': 4096, 'unique': 4096,
                  'prior_manifest_checks': prior, 'prior_unique_dataset_files': len(seen),
                  'processes': 8, 'seconds': time.perf_counter() - started,
                  'source_sha256': S.sha(__file__), 'graph_checker_sha256': S.sha(S.HERE / 'audit_larger_relaxation.py'),
                  'model_outcomes_read': False}
        assert result['seconds'] <= 120
        S.save(DEST / 'independent-input-audit.json', result)
        state.update(status='complete', wall_seconds=result['seconds'])
        S.save(target, state)
        print(json.dumps({'status': 'passed', 'inputs': 4096, 'prior_files': len(seen), 'overlap': 0,
                          'seconds': result['seconds']}, indent=2))
    except BaseException as error:
        if pool is not None:
            for child in list((getattr(pool, '_processes', None) or {}).values()):
                if child.is_alive():
                    child.terminate()
            pool.shutdown(wait=False, cancel_futures=True)
        state.update(status='incomplete', error=repr(error), traceback=traceback.format_exc(),
                     wall_seconds=time.perf_counter() - started)
        S.save(target, state)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
