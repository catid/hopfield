"""Frozen CRAM data-efficiency queues on local GPUs after full Sudoku tests.

All 102 fits use a newly seeded shared reference, explicitly distinguished
from the paper's missing unseeded arrays. This never stops a GPU process.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import traceback

ROOT = Path('/home/catid/hopfield')
SOURCE = ROOT / 'dataeff_source'
UP = ROOT / 'upstream'
DATA = ROOT / 'reproduction/baselines/experiments/runs/trm/data'
OUT = ROOT / 'artifacts/data_efficiency'
CONFIG = ROOT / 'configs/data_efficiency_jobs.json'
KS = [1000, 30, 100, 300, 27, 25, 23, 21, 19, 17, 14, 12, 10, 3, 1, 3000, 10000]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def wait_with_progress(proc, on_progress, interval=15):
    """Refresh live state periodically, but resume dispatch as soon as a child exits."""
    while proc.poll() is None:
        on_progress()
        try:
            proc.wait(timeout=interval)
        except subprocess.TimeoutExpired:
            pass
    return proc.returncode


def build():
    if CONFIG.exists():
        raise FileExistsError('Frozen queue already exists; do not replace it during execution.')
    if SOURCE.exists():
        raise FileExistsError(f'Staging destination already exists: {SOURCE}')
    for folder in ('cram', 'experiments'):
        target = SOURCE / folder
        target.mkdir(parents=True)
        for src in (UP / folder).glob('*.py'):
            shutil.copy2(src, target / src.name)
    source_hashes = {str(p.relative_to(SOURCE)): sha(p) for p in sorted(SOURCE.rglob('*.py'))}
    jobs = [{'id': f'cram-de-v99-recreated42-k{k}-s{s}', 'k': k, 'seed': s, 'lane': i % 3}
            for i, (k, s) in enumerate((k, s) for k in KS for s in range(6))]
    write(CONFIG, {'created_utc': now(), 'jobs': jobs, 'source_sha256': source_hashes,
                  'source_commit': subprocess.check_output(['git', '-C', str(UP), 'rev-parse', 'HEAD'], text=True).strip(),
                  'reference_provenance_sha256': sha(DATA / 'sudoku-extreme-1k-aug-1000/build_provenance.json'),
                  'protocol': {'training_reference': 'Explicit seed42 recreation; original unseeded paper arrays unavailable',
                               'seeds': list(range(6)), 'variants': 99, 'test_n': 10000,
                               'updates': 32, 'warm_fit_repeats': 7, 'test_selection': 'No local tuning; frozen published recipe'},
                  'source': str(SOURCE), 'run_root': 'runs/CRAM-E/DE2-v99-recreated42',
                  'wait_for_existing_training_pids': {0: 1092433, 1: 1092582, 2: 1092432}})
    print(json.dumps({'jobs': len(jobs), 'lanes': 3, 'source_files': len(source_hashes)}))


def gpu(g):
    line = subprocess.check_output(['nvidia-smi', f'--id={g}', '--query-gpu=uuid,name,utilization.gpu,memory.used',
                                    '--format=csv,noheader'], text=True).strip()
    uuid = line.split(',')[0]
    all_owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                          '--format=csv,noheader'], text=True)
    return {'summary': line, 'uuid': uuid, 'owners': [x for x in all_owners.splitlines() if uuid in x]}


def validate_saved_outputs(run, job, directory):
    """Refuse stale fit/test records before an upstream exists-and-skip shortcut."""
    fit_path, test_path = run / 'fit.json', run / 'test10k.json'
    if not fit_path.exists():
        assert not test_path.exists(), 'Existing test has no completed fit; refusing to overwrite or rescore'
        return
    fit = json.loads(fit_path.read_text())
    expected = dict(aug=False, lam=3e-3 / 9, lr=540.0, momentum=0.5,
                    updates=32, split=job['seed'], seed=job['seed'])
    assert fit['config'] == expected and fit['k'] == job['k'] and fit['seed'] == job['seed']
    assert fit['examples'] == job['k'] * 100 and fit['variants'] == 99
    assert Path(fit['data']).resolve() == (directory / 'train').resolve()
    assert fit['model_sha256'] == sha(run / 'model.pt'), 'Saved fit/model mismatch'
    assert (run / 'train.npz').exists(), 'Saved fit has no training archive'
    if test_path.exists():
        result = json.loads(test_path.read_text())
        if 'skipped' in result:
            assert fit['tau'] is None and fit['positive_W'] == 0
            assert result['skipped'] == fit['degenerate']
        else:
            sig = result['signature']
            assert sig['k'] == job['k'] and sig['seed'] == job['seed']
            assert sig['model_sha256'] == fit['model_sha256'] and sig['tau'] == fit['tau']
            assert sig['data'] == 'test10k' and sig['n'] == result['summary']['n'] == 10000
            assert sig['rel'] == [.4, .31, .45] and sig['C'] == 4 and sig['check'] == 4096
            assert sig['cap'] == 4e11 and sig['tol'] == 1e-3
            assert result['per_puzzle']['puzzle'] == list(range(10000, 20000))


def worker(lane):
    config = json.loads(CONFIG.read_text())
    assert socket.gethostname().split('.')[0] == 'foureyes'
    for rel, expected in config['source_sha256'].items():
        assert sha(SOURCE / rel) == expected, rel
    assert sha(DATA / 'sudoku-extreme-1k-aug-1000/build_provenance.json') == config['reference_provenance_sha256']
    state_path = OUT / f'gpu{lane}-worker.json'
    pid = config['wait_for_existing_training_pids'][str(lane)]
    state = {'worker_pid': os.getpid(), 'host': socket.gethostname(), 'gpu': lane,
             'started_utc': now(), 'config_sha256': sha(CONFIG), 'worker_script_sha256': sha(__file__)}
    while Path(f'/proc/{pid}').exists():
        write(state_path, state | {'status': 'waiting_for_main_sudoku_test', 'original_pid': pid, 'utc': now()})
        time.sleep(30)
    # A failed prerequisite remains evidence; data-efficiency has no numerical
    # dependency on that test, so a free GPU can still execute its frozen jobs.
    lock = (ROOT / f'artifacts/gpu{lane}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               CRAM_DATA=str(UP / 'data'), V1_DATA=str(UP / 'data'), PYTHONPATH=str(SOURCE),
               TRM_DATA=str(DATA), DE_VARIANTS='99', CRAM_E_ROOT='DE2-v99-recreated42')
    for job in (j for j in config['jobs'] if j['lane'] == lane):
        while gpu(lane)['owners']:
            write(state_path, state | {'status': 'waiting_for_free_gpu', 'job': job['id'], 'utc': now()})
            time.sleep(30)
        directory = DATA / ('sudoku-extreme-1k-aug-1000' if job['k'] == 1000
                            else f'sudoku-smalldata/s{job["seed"]}-n{job["k"]}')
        marker = directory / ('build_provenance.json' if job['k'] == 1000 else 'subset.json')
        while not marker.exists():
            write(state_path, state | {'status': 'waiting_for_cpu_data', 'job': job['id'], 'data': str(marker), 'utc': now()})
            time.sleep(30)
        run = SOURCE / 'experiments' / config['run_root'] / f'k{job["k"]}-s{job["seed"]}'
        meta_path = OUT / 'jobs' / f'{job["id"]}.json'
        hashes = {name: sha(directory / 'train' / f'all__{name}.npy') for name in ('inputs', 'labels', 'group_indices')}
        manifest = json.loads(marker.read_text())
        expected_hashes = ({name: manifest['arrays'][f'train/all__{name}.npy'] for name in hashes}
                           if job['k'] == 1000 else {name: manifest['sha256'][name] for name in hashes})
        assert hashes == expected_hashes, 'Training arrays differ from their declared preparation hashes'
        if job['k'] != 1000:
            assert manifest['seed'] == job['seed'] and manifest['size'] == job['k']
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert meta['data_sha256'] == hashes and meta['config_sha256'] == sha(CONFIG)
            assert meta['data_manifest_sha256'] == sha(marker), 'Dataset provenance changed'
            validate_saved_outputs(run, job, directory)
            if 'training_archive_sha256' in meta:
                assert sha(run / 'train.npz') == meta['training_archive_sha256']
            if meta.get('status') == 'complete':
                assert set(meta['output_sha256']) == {'model.pt', 'fit.json', 'test10k.json'}
                for name, expected in meta['output_sha256'].items():
                    assert sha(run / name) == expected
                continue
        else:
            assert not (run / 'fit.json').exists(), 'Existing fit has no local provenance'
            assert not (run / 'test10k.json').exists(), 'Existing test has no local provenance'
            meta = job | {'started_utc': now(), 'config_sha256': sha(CONFIG), 'data_sha256': hashes,
                          'data_manifest_sha256': sha(marker), 'phases': [], 'host': socket.gethostname(),
                          'gpu': gpu(lane), 'environment': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'TRM_DATA', 'DE_VARIANTS', 'CRAM_E_ROOT')},
                          'protocol': config['protocol']}
        for phase in ('fit', 'search'):
            argv = [str(UP / '.venv/bin/python'), '-u', 'data_efficiency.py', phase,
                    '--k', str(job['k']), '--seed', str(job['seed'])]
            if phase == 'search':
                argv += ['--data', 'test10k']
            validate_saved_outputs(run, job, directory)
            completed_output = run / ('fit.json' if phase == 'fit' else 'test10k.json')
            if completed_output.exists():
                meta['phases'].append({'phase': phase, 'argv': argv, 'exit_code': 0,
                                      'reused_validated_output': True, 'completed_utc': now()})
                if phase == 'fit':
                    meta['training_archive_sha256'] = sha(run / 'train.npz')
                write(meta_path, meta)
                continue
            meta['status'] = phase
            write(meta_path, meta)
            with (OUT / f'{job["id"]}-{phase}.log').open('a') as log:
                proc = subprocess.Popen(argv, cwd=SOURCE / 'experiments', env=env, stdout=log, stderr=subprocess.STDOUT)
                wait_with_progress(proc, lambda: write(state_path, state | {
                    'status': phase, 'job': job['id'], 'child_pid': proc.pid,
                    'utc': now(), 'gpu_state': gpu(lane)}))
            meta['phases'].append({'phase': phase, 'argv': argv, 'exit_code': proc.returncode, 'completed_utc': now()})
            write(meta_path, meta)
            if proc.returncode:
                write(state_path, state | {'status': 'job_failed', 'job': job['id'], 'utc': now()})
                raise RuntimeError(f'{job["id"]} {phase} failed')
            validate_saved_outputs(run, job, directory)
            if phase == 'fit':
                meta['training_archive_sha256'] = sha(run / 'train.npz')
                write(meta_path, meta)
        fit = json.loads((run / 'fit.json').read_text())
        result = json.loads((run / 'test10k.json').read_text())
        assert fit['examples'] == job['k'] * 100 and fit['variants'] == 99
        if 'skipped' not in result:
            assert result['summary']['n'] == 10000 and result['signature']['n'] == 10000
        meta.update(status='complete', completed_utc=now(), summary=result.get('summary', result),
                    output_sha256={name: sha(run / name) for name in ('model.pt', 'fit.json', 'test10k.json')})
        write(meta_path, meta)
        print(json.dumps({'job': job['id'], 'summary': meta['summary']}), flush=True)
    write(state_path, state | {'status': 'complete', 'utc': now()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('build', 'worker'))
    p.add_argument('--lane', type=int, choices=(0, 1, 2))
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == 'build':
        build()
    else:
        if args.lane is None:
            p.error('--lane required for worker')
        try:
            worker(args.lane)
        except BaseException as error:
            path = OUT / f'gpu{args.lane}-worker.json'
            try:
                previous = json.loads(path.read_text()) if path.exists() else {}
            except (OSError, ValueError):
                previous = {}
            write(path, previous | {'status': 'worker_failed', 'worker_pid': os.getpid(),
                                    'utc': now(), 'error': repr(error), 'traceback': traceback.format_exc()})
            raise
