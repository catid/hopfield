"""Finish main CRAM controls, then run the preassigned data-efficiency lane."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/catid/hopfield')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def owners(gpu):
    uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    return [s for s in subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True).splitlines() if uuid in s]


def validate_marker(name, marker, seed):
    """A filename alone cannot certify a skipped, resumed control evaluation."""
    value = json.loads(marker.read_text())
    if name == 'sudoku-difficulty-zero':
        assert value['status'] == 'complete' and value['seed'] == seed
        assert value['test_size'] == 'full' and value['test_summary']['n'] == 412786
        assert value['test_rows'] == [10000, 422786] and value['stop'] == 'zero'
        assert value['model_sha256'] == sha(marker.parent / f'full-aug-s{seed}/model.pt')
        return
    n = int(name.split('-')[2] if name.startswith('nqueens-free-') else name.split('-')[1])
    assert value['seed'] == seed and value['test']['rows'] == {8: 1240, 10: 18466}[n]
    assert value['test']['chains'] == 20
    if name.startswith('nqueens-free-'):
        assert value['method'] == {8: 'mpl', 10: 'cd'}[n]
        metadata = ROOT / f'artifacts/multi_solution/free-nq{n}-s{seed}-provenance.json'
    else:
        precondition = name.rsplit('-', 1)[1]
        assert value['selection']['config']['precond'] == (precondition == 'on')
        metadata = ROOT / f'artifacts/multi_solution/precond-{precondition}-nq{n}-s{seed}-provenance.json'
    provenance = json.loads(metadata.read_text())
    assert provenance['final_sha256'] == sha(marker)
    assert provenance['model_sha256'] == sha(marker.parent / 'model.pt')
    assert provenance['records_sha256'] == sha(marker.parent / 'records.npz')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--wait-pid', type=int, required=True)
    args = p.parse_args()
    seed = args.seed
    out = ROOT / 'artifacts/main_followups'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'seed{seed}.json'
    record = {'seed': seed, 'gpu': seed, 'worker_pid': os.getpid(), 'wait_pid': args.wait_pid,
              'started_utc': now(), 'phases': [], 'script_sha256': sha(__file__)}
    def save(status):
        record.update(status=status, updated_utc=now())
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(path)
    while Path(f'/proc/{args.wait_pid}').exists():
        save('waiting_for_main_sudoku')
        time.sleep(30)
    lock = (ROOT / f'artifacts/gpu{seed}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    while owners(seed):
        save('waiting_for_free_gpu')
        time.sleep(30)
    python = ROOT / 'upstream/.venv/bin/python'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(seed), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               CRAM_DATA=str(ROOT / 'upstream/data'))
    jobs = [(f'nqueens-free-{n}', [str(python), '-u', str(ROOT / 'scripts/run_nqueens_free_reproduction.py'),
                                 '--n', str(n), '--seed', str(seed)],
             ROOT / f'upstream/experiments/runs/NQ/alt/n{n}/final/{"mpl" if n == 8 else "cd"}-s{seed}/final.json') for n in (8, 10)]
    for n, precond, relative in (
            (8, 'on', f'n8/final/mpl-table-s{seed}'),
            (8, 'off', f'n8/ablation/mpl-table-noprecond-s{seed}'),
            (10, 'off', f'n10/ablation/pl-ising-noprecond-s{seed}')):
        jobs.append((f'nqueens-{n}-precondition-{precond}',
                     [str(python), '-u', str(ROOT / 'scripts/run_nqueens_preconditioning.py'),
                      '--n', str(n), '--seed', str(seed), '--precond', precond],
                     ROOT / f'upstream/experiments/runs/NQ/nq11/{relative}/final.json'))
    jobs.append(('sudoku-difficulty-zero', [str(python), '-u', str(ROOT / 'scripts/run_sudoku_difficulty.py'),
                                          '--seed', str(seed), '--test-size', 'full'],
                 ROOT / f'artifacts/difficulty/full/provenance-s{seed}.json'))
    for name, argv, marker in jobs:
        phase = {'name': name, 'argv': argv, 'runner_sha256': sha(argv[2]), 'started_utc': now()}
        record['phases'].append(phase)
        save(name)
        with (out / f'seed{seed}-{name}.log').open('a') as log:
            proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            phase['pid'] = proc.pid
            while proc.poll() is None:
                phase['gpu_processes'] = owners(seed)
                save(name)
                time.sleep(15)
        phase.update(exit_code=proc.returncode, completed_utc=now())
        if proc.returncode:
            save('job_failed')
            raise RuntimeError(f'{name} failed; retained logs, no downstream job launched')
        if not marker.exists():
            save('completion_marker_missing')
            raise RuntimeError(f'Missing {marker}')
        try:
            validate_marker(name, marker, seed)
        except (AssertionError, KeyError, ValueError, OSError) as error:
            phase['validation_error'] = repr(error)
            save('completion_marker_invalid')
            raise
        phase['marker'] = str(marker)
        phase['marker_sha256'] = sha(marker)
        save(f'{name}_complete')
    lock.close()
    save('data_efficiency')
    proc = subprocess.run(['python3', '-u', str(ROOT / 'scripts/data_efficiency_queue.py'), 'worker', '--lane', str(seed)], cwd=ROOT)
    record['data_efficiency_exit_code'] = proc.returncode
    save('complete' if proc.returncode == 0 else 'data_efficiency_failed')
    proc.check_returncode()


if __name__ == '__main__':
    main()
