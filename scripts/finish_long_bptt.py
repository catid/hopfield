"""Wait for one already-running long BPTT fit, then evaluate its frozen choice.

Uses no GPU until the original process exits and its complete schedule is
verified. No partial schedule is promoted to a full reproduction.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/home/catid/hopfield')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_test_record(test, fit):
    """Validate an existing test before accepting the evaluator's no-rescore shortcut."""
    assert test['test']['n'] == 50000
    assert test['selected_update'] == fit['selected']['update'], 'Test belongs to a different selected update'
    assert test['puzzles'] == [10000, 59999]
    expected = dict(C=4, check=4096, tol=1e-3, period=524288,
                    hi=.4, lo=.31, theta=.45, cap=4e11)
    assert all(test['search'][key] == value for key, value in expected.items())
    assert len(test['flops']) == 50000


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--training-pid', type=int, required=True)
    args = p.parse_args()
    source = ROOT / 'upstream'
    relative = f'runs/ALT/longD3/sudoku-bptt-lr2-s{args.seed}'
    run = source / 'experiments' / relative
    out = ROOT / 'artifacts/long_bptt'
    out.mkdir(parents=True, exist_ok=True)
    previous_path = out / f'seed{args.seed}.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    record = {'seed': args.seed, 'gpu': args.gpu, 'host': socket.gethostname(),
              'training_pid': args.training_pid, 'started_utc': now(), 'run': str(run),
              'source_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
              'training_methods_sha256': sha(source / 'experiments/training_methods.py')}
    def save(status):
        record['status'] = status
        record['updated_utc'] = now()
        target = out / f'seed{args.seed}.json'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(target)
    save('waiting_for_existing_fit')
    while Path(f'/proc/{args.training_pid}').exists():
        time.sleep(30)
    final = run / 'final.json'
    if not final.exists():
        save('training_exited_without_final_record')
        raise RuntimeError('Training process exited before a complete final record')
    fit = json.loads(final.read_text())
    expected = {'task': 'sudoku', 'method': 'bptt', 'lr': 2.0, 'knob': 10.0,
                'l1': 0.1, 'seed': args.seed, 'updates': 3840000}
    for key, value in expected.items():
        if fit['config'][key] != value:
            save('configuration_mismatch')
            raise RuntimeError(f'Unexpected {key}: {fit["config"][key]}')
    if fit['updates'] != 3840000 or max(x['update'] for x in fit['curve']) != 3840000:
        save('incomplete_schedule')
        raise RuntimeError('The complete 3,840,000-update schedule did not finish')
    record['final_sha256'] = sha(final)
    record['selected_update'] = fit['selected']['update']
    record['selected_checkpoint_sha256'] = sha(run / f'model-u{fit["selected"]["update"]}.pt')
    if (run / 'test.json').exists():
        try:
            assert previous is not None, 'Existing test lacks its original launch receipt'
            assert previous.get('status') in ('evaluating_selected_checkpoint', 'complete'), 'Existing test has no successful or interrupted evaluation receipt'
            assert previous.get('final_sha256') == record['final_sha256'], 'Final selection changed since test launch'
            assert previous.get('selected_checkpoint_sha256') == record['selected_checkpoint_sha256'], 'Selected checkpoint changed since test launch'
            assert previous.get('training_methods_sha256') == record['training_methods_sha256'], 'Evaluator source changed since test launch'
            validate_test_record(json.loads((run / 'test.json').read_text()), fit)
        except (AssertionError, KeyError, ValueError) as error:
            record['error'] = repr(error)
            save('existing_test_mismatch')
            raise
    lock = (ROOT / f'artifacts/gpu{args.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    gpu_uuid = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=uuid',
                                        '--format=csv,noheader'], text=True).strip()
    while True:
        owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                          '--format=csv,noheader'], text=True)
        if gpu_uuid not in owners:
            break
        save('waiting_for_free_gpu')
        time.sleep(30)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    argv = [str(source / '.venv/bin/python'), '-u', 'training_methods.py', 'test', '--run', relative]
    record['test_argv'] = argv
    record['gpu_uuid'] = gpu_uuid
    save('evaluating_selected_checkpoint')
    with (out / f'seed{args.seed}-test.log').open('a') as log:
        result = subprocess.run(argv, cwd=source / 'experiments', env=env, stdout=log, stderr=subprocess.STDOUT)
    record['test_exit_code'] = result.returncode
    if result.returncode:
        save('test_failed')
        result.check_returncode()
    test = json.loads((run / 'test.json').read_text())
    try:
        validate_test_record(test, fit)
    except (AssertionError, KeyError, ValueError) as error:
        record['error'] = repr(error)
        save('test_record_invalid')
        raise
    record['test_sha256'] = sha(run / 'test.json')
    record['test'] = test['test']
    save('complete')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
