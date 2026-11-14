#!/usr/bin/env python3
"""Run two checkpoint-probe clients in a private single-GPU MPS session.

The enclosing Figure4 worker owns the physical GPU lock. Every client calls the
unchanged upstream probe on disjoint updates; atomic shard records are merged.
The daemon is fully stopped before this process returns to any subsequent fit.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import time

from objective_queue import digest, gpu_snapshot, now, write
from benchmark_deferred_probes import mps_processes


def mps_conflicts(gpu_uuid, proc_root=Path('/proc'), processes=None):
    """Allow private daemons only when their explicit GPU UUIDs are disjoint.

    Numeric CUDA ordinals and absent/unreadable environments cannot establish
    device isolation. They retain the conservative sequential fallback.
    """
    conflicts = []
    for process in mps_processes() if processes is None else processes:
        try:
            raw = (proc_root / str(process['pid']) / 'environ').read_bytes().split(b'\0')
            env = dict(item.decode().split('=', 1) for item in raw if b'=' in item)
            visible = env.get('CUDA_VISIBLE_DEVICES', '').split(',')
            pipe = env.get('CUDA_MPS_PIPE_DIRECTORY', '')
            explicit = bool(visible) and all(re.fullmatch(r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', v) for v in visible)
            private = False
            if re.fullmatch(r'/tmp/hopfield-probes-[0-9]+', pipe):
                try:
                    meta = Path(pipe).lstat()
                    private = stat.S_ISDIR(meta.st_mode) and stat.S_IMODE(meta.st_mode) == 0o700 and meta.st_uid == os.getuid()
                except OSError:
                    pass
            if not (explicit and private and gpu_uuid.lower() not in [v.lower() for v in visible]):
                conflicts.append({**process, 'visible_devices': visible, 'pipe': pipe})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, UnicodeDecodeError, ValueError) as exc:
            conflicts.append({**process, 'reason': type(exc).__name__})
    return conflicts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--python', required=True)
    p.add_argument('--run', required=True); p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--artifacts', type=Path, required=True); args = p.parse_args()
    rd = args.source / 'experiments' / args.run; out = args.artifacts; out.mkdir(parents=True, exist_ok=True)
    canonical = rd / 'probes.json'
    models = {int(f.stem[7:]): digest(f) for f in sorted(rd.glob('model-u*.pt'))}
    before = json.loads(canonical.read_text()) if canonical.exists() else {}
    assert set(map(int, before)) <= set(models)
    plan_path = out / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan['model_sha256'] == {str(k): v for k, v in models.items()}
        assert plan['run_config_sha256'] == digest(rd / 'config.json')
        assert plan['source_sha256'] == {name: digest(args.source / 'experiments' / name)
                                        for name in ('training_curves.py', 'training_methods.py')}
    else:
        pending = sorted(set(models) - set(map(int, before)))
        plan = {'run': args.run, 'model_sha256': models, 'run_config_sha256': digest(rd / 'config.json'),
                'groups': [pending[::2], pending[1::2]], 'created_utc': now(),
                'source_sha256': {name: digest(args.source / 'experiments' / name)
                                  for name in ('training_curves.py', 'training_methods.py')},
                'clients': 2, 'probe_n': 512, 'fit_concurrency': 0}
        write(plan_path, plan)
    def merge():
        got = dict(before)
        for i in range(2):
            path = out / f'shard{i}' / (rd.name + '.json')
            if path.exists():
                for key, value in json.loads(path.read_text()).items():
                    assert value['n'] == 512 and int(key) in models
                    assert key not in got or got[key] == value, 'A repeated probe changed result'
                    got[key] = value
        write(canonical, got)
        return got
    got = merge()
    if set(map(int, got)) == set(models):
        print(json.dumps({'status': 'verified_complete_probes', 'n': len(got)})); return
    binary = shutil.which('nvidia-cuda-mps-control')
    snapshot = gpu_snapshot(args.gpu); assert not snapshot['owners'], snapshot
    conflicts = mps_conflicts(snapshot['uuid'])
    if binary is None or conflicts:
        # Another owner's daemon is never changed. Preserve correctness and
        # forward progress through the source's ordinary resumable probe.
        argv = [args.python, '-u', 'training_curves.py', 'probe', '--runs', args.run]
        write(out / 'fallback.json', {'utc': now(), 'reason': 'MPS unavailable or device isolation unproven',
                                     'conflicts': conflicts, 'argv': argv})
        subprocess.run(argv, cwd=args.source / 'experiments', check=True); return
    pipe = Path(f'/tmp/hopfield-probes-{os.getpid()}'); pipe.mkdir(mode=0o700)
    logs = out / f'mps-logs-{os.getpid()}'; logs.mkdir()
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': snapshot['uuid'], 'CUDA_MPS_PIPE_DIRECTORY': str(pipe),
           'CUDA_MPS_LOG_DIRECTORY': str(logs)}
    status = {'status': 'starting', 'started_utc': now(), 'gpu_uuid': snapshot['uuid'], 'pipe': str(pipe),
              'logs': str(logs), 'plan_sha256': digest(plan_path), 'argv': [], 'mps_owned_pids': []}
    status_path = out / 'execution.json'; write(status_path, status)
    children = []; begin = time.monotonic(); stopped = False
    def interrupt(_signal, _frame): raise KeyboardInterrupt('MPS probe phase interrupted')
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    try:
        start = subprocess.run([binary, '-d'], env=env, capture_output=True, text=True, timeout=20)
        status['mps_start'] = {'exit_code': start.returncode, 'stdout': start.stdout, 'stderr': start.stderr}
        assert start.returncode == 0, status['mps_start']
        for i, updates in enumerate(plan['groups']):
            if not updates: continue
            argv = [args.python, '-u', 'training_curves.py', 'probe', '--runs', args.run,
                    '--updates', *map(str, updates), '--out', str(out / f'shard{i}')]
            with (out / f'shard{i}.log').open('ab', buffering=0) as log:
                child = subprocess.Popen(argv, cwd=args.source / 'experiments', env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child); status['argv'].append({'argv': argv, 'pid': child.pid})
        last_n, last_progress = len(got), time.monotonic()
        while any(c.poll() is None for c in children):
            got = merge()
            if len(got) != last_n: last_n, last_progress = len(got), time.monotonic()
            if time.monotonic() - last_progress > 600:
                raise TimeoutError('No completed development probe for600seconds')
            sample = {'utc': now(), 'completed_probes': len(got), 'expected_probes': len(models),
                      'gpu': gpu_snapshot(args.gpu), 'elapsed_s': time.monotonic()-begin,
                      'mps_owned_pids': mps_processes(pipe)}
            with (out / 'monitor.jsonl').open('a') as f: f.write(json.dumps(sample) + '\n')
            status.update(status='running', **sample); write(status_path, status)
            time.sleep(5)
        assert all(c.returncode == 0 for c in children), [c.returncode for c in children]
        got = merge(); assert set(map(int, got)) == set(models)
        status.update(status='complete', completed_probes=len(got), finished_utc=now(),
                      wall_seconds=time.monotonic()-begin, probes_sha256=digest(canonical))
    except BaseException as exc:
        status.update(status='failed', error=repr(exc)); raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
        try:
            quit_result = subprocess.run([binary], input='quit\n', env=env, capture_output=True, text=True, timeout=15)
            status['mps_stop'] = {'exit_code': quit_result.returncode, 'stdout': quit_result.stdout, 'stderr': quit_result.stderr}
        except Exception as exc:
            status['mps_stop'] = {'error': repr(exc)}
        for _ in range(30):
            if not mps_processes(pipe): break
            time.sleep(.1)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            owned = mps_processes(pipe)
            if not owned: break
            status['mps_stop'].setdefault('forced_owned_processes', []).extend(owned)
            for process in owned: os.kill(process['pid'], sig)
            time.sleep(1)
        remaining = mps_processes(pipe)
        stopped = not remaining
        status['mps_stop'].update(fully_stopped_before_next_fit=stopped, remaining_owned_processes=remaining)
        write(status_path, status)
        if not stopped:
            raise RuntimeError('Task-owned MPS daemon remains; do not dispatch a fit')
    print(json.dumps({'status': status['status'], 'probes': len(got), 'wall_seconds': status.get('wall_seconds'),
                      'mps_fully_stopped': stopped}), flush=True)


if __name__ == '__main__': main()
