#!/usr/bin/env python3
"""Compare identical development probes sequentially and in two CUDA contexts.

Wait for the active deferred-probe child to finish, benchmark on the same owned
GPU under its physical lock, and restart its Figure 4 worker on every exit.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import shutil
import subprocess
import time
import traceback

from objective_queue import ROOT, digest, gpu_snapshot, now, write
from figure4_queue import validate_curve


def mps_processes(pipe=None):
    result = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            if Path(argv[0].decode()).name not in ('nvidia-cuda-mps-control', 'nvidia-cuda-mps-server'):
                continue
            if pipe is not None:
                env = (proc / 'environ').read_bytes().split(b'\0')
                if ('CUDA_MPS_PIPE_DIRECTORY=' + str(pipe)).encode() not in env:
                    continue
            result.append({'pid': int(proc.name), 'executable': argv[0].decode()})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--updates', type=int, nargs=2, required=True)
    parser.add_argument('--mps', action='store_true')
    args = parser.parse_args()
    host, raw_gpu = args.lane.split(':'); gpu = int(raw_gpu)
    assert host == socket.gethostname().split('.')[0].lower()
    config_path = ROOT / 'configs/figure4_jobs.json'; config = json.loads(config_path.read_text())
    assert args.lane in config['lanes']
    source = Path(config['source']); run = source / 'experiments' / args.run
    models = {f'model-u{k}.pt': digest(run / f'model-u{k}.pt') for k in args.updates}
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + ('-mps' if args.mps else '')
    artifact = ROOT / 'artifacts/figure4/probe_benchmark' / stamp; artifact.mkdir(parents=True)
    state_path = ROOT / 'artifacts/figure4' / f'{host}-gpu{gpu}-state.json'
    status_path = artifact / 'status.json'
    record = {'lane': args.lane, 'run': args.run, 'updates': args.updates, 'started_utc': now(),
              'model_sha256': models, 'source_commit': config['source_commit'],
              'config_sha256': digest(config_path), 'script_sha256': digest(__file__),
              'source_sha256': config['source_sha256'], 'data_manifest': config['data_manifest'],
              'fresh_hbar_prerequisite_sha256': config['fresh_hbar_prerequisite_sha256'],
              'criterion': 'Identical results and at least 20 percent shorter end-to-end wall time before adoption.',
              'mps': 'task-owned GPU UUID only' if args.mps else 'not enabled or modified',
              'status': 'waiting_for_safe_boundary'}
    write(status_path, record)
    parent = None; stopped = False; killed = False; lock = None; children = []
    mps_pipe = None; mps_env = None; mps_binary = shutil.which('nvidia-cuda-mps-control')
    if args.mps and (mps_binary is None or mps_processes()):
        record.update(status='not_attempted', reason='MPS binary absent or an existing MPS daemon makes isolation ambiguous')
        write(status_path, record)
        return
    def stop(_signal, _frame):
        raise KeyboardInterrupt('Probe benchmark interrupted')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        while True:
            state = json.loads(state_path.read_text())
            if state.get('status') == 'running' and state.get('stage') == 1:
                parent = state['worker_pid']; active = state['child_pid']
                parent_cmd = Path(f'/proc/{parent}/cmdline').read_bytes()
                assert b'figure4_queue.py' in parent_cmd and args.lane.encode() in parent_cmd
                os.kill(parent, signal.SIGSTOP); stopped = True
                time.sleep(.02)
                descendants = Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
                if str(active) in descendants:
                    child_cmd = Path(f'/proc/{active}/cmdline').read_bytes()
                    if b'training_curves.py' in child_cmd and b'probe' in child_cmd:
                        break
                os.kill(parent, signal.SIGCONT); stopped = False
            time.sleep(5)
        # The current GPU child remains running; only its idle supervising
        # parent is paused so it cannot dispatch another fit at this boundary.
        record.update(old_worker_pid=parent, current_probe_pid=active, current_job=state['job'])
        while True:
            proc = Path(f'/proc/{active}/stat')
            if not proc.exists() or proc.read_text().split(') ')[1].split()[0] == 'Z':
                break
            sample = {'utc': now(), 'status': 'waiting_for_probe_to_finish', 'gpu': gpu_snapshot(gpu)}
            with (artifact / 'monitor.jsonl').open('a') as f: f.write(json.dumps(sample) + '\n')
            record.update(status=sample['status'], gpu=sample['gpu']); write(status_path, record)
            time.sleep(5)
        job = next(j for j in config['jobs'] if j['id'] == state['job'])
        record['completed_boundary_curve'] = validate_curve(source / 'experiments' / job['run_relative'], job)
        os.kill(parent, signal.SIGKILL); killed = True; stopped = False
        lock_path = Path(config['physical_gpu_locks'][args.lane])
        lock = lock_path.open('a'); fcntl.flock(lock, fcntl.LOCK_EX)
        snapshot = gpu_snapshot(gpu)
        if snapshot['owners']:
            raise RuntimeError(f'GPU still has a compute owner: {snapshot}')
        record.update(status='benchmarking', gpu=snapshot, boundary_utc=now(), lock=str(lock_path))
        write(status_path, record)
        env_changes = {'CUDA_VISIBLE_DEVICES': str(gpu), 'CRAM_DATA': config['data'], 'V1_DATA': config['data'],
                       'FRM_DATA': str(Path(config['data']) / 'frm/split'), 'PYTHONPATH': str(source),
                       'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}
        if args.mps:
            assert not mps_processes(), 'Another MPS daemon appeared; do not change it'
            mps_pipe = Path(f'/tmp/hopfield-mps-{os.getpid()}')
            mps_pipe.mkdir(mode=0o700)
            mps_logs = artifact / 'mps-logs'; mps_logs.mkdir()
            mps_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': snapshot['uuid'],
                       'CUDA_MPS_PIPE_DIRECTORY': str(mps_pipe), 'CUDA_MPS_LOG_DIRECTORY': str(mps_logs)}
            started = subprocess.run([mps_binary, '-d'], env=mps_env, capture_output=True, text=True, timeout=20)
            record['mps_start'] = {'argv': [mps_binary, '-d'], 'gpu_uuid': snapshot['uuid'],
                'pipe_directory': str(mps_pipe), 'log_directory': str(mps_logs), 'exit_code': started.returncode,
                'stdout': started.stdout, 'stderr': started.stderr, 'utc': now()}
            write(status_path, record)
            assert started.returncode == 0, 'Task-owned MPS start failed'
            env_changes.update(CUDA_VISIBLE_DEVICES=snapshot['uuid'], CUDA_MPS_PIPE_DIRECTORY=str(mps_pipe),
                               CUDA_MPS_LOG_DIRECTORY=str(mps_logs))
        record['env'] = env_changes
        for phase in ('sequential', 'parallel'):
            phase_dir = artifact / phase; phase_dir.mkdir()
            groups = [args.updates] if phase == 'sequential' else [[k] for k in args.updates]
            phase_record = {'started_utc': now(), 'processes': []}
            begin = time.monotonic(); children = []
            for index, updates in enumerate(groups):
                dest = phase_dir / str(index)
                argv = [config['python'], '-u', 'training_curves.py', 'probe', '--runs', args.run,
                        '--updates', *map(str, updates), '--out', str(dest)]
                with (phase_dir / f'{index}.log').open('ab', buffering=0) as log:
                    child = subprocess.Popen(argv, cwd=source / 'experiments', env={**os.environ, **env_changes},
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                phase_record['processes'].append({'argv': argv, 'pid': child.pid, 'cwd': str(source / 'experiments'),
                                                  'out': str(dest / (run.name + '.json'))})
            while any(c.poll() is None for c in children):
                if time.monotonic() - begin > 180:
                    raise TimeoutError(f'{phase} probe benchmark exceeded180seconds')
                sample = {'utc': now(), 'phase': phase, 'elapsed_s': time.monotonic() - begin,
                          'gpu': gpu_snapshot(gpu), 'process_poll': [c.poll() for c in children]}
                with (artifact / 'monitor.jsonl').open('a') as f: f.write(json.dumps(sample) + '\n')
                record.update(status='benchmarking', phase=phase, phase_elapsed_s=sample['elapsed_s'])
                write(status_path, record); time.sleep(1)
            phase_record.update(wall_seconds=time.monotonic() - begin, finished_utc=now(),
                                exit_codes=[c.returncode for c in children])
            assert all(c.returncode == 0 for c in children), phase_record
            results = {}
            for process in phase_record['processes']:
                got = json.loads(Path(process['out']).read_text())
                assert not set(results) & set(got)
                results.update(got)
            assert set(results) == {str(k) for k in args.updates}
            assert all(x['n'] == 512 for x in results.values())
            phase_record['results'] = results
            record[phase] = phase_record; write(status_path, record)
            if args.mps:
                server_list = subprocess.run([mps_binary], input='get_server_list\n', env=mps_env,
                                             capture_output=True, text=True, timeout=10)
                phase_record['mps_server_list'] = {'stdout': server_list.stdout, 'stderr': server_list.stderr,
                                                  'exit_code': server_list.returncode}
                phase_record['task_owned_mps_processes'] = mps_processes(mps_pipe)
                write(status_path, record)
        same = record['sequential']['results'] == record['parallel']['results']
        speedup = record['sequential']['wall_seconds'] / record['parallel']['wall_seconds']
        record.update(status='complete', completed_utc=now(), identical_results=same,
                      speedup=speedup, worthwhile=same and speedup >= 1.25,
                      adoption='pending_review' if same and speedup >= 1.25 else 'keep_sequential')
        assert same, 'Concurrent probe solve counts differ'
        write(status_path, record)
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        write(status_path, record)
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=30)
        resume_allowed = True
        if mps_pipe is not None:
            stopped_mps = subprocess.run([mps_binary], input='quit\n', env=mps_env,
                                        capture_output=True, text=True, timeout=15)
            record['mps_stop'] = {'utc': now(), 'exit_code': stopped_mps.returncode,
                                  'stdout': stopped_mps.stdout, 'stderr': stopped_mps.stderr}
            for _ in range(30):
                if not mps_processes(mps_pipe):
                    break
                time.sleep(.1)
            remaining = mps_processes(mps_pipe)
            if remaining:
                # Only processes whose own environment proves ownership of our
                # private pipe can be stopped; no global daemon is touched.
                record['mps_stop']['forced_owned_pids'] = remaining
                for process in remaining:
                    os.kill(process['pid'], signal.SIGTERM)
                time.sleep(1)
            remaining = mps_processes(mps_pipe)
            if remaining:
                for process in remaining:
                    os.kill(process['pid'], signal.SIGKILL)
                time.sleep(1)
            remaining = mps_processes(mps_pipe)
            record['mps_stop']['remaining_owned_processes'] = remaining
            resume_allowed = not remaining
            record['mps_stop']['fully_stopped_before_resume'] = resume_allowed
            write(status_path, record)
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
        if killed and resume_allowed:
            with (artifact / 'worker-resume.log').open('ab', buffering=0) as log:
                resumed = subprocess.Popen(['/usr/bin/python3', str(ROOT / 'scripts/figure4_queue.py'),
                    'worker', '--lane', args.lane], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['resumed_worker_pid'] = resumed.pid; record['resumed_utc'] = now(); write(status_path, record)
        elif killed:
            record.update(status='resume_blocked', reason='Task-owned MPS processes remain; no fit dispatched')
            write(status_path, record)
        elif stopped:
            os.kill(parent, signal.SIGCONT)


if __name__ == '__main__':
    main()
