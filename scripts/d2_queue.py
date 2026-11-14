#!/usr/bin/env python3
"""Run nine development-only implicit long fits between Table 7 and Figure 4."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from objective_queue import ROOT, digest, gpu_snapshot, now, progress, verify, write


def worker(args):
    config_path = Path(args.config); config = json.loads(config_path.read_text())
    host, index = args.lane.split(':'); gpu = int(index)
    assert host == socket.gethostname().split('.')[0].lower() and args.lane in config['lanes']
    artifacts = ROOT / 'artifacts/d2'; artifacts.mkdir(parents=True, exist_ok=True)
    state_path = artifacts / f'{host}-gpu{gpu}-state.json'
    figure_state = ROOT / 'artifacts/figure4' / f'{host}-gpu{gpu}-state.json'
    objective_state = ROOT / 'artifacts/objectives' / f'{host}-gpu{gpu}-state.json'
    base = {'lane': args.lane, 'worker_pid': os.getpid(), 'source_commit': config['source_commit'],
            'configuration_sha256': digest(config_path), 'started_utc': now()}
    while not objective_state.exists() or json.loads(objective_state.read_text())['status'] != 'queue_complete':
        write(state_path, {**base, 'status': 'queued_after_table7', 'utc': now()})
        write(figure_state, {**base, 'status': 'queued_after_d2', 'utc': now()})
        time.sleep(30)
    lock = (ROOT / 'artifacts/objectives' / f'gpu-{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    base['validation'] = verify(config)
    base['runtime'] = json.loads(subprocess.check_output([config['python'], '-c',
        'import json,platform,torch,triton; print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__)))'], text=True))
    write(artifacts / f'{host}-gpu{gpu}-worker.json', base)
    source = Path(config['source'])
    for job in config['jobs']:
        if job['lane'] != args.lane:
            continue
        run = source / 'experiments' / job['run_relative']
        path = artifacts / 'jobs' / job['id'] / 'execution.json'
        identity = hashlib.sha256(json.dumps({'job': job, 'source': config['source_sha256'],
                    'data': config['data_manifest'], 'hbar': config['fresh_hbar_prerequisite_sha256']}, sort_keys=True).encode()).hexdigest()
        owner = run / 'd2_provenance.json'
        if owner.exists():
            assert json.loads(owner.read_text())['identity'] == identity
        elif run.exists() and any(run.iterdir()):
            raise RuntimeError(f'Refusing unprovenanced output: {run}')
        else:
            write(owner, {**base, 'identity': identity, 'job': job})
        record = json.loads(path.read_text()) if path.exists() else {**base, 'job': job, 'identity': identity}
        if record.get('status') == 'complete':
            assert all(digest(run / name) == sha for name, sha in record['outputs'].items())
            continue
        while gpu_snapshot(gpu)['owners']:
            write(state_path, {**base, 'status': 'waiting_for_gpu', 'job': job['id'], 'utc': now()}); time.sleep(15)
        env_changes = {**job['env'], 'CUDA_VISIBLE_DEVICES': str(gpu), 'WANDB_MODE': 'disabled'}
        record.update(status='running', env=env_changes, gpu=gpu_snapshot(gpu), started_utc=now()); write(path, record)
        start = time.monotonic()
        if not (run / 'final.json').exists():
            with (path.parent / 'train.log').open('ab', buffering=0) as log:
                child = subprocess.Popen(job['argv'], cwd=job['cwd'], env={**os.environ, **env_changes},
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    state = {**base, 'status': 'running', 'job': job['id'], 'child_pid': child.pid,
                             'gpu': gpu_snapshot(gpu), 'progress': progress(run), 'elapsed_s': time.monotonic()-start, 'utc': now()}
                    write(state_path, state)
                    write(figure_state, {**base, 'status': 'queued_after_d2', 'd2_job': job['id'], 'utc': now()})
                    with (path.parent / 'monitor.jsonl').open('a') as f: f.write(json.dumps(state) + '\n')
                    try: child.wait(timeout=15)
                    except subprocess.TimeoutExpired: pass
                record.update(exit_code=child.returncode, wall_seconds=time.monotonic()-start)
                if child.returncode != 0 or not (run / 'final.json').exists():
                    record.update(status='failed'); write(path, record)
                    raise RuntimeError(f'D.2 child failed: {job["id"]}')
        fin = json.loads((run / 'final.json').read_text())
        cfg = json.loads((run / 'config.json').read_text())
        for key in ('method', 'seed', 'lr', 'knob', 'l1'):
            assert cfg[key] == job[key], (key, cfg[key], job[key])
        assert cfg['updates_resolved'] == 60000
        last = progress(run)['last_update']['update']
        curve = fin['curve']
        valid = [r for r in curve if r['n'] == 512]
        assert all(r['n'] in (1, 512) for r in curve)
        assert not (run / 'test.json').exists(), 'D.2 jobs never score test puzzles'
        record.update(status='complete', finished_utc=now(), scheduled_updates=60000, completed_updates=last,
                      schedule_completed=last == 60000, stop=fin['stop'], probe_n=512, test_evaluated=False,
                      max_probe_solved_percent=max((100*r['solved']/r['n'] for r in valid), default=None),
                      probe_curve=curve, fit_s=fin['fit_s'],
                      outputs={name: digest(run / name) for name in ('config.json', 'final.json', 'log.jsonl')})
        write(path, record)
        write(state_path, {**base, 'status': 'complete', 'job': job['id'], 'utc': now()})
    argv = ['/usr/bin/python3', str(ROOT / 'scripts/figure4_queue.py'), 'worker', '--lane', args.lane]
    write(state_path, {**base, 'status': 'queue_complete', 'handoff_argv': argv, 'utc': now()})
    fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    os.execv(argv[0], argv)


def collect(args):
    config = json.loads(Path(args.config).read_text()); root = ROOT / 'artifacts/d2'
    def fetch(host):
        dest = root / host; dest.mkdir(parents=True, exist_ok=True)
        argv = ['rsync', '-a', '--partial', '-e', 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes',
                '--include=*/', '--include=*.json', '--include=*.jsonl', '--include=*.log', '--exclude=*']
        subprocess.run(argv + [f'{host}:{ROOT}/artifacts/d2/', str(dest) + '/'], check=True, capture_output=True, text=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch, h) for h in ('loki', 'work')]
        for f in futures: f.result()
    jobs, lanes = [], {}
    for lane in config['lanes']:
        host, gpu = lane.split(':'); p = root / host / f'{host}-gpu{gpu}-state.json'
        if p.exists(): lanes[lane] = json.loads(p.read_text())
    for job in config['jobs']:
        p = root / job['lane'].split(':')[0] / 'jobs' / job['id'] / 'execution.json'
        jobs.append(json.loads(p.read_text()) if p.exists() else {**job, 'status': 'queued_after_table7'})
    counts = {s: sum(j['status'] == s for j in jobs) for s in sorted({j['status'] for j in jobs})}
    write(root / 'summary.json', {'generated_utc': now(), 'jobs': jobs, 'lanes': lanes, 'status_counts': counts,
          'claim': config['claim'], 'no_test_evaluation': True})
    print(json.dumps({'summary': str(root / 'summary.json'), 'status_counts': counts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('worker'); p.add_argument('--lane', required=True)
    p.add_argument('--config', default=str(ROOT / 'configs/d2_supplemental_jobs.json'))
    p = sub.add_parser('collect'); p.add_argument('--config', default=str(ROOT / 'configs/d2_supplemental_jobs.json'))
    args = parser.parse_args(); globals()[args.command](args)
