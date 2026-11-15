"""Launch only the two H2br timed workers; archive device and owned-process counters."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / 'artifacts/research/20261114/work/bp_first_endpoint_dispatch'
parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['timing'])
args = parser.parse_args()
env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
logs = [(DEST / f'{args.mode}-gpu{g}.log').open('x') for g in [0, 1]]
children = [subprocess.Popen([sys.executable, str(Path(__file__).with_name('endpoint_dispatch_study.py')),
             args.mode, '--gpu', str(g)], cwd=ROOT, env=env, stdout=logs[g], stderr=subprocess.STDOUT)
            for g in [0, 1]]
started = time.monotonic()
limit = 3000 + 10
samples = []
forced = []
while True:
    sample = {'utc_seconds': time.time(), 'monitor_wall_seconds': time.monotonic() - started,
              'worker_pids': [c.pid for c in children], 'exit_codes': [c.poll() for c in children]}
    for key, query in [('gpu', '--query-gpu=index,uuid,utilization.gpu,memory.used'),
                       ('gpu_processes', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory')]:
        sample[key] = subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader'], text=True).splitlines()
    sample['workers'], sample['owned_process_counters'] = {}, {}
    for g, child in enumerate(children):
        path = DEST / f'{args.mode}-worker{g}.json'
        if path.exists():
            try:
                sample['workers'][str(g)] = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        proc = Path('/proc') / str(child.pid)
        if proc.exists():
            try:
                status = (proc / 'status').read_text().splitlines()
                sample['owned_process_counters'][str(child.pid)] = {
                    'stat': (proc / 'stat').read_text().strip(),
                    'status_memory_threads': [line for line in status if line.startswith(('VmRSS:', 'VmHWM:', 'Threads:'))]}
            except FileNotFoundError:
                pass
    samples.append(sample)
    target = DEST / f'live-{args.mode}.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'samples': samples, 'forced_owned_signals': forced}, indent=2) + '\n')
    temporary.replace(target)
    failed = any(c.poll() not in [None, 0] for c in children)
    for child in children:
        if child.poll() is None and (failed or time.monotonic() - started > limit):
            if not any(r['pid'] == child.pid for r in forced):
                child.send_signal(signal.SIGALRM)
                forced.append({'pid': child.pid, 'signal': 'SIGALRM', 'at_wall_seconds': time.monotonic() - started,
                               'reason': 'Peer failure' if failed else 'Monitor wall backstop'})
            else:
                first = next(r for r in forced if r['pid'] == child.pid)
                if time.monotonic() - started - first['at_wall_seconds'] > 7:
                    child.kill()
                    forced.append({'pid': child.pid, 'signal': 'SIGKILL', 'at_wall_seconds': time.monotonic() - started,
                                   'reason': 'Owned process ignored bounded graceful failure'})
    if all(c.poll() is not None for c in children):
        break
    time.sleep(1)
for log in logs:
    log.close()
result = {'exit_codes': [c.returncode for c in children], 'samples': len(samples),
          'wall_seconds': time.monotonic() - started, 'forced_owned_signals': forced,
          'independent_CPU_audit_overlapped': False}
(DEST / f'launcher-{args.mode}.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if any(c.returncode for c in children):
    raise SystemExit(1)
