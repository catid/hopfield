"""Bounded four-worker queue launcher; only owns its four process groups."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/research/20261114/foureyes/logical_completion'


def main():
    p = json.loads((HERE/'logical_completion.json').read_text())
    start = time.monotonic();handles = [];workers = [];samples = []
    try:
        for gpu in p['budget']['gpus']:
            log = (OUT/f'gpu{gpu}.log').open('x');handles.append(log)
            workers.append(subprocess.Popen([str(ROOT/'upstream/.venv/bin/python'), str(HERE/'logical_completion_search.py'), '--gpu', str(gpu)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        while any(w.poll() is None for w in workers):
            elapsed = time.monotonic()-start
            if elapsed > p['budget']['max_queue_wall_seconds'] or any(w.poll() not in [None, 0] for w in workers):
                raise RuntimeError('queue deadline or worker failure')
            sample = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader'], text=True)
            queue = json.loads((OUT/'queue.json').read_text())
            samples.append(dict(elapsed=elapsed, gpu=sample, completed=sum(j['status'] == 'complete' for j in queue), running=sum(j['status'] == 'running' for j in queue)))
            time.sleep(2)
        assert all(w.returncode == 0 for w in workers)
        result = dict(status='complete', wall_seconds=time.monotonic()-start, worker_pids=[w.pid for w in workers],
            returncodes=[w.returncode for w in workers], samples=samples,
            launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (OUT/'launch-result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'samples'}))
    finally:
        for worker in workers:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
                try:worker.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(worker.pid, signal.SIGKILL);worker.wait()
        for handle in handles:handle.close()


if __name__ == '__main__':
    main()
