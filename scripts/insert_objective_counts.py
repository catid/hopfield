#!/usr/bin/env python3
"""Restart exactly one owned queue at a child boundary to insert count work."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

root = Path('/home/catid/hopfield')
art = root / 'artifacts/objectives'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lane', required=True)
args = parser.parse_args()
host, gpu = args.lane.split(':')
assert host == socket.gethostname().split('.')[0].lower()
state_path = art / f'{host}-gpu{gpu}-state.json'
worker = json.loads((art / f'{host}-gpu{gpu}-worker.json').read_text())
pid = worker['worker_pid']
cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert b'worker' in cmd and args.lane.encode() in cmd
assert any(x.endswith(b'/scripts/objective_queue.py') for x in cmd)
start = time.monotonic()
restarted = False
try:
    while time.monotonic() - start < 1800:
        os.kill(pid, signal.SIGSTOP)
        time.sleep(0.01)
        ids = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        active = []
        for child in ids:
            try:
                child_cmd = Path(f'/proc/{child}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            except FileNotFoundError:
                continue
            if child_cmd and not child_cmd.startswith('nvidia-smi '):
                active.append(child_cmd)
        if not active:
            record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'old_worker_pid': pid, 'previous_state': json.loads(state_path.read_text()),
                      'reason': 'Insert operation counts at child boundary; no training or test child was active.'}
            # The stopped parent cannot dispatch a new job between our child
            # check and its exit. This releases flock without touching a fit.
            os.kill(pid, signal.SIGKILL)
            restarted = True
            with (art / f'worker-{host}-{gpu}-count-restart.log').open('ab', buffering=0) as log:
                child = subprocess.Popen(['/usr/bin/python3', str(root / 'scripts/objective_queue.py'),
                    'worker', '--lane', args.lane, '--count-first'], cwd=root,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['new_worker_pid'] = child.pid
            (art / 'count-insertion.json').write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps(record), flush=True)
            break
        os.kill(pid, signal.SIGCONT)
        time.sleep(.2)
    else:
        raise RuntimeError('No safe queue boundary observed')
finally:
    if not restarted:
        os.kill(pid, signal.SIGCONT)
