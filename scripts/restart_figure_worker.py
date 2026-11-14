#!/usr/bin/env python3
"""Reload a Figure4 worker only after its active deferred probe completes."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from objective_queue import ROOT, digest, now, write
from figure4_queue import validate_curve

p = argparse.ArgumentParser(description=__doc__); p.add_argument('--lane', required=True); args = p.parse_args()
host, gpu = args.lane.split(':'); a = ROOT / 'artifacts/figure4'
assert host == socket.gethostname().split('.')[0].lower(), 'Lane belongs to another host'
state_path = a / f'{host}-gpu{gpu}-state.json'
record_path = a / f'reload-{host}-{gpu}-{int(time.time())}.json'
parent = None; stopped = False; killed = False
record = {'lane': args.lane, 'status': 'waiting_for_probe_boundary', 'started_utc': now(),
          'script_sha256': digest(__file__)}; write(record_path, record)
def interrupt(_sig, _frame): raise KeyboardInterrupt('Worker reload interrupted')
signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
try:
    while True:
        state = json.loads(state_path.read_text())
        if state.get('status') == 'running' and state.get('stage') == 1:
            parent, child = state['worker_pid'], state['child_pid']
            parent_args = Path(f'/proc/{parent}/cmdline').read_bytes().decode().strip('\0').split('\0')
            assert str(ROOT / 'scripts/figure4_queue.py') in parent_args
            assert parent_args[parent_args.index('--lane') + 1] == args.lane
            os.kill(parent, signal.SIGSTOP); stopped = True; time.sleep(.02)
            if str(child) in Path(f'/proc/{parent}/task/{parent}/children').read_text().split():
                config = json.loads((ROOT / 'configs/figure4_jobs.json').read_text())
                job = next(j for j in config['jobs'] if j['id'] == state['job'])
                assert job['lane'] == args.lane and not job['dense']
                child_args = Path(f'/proc/{child}/cmdline').read_bytes().decode().strip('\0').split('\0')
                if 'training_curves.py' in child_args:
                    assert 'probe' in child_args and child_args[child_args.index('--runs') + 1] == job['run_relative']
                else:
                    assert str(ROOT / 'scripts/run_mps_probes.py') in child_args
                    assert child_args[child_args.index('--run') + 1] == job['run_relative']
                    assert child_args[child_args.index('--gpu') + 1] == gpu
                break
            os.kill(parent, signal.SIGCONT); stopped = False
        time.sleep(5)
    record.update(status='waiting_for_existing_probe', old_worker_pid=parent, probe_pid=child, job=state['job']); write(record_path, record)
    while Path(f'/proc/{child}/stat').exists() and Path(f'/proc/{child}/stat').read_text().split(') ')[1].split()[0] != 'Z':
        time.sleep(5)
    config = json.loads((ROOT / 'configs/figure4_jobs.json').read_text())
    job = next(j for j in config['jobs'] if j['id'] == state['job'])
    run = Path(config['source']) / 'experiments' / job['run_relative']
    validation = validate_curve(run, job)
    proc = Path(f'/proc/{child}/stat')
    if proc.exists(): assert int(proc.read_text().split(') ')[1].split()[49]) == 0
    execution_path = a / 'jobs' / job['id'] / 'execution.json'
    execution = json.loads(execution_path.read_text())
    # Record the completed native probe before replacing its paused supervisor.
    execution.update(status='complete', validation=validation, finished_utc=now(),
        supervisor_boundary_receipt=str(record_path),
        outputs={name: digest(run / name) for name in ('config.json', 'final.json', 'log.jsonl', 'probes.json')})
    write(execution_path, execution)
    os.kill(parent, signal.SIGKILL); killed = True; stopped = False
    record.update(status='reloading', completed_curve=job['id'], validated_probes=validation['n_probes']); write(record_path, record)
finally:
    if killed:
        with record_path.with_suffix('.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(['/usr/bin/python3', str(ROOT / 'scripts/figure4_queue.py'), 'worker', '--lane', args.lane],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(status='resumed', new_worker_pid=child.pid, config_sha256=digest(ROOT / 'configs/figure4_jobs.json'), resumed_utc=now())
        write(record_path, record)
    elif stopped: os.kill(parent, signal.SIGCONT)
