"""Admit one detached continuation with a durable host-side launch receipt."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--devices', required=True)
    parser.add_argument('--host', required=True)
    args = parser.parse_args()
    assert socket.gethostname() == args.host
    base = args.base.resolve()
    assert base.is_relative_to(ROOT / 'artifacts/data_efficiency/migrations')
    assert json.loads((base / 'handoff.json').read_text())['original_writer_stopped'] is True
    assert json.loads((base / 'proof.json').read_text())['status'] == 'validated'
    argv = [str(ROOT / 'upstream/.venv/bin/python'), '-u', str(ROOT / 'scripts/resume_sharded_data_efficiency.py'),
            '--run-dir', str(base / f'k1-s{args.seed}'), '--seed', str(args.seed), '--devices', args.devices,
            '--proof', str(base / 'proof.json'), '--handoff', str(base / 'handoff.json')]
    record = {'host': args.host, 'seed': args.seed, 'argv': argv,
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    # Exclusive intent makes uncertain launch replies safe: inspect the durable
    # receipt/process instead of dispatching another writer.
    with (base / 'launch-intent.json').open('x') as f:
        json.dump(record, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    env = dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    with (base / 'search.log').open('ab', buffering=0) as log:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    stat = Path(f'/proc/{child.pid}/stat').read_text()
    record.update(pid=child.pid, start_ticks=int(stat[stat.rfind(')') + 2:].split()[19]))
    path = base / 'launch-receipt.json'
    temp = path.with_suffix('.tmp.json')
    with temp.open('x') as f:
        json.dump(record, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
