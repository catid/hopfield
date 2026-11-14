"""Monitor migrated searches and import finished receipts; never launch or signal work."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

import data_efficiency_queue as Q
from data_efficiency_extra import SSH

ROOT = Q.ROOT
HELPER_SHA = '5c2cea80dcb19d473e19893dafcc2a852e4dcf56107fe62452b7f676d6f9beb3'
HELPER = Q.OUT / f'handoff-helper-{HELPER_SHA}.py'


def snapshot(host, items):
    code = '''
import json,subprocess,socket
from pathlib import Path
out={'host':socket.gethostname(),'jobs':[]}
for item in ITEMS:
    path=Path(item['isolated_run']).parent/('sharded-search-s'+str(item['seed'])+'.json')
    row={'seed':item['seed'],'receipt_path':str(path),'receipt':json.loads(path.read_text()) if path.exists() else None}
    receipt=row['receipt']
    if receipt:
        receipt.pop('final_moves',None);receipt.pop('search_seconds',None)
    if receipt and receipt.get('pid'):
        try:
            p=Path('/proc')/str(receipt['pid']);raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
            row['process']={'pid':receipt['pid'],'ppid':int(fields[1]),'state':fields[0],'startticks':int(fields[19]),'cpu_ticks':int(fields[11])+int(fields[12]),'argv':(p/'cmdline').read_bytes().decode().rstrip('\\0').split('\\0')}
        except (FileNotFoundError,ProcessLookupError):row['process']=None
    out['jobs'].append(row)
for name,args in [('gpus',['--query-gpu=index,uuid,name,utilization.gpu,memory.used,power.draw','--format=csv,noheader']),('gpu_processes',['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'])]:
    proc=subprocess.run(['nvidia-smi']+args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
    out[name]={'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}
print(json.dumps(out))
'''.replace('ITEMS', repr(items))
    argv = ['python3', '-c', code]
    if host != socket.gethostname().split('.')[0]:
        argv = SSH + [host, shlex.join(argv)]
    return json.loads(subprocess.check_output(argv, text=True, timeout=35))


def observe(host, items):
    try:
        return snapshot(host, items)
    except Exception as error:
        return {'host': host, 'observation_error': repr(error), 'jobs': []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    journal = args.journal.resolve()
    assert journal.is_relative_to((Q.OUT / 'migrations').resolve())
    assert Q.sha(HELPER) == HELPER_SHA
    base = journal.parent
    admission = (base / 'import-watch.lock').open('a')
    fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'watching', 'started_utc': Q.now(), 'pid': os.getpid(),
             'argv': sys.argv, 'helper_path': str(HELPER), 'helper_sha256': HELPER_SHA,
             'policy': 'Read-only process/GPU snapshots; import completed results through reviewed helper; no search launches or signals.',
             'imports': {}, 'previous_progress': {}}
    Q.write(base / 'import-watch.json', state)
    with ThreadPoolExecutor(max_workers=4) as pool, (base / 'progress-audit.jsonl').open('a', buffering=1) as history:
        while True:
            record = json.loads(journal.read_text())
            groups = {host: [] for host in ('foureyes', 'ripper', 'work', 'loki')}
            for item in record['jobs']:
                if item.get('isolated_run'):
                    groups[item['target_host']].append(item)
            snapshots = list(pool.map(lambda pair: observe(*pair), groups.items()))
            audit = {'observed_utc': Q.now(), 'journal_status': record['status'], 'hosts': snapshots}
            for host in snapshots:
                for row in host['jobs']:
                    receipt = row['receipt']
                    if not receipt:
                        continue
                    seed = row['seed']
                    progress = receipt.get('progress', {})
                    current = {'moves': progress.get('moves_before_check', receipt.get('starting_move')),
                               'active_puzzles': progress.get('active_puzzles', receipt.get('starting_active_puzzles')),
                               'cpu_ticks': (row.get('process') or {}).get('cpu_ticks'),
                               'receipt_status': receipt['status']}
                    previous = state['previous_progress'].get(str(seed), {})
                    row['forward_progress'] = {'previous': previous, 'current': current,
                        'moves_increased': current['moves'] is not None and previous.get('moves') is not None and current['moves'] > previous['moves'],
                        'cpu_ticks_increased': current['cpu_ticks'] is not None and previous.get('cpu_ticks') is not None and current['cpu_ticks'] > previous['cpu_ticks']}
                    state['previous_progress'][str(seed)] = current
                    if receipt['status'] != 'complete' or str(seed) in state['imports']:
                        continue
                    assert Q.sha(HELPER) == HELPER_SHA
                    argv = [str(Q.UP / '.venv/bin/python'), str(HELPER), 'import', '--journal', str(journal), '--seed', str(seed)]
                    env = dict(os.environ, PYTHONPATH=str(ROOT / 'scripts'))
                    try:
                        imported = subprocess.run(argv, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
                        attempt = {'seed': seed, 'utc': Q.now(), 'returncode': imported.returncode, 'stdout': imported.stdout, 'stderr': imported.stderr}
                    except subprocess.TimeoutExpired as error:
                        attempt = {'seed': seed, 'utc': Q.now(), 'error': str(error)}
                    row['import_attempt'] = attempt
                    if attempt.get('returncode') == 0:
                        state['imports'][str(seed)] = attempt
            history.write(json.dumps(audit) + '\n')
            state.update(updated_utc=Q.now(), latest_audit=audit)
            if len(state['imports']) == len(record['jobs']):
                state.update(status='complete', completed_utc=Q.now())
            Q.write(base / 'import-watch.json', state)
            if state['status'] == 'complete':
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
