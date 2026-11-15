"""Run two bounded H2bk workers and overlap audits only after dependencies exist."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'artifacts/research/20261114/work/fallback_confirmation'
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['gates','evaluate']);args=parser.parse_args()
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
logs=[(DEST/f'{args.mode}-gpu{g}.log').open('w') for g in [0,1]]
commands=[[sys.executable,str(Path(__file__).with_name('preflight_fallback_confirmation.py')),'--gpu',str(g)] if args.mode=='gates'
          else [sys.executable,str(Path(__file__).with_name('fallback_confirmation_study.py')),'evaluate','--gpu',str(g)] for g in [0,1]]
children=[subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=logs[g],stderr=subprocess.STDOUT) for g,cmd in enumerate(commands)]
samples=[];audit=None;auditlog=None
while True:
    sample={'time':time.time(),'worker_pids':[c.pid for c in children],'exit_codes':[c.poll() for c in children]}
    sample['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True).splitlines()
    sample['processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True).splitlines();sample['workers']={}
    for g in [0,1]:
        path=DEST/(f'gate-worker{g}.json' if args.mode=='gates' else f'eval-worker{g}.json')
        if path.exists():
            try:sample['workers'][str(g)]=json.loads(path.read_text())
            except (OSError,json.JSONDecodeError):pass
    if args.mode=='evaluate' and audit is None and not any(c.poll() not in [None,0] for c in children):
        p=json.loads((DEST/'protocol.json').read_text());first=p['jobs'][:16]
        if all((DEST/'runs'/j['id']/'result.json').exists() for j in first):
            auditlog=(DEST/'audit.log').open('w')
            audit=subprocess.Popen([sys.executable,str(Path(__file__).with_name('audit_fallback_confirmation.py')),str(DEST),str(ROOT),'--workers','16'],cwd=ROOT,env=env,stdout=auditlog,stderr=subprocess.STDOUT)
    if audit is not None:sample['audit_pid']=audit.pid;sample['audit_exit_code']=audit.poll();sample['audit_rows_complete']=len(list((DEST/'partial-rows').glob('*.json')))
    samples.append(sample);(DEST/f'live-{args.mode}.json').write_text(json.dumps({'samples':samples},indent=2)+'\n')
    if all(c.poll() is not None for c in children) and (audit is None or audit.poll() is not None):break
    time.sleep(1. if args.mode=='gates' else 5.)
for log in logs:log.close()
if auditlog:auditlog.close()
status={'exit_codes':[c.returncode for c in children],'samples':len(samples),'audit_exit_code':audit.returncode if audit else None}
print(json.dumps(status))
if any(c.returncode for c in children) or (audit is not None and audit.returncode):raise SystemExit(1)
