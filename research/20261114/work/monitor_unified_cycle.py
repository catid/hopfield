"""Read-only GPU/process/progress sampling; outside scientific inference timing."""
import json
import subprocess
import time
from pathlib import Path

D=Path(__file__).resolve().parents[3]/'artifacts/research/20261114/work/unified_cycle'
started=time.monotonic();samples=[]
while time.monotonic()-started<650:
    states=[]
    for gpu in [0,1]:
        path=D/f'eval-worker{gpu}.json';s=json.loads(path.read_text()) if path.exists() else {};states.append({k:s.get(k) for k in ['gpu','pid','status','current_job','completed_inputs','jobs_complete','worker_elapsed_seconds','wall_seconds','updated_utc']})
    sm=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    samples.append({'elapsed_seconds':time.monotonic()-started,'workers':states,'gpu_counters':sm,'processes':processes});p=D/'live-progress.json';q=p.with_suffix('.tmp');q.write_text(json.dumps({'read_only':True,'samples':samples},indent=2)+'\n');q.replace(p)
    if all(s['status']=='complete' for s in states):break
    time.sleep(5)
