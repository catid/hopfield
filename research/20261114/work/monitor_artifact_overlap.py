"""Read-only both-GPU/PID/worker counters during H2bi gates and timing."""
import json
from pathlib import Path
import subprocess
import time

D=Path(__file__).resolve().parents[3]/'artifacts/research/20261114/work/artifact_overlap';started=time.monotonic();samples=[]
while time.monotonic()-started<650:
    workers=[]
    for kind in ['gate','timing']:
        for gpu in [0,1]:
            path=D/f'{kind}-worker{gpu}.json';r=json.loads(path.read_text()) if path.exists() else {};workers.append({'kind':kind,**{k:r.get(k) for k in ['gpu','pid','status','phase','watchdog_checks','completed_replays','worker_seconds','updated_utc']}})
    sm=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True);processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True);samples.append({'elapsed_seconds':time.monotonic()-started,'workers':workers,'gpu_counters':sm,'processes':processes});target=D/'live-progress.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps({'read_only':True,'samples':samples},indent=2)+'\n');tmp.replace(target)
    if all(r['status']=='complete' for r in workers if r['kind']=='timing'):break
    time.sleep(3)
