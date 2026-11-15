"""Launch the two authorized H2bj workers and retain live device progress."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'artifacts/research/20261114/work/certificate_fallback'
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
logs=[(DEST/f'gpu{g}.log').open('w') for g in [0,1]]
children=[subprocess.Popen([sys.executable,str(Path(__file__).with_name('certificate_fallback_study.py')),
                           'evaluate','--gpu',str(g)],cwd=ROOT,env=env,stdout=logs[g],stderr=subprocess.STDOUT)
          for g in [0,1]]
samples=[]
while True:
    sample={'time':time.time(),'worker_pids':[c.pid for c in children],'exit_codes':[c.poll() for c in children]}
    sample['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
                                          '--format=csv,noheader'],text=True).splitlines()
    sample['processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                                '--format=csv,noheader'],text=True).splitlines()
    sample['workers']={}
    for g in [0,1]:
        path=DEST/f'worker{g}.json'
        if path.exists():
            try:sample['workers'][str(g)]=json.loads(path.read_text())
            except (OSError,json.JSONDecodeError):pass
    samples.append(sample)
    (DEST/'live-progress.json').write_text(json.dumps({'samples':samples},indent=2)+'\n')
    if all(c.poll() is not None for c in children):break
    time.sleep(.25)
for log in logs:log.close()
print(json.dumps({'exit_codes':[c.returncode for c in children],'samples':len(samples)}))
if any(c.returncode for c in children):raise SystemExit(1)
