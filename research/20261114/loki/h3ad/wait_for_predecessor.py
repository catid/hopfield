"""Orchestration only: no GPU overlap and no changes to either frozen protocol."""
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/'artifacts/research/20261114/loki/h3ad';PRE=ROOT/'artifacts/research/20261114/loki/h3ac';start=time.time()
def read(path):return json.loads(path.read_text())
while True:
    gate=OUT/'enumeration-gate.json'
    if gate.exists()and read(gate)['status']!='complete_verified':raise RuntimeError('H3ad enumeration failed; no GPU launch')
    ready=OUT/'cpu-preparation.json'
    if ready.exists()and read(ready)['status']=='complete'and read(PRE/'campaign.json')['status']=='complete':break
    if time.time()>read(PRE/'global-budget.json')['deadline_unix']+30:raise TimeoutError('Predecessor did not complete within its own envelope')
    time.sleep(2)
receipt=dict(status='launching_after_predecessor_completion',waiting_wall_seconds=time.time()-start,predecessor_completed=read(PRE/'campaign.json'),no_gpu_overlap=True)
(OUT/'queue-release.json').write_text(json.dumps(receipt,indent=2)+'\n')
subprocess.run([sys.executable,str(ROOT/'research/20261114/loki/h3ad/campaign.py'),'gpu'],check=True)
