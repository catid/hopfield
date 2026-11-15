"""Record live useful progress and enforce the allocated four-GPU time envelope."""
import datetime
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from run import ROOT,now

OUT=ROOT/'artifacts/research/20261114/ripper/rate_transfer'


def main():
    while True:
        q=json.loads((OUT/'queue.json').read_text())
        running=[j for j in q if j['status']=='running']
        starts=[datetime.datetime.fromisoformat(j['started_utc']) for j in q if 'started_utc' in j]
        elapsed=(datetime.datetime.now(datetime.timezone.utc)-min(starts)).total_seconds() if starts else 0.
        progress={}
        for j in running:
            path=OUT/'jobs'/j['job_id']/'progress.json'
            if path.exists():progress[j['job_id']]=json.loads(path.read_text())
        snapshot={'utc':now(),'queue':{s:sum(j['status']==s for j in q) for s in ['pending','running','complete','failed']},
            'allocated_gpu_envelope_s':4*elapsed,'progress':progress,
            'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True)}
        with(OUT/'utilization.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
        if not any(j['status'] in ('pending','running') for j in q):break
        if elapsed>=450:
            stopped=[]
            for j in running:
                pid=j['pid'];path=Path(f'/proc/{pid}/cmdline')
                if path.exists() and b'rate_transfer_search.py' in path.read_bytes():
                    os.kill(pid,signal.SIGTERM);stopped.append(pid)
            (OUT/'budget_exhausted.json').write_text(json.dumps({'utc':now(),'stopped_owned_pids':stopped,'snapshot':snapshot},indent=2)+'\n')
            break
        time.sleep(5)


if __name__=='__main__':main()
