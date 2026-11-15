"""H3ab frozen-kernel reuse and shared whole-study compute envelope."""
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3ab';OUT=ROOT/'artifacts/research/20261114/loki/h3ab';OLD=ROOT/'artifacts/research/20261114/loki/h3aa';BASE=ROOT/'artifacts/research/20261114/loki'
sys.path.insert(0,str(SRC.parent/'h3aa'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def frozen():
    p=read(SRC/'protocol.json')
    for group in['source_sha256','data_sha256']:
        for rel,h in p[group].items():assert sha(ROOT/rel)==h,rel
    return p
def budget():
    with(OUT/'global-budget.json').open('a+')as file:
        fcntl.flock(file,fcntl.LOCK_EX);file.seek(0);value=file.read()
        if value:deadline=json.loads(value)['deadline_unix']
        else:
            deadline=time.time()+600;file.write(json.dumps(dict(started_utc=now(),deadline_unix=deadline,whole_study_wall_seconds=600,aggregate_GPU_worker_seconds_envelope=1200,includes_calibration_warmups_refinement_and_target=True,protocol_sha256=sha(SRC/'protocol.json')))+'\n');file.flush()
    def expired(*args):raise TimeoutError('Shared whole-study600second wall/1200GPU-worker-second envelope reached')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()));return deadline
