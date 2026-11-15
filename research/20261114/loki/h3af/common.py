"""H3af immutable inputs and whole-study envelope, including fitting."""
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3af';BASE=ROOT/'artifacts/research/20261114/loki';OUT=BASE/'h3af';OLD=BASE/'h3ae'
sys.path.insert(0,str(SRC.parent/'h3aa'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def frozen(target=False):
    p=read(SRC/'protocol.json')
    for group in ['source_sha256','source_data_sha256']+(['target_data_sha256']if target else[]):
        for rel,h in p[group].items():assert sha(ROOT/rel)==h,rel
    return p
def budget():
    with(OUT/'global-budget.json').open('a+')as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);raw=f.read()
        if raw:deadline=json.loads(raw)['deadline_unix']
        else:
            deadline=time.time()+360;f.write(json.dumps(dict(started_utc=now(),deadline_unix=deadline,whole_study_wall_seconds=360,aggregate_GPU_worker_seconds_envelope=600,per_device_worker_seconds_ceiling=300,includes_fits_calibration_warmups_and_target=True,protocol_sha256=sha(SRC/'protocol.json')))+'\n');f.flush()
    def expired(*args):raise TimeoutError('H3af shared360second wall/600GPU-worker-second envelope reached')
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);used=0.
    for path in OUT.glob(f'b*/target-worker-g{gpu}.json'):
        worker=read(path)
        if worker['status']=='complete':used+=(dt.datetime.fromisoformat(worker['completed_utc'])-dt.datetime.fromisoformat(worker['started_utc'])).total_seconds()
    assert used<300;effective=min(deadline,time.time()+300-used)
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,effective-time.time()));return deadline
