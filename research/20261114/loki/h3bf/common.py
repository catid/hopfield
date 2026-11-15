"""Hash-sealed source-only fitting and a monitored whole-study envelope."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3bf';BASE=ROOT/'artifacts/research/20261114/loki';OUT=BASE/'h3bf'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def frozen(target=False):
    p=read(SRC/'protocol.json')
    for group in ['source_sha256','source_data_sha256']+(['target_data_sha256']if target else[]):
        for rel,digest in p[group].items():assert sha(ROOT/rel)==digest,rel
    return p
def budget():return read(OUT/'study-start.json')['deadline_unix']
