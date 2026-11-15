"""Frozen inputs for the CPU-only paired full compiler benchmark."""
import datetime as dt
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3aj';OUT=ROOT/'artifacts/research/20261114/loki/h3aj'
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def frozen():
    p=read(SRC/'protocol.json')
    for group in ['source_sha256','input_sha256']:
        for rel,h in p[group].items():assert sha(ROOT/rel)==h,rel
    return p
