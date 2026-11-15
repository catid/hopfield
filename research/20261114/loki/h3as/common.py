"""Source-only finite-population reliability study paths and frozen hashes."""
import datetime as dt
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/'research/20261114/loki/h3as'
OUT=ROOT/'artifacts/research/20261114/loki/h3as'
def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def frozen():
    p=read(SRC/'protocol.json')
    for group in ('source_sha256','source_data_sha256'):
        for rel,digest in p[group].items(): assert sha(ROOT/rel)==digest,rel
    return p
