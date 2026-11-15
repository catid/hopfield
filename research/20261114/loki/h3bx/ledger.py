"""Append-only, fsynced, hash-chained evidence; query must precede oracle."""
import hashlib
import json
import os
import time
from common import now


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


class Ledger:
    def __init__(self,path):
        self.path=path;self.handle=path.open('xb');self.sequence=0;self.previous='0'*64

    def append(self,kind,data):
        record=dict(sequence=self.sequence,previous_sha256=self.previous,utc=now(),monotonic_ns=time.monotonic_ns(),kind=kind,data=data)
        record['sha256']=hashlib.sha256(canonical(record)).hexdigest();self.handle.write(canonical(record)+b'\n');self.handle.flush();os.fsync(self.handle.fileno());self.sequence+=1;self.previous=record['sha256'];return record

    def close(self):self.handle.close();self.path.chmod(0o444)


def audit_bytes(data):
    records=[];previous='0'*64;last_time=-1;last_utc=''
    for i,line in enumerate(data.splitlines()):
        record=json.loads(line);digest=record.pop('sha256');assert digest==hashlib.sha256(canonical(record)).hexdigest();assert record['sequence']==i and record['previous_sha256']==previous and record['monotonic_ns']>last_time and record['utc']>=last_utc
        record['sha256']=digest
        if record['kind']=='answer':assert records and records[-1]['kind']=='query' and record['data']['query_sha256']==records[-1]['sha256']
        records.append(record);previous=digest;last_time=record['monotonic_ns'];last_utc=record['utc']
    assert records;return records
