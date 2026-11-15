"""Export audited validity/stopping bits, without source labels or returned boards."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/research/20261114/foureyes/noisy_pl'
PUBLIC=ROOT/'results/research/foureyes/noisy_pl'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary=json.loads((PUBLIC/'summary.json').read_text());assert summary['status']=='validated'
    q=json.loads((OUT/'queue.json').read_text());assert len(q)==summary['unique_searches']==53
    valid=[];stopped=[];ids=[]
    for job in q:
        path=OUT/job['job_id']/'moves-262144/boards.npz'
        assert sha(path)==summary['input_sha256'][str(path.relative_to(ROOT))]
        z=np.load(path);valid.append(z['valid']);stopped.append(z['stopped']);ids.append(job['job_id'])
    valid=np.stack(valid);stopped=np.stack(stopped)
    target=PUBLIC/'audited-outcome-bits.npz'
    np.savez_compressed(target,valid=np.packbits(valid,axis=1),stopped=np.packbits(stopped,axis=1),job_ids=np.asarray(ids))
    z=np.load(target,allow_pickle=False)
    assert np.array_equal(valid,np.unpackbits(z['valid'],axis=1,count=10000))
    assert np.array_equal(stopped,np.unpackbits(z['stopped'],axis=1,count=10000))
    receipt=dict(scope='Only validated returned-board validity and stopping flags; no boards, clues or source solutions.',shape=list(valid.shape),
                 sha256=sha(target),audited_summary_sha256=sha(PUBLIC/'summary.json'),exporter_sha256=sha(__file__))
    (PUBLIC/'outcome-export.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='exported',bytes=target.stat().st_size,shape=list(valid.shape))))


if __name__=='__main__':main()
