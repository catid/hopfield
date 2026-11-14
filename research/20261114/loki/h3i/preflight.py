"""Independent construction and split audit before H3i search outcomes."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3i';OUT=ROOT/'artifacts/research/20261114/loki/h3i'
HASHES={}


def sha(path):
    path=Path(path)
    with path.open('rb') as f:v=hashlib.file_digest(f,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=v;return v


def main():
    torch.set_num_threads(1);p=json.loads((SRC/'protocol.json').read_text());sha(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    d=dict(np.load(OUT/'data.npz'));solutions=d['all_solutions'];assert solutions.shape==(14200,144)
    keys={row.tobytes():i for i,row in enumerate(solutions)};assert len(keys)==14200
    for row in solutions:
        cells=np.flatnonzero(row);r=cells//12;c=cells%12
        assert len(cells)==len(set(r))==len(set(c))==len(set(r-c))==len(set(r+c))==12
    available=np.setdiff1d(np.arange(14200),d['prior_training_bank_indices'])
    raw=np.random.default_rng(2026111415).permutation(available)[:1024]
    assert np.array_equal(raw,d['raw1024_indices']) and np.array_equal(raw[:128],d['raw128_indices'])
    transformed=[]
    for k in range(4):
        for mirrored in [False,True]:
            ids=[]
            for row in solutions[raw[:128]]:
                cells=np.flatnonzero(row);r=cells//12;c=cells%12
                if mirrored:c=11-c
                for _ in range(k):r,c=11-c,r
                board=np.zeros(144,bool);board[r*12+c]=True;ids.append(keys[board.tobytes()])
            transformed.append(ids)
    assert np.array_equal(transformed,d['dihedral_image_indices'])
    aug=np.unique(transformed);assert np.array_equal(aug,d['augmented128_indices']) and len(aug)==996
    union=np.union1d(raw,aug);assert np.array_equal(union,d['union_indices']) and len(union)==1839
    inputs=d['test_inputs'];assert len(inputs)==len(np.unique(d['test_keys']))==4096
    assert not np.isin(d['test_keys'],d['prior_test_keys']).any()
    for row,key,count in zip(inputs,d['test_keys'],d['test_solution_counts']):
        cells=np.flatnonzero(row);assert sum((int(c)+1)*145**(2-j) for j,c in enumerate(cells))==key
        compatible=solutions[:,cells].all(1);assert compatible.sum()==count>0 and not compatible[union].any()
    i,j=np.triu_indices(144,1);diags=[]
    for cfg in p['arms']:
        arm=cfg['arm'];folder=OUT/f'models/{arm}';rec=json.loads((folder/'model.json').read_text());sha(folder/'model.json')
        assert rec['model_sha256']==sha(folder/'model.pt') and rec['construction_sha256']==sha(folder/'construction.npz')
        assert rec['source_sha256']==sha(SRC/'driver.py') and rec['protocol_sha256']==sha(SRC/'protocol.json')
        z=dict(np.load(folder/'construction.npz'));labels=solutions[d[cfg['indices']]];counts=np.zeros((144,144),np.int64)
        for row in labels:
            for a,b in itertools.combinations(np.flatnonzero(row),2):counts[a,b]+=1;counts[b,a]+=1
        assert np.array_equal(counts[i,j],z['raw_cooccurrence']) and np.array_equal(z['pattern_indices'],d[cfg['indices']])
        if cfg['pooling']=='individual':forbidden=counts[i,j]==0
        else:
            pooled={}
            for a,b in zip(i,j):
                offset=(b//12-a//12,b%12-a%12);pooled[offset]=pooled.get(offset,0)+counts[a,b]
            forbidden=np.array([pooled[(b//12-a//12,b%12-a%12)]==0 for a,b in zip(i,j)])
        assert np.array_equal(forbidden,z['forbidden'])
        J=np.zeros((144,144),np.int64);J[i,j]=12*forbidden-1;J[j,i]=J[i,j]
        m=torch.load(folder/'model.pt',weights_only=False);assert m['q']==12 and np.array_equal(J,m['S'][:,:,1,1].numpy())
        assert not m['S'][:,:,0,:].any() and not m['S'][:,:,:,0].any() and not m['B'].any()
        x=labels.astype(np.float64);fields=x@J
        assert m['delta']==float(((1-2*x)*fields).min())==11 and m['tau']==float((.5*x*fields).sum(1).max())==-66
        assert m['unit']==float(np.abs(fields).mean())
        attack=(i//12==j//12)|(i%12==j%12)|(np.abs(i//12-j//12)==np.abs(i%12-j%12))
        assert not (attack&~forbidden).any()
        diag=dict(arm=arm,unique_patterns=len(labels),forbidden_pairs=int(forbidden.sum()),false_forbidden=int((forbidden&~attack).sum()),missed_forbidden=0)
        assert all(diag[k]==rec[k] for k in ['unique_patterns','forbidden_pairs','false_forbidden','missed_forbidden'])
        diags.append(diag)
    sha(__file__)
    rec=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='Independent data and construction checks only; no search outcomes read',
        models=diags,unique_inputs=4096,all_test_completions_outside_union=True,dihedral_membership_and_coordinate_map_checked=True,input_sha256=HASHES)
    (OUT/'independent-preflight.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))


if __name__=='__main__':main()
