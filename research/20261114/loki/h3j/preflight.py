"""Independently audit N12 line inference and N13 model/data separation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3j';OUT=ROOT/'artifacts/research/20261114/loki/h3j';TRAIN=ROOT/'artifacts/research/20261114/loki/h3i'
HASHES={}


def sha(path):
    path=Path(path)
    with path.open('rb') as f:v=hashlib.file_digest(f,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=v;return v


def main():
    torch.set_num_threads(1);p=json.loads((SRC/'protocol.json').read_text());sha(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    train=dict(np.load(TRAIN/'data.npz'));labels=train['all_solutions'][train['raw128_indices']]
    offsets={(dr,dc):0 for dr in range(12) for dc in range(-11,12) if dr>0 or dc>0}
    for label in labels:
        for a,b in itertools.combinations(np.flatnonzero(label),2):offsets[(b//12-a//12,b%12-a%12)]+=1
    source=dict(np.load(OUT/'source-support.npz'));sha(OUT/'source-support.npz')
    assert np.array_equal(source['training_indices'],train['raw128_indices'])
    assert all(offsets[tuple(o)]==n for o,n in zip(source['offsets'],source['counts'])) and len(offsets)==len(source['counts'])==264
    candidates=[];retained=[]
    for a,b in itertools.product(range(-2,3),repeat=2):
        if (a,b)==(0,0) or math.gcd(abs(a),abs(b))!=1 or next(x for x in [a,b] if x)!=abs(next(x for x in [a,b] if x)):continue
        members=[count for (r,c),count in offsets.items() if a*r+b*c==0]
        nf=sum(x==0 for x in members);na=sum(x>0 for x in members);keep=nf>=3 and na==0
        candidates.append(dict(a=a,b=b,forbidden_offsets=nf,allowed_offsets=na,retained=keep))
        if keep:retained.append([a,b])
    fit=json.loads((OUT/'line-fit.json').read_text());sha(OUT/'line-fit.json')
    assert fit['candidates']==candidates and fit['retained_lines']==retained
    data=dict(np.load(OUT/'data.npz'));solutions=data['all_solutions'];assert solutions.shape==(73712,169)
    assert len({row.tobytes() for row in solutions})==73712
    for row in solutions:
        cells=np.flatnonzero(row);r=cells//13;c=cells%13
        assert len(cells)==len(set(r))==len(set(c))==len(set(r-c))==len(set(r+c))==13
    # Independent bit-mask enumeration counts the complete solution universe.
    full=(1<<13)-1
    def count_boards(columns,left,right):
        if columns==full:return 1
        available=full&~(columns|left|right);total=0
        while available:
            bit=available&-available;available^=bit;total+=count_boards(columns|bit,(left|bit)<<1,(right|bit)>>1)
        return total
    assert count_boards(0,0,0)==len(solutions)
    pool=data['full_universe_keys'];expected=np.sort(np.random.default_rng(2026111417).permutation(pool)[:4096])
    assert np.array_equal(expected,data['test_keys']) and len(np.unique(expected))==4096
    for given,key,n in zip(data['test_inputs'],data['test_keys'],data['test_solution_counts']):
        cells=np.flatnonzero(given);assert sum((int(c)+1)*170**(2-j) for j,c in enumerate(cells))==key
        assert solutions[:,cells].all(1).sum()==n>0
    i,j=np.triu_indices(169,1);r=j//13-i//13;c=j%13-i%13;models={};diags=[]
    for cfg in p['arms']:
        arm=cfg['arm'];folder=OUT/f'models/{arm}';rec=json.loads((folder/'model.json').read_text());sha(folder/'model.json')
        assert rec['model_sha256']==sha(folder/'model.pt') and rec['construction_sha256']==sha(folder/'construction.npz')
        if arm=='line-extrapolation':mask=np.array([any(a*x+b*y==0 for a,b in retained) for x,y in zip(r,c)])
        elif arm=='direct-offset':mask=np.array([(x,y) in offsets and offsets[x,y]==0 for x,y in zip(r,c)])
        else:mask=(r==0)|(c==0)|(np.abs(r)==np.abs(c))
        saved=np.load(folder/'construction.npz');assert np.array_equal(mask,saved['forbidden'])
        J=np.zeros((169,169),np.int64);J[i,j]=13*mask-1;J[j,i]=J[i,j]
        m=torch.load(folder/'model.pt',weights_only=False);models[arm]=m
        assert np.array_equal(J,m['S'][:,:,1,1].numpy()) and not m['S'][:,:,0,:].any() and not m['S'][:,:,:,0].any() and not m['B'].any()
        assert m['q']==13 and m['delta']==12 and m['tau']==-78 and m['unit']==p['calibration']['unit']
        assert rec['target_training_patterns']==0
        attack=(r==0)|(c==0)|(np.abs(r)==np.abs(c))
        diag=dict(arm=arm,forbidden_pairs=int(mask.sum()),false_forbidden=int((mask&~attack).sum()),missed_forbidden=int((~mask&attack).sum()))
        assert all(diag[k]==rec[k] for k in ['forbidden_pairs','false_forbidden','missed_forbidden']);diags.append(diag)
    a=models['line-extrapolation'];b=models['true-graph-oracle']
    assert all(torch.equal(a[k],b[k]) for k in ['S','B']) and all(a[k]==b[k] for k in ['q','delta','tau','unit'])
    sha(__file__)
    rec=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='Independent training-only line selection, synthetic representation and evaluation-data checks; no target search outcomes read',
        retained_lines=retained,models=diags,complete_target_solutions=73712,independent_exhaustive_count=73712,unique_inputs=4096,
        oracle_solver_identity=dict(exact=True,model_keys=['S','B','q','delta','tau','unit'],backend_sha256=sha(SRC.parent/'h3i/backend.py'),
            same_inputs=True,same_seeds=[0,1,2],interpretation='The two model names have exactly equal solver-used parameters, so matched-seed outputs are identical under this unchanged deterministic seeded backend.'),input_sha256=HASHES)
    (OUT/'independent-preflight.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:v for k,v in rec.items() if k!='input_sha256'}))


if __name__=='__main__':main()
