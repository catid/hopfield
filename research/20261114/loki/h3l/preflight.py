"""Audit superqueen source counts, retained zero offsets and target representations."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3l';OUT=ROOT/'artifacts/research/20261114/loki/h3l'
sys.path.insert(0,str(ROOT/'upstream'))
from cram import cram_nqueens as Q
HASHES={}


def sha(path):
    path=Path(path)
    with path.open('rb') as f:v=hashlib.file_digest(f,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=v;return v


def forbidden(dr,dc):
    return dr==0 or dc==0 or abs(dr)==abs(dc) or (abs(dr),abs(dc)) in [(1,2),(2,1)]


def literal_counts(solutions,n):
    offsets={(r,c):0 for r in range(n) for c in range(1-n,n) if r>0 or c>0}
    for row in solutions:
        cells=np.flatnonzero(row);assert len(cells)==n
        for a,b in itertools.combinations(cells,2):
            dr=b//n-a//n;dc=b%n-a%n;assert not forbidden(dr,dc)
            offsets[dr,dc]+=1
    return offsets


def main():
    torch.set_num_threads(1);p=json.loads((SRC/'protocol.json').read_text());sha(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    train=dict(np.load(OUT/'training.npz'));data=dict(np.load(OUT/'data.npz'))
    assert train['source_solutions'].shape==(156,144) and data['all_solutions'].shape==(1876,169)
    assert np.array_equal(train['training_indices'],np.random.default_rng(2026112501).permutation(156)[:128])
    assert np.array_equal(train['training_labels'],train['source_solutions'][train['training_indices']])
    sample=literal_counts(train['training_labels'],12);all_source=literal_counts(train['source_solutions'],12)
    target_counts=literal_counts(data['all_solutions'],13)
    feasibility=json.loads((OUT/'feasibility.json').read_text())
    for n,name,array in [(12,'solutions-n12.npy',train['source_solutions']),(13,'solutions-n13.npy',data['all_solutions'])]:
        assert np.array_equal(array,np.load(OUT/name))
        assert sha(OUT/name)==feasibility['enumerations'][str(n)]['array_sha256']
    source=dict(np.load(OUT/'source-support.npz'));sha(OUT/'source-support.npz')
    assert all(sample[tuple(o)]==n for o,n in zip(source['offsets'],source['counts']))
    candidates=[];retained=[]
    for a,b in itertools.product(range(-2,3),repeat=2):
        if (a,b)==(0,0) or math.gcd(abs(a),abs(b))!=1 or a<0 or (a==0 and b<0):continue
        members=[value for (r,c),value in sample.items() if a*r+b*c==0]
        nf=sum(v==0 for v in members);na=sum(v>0 for v in members);keep=nf>=3 and na==0
        candidates.append(dict(a=a,b=b,forbidden_offsets=nf,allowed_offsets=na,retained=keep))
        if keep:retained.append([a,b])
    fit=json.loads((OUT/'line-fit.json').read_text());sha(OUT/'line-fit.json')
    assert fit['candidates']==candidates and fit['retained_lines']==retained
    extra=[]
    for (r,c),count in sample.items():
        if count==0 and not forbidden(r,c):extra.append(dict(dr=r,dc=c,training_count=0,full_N12_count=all_source[r,c],
            source_size_global_zero=all_source[r,c]==0,full_N13_count=target_counts.get((r,c),0)))
    support_diagnostic=dict(source_zero_offsets=sum(v==0 for v in sample.values()),true_source_rule_offsets=sum(forbidden(r,c) for r,c in sample),
        extra_source_zero_offsets=len(extra),extra_offsets_globally_absent_at_N12=sum(v['source_size_global_zero'] for v in extra),
        extra_offsets_absent_only_from_sample=sum(not v['source_size_global_zero'] for v in extra),
        global_source_zero_offsets_observed_at_N13=sum(v['source_size_global_zero'] and v['full_N13_count']>0 for v in extra),extra_offsets=extra,
        use='Post-construction diagnosis only; no offset is removed or modified.')
    universe=dict(np.load(OUT/'input-universe.npz'));sha(OUT/'input-universe.npz')
    expected=np.sort(np.random.default_rng(2026112502).permutation(universe['eligible_keys'])[:4096])
    assert np.array_equal(expected,data['test_keys']) and not np.isin(expected,data['prior_target_test_keys']).any()
    for given,key,count in zip(data['test_inputs'],expected,data['test_solution_counts']):
        cells=np.flatnonzero(given);assert sum((int(c)+1)*170**(2-j) for j,c in enumerate(cells))==key
        assert data['all_solutions'][:,cells].all(1).sum()==count>0
    i,j=np.triu_indices(169,1);dr=j//13-i//13;dc=j%13-i%13;models={};diags=[]
    for cfg in p['arms']:
        arm=cfg['arm'];folder=OUT/f'models/{arm}';rec=json.loads((folder/'model.json').read_text());sha(folder/'model.json')
        assert rec['model_sha256']==sha(folder/'model.pt') and rec['construction_sha256']==sha(folder/'construction.npz')
        line=np.array([any(a*r+b*c==0 for a,b in retained) for r,c in zip(dr,dc)])
        finite=np.array([(r,c) in sample and sample[r,c]==0 for r,c in zip(dr,dc)])
        oracle=np.array([forbidden(r,c) for r,c in zip(dr,dc)])
        mask=line if arm=='line-only' else line|finite if arm=='hybrid' else finite if arm=='direct-offset' else oracle
        assert np.array_equal(mask,np.load(folder/'construction.npz')['forbidden'])
        J=np.zeros((169,169),np.int64);J[i,j]=13*mask-1;J[j,i]=J[i,j]
        m=torch.load(folder/'model.pt',weights_only=False);models[arm]=m
        assert np.array_equal(J,m['S'][:,:,1,1].numpy()) and not m['S'][:,:,0,:].any() and not m['S'][:,:,:,0].any() and not m['B'].any()
        assert m['q']==13 and m['delta']==12 and m['tau']==-78 and m['unit']==p['calibration']['unit']
        diags.append(dict(arm=arm,forbidden_pairs=int(mask.sum()),false_forbidden=int((mask&~oracle).sum()),missed_forbidden=int((~mask&oracle).sum()),
            missed_knight_pairs=int((~mask&(((dr==1)&(np.abs(dc)==2))|((dr==2)&(np.abs(dc)==1)))).sum())))
    groups=json.loads((OUT/'energy-groups.json').read_text());sha(OUT/'energy-groups.json')
    fixtures=np.random.default_rng(83116).integers(0,2,(13,169)).astype(bool);members=[]
    for group in groups['groups']:
        m=models[group['representative']];J=m['S'][:,:,1,1].numpy().astype(np.int64)
        literal=(fixtures[:,i]*fixtures[:,j]*J[i,j]).sum(1)
        assert np.array_equal(literal,Q.board_energy(m['S'].double(),m['B'].double(),torch.tensor(fixtures,dtype=torch.long)).numpy())
        for board,energy in zip(fixtures,literal):
            for cell in range(169):
                other=board.copy();other[cell]=~other[cell]
                assert (other[i]*other[j]*J[i,j]).sum()-energy==(1-2*int(board[cell]))*(board@J[:,cell])
        for arm in group['members']:
            other=models[arm];assert all(torch.equal(other[k],m[k]) for k in ['S','B']) and all(other[k]==m[k] for k in ['q','delta','tau','unit'])
            members.append(arm)
    assert sorted(members)==sorted(cfg['arm'] for cfg in p['arms'])
    spec=importlib.util.spec_from_file_location('h3l_runner',SRC/'driver.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    positive=data['all_solutions'][:64];assert runner.super_valid(positive[:,None,:],np.zeros_like(positive)).all()
    ordinary_path=ROOT/'artifacts/research/20261114/loki/h3j/data.npz';sha(ordinary_path)
    ordinary=np.load(ordinary_path)['all_solutions'];target_keys={row.tobytes() for row in data['all_solutions']}
    negative=np.array([row for row in ordinary[:128] if row.tobytes() not in target_keys]);assert len(negative)>0
    assert not runner.super_valid(negative[:,None,:],np.zeros_like(negative)).any()
    sha(__file__)
    rec=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='Source counts and all finite zeros retained; target graph/solution comparisons are post-construction diagnostics only',
        retained_lines=retained,source_support_diagnostic=support_diagnostic,target_models=diags,identity_groups=groups,
        all_identities_exact=True,scoring_positive_fixture_count=len(positive),scoring_knight_violation_fixture_count=len(negative),input_sha256=HASHES)
    (OUT/'independent-preflight.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(dict(status='passed',retained_lines=retained,target_models=diags,source_support={k:v for k,v in support_diagnostic.items() if k!='extra_offsets'})))


if __name__=='__main__':main()
