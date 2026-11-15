"""Independent literal audit of mixed-size source evidence and target energies."""
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
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3m';OUT=ROOT/'artifacts/research/20261114/loki/h3m'
sys.path.insert(0,str(ROOT/'upstream'))
from cram import cram_nqueens as Q
from cram import nqueens_data as D
HASHES={}
def sha(path):
    path=Path(path)
    with path.open('rb') as f:v=hashlib.file_digest(f,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=v;return v
def forbidden(dr,dc):return dr==0 or dc==0 or abs(dr)==abs(dc) or (abs(dr),abs(dc)) in [(1,2),(2,1)]
def literal_counts(solutions,n,domain=12):
    counts={(r,c):0 for r in range(domain) for c in range(1-domain,domain) if r>0 or c>0}
    for row in solutions:
        cells=np.flatnonzero(row);assert len(cells)==n
        for a,b in itertools.combinations(cells,2):
            dr=b//n-a//n;dc=b%n-a%n;assert not forbidden(dr,dc)
            counts[dr,dc]+=1
    return counts

def main():
    torch.set_num_threads(1);p=json.loads((SRC/'protocol.json').read_text());sha(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    train=dict(np.load(OUT/'training.npz'));data=dict(np.load(OUT/'data.npz'))
    parent=dict(np.load(ROOT/'artifacts/research/20261114/loki/h3l/training.npz'))
    assert train['n11_all'].shape==(44,121) and train['n12_all'].shape==(156,144) and data['all_solutions'].shape==(1876,169)
    assert np.array_equal(train['n12_base'],parent['training_labels']) and np.array_equal(train['n12_base_indices'],parent['training_indices'])
    assert np.array_equal(train['n12_all'],parent['source_solutions'])
    feasibility=json.loads((OUT/'feasibility.json').read_text())
    assert feasibility['solutions_sha256']==sha(OUT/'solutions-n11.npy')
    assert np.array_equal(train['n11_all'],np.load(OUT/'solutions-n11.npy'))
    ordinary=np.asarray(D.completions(np.zeros(121,bool),11),bool);columns=ordinary.reshape(-1,11,11).argmax(2)
    keep=(np.abs(columns[:,1:]-columns[:,:-1])!=2).all(1)&(np.abs(columns[:,2:]-columns[:,:-2])!=1).all(1)
    assert len(ordinary)==2680 and {r.tobytes() for r in ordinary[keep]}=={r.tobytes() for r in train['n11_all']}
    blocks={key:literal_counts(train[key],n) for key,n in [('n11_all',11),('n12_base',12),('n12_all',12)]}
    sample=blocks['n12_base'];all_source=blocks['n12_all'];n11=blocks['n11_all']
    evidence={'n12-only':sample,'multisize':{o:sample[o]+n11[o] for o in sample},'n12-full':all_source}
    target_counts=literal_counts(data['all_solutions'],13,13)
    source=dict(np.load(OUT/'source-support.npz'));sha(OUT/'source-support.npz')
    for key,counts in {**blocks,**evidence}.items():assert all(counts[tuple(o)]==v for o,v in zip(source['offsets'],source[key])),key
    fit=json.loads((OUT/'source-fits.json').read_text());sha(OUT/'source-fits.json');retained={}
    for arm,counts in evidence.items():
        candidates=[];lines=[]
        for a,b in itertools.product(range(-2,3),repeat=2):
            if (a,b)==(0,0) or math.gcd(abs(a),abs(b))!=1 or a<0 or (a==0 and b<0):continue
            members=[v for (r,c),v in counts.items() if a*r+b*c==0]
            nf=sum(v==0 for v in members);na=sum(v>0 for v in members);keep=nf>=3 and na==0
            candidates.append(dict(a=a,b=b,forbidden_offsets=nf,allowed_offsets=na,retained=keep))
            if keep:lines.append([a,b])
        assert fit['arms'][arm]==dict(candidates=candidates,retained_lines=lines,source_zero_offsets=sum(v==0 for v in counts.values()))
        retained[arm]=lines
    zeros=[]
    for (r,c),count in sample.items():
        if count==0:zeros.append(dict(dr=r,dc=c,true_local_rule=forbidden(r,c),n12_base_count=0,n11_all_count=n11[r,c],
            n12_full_count=all_source[r,c],source_size_global_zero=all_source[r,c]==0,full_N13_count=target_counts.get((r,c),0)))
    support_diagnostic=dict(source_zero_offsets=len(zeros),true_source_rule_offsets=sum(v['true_local_rule'] for v in zeros),
        extra_source_zero_offsets=sum(not v['true_local_rule'] for v in zeros),
        source_zeros_resolved_by_n11=sum(v['n11_all_count']>0 for v in zeros),source_zeros_resolved_by_full_n12=sum(v['n12_full_count']>0 for v in zeros),
        all_zero_offsets=zeros,extra_offsets=[v for v in zeros if not v['true_local_rule']],
        use='Post-construction diagnosis only; no offset removed or modified.')
    universe_path=ROOT/'artifacts/research/20261114/loki/h3l/input-universe.npz';universe=dict(np.load(universe_path));sha(universe_path)
    previous=np.load(ROOT/'artifacts/research/20261114/loki/h3l/data.npz')
    excluded=np.union1d(universe['prior_target_keys'],previous['test_keys']);assert np.array_equal(excluded,data['prior_target_test_keys'])
    eligible=universe['keys'][~np.isin(universe['keys'],excluded)];assert len(eligible)==185022
    expected=np.sort(np.random.default_rng(2026112503).permutation(eligible)[:4096])
    assert np.array_equal(expected,data['test_keys']) and not np.isin(expected,data['prior_target_test_keys']).any()
    for given,key,count in zip(data['test_inputs'],expected,data['test_solution_counts']):
        cells=np.flatnonzero(given);assert sum((int(c)+1)*170**(2-j) for j,c in enumerate(cells))==key
        assert data['all_solutions'][:,cells].all(1).sum()==count>0
    assert np.array_equal(data['all_solutions'],np.load(ROOT/'artifacts/research/20261114/loki/h3l/solutions-n13.npy'))
    i,j=np.triu_indices(169,1);dr=j//13-i//13;dc=j%13-i%13;models={};diags=[]
    oracle=np.array([forbidden(r,c) for r,c in zip(dr,dc)])
    for cfg in p['arms']:
        arm=cfg['arm'];folder=OUT/f'models/{arm}';rec=json.loads((folder/'model.json').read_text());sha(folder/'model.json')
        assert rec['model_sha256']==sha(folder/'model.pt') and rec['construction_sha256']==sha(folder/'construction.npz')
        if arm=='true-graph-oracle':mask=oracle
        else:
            counts=evidence[arm];lines=retained[arm]
            mask=np.array([any(a*r+b*c==0 for a,b in lines) or ((r,c) in counts and counts[r,c]==0) for r,c in zip(dr,dc)])
        assert np.array_equal(mask,np.load(folder/'construction.npz')['forbidden'])
        J=np.zeros((169,169),np.int64);J[i,j]=13*mask-1;J[j,i]=J[i,j]
        m=torch.load(folder/'model.pt',weights_only=False);models[arm]=m
        assert np.array_equal(J,m['S'][:,:,1,1].numpy()) and not m['S'][:,:,0,:].any() and not m['S'][:,:,:,0].any() and not m['B'].any()
        assert m['q']==13 and m['delta']==12 and m['tau']==-78 and m['unit']==p['calibration']['unit']
        diags.append(dict(arm=arm,forbidden_pairs=int(mask.sum()),false_forbidden=int((mask&~oracle).sum()),missed_forbidden=int((~mask&oracle).sum())))
    parent_model=torch.load(ROOT/'artifacts/research/20261114/loki/h3l/models/hybrid/model.pt',weights_only=False)
    assert all(torch.equal(parent_model[k],models['n12-only'][k]) for k in ['S','B']) and all(parent_model[k]==models['n12-only'][k] for k in ['q','delta','tau','unit'])
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
    for a,b in itertools.combinations(groups['groups'],2):assert not torch.equal(models[a['representative']]['S'],models[b['representative']]['S'])
    spec=importlib.util.spec_from_file_location('h3m_runner',SRC/'driver.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    positive=data['all_solutions'][:64];assert runner.super_valid(positive[:,None,:],np.zeros_like(positive)).all()
    ordinary_path=ROOT/'artifacts/research/20261114/loki/h3j/data.npz';sha(ordinary_path)
    ordinary=np.load(ordinary_path)['all_solutions'];target_keys={row.tobytes() for row in data['all_solutions']}
    negative=np.array([row for row in ordinary[:128] if row.tobytes() not in target_keys]);assert len(negative)>0
    assert not runner.super_valid(negative[:,None,:],np.zeros_like(negative)).any();sha(__file__)
    rec=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='Literal per-board-size source counts and all finite zeros retained; target graph/solution comparisons are diagnostic only',
        retained_lines=retained,source_support_diagnostic=support_diagnostic,target_models=diags,identity_groups=groups,
        n12_only_exact_h3l_hybrid=True,all_identities_exact=True,scoring_positive_fixture_count=len(positive),scoring_knight_violation_fixture_count=len(negative),input_sha256=HASHES)
    (OUT/'independent-preflight.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(dict(status='passed',retained_lines=retained,target_models=diags,identity_groups=groups,source_support={k:v for k,v in support_diagnostic.items() if k not in ['extra_offsets','all_zero_offsets']})))
if __name__=='__main__':main()
