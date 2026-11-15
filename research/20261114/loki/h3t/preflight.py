"""Independent literal audit of source-only exposure scores for the queen-plus-camel family and target energies."""
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
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3t';OUT=ROOT/'artifacts/research/20261114/loki/h3t'
sys.path.insert(0,str(ROOT/'upstream'))
from cram import cram_nqueens as Q
from cram import nqueens_data as D
HASHES={}
def sha(path):
    path=Path(path)
    with path.open('rb') as f:v=hashlib.file_digest(f,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=v;return v
def forbidden(dr,dc):return dr==0 or dc==0 or abs(dr)==abs(dc) or (abs(dr),abs(dc)) in [(1,3),(3,1)]
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
    assert train['n12_all'].shape==(176,144) and data['all_solutions'].shape==(1932,169)
    feasibility=json.loads((OUT/'feasibility.json').read_text());assert feasibility['status']=='feasible_validated_no_fit'
    for n,labels in [(12,train['n12_all']),(13,data['all_solutions'])]:
        rec=feasibility['enumerations'][str(n)];assert rec['array_sha256']==sha(OUT/f'solutions-n{n}.npy')
        assert rec['independent_enumerations_match'] and rec['every_board_pairwise_valid'] and len(labels)==rec['complete_solutions']
        assert np.array_equal(labels,np.load(OUT/f'solutions-n{n}.npy'))
    assert feasibility['seconds']<feasibility['cpu_wall_seconds_envelope']==120
    for group in ['source_sha256','prior_data_sha256']:
        for path,digest in feasibility[group].items():assert sha(ROOT/path)==digest,path
    sample=literal_counts(train['n12_all'],12,12);target_counts=literal_counts(data['all_solutions'],13,13)
    source=dict(np.load(OUT/'source-support.npz'));sha(OUT/'source-support.npz')
    marginal=np.zeros(144,np.int64)
    for board in train['n12_all']:
        for cell in np.flatnonzero(board):marginal[cell]+=1
    assert np.array_equal(marginal,source['marginal_counts'])
    exposure_numerator={}
    for dr,dc in sample:
        total=0
        for row in range(12):
            for col in range(12):
                if 0<=row+dr<12 and 0<=col+dc<12:total+=int(marginal[row*12+col])*int(marginal[(row+dr)*12+col+dc])
        exposure_numerator[dr,dc]=total
    expected={o:v/176 for o,v in exposure_numerator.items()};threshold=math.log(len(sample)/.01);assert len(sample)==264==p['exposure']['K']
    keep={o:sample[o]==0 and expected[o]>=threshold for o in sample}
    for o,n,numerator,e,k in zip(source['offsets'],source['counts'],source['exposure_numerator'],source['expected_count'],source['retained_finite_zero']):
        o=tuple(o);assert sample[o]==n and exposure_numerator[o]==numerator and expected[o]==e and keep[o]==k
    fit=json.loads((OUT/'source-fits.json').read_text());sha(OUT/'source-fits.json');candidates=[];lines=[]
    for a,b in itertools.product(range(-2,3),repeat=2):
        if (a,b)==(0,0) or math.gcd(abs(a),abs(b))!=1 or a<0 or (a==0 and b<0):continue
        members=[v for (r,c),v in sample.items() if a*r+b*c==0]
        nf=sum(v==0 for v in members);na=sum(v>0 for v in members);ret=nf>=3 and na==0
        candidates.append(dict(a=a,b=b,forbidden_offsets=nf,allowed_offsets=na,retained=ret))
        if ret:lines.append([a,b])
    assert fit['candidates']==candidates and fit['retained_lines']==lines and fit['threshold']==threshold
    assert fit['source_zero_offsets']==sum(v==0 for v in sample.values()) and fit['retained_finite_zero_offsets']==sum(keep.values())
    offsets=[]
    for (r,c),count in sample.items():
        offsets.append(dict(dr=r,dc=c,source_count=count,exposure_numerator=exposure_numerator[r,c],source_patterns=176,expected_count=expected[r,c],threshold=threshold,
            finite_zero_retained=keep[r,c],removed_zero=count==0 and not keep[r,c],covered_by_unchanged_line=any(a*r+b*c==0 for a,b in lines),
            true_local_rule=forbidden(r,c),target_pair_count=max(0,13-r)*max(0,13-abs(c)),full_N13_count=target_counts.get((r,c),0)))
    support_diagnostic=dict(K=264,threshold=threshold,all_offsets=offsets,removed_zero_offsets=[v for v in offsets if v['removed_zero']],
        extra_source_zero_offsets=[v for v in offsets if v['source_count']==0 and not v['true_local_rule']],
        heuristic_scope='Independent-cell exposure assumption only; no calibrated p-values, coverage or Bonferroni guarantee for constrained patterns.',
        use='Threshold and score frozen before computation; target truth diagnostics do not change any score, line or coefficient.')
    retained=lines
    universe_path=OUT/'input-universe.npz';universe=dict(np.load(universe_path));sha(universe_path)
    excluded=np.zeros(0,np.int64)
    for study in ['h3j','h3k','h3l','h3m','h3n']:excluded=np.union1d(excluded,np.load(ROOT/f'artifacts/research/20261114/loki/{study}/data.npz')['test_keys'])
    assert np.array_equal(data['prior_target_test_keys'],excluded) and np.array_equal(universe['prior_target_keys'],excluded)
    eligible=universe['keys'][~np.isin(universe['keys'],excluded)];assert len(eligible)==181793 and np.array_equal(eligible,universe['eligible_keys'])
    expected=np.sort(np.random.default_rng(2026112510).permutation(eligible)[:4096])
    assert np.array_equal(expected,data['test_keys']) and not np.isin(expected,excluded).any()
    for given,key,count in zip(data['test_inputs'],expected,data['test_solution_counts']):
        cells=np.flatnonzero(given);assert sum((int(c)+1)*170**(2-j) for j,c in enumerate(cells))==key
        assert data['all_solutions'][:,cells].all(1).sum()==count>0
        assert universe['counts'][np.searchsorted(universe['keys'],key)]==count
    i,j=np.triu_indices(169,1);dr=j//13-i//13;dc=j%13-i%13;models={};diags=[]
    oracle=np.array([forbidden(r,c) for r,c in zip(dr,dc)])
    for cfg in p['arms']:
        arm=cfg['arm'];folder=OUT/f'models/{arm}';rec=json.loads((folder/'model.json').read_text());sha(folder/'model.json')
        assert rec['model_sha256']==sha(folder/'model.pt') and rec['construction_sha256']==sha(folder/'construction.npz')
        if arm=='true-graph-oracle':mask=oracle
        else:
            finite={o:sample[o]==0 for o in sample} if arm=='n12-full' else keep
            mask=np.array([any(a*r+b*c==0 for a,b in lines) or finite.get((r,c),False) for r,c in zip(dr,dc)])
        assert np.array_equal(mask,np.load(folder/'construction.npz')['forbidden'])
        J=np.zeros((169,169),np.int64);J[i,j]=13*mask-1;J[j,i]=J[i,j]
        m=torch.load(folder/'model.pt',weights_only=False);models[arm]=m
        assert np.array_equal(J,m['S'][:,:,1,1].numpy()) and not m['S'][:,:,0,:].any() and not m['S'][:,:,:,0].any() and not m['B'].any()
        assert m['q']==13 and m['delta']==12 and m['tau']==-78 and m['unit']==p['calibration']['unit']
        diags.append(dict(arm=arm,forbidden_pairs=int(mask.sum()),false_forbidden=int((mask&~oracle).sum()),missed_forbidden=int((~mask&oracle).sum())))
    assert dt.datetime.fromisoformat(p['frozen_utc'])<dt.datetime.fromisoformat(fit['utc'])
    groups=json.loads((OUT/'energy-groups.json').read_text());sha(OUT/'energy-groups.json')
    fixtures=np.random.default_rng(83116).integers(0,2,(15,169)).astype(bool);members=[]
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
    spec=importlib.util.spec_from_file_location('h3t_runner',SRC/'driver.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    positive=data['all_solutions'][:64];assert runner.camel_valid(positive[:,None,:],np.zeros_like(positive)).all()
    ordinary=np.asarray(D.completions(np.zeros(169,bool),13,limit=128));target_keys={row.tobytes() for row in data['all_solutions']}
    negative=np.array([row for row in ordinary if row.tobytes() not in target_keys]);assert len(negative)>0
    assert not runner.camel_valid(negative[:,None,:],np.zeros_like(negative)).any();sha(__file__)
    rec=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='Literal per-board-size source counts and all finite zeros retained; target graph/solution comparisons are diagnostic only',
        retained_lines=retained,source_support_diagnostic=support_diagnostic,target_models=diags,identity_groups=groups,
        exposure_heuristic_unchanged_from_h3r=True,all_identities_exact=True,scoring_positive_fixture_count=len(positive),scoring_camel_violation_fixture_count=len(negative),input_sha256=HASHES)
    (OUT/'independent-preflight.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(dict(status='passed',retained_lines=retained,target_models=diags,identity_groups=groups,source_support={k:v for k,v in support_diagnostic.items() if k not in ['extra_source_zero_offsets','all_offsets','removed_zero_offsets']})))
if __name__=='__main__':main()
