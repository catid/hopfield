"""Independent literal audit of source-only exposure scores and resulting target energies and target energies."""
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
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3r';OUT=ROOT/'artifacts/research/20261114/loki/h3r'
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
    parent=dict(np.load(ROOT/'artifacts/research/20261114/loki/h3n/training.npz'))
    assert train['n14_all'].shape==(5180,196) and data['all_solutions'].shape==(32516,225)
    assert np.array_equal(train['n14_all'],parent['n14_all'])
    feasibility=json.loads((OUT/'feasibility.json').read_text())
    assert feasibility['solutions_sha256']==sha(OUT/'solutions-n15.npy')
    assert feasibility['independent_enumerations_match'] and feasibility['every_board_pairwise_valid']
    assert feasibility['complete_superqueen_solutions']==32516 and feasibility['complete_ordinary_queen_solutions']==2279184
    assert feasibility['seconds']<feasibility['cpu_wall_seconds_envelope']==300
    assert np.array_equal(data['all_solutions'],np.load(OUT/'solutions-n15.npy'))
    for path,digest in feasibility['source_sha256'].items():assert sha(ROOT/path)==digest,path
    sample=literal_counts(train['n14_all'],14,14);target_counts=literal_counts(data['all_solutions'],15,15)
    source=dict(np.load(OUT/'source-support.npz'));sha(OUT/'source-support.npz')
    marginal=np.zeros(196,np.int64)
    for board in train['n14_all']:
        for cell in np.flatnonzero(board):marginal[cell]+=1
    assert np.array_equal(marginal,source['marginal_counts'])
    exposure_numerator={}
    for dr,dc in sample:
        total=0
        for row in range(14):
            for col in range(14):
                if 0<=row+dr<14 and 0<=col+dc<14:total+=int(marginal[row*14+col])*int(marginal[(row+dr)*14+col+dc])
        exposure_numerator[dr,dc]=total
    expected={o:v/5180 for o,v in exposure_numerator.items()};threshold=math.log(len(sample)/.01);assert len(sample)==364==p['exposure']['K']
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
    parent_fit=json.loads((ROOT/'artifacts/research/20261114/loki/h3o/source-fits.json').read_text())['arms']['n14-full']
    assert lines==parent_fit['retained_lines'] and candidates==parent_fit['candidates']
    parent_support=np.load(ROOT/'artifacts/research/20261114/loki/h3o/source-support.npz')
    assert np.array_equal(source['offsets'],parent_support['n14-full_offsets']) and np.array_equal(source['counts'],parent_support['n14-full_counts'])
    offsets=[]
    for (r,c),count in sample.items():
        offsets.append(dict(dr=r,dc=c,source_count=count,exposure_numerator=exposure_numerator[r,c],source_patterns=5180,expected_count=expected[r,c],threshold=threshold,
            finite_zero_retained=keep[r,c],removed_zero=count==0 and not keep[r,c],covered_by_unchanged_line=any(a*r+b*c==0 for a,b in lines),
            true_local_rule=forbidden(r,c),target_pair_count=max(0,15-r)*max(0,15-abs(c)),full_N15_count=target_counts.get((r,c),0)))
    support_diagnostic=dict(K=364,threshold=threshold,all_offsets=offsets,removed_zero_offsets=[v for v in offsets if v['removed_zero']],
        extra_source_zero_offsets=[v for v in offsets if v['source_count']==0 and not v['true_local_rule']],
        heuristic_scope='Independent-cell exposure assumption only; no calibrated p-values, coverage or Bonferroni guarantee for constrained patterns.',
        use='Threshold and score frozen before computation; target truth diagnostics do not change any score, line or coefficient.')
    retained=lines
    universe_path=OUT/'input-universe.npz';universe=dict(np.load(universe_path));sha(universe_path)
    initial=np.load(ROOT/'artifacts/research/20261114/loki/h3o/data.npz');confirmation=np.load(ROOT/'artifacts/research/20261114/loki/h3p/data.npz');excluded=np.union1d(initial['test_keys'],confirmation['test_keys'])
    assert np.array_equal(data['prior_target_test_keys'],excluded)
    eligible=universe['keys'][~np.isin(universe['keys'],excluded)];assert len(eligible)==815589
    expected=np.sort(np.random.default_rng(2026112509).permutation(eligible)[:4096])
    assert np.array_equal(expected,data['test_keys']) and not np.isin(expected,excluded).any()
    for given,key,count in zip(data['test_inputs'],expected,data['test_solution_counts']):
        cells=np.flatnonzero(given);assert sum((int(c)+1)*226**(2-j) for j,c in enumerate(cells))==key
        assert data['all_solutions'][:,cells].all(1).sum()==count>0
        assert universe['counts'][np.searchsorted(universe['keys'],key)]==count
    i,j=np.triu_indices(225,1);dr=j//15-i//15;dc=j%15-i%15;models={};diags=[]
    oracle=np.array([forbidden(r,c) for r,c in zip(dr,dc)])
    for cfg in p['arms']:
        arm=cfg['arm'];folder=OUT/f'models/{arm}';rec=json.loads((folder/'model.json').read_text());sha(folder/'model.json')
        assert rec['model_sha256']==sha(folder/'model.pt') and rec['construction_sha256']==sha(folder/'construction.npz')
        if arm=='true-graph-oracle':mask=oracle
        else:
            finite={o:sample[o]==0 for o in sample} if arm=='n14-full' else keep
            mask=np.array([any(a*r+b*c==0 for a,b in lines) or finite.get((r,c),False) for r,c in zip(dr,dc)])
        assert np.array_equal(mask,np.load(folder/'construction.npz')['forbidden'])
        J=np.zeros((225,225),np.int64);J[i,j]=15*mask-1;J[j,i]=J[i,j]
        m=torch.load(folder/'model.pt',weights_only=False);models[arm]=m
        assert np.array_equal(J,m['S'][:,:,1,1].numpy()) and not m['S'][:,:,0,:].any() and not m['S'][:,:,:,0].any() and not m['B'].any()
        assert m['q']==15 and m['delta']==14 and m['tau']==-105 and m['unit']==p['calibration']['unit']
        diags.append(dict(arm=arm,forbidden_pairs=int(mask.sum()),false_forbidden=int((mask&~oracle).sum()),missed_forbidden=int((~mask&oracle).sum())))
    for arm in ['n14-full','true-graph-oracle']:
        prior=torch.load(ROOT/f'artifacts/research/20261114/loki/h3o/models/{arm}/model.pt',weights_only=False)
        assert all(torch.equal(prior[k],models[arm][k]) for k in ['S','B']) and all(prior[k]==models[arm][k] for k in ['q','delta','tau','unit'])
    assert dt.datetime.fromisoformat(p['frozen_utc'])<dt.datetime.fromisoformat(fit['utc'])
    groups=json.loads((OUT/'energy-groups.json').read_text());sha(OUT/'energy-groups.json')
    fixtures=np.random.default_rng(83116).integers(0,2,(15,225)).astype(bool);members=[]
    for group in groups['groups']:
        m=models[group['representative']];J=m['S'][:,:,1,1].numpy().astype(np.int64)
        literal=(fixtures[:,i]*fixtures[:,j]*J[i,j]).sum(1)
        assert np.array_equal(literal,Q.board_energy(m['S'].double(),m['B'].double(),torch.tensor(fixtures,dtype=torch.long)).numpy())
        for board,energy in zip(fixtures,literal):
            for cell in range(225):
                other=board.copy();other[cell]=~other[cell]
                assert (other[i]*other[j]*J[i,j]).sum()-energy==(1-2*int(board[cell]))*(board@J[:,cell])
        for arm in group['members']:
            other=models[arm];assert all(torch.equal(other[k],m[k]) for k in ['S','B']) and all(other[k]==m[k] for k in ['q','delta','tau','unit'])
            members.append(arm)
    assert sorted(members)==sorted(cfg['arm'] for cfg in p['arms'])
    for a,b in itertools.combinations(groups['groups'],2):assert not torch.equal(models[a['representative']]['S'],models[b['representative']]['S'])
    spec=importlib.util.spec_from_file_location('h3r_runner',SRC/'driver.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    positive=data['all_solutions'][:64];assert runner.super_valid(positive[:,None,:],np.zeros_like(positive)).all()
    ordinary=np.asarray(D.completions(np.zeros(225,bool),15,limit=128));target_keys={row.tobytes() for row in data['all_solutions']}
    negative=np.array([row for row in ordinary if row.tobytes() not in target_keys]);assert len(negative)>0
    assert not runner.super_valid(negative[:,None,:],np.zeros_like(negative)).any();sha(__file__)
    rec=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='Literal per-board-size source counts and all finite zeros retained; target graph/solution comparisons are diagnostic only',
        retained_lines=retained,source_support_diagnostic=support_diagnostic,target_models=diags,identity_groups=groups,
        source_counts_exact_h3o=True,control_solver_parameters_exact_h3o=True,all_identities_exact=True,scoring_positive_fixture_count=len(positive),scoring_knight_violation_fixture_count=len(negative),input_sha256=HASHES)
    (OUT/'independent-preflight.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(dict(status='passed',retained_lines=retained,target_models=diags,identity_groups=groups,source_support={k:v for k,v in support_diagnostic.items() if k not in ['extra_source_zero_offsets','all_offsets','removed_zero_offsets']})))
if __name__=='__main__':main()
