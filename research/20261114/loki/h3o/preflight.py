"""Independent literal audit of unchanged source evidence for upward transfer and target energies."""
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
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3o';OUT=ROOT/'artifacts/research/20261114/loki/h3o'
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
    evidence={cfg['arm']:literal_counts(train[cfg['archive']],cfg['source_n'],cfg['source_n']) for cfg in p['arms'] if cfg['archive'] is not None}
    target_counts=literal_counts(data['all_solutions'],15,15)
    source=dict(np.load(OUT/'source-support.npz'));sha(OUT/'source-support.npz')
    for arm,counts in evidence.items():
        assert {tuple(o):int(v) for o,v in zip(source[f'{arm}_offsets'],source[f'{arm}_counts'])}==counts,arm
    fit=json.loads((OUT/'source-fits.json').read_text());sha(OUT/'source-fits.json');retained={};extras=[]
    for cfg in p['arms']:
        arm=cfg['arm']
        if arm not in evidence:continue
        counts=evidence[arm];candidates=[];lines=[]
        for a,b in itertools.product(range(-2,3),repeat=2):
            if (a,b)==(0,0) or math.gcd(abs(a),abs(b))!=1 or a<0 or (a==0 and b<0):continue
            members=[v for (r,c),v in counts.items() if a*r+b*c==0]
            nf=sum(v==0 for v in members);na=sum(v>0 for v in members);keep=nf>=3 and na==0
            candidates.append(dict(a=a,b=b,forbidden_offsets=nf,allowed_offsets=na,retained=keep))
            if keep:lines.append([a,b])
        assert fit['arms'][arm]==dict(source_n=cfg['source_n'],source_patterns=cfg['source_patterns'],candidates=candidates,retained_lines=lines,source_zero_offsets=sum(v==0 for v in counts.values()))
        retained[arm]=lines
        for (r,c),count in counts.items():
            if count==0 and not forbidden(r,c):extras.append(dict(arm=arm,source_n=cfg['source_n'],dr=r,dc=c,source_count=0,
                n14_full_count=evidence['n14-full'].get((r,c),0),target_pair_count=max(0,15-r)*max(0,15-abs(c)),full_N15_count=target_counts.get((r,c),0)))
    parent_support=np.load(ROOT/'artifacts/research/20261114/loki/h3n/source-support.npz')
    assert np.array_equal(source['n14-full_offsets'],parent_support['n14-full_offsets']) and np.array_equal(source['n14-full_counts'],parent_support['n14-full_counts'])
    support_diagnostic=dict(extra_source_zero_offsets=extras,all_extra_offsets_have_N15_witnesses=all(v['full_N15_count']>0 for v in extras),
        total_target_pair_exclusions_from_extra_offsets=sum(v['target_pair_count'] for v in extras),
        use='Post-construction diagnosis only; no offset removed or modified.')
    universe_path=OUT/'input-universe.npz';universe=dict(np.load(universe_path));sha(universe_path)
    expected=np.sort(np.random.default_rng(2026112506).permutation(universe['keys'])[:4096])
    assert np.array_equal(expected,data['test_keys']) and len(data['prior_target_test_keys'])==0
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
            counts=evidence[arm];lines=retained[arm]
            mask=np.array([any(a*r+b*c==0 for a,b in lines) or ((r,c) in counts and counts[r,c]==0) for r,c in zip(dr,dc)])
        assert np.array_equal(mask,np.load(folder/'construction.npz')['forbidden'])
        J=np.zeros((225,225),np.int64);J[i,j]=15*mask-1;J[j,i]=J[i,j]
        m=torch.load(folder/'model.pt',weights_only=False);models[arm]=m
        assert np.array_equal(J,m['S'][:,:,1,1].numpy()) and not m['S'][:,:,0,:].any() and not m['S'][:,:,:,0].any() and not m['B'].any()
        assert m['q']==15 and m['delta']==14 and m['tau']==-105 and m['unit']==p['calibration']['unit']
        diags.append(dict(arm=arm,forbidden_pairs=int(mask.sum()),false_forbidden=int((mask&~oracle).sum()),missed_forbidden=int((~mask&oracle).sum())))
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
    spec=importlib.util.spec_from_file_location('h3o_runner',SRC/'driver.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    positive=data['all_solutions'][:64];assert runner.super_valid(positive[:,None,:],np.zeros_like(positive)).all()
    ordinary=np.asarray(D.completions(np.zeros(225,bool),15,limit=128));target_keys={row.tobytes() for row in data['all_solutions']}
    negative=np.array([row for row in ordinary if row.tobytes() not in target_keys]);assert len(negative)>0
    assert not runner.super_valid(negative[:,None,:],np.zeros_like(negative)).any();sha(__file__)
    rec=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='Literal per-board-size source counts and all finite zeros retained; target graph/solution comparisons are diagnostic only',
        retained_lines=retained,source_support_diagnostic=support_diagnostic,target_models=diags,identity_groups=groups,
        source_counts_exact_h3n=True,all_identities_exact=True,scoring_positive_fixture_count=len(positive),scoring_knight_violation_fixture_count=len(negative),input_sha256=HASHES)
    (OUT/'independent-preflight.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(dict(status='passed',retained_lines=retained,target_models=diags,identity_groups=groups,source_support={k:v for k,v in support_diagnostic.items() if k!='extra_source_zero_offsets'})))
if __name__=='__main__':main()
