"""Independently audit all source-bank fits, target coefficients and deduplication."""
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

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3k';OUT=ROOT/'artifacts/research/20261114/loki/h3k'
sys.path.insert(0,str(ROOT/'upstream'))
from cram import cram_nqueens as Q
spec=importlib.util.spec_from_file_location('stats',SRC.parent/'analyze.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
HASHES={}


def sha(path):
    path=Path(path)
    with path.open('rb') as f:v=hashlib.file_digest(f,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=v;return v


def main():
    torch.set_num_threads(1);p=json.loads((SRC/'protocol.json').read_text());sha(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    d=dict(np.load(OUT/'data.npz'));fits=json.loads((OUT/'source-fits.json').read_text());sha(OUT/'source-fits.json')
    assert fits['array_sha256']==sha(OUT/'source-fits.npz') and fits['source_sha256']==sha(SRC/'driver.py')
    z=dict(np.load(OUT/'source-fits.npz'));assert len(fits['rows'])==2560
    assert not np.isin(d['source_draw_indices'],d['source_excluded_indices']).any()
    for seed,draw in zip(d['source_draw_seeds'],d['source_draw_indices']):
        assert np.array_equal(draw,np.random.default_rng(int(seed)).choice(d['source_available_indices'],16,replace=False))
    offsets=[tuple(o) for o in z['offsets']];candidates=[]
    for a,b in itertools.product(range(-2,3),repeat=2):
        if (a,b)==(0,0) or math.gcd(abs(a),abs(b))!=1 or a<0 or (a==0 and b<0):continue
        candidates.append((a,b))
    assert np.array_equal(candidates,z['candidates'])
    truth={(0,1),(1,0),(1,1),(1,-1)};outcomes=[]
    for draw,indices in enumerate(d['source_draw_indices']):
        counts={o:0 for o in offsets}
        for k,index in enumerate(indices,1):
            label=d['source_solutions'][index];cells=np.flatnonzero(label);r=cells//12;c=cells%12
            assert len(cells)==len(set(r))==len(set(c))==len(set(r-c))==len(set(r+c))==12
            for a,b in itertools.combinations(cells,2):counts[b//12-a//12,b%12-a%12]+=1
            if k not in p['source_sizes']:continue
            row_index=draw*5+p['source_sizes'].index(k);row=fits['rows'][row_index]
            assert np.array_equal([counts[o] for o in offsets],z['counts'][row_index])
            retained=[]
            for a,b in candidates:
                members=[v for (r,c),v in counts.items() if a*r+b*c==0]
                if sum(v==0 for v in members)>=3 and not any(v>0 for v in members):retained.append((a,b))
            assert [list(v) for v in retained]==row['retained_lines']
            assert np.array_equal([candidate in retained for candidate in candidates],z['retained'][row_index])
            inferred=set(retained);outcomes.append(dict(draw=draw,source_seed=int(d['source_draw_seeds'][draw]),source_patterns=k,
                exact_four_line_recovery=inferred==truth,spurious_lines=len(inferred-truth),missing_true_lines=len(truth-inferred),line_signature=row['line_signature']))
    recovery=[]
    for size in p['source_sizes']:
        rows=[r for r in outcomes if r['source_patterns']==size];success=sum(r['exact_four_line_recovery'] for r in rows);n=len(rows)
        lower=1-A.cp_upper(n-success,n);upper=A.cp_upper(success,n)
        recovery.append(dict(source_patterns=size,independent_draws=n,exact_recovery_successes=success,recovery_fraction=success/n,
            exact95_lower=lower,exact95_upper=upper,missing_true_line_models=sum(r['missing_true_lines']>0 for r in rows),
            spurious_line_models=sum(r['spurious_lines']>0 for r in rows),primary=size==8,
            decision=('supported' if lower>=.99 else 'falsified' if upper<.99 else 'inconclusive') if size==8 else 'descriptive'))
    assert len(d['test_keys'])==len(np.unique(d['test_keys']))==2048 and not np.isin(d['test_keys'],d['prior_target_test_keys']).any()
    for given,key,count in zip(d['test_inputs'],d['test_keys'],d['test_solution_counts']):
        cells=np.flatnonzero(given);assert sum((int(c)+1)*170**(2-j) for j,c in enumerate(cells))==key
        assert d['all_solutions'][:,cells].all(1).sum()==count>0
    i,j=np.triu_indices(169,1);dr=j//13-i//13;dc=j%13-i%13;models={};model_audits=[]
    for cfg in p['arms']:
        arm=cfg['arm'];folder=OUT/f'models/{arm}';rec=json.loads((folder/'model.json').read_text());sha(folder/'model.json')
        assert rec['model_sha256']==sha(folder/'model.pt') and rec['construction_sha256']==sha(folder/'construction.npz')
        row=fits['rows'][cfg['draw']*5+p['source_sizes'].index(cfg['source_patterns'])]
        assert rec['retained_lines']==row['retained_lines'] and rec['draw']==cfg['draw'] and rec['source_patterns']==cfg['source_patterns']
        mask=np.array([any(a*r+b*c==0 for a,b in row['retained_lines']) for r,c in zip(dr,dc)])
        assert np.array_equal(mask,np.load(folder/'construction.npz')['forbidden'])
        J=np.zeros((169,169),np.int64);J[i,j]=13*mask-1;J[j,i]=J[i,j]
        m=torch.load(folder/'model.pt',weights_only=False);models[arm]=m
        assert np.array_equal(J,m['S'][:,:,1,1].numpy()) and not m['S'][:,:,0,:].any() and not m['S'][:,:,:,0].any() and not m['B'].any()
        assert m['q']==13 and m['delta']==12 and m['tau']==-78 and m['unit']==p['calibration']['unit']
        attack=(dr==0)|(dc==0)|(np.abs(dr)==np.abs(dc));assert not (attack&~mask).any()
        model_audits.append(dict(arm=arm,source_patterns=cfg['source_patterns'],draw=cfg['draw'],false_forbidden=int((mask&~attack).sum()),missed_forbidden=0,
            exact_four_line_recovery=set(map(tuple,row['retained_lines']))==truth))
    groups=json.loads((OUT/'energy-groups.json').read_text());sha(OUT/'energy-groups.json')
    all_members=[];fixtures=np.random.default_rng(83115).integers(0,2,(13,169)).astype(bool)
    for group in groups['groups']:
        base=models[group['representative']];J=base['S'][:,:,1,1].numpy().astype(np.int64)
        dense=Q.board_energy(base['S'].double(),base['B'].double(),torch.tensor(fixtures,dtype=torch.long)).numpy()
        literal=(fixtures[:,i]*fixtures[:,j]*J[i,j]).sum(1);assert np.array_equal(dense,literal)
        for board,energy in zip(fixtures,literal):
            for cell in range(169):
                changed=board.copy();changed[cell]=~changed[cell]
                assert (changed[i]*changed[j]*J[i,j]).sum()-energy==(1-2*int(board[cell]))*(board@J[:,cell])
        for arm in group['members']:
            m=models[arm];assert all(torch.equal(m[k],base[k]) for k in ['S','B']) and all(m[k]==base[k] for k in ['q','delta','tau','unit'])
            all_members.append(arm)
    assert sorted(all_members)==sorted(cfg['arm'] for cfg in p['arms']) and len(all_members)==12
    sha(__file__);sha(SRC.parent/'analyze.py')
    report=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='All2560 source fits, fixed target cohort,12 model representations and full identity groups audited before target inference',
        primary=next(r for r in recovery if r['primary']),source_recovery=recovery,source_outcomes=outcomes,target_models=model_audits,
        identity_groups=groups,identity_solver_keys=['S','B','q','delta','tau','unit'],all_identities_exact=True,input_sha256=HASHES)
    (OUT/'independent-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',source_recovery=recovery,target_models=model_audits,distinct_energies=groups['distinct_energies'])))


if __name__=='__main__':main()
