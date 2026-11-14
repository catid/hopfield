"""CPU audit and preregistered inference for H1c, from saved outputs only."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/research/20261114/ripper'
OUT=BASE/'mechanism'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def interval(diff,seed=20261114):
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(5000):
        s=rng.integers(0,diff.shape[0],diff.shape[0]);i=rng.integers(0,diff.shape[2],diff.shape[2])
        boot.append(float(100*diff[s][:,:,i].mean()))
    return np.quantile(boot,[.025,.975]).tolist()

def main():
    ppath=HERE/'protocol-mechanism.json';p=json.loads(ppath.read_text());n=2048
    rows=[];arrays={};models={};hashes={};units={};proofs=[]
    for k in p['primary_k']:
      for seed in p['training_seeds']:
       for arm in p['arms']:
        d=OUT/f'jobs/k{k}-s{seed}-{arm}';r=json.loads((d/'receipt.json').read_text())
        assert r['status']=='complete' and r['protocol_sha256']==sha(ppath)
        assert r['wrapper_sha256']==sha(HERE/'mechanism.py') and r['capture_helper_sha256']==sha(HERE/'run.py')
        assert r['development_rows_half_open']==[2560,4608] and r['test_rows']==n
        for name,h in r['output_sha256'].items():assert sha(d/name)==h;hashes[str((d/name).relative_to(ROOT))]=h
        for name,h in r['source_sha256'].items():assert sha(ROOT/name)==h;hashes[name]=h
        hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
        entry=next(e for e in p['frozen_inputs'] if e['k']==k and e['seed']==seed)
        for name,h in entry['sha256'].items():
            q=ROOT/entry['input_dir']/name;assert sha(q)==h;hashes[str(q.relative_to(ROOT))]=h
        z=np.load(d/'boards.npz');b=z['boards'];y=z['solutions'];f=z['free'];st=z['stopped'];moves=z['moves']
        assert b.shape==(n,81) and np.array_equal(z['puzzle'],np.arange(2560,4608))
        exact=(b==y).all(1);valid=[]
        for board,target,free in zip(b,y,f):
            q=board.reshape(9,9);want=set(range(9));good=all(set(q[i].tolist())==want and set(q[:,i].tolist())==want for i in range(9))
            good=good and all(set(q[i:i+3,j:j+3].reshape(-1).tolist())==want for i in [0,3,6] for j in [0,3,6]) and bool(np.all((board==target)|free));valid.append(good)
        assert np.array_equal(valid,exact) and np.array_equal(z['valid'],valid) and np.array_equal(z['exact'],exact)
        assert int(exact.sum())==r['solved'] and int((st&~exact).sum())==r['wrong_stops'] and int((~st).sum())==r['budget_failures']
        assert np.all((moves>0)&(moves<=262144)) and np.all(moves[~st]==r['terminal_move'])
        assert r['terminal_move']==262144 or bool(st.all())
        result=json.loads((d/'result.json').read_text())['per_puzzle']
        assert np.array_equal(result['solved'],exact) and np.array_equal(result['stopped'],st)
        assert np.allclose(result['flops'],moves*r['fl_move']*4,rtol=0,atol=0)
        units.setdefault((k,seed),[]).append(r['used_hbar'])
        models[k,seed,arm]=torch.load(d/'searched_model.pt',map_location='cpu',weights_only=True)
        arrays[k,seed,arm]=exact.astype(float);proofs.append(r['gauge_proof'])
        row={key:r[key] for key in ['job_id','k','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s','fl_move','used_hbar']}
        row['D']=r['statistics']['D'];row['mean_flop_proxy']=float(np.mean(result['flops']));rows.append(row)
    assert all(len(set(u))==1 for u in units.values())
    eye=torch.eye(9,dtype=torch.bool)
    for k in p['primary_k']:
      for s in p['training_seeds']:
        raw=models[k,s,'projected'];can=models[k,s,'canonical'];rep=models[k,s,'canonical_repulsive']
        w=raw['W'].double();diag=w[:,eye].mean(-1);off=w[:,~eye].mean(-1);j=diag-off
        expected=j[:,None,None]*eye
        assert torch.allclose(can['W'].double(),expected,atol=2e-7,rtol=0)
        assert torch.allclose(rep['W'].double(),j.clamp_min(0)[:,None,None]*eye,atol=2e-7,rtol=0)
        assert not torch.any(can['B']) and not torch.any(rep['B']) and not torch.any(rep['W']<0)
        residual=(w-can['W'].double()-off[:,None,None]).abs().flatten(1).max(1).values.sum()
        assert residual<1e-4
    assert all(q['all_board_float32_coefficient_residual_bound']<1e-4 and q['max_training_energy_shift_error']<1e-4 and q['training_threshold_shift_error']<1e-4 for q in proofs)
    diff=np.array([[arrays[k,s,'canonical_repulsive']-arrays[k,s,'canonical'] for k in p['primary_k']] for s in p['training_seeds']])
    gauge=np.array([[arrays[k,s,'canonical']-arrays[k,s,'projected'] for k in p['primary_k']] for s in p['training_seeds']])
    ci=interval(diff);gci=interval(gauge);effect=float(100*diff.mean());geffect=float(100*gauge.mean())
    decision='supported_for_frozen_protocol' if ci[0]>=10 else 'falsified_for_frozen_protocol' if ci[1]<10 else 'inconclusive'
    grouped=[]
    for k in p['primary_k']:
      for arm in p['arms']:
        rs=[r for r in rows if r['k']==k and r['arm']==arm]
        grouped.append({'k':k,'arm':arm,'solved_per_seed':[r['solved'] for r in rs],'mean_accuracy_pct':float(100*np.mean([r['solved']/n for r in rs])),'wrong_stops':sum(r['wrong_stops'] for r in rs),'mean_moves':float(np.mean([r['mean_moves'] for r in rs])),'mean_flop_proxy':float(np.mean([r['mean_flop_proxy'] for r in rs])),'D_per_seed':[r['D'] for r in rs]})
    for q in [ppath,HERE/'mechanism.py',HERE/'run.py',Path(__file__),BASE/'capture-parity.json']:
        hashes[str(q.relative_to(ROOT))]=sha(q)
    rec={'status':'validated','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':54,'saved_boards_verified':54*n,'primary_effect_pp':effect,'primary_ci95_pp':ci,'primary_decision':decision,'primary_per_seed_pp':(100*diff.mean((1,2))).tolist(),'secondary_gauge_effect_pp':geffect,'secondary_gauge_ci95_pp':gci,'grouped':grouped,'rows':rows,'same_hbar_all_three_arms':True,'gauge_proofs_checked':54,'max_energy_shift_error':max(q['max_training_energy_shift_error'] for q in proofs),'input_sha256':hashes,'new_inference':False,'limitations':['Six training-seed clusters; paired two-way bootstrap remains approximate.','All comparisons on new development rows2560:4608, not an external test distribution.','Equal energy up to a constant does not preserve global-ebar focus dynamics.','Logical flop proxy differs from full paper accounting and is not measured isolated wall time.','No Sudoku geometry supplied to transforms.']}
    (OUT/'audit.json').write_text(json.dumps(rec,indent=2)+'\n')
    lines=['# H1c: gauge and repulsion mechanism','',f'All54 searches and110,592 saved boards validated. Clipping negative canonical couplings changes accuracy by **{effect:.4f}pp** versus the signed canonical arm; paired two-way bootstrap95% interval **[{ci[0]:.4f}, {ci[1]:.4f}]pp**. The preregistered >=10pp hypothesis is **{decision}**.','',f'Secondary canonical-versus-raw-projection effect: **{geffect:.4f}pp**, interval[{gci[0]:.4f}, {gci[1]:.4f}]. Gauge equivalence concerns energies; the global focus reference can change trajectories. All arms use the same raw-projection thermal/focus scale and recompute training-only tau/ebar.','',
      '| k | Arm | Solved per seed (of2048) | Mean accuracy (%) | Wrong stops |','|---|---|---|---:|---:|','| — | BP AdamW: outside post-training protocol | Not run | — | — |']
    for r in grouped:lines.append(f"| {r['k']} | {r['arm']} | {' / '.join(map(str,r['solved_per_seed']))} | {r['mean_accuracy_pct']:.4f} | {r['wrong_stops']} |")
    lines+=['','No Sudoku spatial peer graph enters any intervention. The canonical transformation is explicit; broad invariance under arbitrary original pair/unary gauge changes is not claimed. Paired seeds use the same initialization but model-specific compaction may change later random streams.','',
      '[Frozen protocol](protocol-mechanism.json) · [Runner](mechanism.py) · [CPU audit](audit_mechanism.py) · [Receipt](../../../artifacts/research/20261114/ripper/mechanism/audit.json)','']
    (HERE/'REPORT-mechanism.md').write_text('\n'.join(lines))
    print(json.dumps({key:rec[key] for key in ['status','jobs','saved_boards_verified','primary_effect_pp','primary_ci95_pp','primary_decision','secondary_gauge_effect_pp','secondary_gauge_ci95_pp']}))

if __name__=='__main__':main()
