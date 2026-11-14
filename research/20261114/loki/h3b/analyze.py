"""Independent saved-output validation and frozen H3b analysis; CPU only."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
from pathlib import Path
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3b';OUT=ROOT/'artifacts/research/20261114/loki/h3b'
sys.path.insert(0,str(ROOT/'upstream'))
from cram import nqueens_data as D
spec=importlib.util.spec_from_file_location('h3_analysis',SRC.parent/'analyze.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
H={}
def sha(p):
    p=Path(p)
    with p.open('rb') as f:v=hashlib.file_digest(f,'sha256').hexdigest()
    H[str(p.relative_to(ROOT))]=v;return v
def read(p):sha(p);return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def validate_record(arm,stage,seed,suffix,inputs,solutions,keys):
    folder=OUT/f'{stage}/{arm}/s{seed}';r=read(folder/f'{suffix}.json');assert r['status']=='complete'
    assert r['protocol_sha256']==sha(SRC/'protocol.json') and r['source_sha256']==sha(SRC/'driver.py') and r['capture_source_sha256']==sha(SRC/'capture_solve.py')
    assert r['model_sha256']==sha(ROOT/r['model_path']) and r['records_sha256']==sha(folder/f'{suffix}.npz')
    z=dict(np.load(folder/f'{suffix}.npz'));b=np.unpackbits(z['boards'],axis=-1)[...,:100].astype(bool)
    assert np.array_equal(inputs,z['inputs'])
    ok=np.array([[row.tobytes() in keys and row[givens].all() for row in samples] for givens,samples in zip(inputs,b)])
    assert np.array_equal(ok,z['valid'])
    counts=np.array([solutions[:,row].all(1).sum() for row in inputs]);denom=np.minimum(counts,20)
    found=np.array([min(len({row.tobytes() for row in samples[valid]}),cap) for samples,valid,cap in zip(b,ok,denom)])
    pick=z['energy'].argmin(1);n20=ok[np.arange(len(inputs)),pick]
    assert np.array_equal(found,z['found']) and np.array_equal(n20,z['n20']) and np.array_equal(denom,z['denom']) and np.array_equal(counts,z['solution_counts'])
    assert np.array_equal(ok[:,0],z['n1'])
    cap=80000 if stage=='secondary' and suffix=='final' else 20000
    assert ((z['moves']>=0)&(z['moves']<=cap)).all() and (z['moves'][~z['stopped']]==cap).all()
    assert np.array_equal(z['flop'],20201+514*z['moves'])
    assert int((z['stopped']&~ok).sum())==r['wrong_stops'] and int((~z['stopped']).sum())==r['budget_failures']
    assert r['n20_accuracy_pct']==100*float(n20.mean()) and r['coverage_pct']==100*float(found.sum()/denom.sum())
    z['unpacked']=b
    return r,z


def gap_diagnostics(z,model):
    """Exact Ising returned-board single-flip margins, post hoc only."""
    J=model['S'][:,:,1,1].numpy().astype(np.float64);h=model['B'][:,1].numpy().astype(np.float64)
    boards=z['unpacked'].reshape(-1,100);given=np.repeat(z['inputs'],20,axis=0)
    gaps=[]
    for lo in range(0,len(boards),4096):
        x=boards[lo:lo+4096].astype(np.float64);field=x@J+h
        margin=(1-2*x)*field;margin[given[lo:lo+4096]]=np.inf
        gaps.append(margin.min(1)-(float(model['delta'])-.001))
    gaps=np.concatenate(gaps);valid=z['valid'].reshape(-1);stopped=z['stopped'].reshape(-1)
    out={}
    for name,sel in [('correct_stopped',valid&stopped),('wrong_stopped',~valid&stopped),('correct_capped',valid&~stopped),('wrong_capped',~valid&~stopped)]:
        values=gaps[sel]
        out[name]=dict(n=int(sel.sum()),gap_quantiles=np.quantile(values,[0,.01,.1,.5,.9,.99,1]).tolist() if len(values) else None,
            negative_gap_count=int((values<0).sum()),below_minus1e3_count=int((values<-.001).sum()))
    return dict(unit=float(model['unit']),delta=float(model['delta']),quantiles=[0,.01,.1,.5,.9,.99,1],groups=out,
        scope='Float64 direct Ising fields on saved returned boards. Capped output is the best-energy visited board, not necessarily the last state. Labels/rules used only after inference to classify correctness.')


def main():
    start=time.perf_counter();torch.set_num_threads(1)
    p=read(SRC/'protocol.json');validation=read(OUT/'validation.json');assert validation['status']=='passed'
    assert all(validation['prefix_fields_equal']) and all(validation['final_fields_equal']) and validation['multi_block_capture_rejected']
    proof=read(OUT/'production-prefix-parity.json');assert proof['status']=='passed' and all(proof['fields_equal'])
    for path,h in p['source_sha256'].items():assert sha(ROOT/path)==h,path
    for path,h in p['data_sha256'].items():assert sha(ROOT/path)==h,path
    for path,h in p['cohorts']['10']['data_sha256'].items():assert sha(ROOT/path)==h,path
    for cfg in p['arms'].values():
        if 'cached_model' in cfg:assert sha(ROOT/cfg['cached_model'])==cfg['model_sha256'] and sha(ROOT/cfg['cached_fit'])==cfg['fit_sha256']
    raw=np.load(OUT/'test-inputs.npz')['inputs']==2;solutions=np.array(D.completions(np.zeros(100,bool),10),bool);assert len(solutions)==724
    keys={row.tobytes() for row in solutions};rows=[];arrays={};diagnostics={};fits={}
    for arm in ('adamw-decay','adam-nodecay','sgd-pre'):
        cfg=p['arms'][arm];fitpath=ROOT/cfg['cached_fit'] if 'cached_fit' in cfg else OUT/'n10/fits/adam-nodecay/fit.json'
        fit=read(fitpath);fits[arm]=fit
        assert fit['updates']==4000 and fit['rows']==97554
        for stage,suffixes in [('primary',['final']),('secondary',['prefix20k','final'])]:
            inputs=raw[:p[stage]['n']]
            loaded={}
            for suffix in suffixes:
                recs=[];zs=[]
                for seed in (0,1,2):
                    rec,z=validate_record(arm,stage,seed,suffix,inputs,solutions,keys);recs.append(rec);zs.append(z)
                    if stage=='primary':
                        model=torch.load(ROOT/rec['model_path'],map_location='cpu',weights_only=False)
                        diagnostics[f'{arm}/s{seed}']=gap_diagnostics(z,model)
                loaded[suffix]=zs;arr={k:np.stack([z[k] for z in zs]) for k in ('n20','n1','found')};denom=zs[0]['denom']
                arrays[arm,stage,suffix]=(arr,denom)
                rows.append(dict(arm=arm,stage=stage,budget=80000 if stage=='secondary' and suffix=='final' else 20000,
                    suffix=suffix,unique_inputs=len(inputs),search_seeds=[0,1,2],n20_accuracy_pct=float(arr['n20'].mean()*100),
                    n1_accuracy_pct=float(arr['n1'].mean()*100),coverage_pct=float(arr['found'].sum(axis=1).mean()/denom.sum()*100),
                    mean_moves=float(np.mean([r['mean_moves'] for r in recs])),wrong_stops=sum(r['wrong_stops'] for r in recs),
                    budget_failures=sum(r['budget_failures'] for r in recs),search_seconds_total=sum(r['search_seconds'] for r in recs) if suffix=='final' else None))
            if stage=='secondary':
                for old,new in zip(loaded['prefix20k'],loaded['final']):
                    stopped=old['stopped'];assert new['stopped'][stopped].all()
                    for k in ('boards','energy','moves','flop'):assert np.array_equal(old[k][stopped],new[k][stopped]),k
                    assert (new['moves'][~stopped]>20000).all()
    comparisons=[]
    for arm in ('adam-nodecay','adamw-decay'):
        a,denom=arrays[arm,'primary','final'];b,_=arrays['sgd-pre','primary','final'];stats=A.bootstrap(a,b,denom)
        loss=(b['n20']&~a['n20']).any(axis=0);upper=A.cp_upper(int(loss.sum()),len(loss))
        supported=upper<.001 and stats['coverage_one_sided_95_lower_pp']>-2
        falsified=stats['accuracy_one_sided_95_upper_pp']<-.1 or stats['coverage_one_sided_95_upper_pp']<-2
        comparisons.append(dict(arm=arm,control='sgd-pre',decision='supported' if supported else 'falsified' if falsified else 'inconclusive',
            any_regression_inputs=int(loss.sum()),regression_probability_exact95_upper=upper,conservative_accuracy95_lower_pp=-100*upper,**stats))
    tradeoffs=[]
    for arm in p['arms']:
        a,denom=arrays[arm,'secondary','final'];b,_=arrays[arm,'secondary','prefix20k'];stats=A.bootstrap(a,b,denom)
        r20=next(r for r in rows if r['arm']==arm and r['stage']=='secondary' and r['budget']==20000)
        r80=next(r for r in rows if r['arm']==arm and r['stage']=='secondary' and r['budget']==80000)
        tradeoffs.append(dict(arm=arm,mean_move_ratio_80k_over20k=r80['mean_moves']/r20['mean_moves'],**stats))
    decision=next(c['decision'] for c in comparisons if c['arm']=='adam-nodecay')
    sha(SRC.parent/'analyze.py');sha(Path(__file__))
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3b',decision=decision,
      rows=rows,primary_comparisons=comparisons,secondary_budget_tradeoffs=tradeoffs,fit_summaries={k:{f:v[f] for f in ['complete_fit_seconds','final_loss','unit','delta','tau','offsets']} for k,v in fits.items()},
      diagnostics=diagnostics,input_sha256=H,hashed_files=len(H),new_training_or_inference=False,audit_seconds=time.perf_counter()-start,
      limits=['One deterministic fitted model per arm, three repeated search seeds per unique input, not training-seed replication.',
        'Officialtest used once per declared arm/protocol; no outcome-based model or temperature selection.',
        'Secondary1024inputs are the fixed canonical prefix, not a random representative sample of all6513.',
        'Budget-prefix parity is valid only inside one original search block; primary test retains original multiblock behavior.',
        'Confirmation is conditional on fixedLR0.3,4000updates,geometry-sharedIsing energy and this task; it does not establish optimizer superiority across tasks.'])
    save(OUT/'summary.json',summary)
    lines=['# H3b: weight decay and finite search budget','',f"**Preregistered result: {decision}.** Removing weight decay at fixed learning rate 0.3 was compared with the original AdamW control and preconditioned SGD on all 6,513 unique official n=10 test inputs. Each arm used 20 chains under three search seeds; all fits used the same 97,554 fitting rows and 4,000 updates. No settings were selected on test results.",'',
      '| Method | N20 accuracy (%) | Coverage (%) | Wrong stopped chains | Capped chains | Mean moves / chain |',
      '|---|---:|---:|---:|---:|---:|']
    names={'adamw-decay':'BP AdamW, decay 0.01','adam-nodecay':'BP Adam, no decay','sgd-pre':'SGD + geometry preconditioner'}
    for arm in ('adamw-decay','adam-nodecay','sgd-pre'):
        r=next(r for r in rows if r['arm']==arm and r['stage']=='primary')
        lines.append(f"| {names[arm]} | {r['n20_accuracy_pct']:.6f} | {r['coverage_pct']:.6f} | {r['wrong_stops']:,} | {r['budget_failures']:,} | {r['mean_moves']:.2f} |")
    c=next(c for c in comparisons if c['arm']=='adam-nodecay')
    lines+=['',f"No-decay Adam versus preconditioned SGD: accuracy difference {c['accuracy_difference_pp']:+.6f} percentage points, coverage difference {c['coverage_difference_pp']:+.6f} points. One-sided 95% bootstrap lower/upper bounds are [{c['accuracy_one_sided_95_lower_pp']:.6f}, {c['accuracy_one_sided_95_upper_pp']:.6f}] for accuracy and [{c['coverage_one_sided_95_lower_pp']:.6f}, {c['coverage_one_sided_95_upper_pp']:.6f}] for coverage. These are the two one-sided endpoints, equivalent to a 90% two-sided interval.",'',
      f"The exact regression safeguard observed {c['any_regression_inputs']} unique inputs with any paired accuracy regression; its one-sided 95% upper probability is {100*c['regression_probability_exact95_upper']:.6f}%. The preregistered noninferiority margins are 0.1 point for accuracy and 2 points for coverage. Search repeats stay grouped within each input for uncertainty.",'',
      '## Same-trajectory budget comparison','',
      'The following uses the first 1,024 canonical unique test inputs. The 20,000-move output is captured from the exact trajectory continued to 80,000 moves. Previously stopped boards, energies, moves and FLOPs remain unchanged. This is a fixed subset; it is separate from the differently batched primary full-test run.','',
      '| Method | Budget | N20 accuracy (%) | Coverage (%) | Mean moves / chain | Capped chains |',
      '|---|---:|---:|---:|---:|---:|']
    for arm in ('adamw-decay','adam-nodecay','sgd-pre'):
        for cap in (20000,80000):
            r=next(r for r in rows if r['arm']==arm and r['stage']=='secondary' and r['budget']==cap)
            lines.append(f"| {names[arm]} | {cap:,} | {r['n20_accuracy_pct']:.6f} | {r['coverage_pct']:.6f} | {r['mean_moves']:.2f} | {r['budget_failures']:,} |")
    lines+=['','The CPU parity check caught upstream RNG coupling across blocks before production: extending an earlier block shifts the RNG stream of later blocks. The snapshot solver therefore rejects multi-block cohorts. The declared secondary cohort fits one block; a full 1,024-input, 20-chain production comparison also exactly matches the original 20,000-move search. Both proof and archived prelaunch revision are retained.','',
      'Post-inference diagnostics in the JSON recompute float64 single-cell certificate gaps on saved returned boards, separated into correct/wrong and stopped/capped groups. These diagnostics never steer the solver. For capped chains, the returned board is the best-energy visited board, not necessarily the last state.','',
      'No-decay Adam uses the same tested AdamW update routine with weight decay set to zero; direct updates were verified bitwise equal to `torch.optim.Adam`. This intervention changes only weight decay. The two cached control fits are hash-identical to H3.','',
      '[Frozen protocol](protocol.json), [runner](driver.py), [snapshot solver](capture_solve.py), [auditor](analyze.py), [machine-readable evidence](../../../../artifacts/research/20261114/loki/h3b/summary.json).']
    (SRC/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='validated',decision=decision,primary=c,hashed_files=len(H),audit_seconds=summary['audit_seconds'])))


if __name__=='__main__':main()
