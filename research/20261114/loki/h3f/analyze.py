"""Independent support, training-bank and all-returned-board audit for H3f."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/'research/20261114/loki/h3f'
OUT=ROOT/'artifacts/research/20261114/loki/h3f'
OLD=ROOT/'artifacts/research/20261114/loki/h3e'
PUBLIC=ROOT/'results/research/loki/h3f'
spec=importlib.util.spec_from_file_location('h3stats',SRC.parent/'analyze.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
HASHES={}


def sha(path):
    path=Path(path)
    with path.open('rb') as handle:value=hashlib.file_digest(handle,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=value;return value
def read(path):sha(path);return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def support(data):
    rec=read(OUT/'construction.json')
    assert rec['protocol_sha256']==sha(SRC/'protocol.json') and rec['source_sha256']==sha(SRC/'driver.py')
    assert rec['model_sha256']==sha(OUT/'model.pt') and rec['support_sha256']==sha(OUT/'training-support.npz')
    assert rec['bank_sha256']==sha(OUT/'training-bank.npy')
    z=dict(np.load(OUT/'training-support.npz'));model=torch.load(OUT/'model.pt',weights_only=False)
    labels=data['solutions'][data['rows_per_solution']>0];weights=data['rows_per_solution'][data['rows_per_solution']>0]
    assert labels.shape==(14200,144) and np.array_equal(np.unpackbits(np.load(OUT/'training-bank.npy'),axis=1)[:,:144],labels)
    q=np.unique(labels.sum(1));assert len(q)==1 and int(q[0])==rec['q_from_training']==12
    # Rebuild weighted counts by iterating coordinate offsets and complete-board intersections.
    counts=[]
    for dr,dc in z['offsets']:
        first=[];second=[]
        for r in range(12):
            for c in range(12):
                rr,cc=r+int(dr),c+int(dc)
                if 0<=rr<12 and 0<=cc<12:first.append(r*12+c);second.append(rr*12+cc)
        cooccur=(labels[:,first]&labels[:,second]).sum(1)
        counts.append(int(cooccur@weights))
    counts=np.array(counts);assert np.array_equal(counts,z['counts'])
    forbidden=counts==0;assert np.array_equal(forbidden,z['forbidden'])
    assert np.array_equal(model['A'].numpy(),12*forbidden.astype(np.float32)-1) and float(model['b'][0])==0
    offset_values={tuple(o):float(v) for o,v in zip(z['offsets'],model['A'].numpy())}
    J=model['S'][:,:,1,1].numpy().astype(np.float64)
    for i in range(144):
        for j in range(i+1,144):assert J[i,j]==J[j,i]==offset_values[j//12-i//12,j%12-i%12]
    assert not np.count_nonzero(np.diag(J)) and not np.count_nonzero(model['B'].numpy())
    x=labels.astype(np.float64);fields=x@J;margins=(1-2*x)*fields;energies=.5*(x*fields).sum(1)
    assert np.all(energies==-66) and float(margins.min())==11==model['delta']
    assert np.array_equal(margins.min(1),z['training_margin_minima'])
    unit=float((np.abs(fields).mean(1)*weights).sum()/weights.sum())
    assert unit==model['unit'] and model['tau']==-66
    return model,rec


def check(path,arrays,arm,updates,seed,data,keys,model=None):
    rec=read(path);assert rec['status']=='complete';assert rec['records_sha256']==sha(arrays)
    z=dict(np.load(arrays));boards=np.unpackbits(z['boards'],axis=-1)[...,:144].astype(bool);inputs=data['test_inputs']
    assert np.array_equal(z['inputs'],inputs)
    valid=np.array([[row.tobytes() in keys and row[given].all() for row in samples] for given,samples in zip(inputs,boards)])
    assert np.array_equal(valid,z['valid'])
    denom=np.minimum(data['test_solution_counts'],20)
    found=np.array([min(len({row.tobytes() for row in samples[ok]}),d) for samples,ok,d in zip(boards,valid,denom)])
    assert np.array_equal(found,z['found']) and np.array_equal(denom,z['denom'])
    original=z['energy'].argmin(1);selected=[]
    for energy,flags in zip(z['energy'],z['stopped']):
        pool=np.flatnonzero(flags)
        if not len(pool):pool=np.arange(20)
        selected.append(int(pool[energy[pool].argmin()]))
    selected=np.array(selected);ids=np.arange(len(inputs))
    assert np.array_equal(original,z['original_index']) and np.array_equal(selected,z['certificate_index'])
    assert np.array_equal(valid[ids,original],z['n20']) and np.array_equal(valid[ids,selected],z['n20_certificate'])
    assert np.array_equal(valid[:,0],z['n1'])
    if arm!='training-bank':
        assert np.array_equal(z['flop'],41761+734*z['moves'])
        assert (z['moves'][~z['stopped']]==20000).all() and ((z['moves']>=0)&(z['moves']<=20000)).all()
        assert rec['wrong_stops']==int((z['stopped']&~valid).sum()) and rec['capped_chains']==int((~z['stopped']).sum())
    else:
        assert not np.count_nonzero(z['moves']) and not np.count_nonzero(z['flop']) and valid.all()
        bank=np.unpackbits(np.load(OUT/'training-bank.npy'),axis=1)[:,:144].astype(bool);rng=np.random.default_rng(seed)
        for given,samples in zip(inputs,boards):
            compatible=np.flatnonzero(bank[:,np.flatnonzero(given)].all(1))
            assert np.array_equal(samples,bank[rng.choice(compatible,size=20,replace=True)])
    counts=boards.sum(-1);occupancies={}
    for group,mask in [('all_invalid',~valid),('stopped_invalid',~valid&z['stopped']),('capped_invalid',~valid&~z['stopped'])]:
        occupancies[group]={name:int((mask&condition).sum()) for name,condition in [('below_q',counts<12),('equal_q',counts==12),('above_q',counts>12)]}
    diagnostic=dict(arm=arm,updates=updates,seed=seed,occupancies=occupancies)
    if arm=='exclusion':
        assert np.array_equal(counts,z['queen_counts']) and rec['wrong_stops_by_occupancy']==occupancies['stopped_invalid']
        J=model['S'][:,:,1,1].numpy().astype(np.float64);flat=boards.reshape(-1,144);given=np.repeat(inputs,20,0);gaps=[]
        for lo in range(0,len(flat),4096):
            x=flat[lo:lo+4096].astype(np.float64);margin=(1-2*x)*(x@J);margin[given[lo:lo+4096]]=np.inf
            gaps.append(margin.min(1)-(model['delta']-.001))
        gaps=np.concatenate(gaps);stopped=gaps[z['stopped'].ravel()]
        diagnostic.update(stopped_gap_min=float(stopped.min()) if len(stopped) else None,
                          stopped_strict_certificate_violations=int((stopped<0).sum()))
    row=dict(arm=arm,updates=updates,seed=seed,unique_inputs=8192,chains=20,original_correct=int(z['n20'].sum()),
        certificate_first_correct=int(z['n20_certificate'].sum()),n1_correct=int(z['n1'].sum()),
        coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),wrong_stops=rec['wrong_stops'],
        capped_chains=rec['capped_chains'],mean_moves=rec['mean_moves'],search_seconds=rec['search_seconds'],
        records_sha256=rec['records_sha256'],model_sha256=rec['model_sha256'],result_sha256=sha(path))
    return z,row,diagnostic


def main():
    started=time.perf_counter();torch.set_num_threads(1);PUBLIC.mkdir(parents=True,exist_ok=True)
    p=read(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    data=dict(np.load(OLD/'data.npz'));keys={row.tobytes() for row in data['solutions']};assert len(keys)==14200
    model,construction=support(data);evidence={};perseed=[];diagnostics=[]
    for cache in p['cached_comparisons']:
        path=ROOT/cache['path'];assert sha(path)==cache['sha256'] and sha(ROOT/cache['model_path'])==cache['model_sha256']
        z,row,diag=check(path,path.parent/'records.npz',cache['arm'],cache['updates'],cache['seed'],data,keys)
        assert row['records_sha256']==cache['records_sha256']
        evidence.setdefault((cache['arm'],cache['updates']),[]).append(z);perseed.append(row);diagnostics.append(diag)
    for arm in ['exclusion','training-bank']:
        for seed in [0,1,2]:
            path=OUT/f'{arm}/s{seed}/result.json';rec=read(path)
            assert rec['protocol_sha256']==sha(SRC/'protocol.json') and rec['source_sha256']==sha(SRC/'driver.py')
            assert rec['model_sha256']==sha(OUT/('model.pt' if arm=='exclusion' else 'training-bank.npy'))
            z,row,diag=check(path,path.parent/'records.npz',arm,0,seed,data,keys,model)
            evidence.setdefault((arm,0),[]).append(z);perseed.append(row);diagnostics.append(diag)
    rows=[];stacked={}
    for (arm,updates),zs in evidence.items():
        arr={key:np.stack([z[key] for z in zs]) for key in ['n20','n20_certificate','n1','found']};stacked[arm,updates]=arr
        current=[r for r in perseed if r['arm']==arm and r['updates']==updates]
        rows.append(dict(arm=arm,updates=updates,unique_inputs=8192,seeds=[0,1,2],
            original_accuracy_pct=100*float(arr['n20'].mean()),certificate_first_accuracy_pct=100*float(arr['n20_certificate'].mean()),
            n1_accuracy_pct=100*float(arr['n1'].mean()),coverage_pct=100*float(arr['found'].sum(1).mean()/zs[0]['denom'].sum()),
            wrong_stops=sum(r['wrong_stops'] for r in current) if arm!='training-bank' else None,
            capped_chains=sum(r['capped_chains'] for r in current) if arm!='training-bank' else None,
            mean_moves=float(np.mean([r['mean_moves'] for r in current])) if arm!='training-bank' else None,
            seconds_mean=float(np.mean([r['search_seconds'] for r in current])),
            timing_scope='CPU compatible-bank matching and sampling' if arm=='training-bank' else 'Instrumented GPU search; not directly comparable with CPU bank time'))
    a=stacked['exclusion',0];b=stacked['adam-nodecay',250];denom=np.minimum(data['test_solution_counts'],20)
    stats=A.bootstrap(dict(n20=a['n20_certificate'],found=a['found']),dict(n20=b['n20_certificate'],found=b['found']),denom)
    regressions=(b['n20_certificate']&~a['n20_certificate']).any(0);upper=A.cp_upper(int(regressions.sum()),len(regressions))
    supported=upper<.001 and stats['coverage_one_sided_95_lower_pp']>-2
    falsified=stats['accuracy_one_sided_95_upper_pp']<-.1 or stats['coverage_one_sided_95_upper_pp']<-2
    assert not(supported and falsified);decision='supported' if supported else 'falsified' if falsified else 'inconclusive'
    primary=dict(decision=decision,any_regression_inputs=int(regressions.sum()),regression_probability_exact95_upper=upper,
                 conservative_accuracy95_lower_pp=-100*upper,**stats)
    sha(__file__);sha(SRC.parent/'analyze.py')
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3f',decision=decision,
        primary=primary,rows=rows,construction=construction,occupancy_and_certificate_diagnostics=diagnostics,
        independent_saved_boards=len(perseed)*8192*20,limits=p['limits'],input_sha256=HASHES,audit_seconds=time.perf_counter()-started,
        bank_inference='Uniform-with-replacement retrieval from compatible complete training labels; verified draw-for-draw from declared RNG.',
        added_energy_design_prior=p['added_prior'])
    save(OUT/'summary.json',summary);save(PUBLIC/'summary.json',summary)
    with (PUBLIC/'per_seed.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(perseed[0]));writer.writeheader();writer.writerows(perseed)
    names={'adamw-decay':'BP AdamW, decay0.01','adam-nodecay':'BP Adam, no decay','sgd-pre':'SGD + geometry preconditioner',
           'exclusion':'Training-count exclusion + fixed energy','training-bank':'Compatible training-bank retrieval'}
    lines=['# H3f: training-count and complete-bank controls','',
        f"**Frozen exclusion-energy noninferiority hypothesis: {decision}.** All 14,200 complete n=12 solution patterns already occur among the positive-weight fitting labels. The 8,192 evaluation partial inputs are held out, but their complete solution patterns are not. The compatible-bank control makes this support overlap explicit.",'',
        '| Method | Optimizer updates | Original N20 (%) | Certificate-first N20 (%) | Coverage (%) | Wrong stops | Capped chains | Mean search moves |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        fmt=lambda x:'—' if x is None else f'{x:,}'
        lines.append(f"| {names[row['arm']]} | {row['updates']:,} | {row['original_accuracy_pct']:.6f} | {row['certificate_first_accuracy_pct']:.6f} | {row['coverage_pct']:.6f} | {fmt(row['wrong_stops'])} | {fmt(row['capped_chains'])} | {'—' if row['mean_moves'] is None else f'{row["mean_moves"]:.2f}'} |")
    lines+=['',f"Exclusion versus Adam250: accuracy difference {stats['accuracy_difference_pp']:+.6f} percentage points, one-sided 95% endpoints [{stats['accuracy_one_sided_95_lower_pp']:.6f}, {stats['accuracy_one_sided_95_upper_pp']:.6f}]; coverage difference {stats['coverage_difference_pp']:+.6f} points, endpoints [{stats['coverage_one_sided_95_lower_pp']:.6f}, {stats['coverage_one_sided_95_upper_pp']:.6f}]. Exact upper probability for any paired input regression: {100*upper:.6f}% ({int(regressions.sum())} observed). Frozen margins: 0.1 point accuracy, 2 points coverage.",'',
        'The constructor counts offset co-occurrences using only complete fitting labels and their exact positive expanded-row multiplicities. It infers q=12 from their common cardinality. Zero observed count marks an offset forbidden; coefficients are fixed as J=q×forbidden−1 and unary0. Rewarding every allowed pair by−1 and penalizing every forbidden pair byq−1 is an added fixed energy-design prior, not learned-weight recovery. The offset architecture supplies geometry. No attack category or held-out label chooses coefficients.', '',
        f"The separate post hoc rule audit finds {construction['forbidden_offsets']} inferred forbidden offsets and {construction['allowed_offsets']} allowed offsets, with exact attack-support recovery. All training solutions have energy−66 and single-cell margin at least11. Temperature unit {construction['unit']:.9f}, threshold and margin are training-derived. The saved-output audit explicitly counts invalid boards below, equal to and above q, separately for stopped and capped chains; dense invalid occupancies are not silently ignored.",'',
        f"The bank stores all14,200 complete training boards in {construction['bank_packed_bytes']:,} packed bytes. For each partial input it filters compatible stored boards and samples20 times with replacement. It does not consult puzzle rules or a held-out solution list. Every sampling identity was independently replayed. Its CPU timing is retained separately from GPU search timing; no cross-hardware speedup is claimed. Bank output storage flags are not learned stopping certificates.",'',
        f"All {len(perseed)*8192*20:,} saved boards, learned comparisons, selection identities, support counts and bank draws passed the independent audit. Original learned-model searches were reused only with pinned hashes and unchanged full settings. This is adaptive reuse of the H3e evaluation split, not new confirmation or solution-pattern generalization.",'',
        '[Frozen protocol](protocol.json), [constructor and workers](driver.py), [auditor](analyze.py), [tracked summary](../../../../results/research/loki/h3f/summary.json), [per-seed counts](../../../../results/research/loki/h3f/per_seed.csv).']
    (SRC/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (PUBLIC/'README.md').write_text('# H3f: training-count and complete-bank controls\n\n'+
        f"Frozen no-optimizer exclusion-energy noninferiority result: **{decision}**. All14,200 n=12 complete solution patterns occur in training. Evaluation holds out partial inputs, not complete patterns; the compatible-training-bank baseline makes that limitation explicit.\n\n"+
        'The inferred exclusion support is combined with a declared fixed energy-design prior and supplied offset geometry. It is not learned-weight recovery.\n\n'+
        '[Full report and protocol](../../../../research/20261114/loki/h3f/REPORT.md), [summary and hashes](summary.json), [per-seed counts](per_seed.csv).\n')
    print(json.dumps(dict(status='validated',decision=decision,primary=primary,saved_boards=summary['independent_saved_boards'],
        audit_seconds=summary['audit_seconds'])))


if __name__=='__main__':main()
