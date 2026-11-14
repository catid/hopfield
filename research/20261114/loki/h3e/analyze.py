"""Independent CPU audit and compact public results for n12 fixed-recipe transfer."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
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

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'research/20261114/loki/h3e'
OUT = ROOT / 'artifacts/research/20261114/loki/h3e'
PUBLIC = ROOT / 'results/research/loki/h3e'
sys.path.insert(0,str(ROOT/'upstream'))
spec=importlib.util.spec_from_file_location('stats',SRC.parent/'analyze.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
HASHES={}


def sha(path):
    path=Path(path)
    with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=digest;return digest
def read(path):sha(path);return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_data(data,p):
    solutions=data['solutions'];cols=data['solution_columns'];n=12
    assert solutions.shape==(14200,144) and cols.shape==(14200,12)
    assert np.all(np.sort(cols,axis=1)==np.arange(n))
    assert all(len(set(np.arange(n)-row))==n and len(set(np.arange(n)+row))==n for row in cols)
    rebuilt=np.zeros_like(solutions);rebuilt[np.arange(len(cols))[:,None],np.arange(n)[None,:]*n+cols]=True
    assert np.array_equal(rebuilt,solutions) and len(set(map(bytes,solutions)))==14200
    assert not set(data['fit_keys'].tolist()) & set(data['test_keys'].tolist())
    assert len(np.unique(data['fit_keys']))==100000 and len(np.unique(data['test_keys']))==8192
    assert not set(map(bytes,data['fit_inputs'])) & set(map(bytes,data['test_inputs']))
    for split in ['fit','test']:
        encoded=np.array([sum((int(cell)+1)*145**(2-j) for j,cell in enumerate(np.flatnonzero(row))) for row in data[f'{split}_inputs']])
        assert np.array_equal(encoded,data[f'{split}_keys'])
        assert dict(zip(['1','2','3'],[(data[f'{split}_inputs'].sum(1)==k).sum() for k in [1,2,3]]))==p['dataset'][f'{split}_unique_by_clues']
    rows=data['expanded_fit_solution_indices'];keys=data['expanded_fit_keys']
    assert len(rows)==2049002 and np.isin(keys,data['fit_keys']).all()
    multiplicities=np.bincount(rows,minlength=14200)
    counts=np.repeat(multiplicities[:,None],144,axis=1)
    for power in [2,1,0]:
        cells=keys//145**power%145;keep=cells>0
        # Every expanded row's givens must occur in its complete label, not just a sample.
        assert solutions[rows[keep],cells[keep]-1].all()
        np.add.at(counts,(rows[keep],cells[keep]-1),-1)
    assert np.array_equal(counts,data['free_counts'])
    assert np.array_equal(multiplicities,data['rows_per_solution'])
    assert int(counts.sum())==p['dataset']['free_cell_weight_total']
    exact=np.array([solutions[:,np.flatnonzero(given)].all(1).sum() for given in data['test_inputs']])
    assert np.array_equal(exact,data['test_solution_counts'])
    return {row.tobytes() for row in solutions}


def validate_model(arm,updates,data):
    folder=OUT/f'fits/{arm}/u{updates}';fit=read(folder/'fit.json')
    assert fit['model_sha256']==sha(folder/'model.pt')
    assert fit['protocol_sha256']==sha(SRC/'protocol.json') and fit['source_sha256']==sha(SRC/'driver.py')
    assert fit['updates']==updates and fit['rows']==2049002 and fit['distinct_solutions']==14200
    model=torch.load(folder/'model.pt',weights_only=False)
    S=model['S'].numpy().astype(np.float64);J=S[:,:,1,1];h=model['B'][:,1].numpy().astype(np.float64)
    assert np.array_equal(J,J.T) and not np.count_nonzero(np.diag(J))
    assert not np.count_nonzero(S[:,:,0,:]) and not np.count_nonzero(S[:,:,:,0])
    assert not np.count_nonzero(model['B'][:,0].numpy())
    # Map offsets directly from cell coordinates, independent of the fitting helper.
    all_offsets=[(dr,dc) for dr in range(12) for dc in range(-11,12) if dr>0 or dc>0]
    lookup={offset:i for i,offset in enumerate(all_offsets)}
    values=model['A'].numpy()
    for i in range(144):
        for j in range(i+1,144):assert J[i,j]==values[lookup[(j//12-i//12,j%12-i%12)]]
    assert np.all(h==float(model['b'][0]))
    x=data['solutions'].astype(np.float64);fields=x@J+h
    computed=dict(tau=float((.5*(x*fields).sum(1)+.5*(x*h).sum(1)).max()),
        delta=float(((1-2*x)*fields).min()),
        unit=float((np.abs(fields).mean(1)*data['rows_per_solution']).sum()/data['rows_per_solution'].sum()))
    errors={key:abs(computed[key]-float(model[key])) for key in computed}
    assert max(errors.values())<2e-4,errors
    return model,dict(arm=arm,updates=updates,computed_float64=computed,absolute_errors=errors,
                     fit_loss=fit['final_loss'],model_sha256=fit['model_sha256'])


def validate_record(arm,updates,seed,data,keys,model):
    folder=OUT/f'search/{arm}/u{updates}/s{seed}';rec=read(folder/'result.json')
    assert rec['status']=='complete' and rec['protocol_sha256']==sha(SRC/'protocol.json')
    assert rec['source_sha256']==sha(SRC/'driver.py') and rec['records_sha256']==sha(folder/'records.npz')
    assert rec['model_sha256']==sha(ROOT/rec['model_path'])
    z=dict(np.load(folder/'records.npz'));inputs=data['test_inputs'];boards=np.unpackbits(z['boards'],axis=-1)[...,:144].astype(bool)
    assert np.array_equal(z['inputs'],inputs)
    valid=np.array([[b.tobytes() in keys and b[given].all() for b in samples] for given,samples in zip(inputs,boards)])
    assert np.array_equal(valid,z['valid'])
    denom=np.minimum(data['test_solution_counts'],20)
    found=np.array([min(len({b.tobytes() for b in samples[ok]}),d) for samples,ok,d in zip(boards,valid,denom)])
    assert np.array_equal(found,z['found']) and np.array_equal(denom,z['denom'])
    assert np.array_equal(z['solution_counts'],data['test_solution_counts'])
    original=z['energy'].argmin(1);selected=[]
    for values,stopped in zip(z['energy'],z['stopped']):
        subset=np.flatnonzero(stopped)
        if not len(subset):subset=np.arange(20)
        selected.append(int(subset[np.argmin(values[subset])]))
    selected=np.array(selected);ids=np.arange(len(inputs))
    assert np.array_equal(original,z['original_index']) and np.array_equal(selected,z['certificate_index'])
    assert np.array_equal(valid[ids,original],z['n20']) and np.array_equal(valid[ids,selected],z['n20_certificate'])
    assert np.array_equal(valid[:,0],z['n1'])
    assert ((z['moves']>=0)&(z['moves']<=20000)).all() and (z['moves'][~z['stopped']]==20000).all()
    assert np.array_equal(z['flop'],41761+734*z['moves'])
    assert int((z['stopped']&~valid).sum())==rec['wrong_stops'] and int((~z['stopped']).sum())==rec['capped_chains']
    assert rec['original_accuracy_pct']==100*float(z['n20'].mean())
    assert rec['certificate_accuracy_pct']==100*float(z['n20_certificate'].mean())
    assert rec['coverage_pct']==100*float(found.sum()/denom.sum())
    J=model['S'][:,:,1,1].numpy().astype(np.float64);h=model['B'][:,1].numpy().astype(np.float64)
    flat=boards.reshape(-1,144);givens=np.repeat(inputs,20,0);flags=z['stopped'].ravel();gap=[]
    for low in range(0,len(flat),4096):
        x=flat[low:low+4096].astype(np.float64);m=(1-2*x)*(x@J+h);m[givens[low:low+4096]]=np.inf
        gap.append(m.min(1)-(float(model['delta'])-.001))
    gap=np.concatenate(gap);stopped_gaps=gap[flags]
    diag=dict(stopped_count=int(flags.sum()),strict_negative_count=int((stopped_gaps<0).sum()),
        deficit_beyond_1e3_count=int((stopped_gaps<-.001).sum()),minimum_stopped_gap=float(stopped_gaps.min()) if len(stopped_gaps) else None)
    perseed=dict(arm=arm,updates=updates,seed=seed,unique_inputs=len(inputs),chains=20,
        original_correct=int(z['n20'].sum()),certificate_first_correct=int(z['n20_certificate'].sum()),n1_correct=int(z['n1'].sum()),
        coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),wrong_stops=rec['wrong_stops'],
        capped_chains=rec['capped_chains'],mean_moves=rec['mean_moves'],search_seconds=rec['search_seconds'],
        model_sha256=rec['model_sha256'],records_sha256=rec['records_sha256'],result_sha256=sha(folder/'result.json'))
    strata=[]
    for clue in [1,2,3]:
        mask=inputs.sum(1)==clue
        strata.append(dict(arm=arm,updates=updates,seed=seed,clues=clue,unique_inputs=int(mask.sum()),
            original_correct=int(z['n20'][mask].sum()),certificate_first_correct=int(z['n20_certificate'][mask].sum()),
            coverage_numerator=int(found[mask].sum()),coverage_denominator=int(denom[mask].sum())))
    return z,perseed,strata,diag


def main():
    started=time.perf_counter();torch.set_num_threads(1);PUBLIC.mkdir(parents=True,exist_ok=True)
    p=read(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    cpu=read(OUT/'cpu-validation.json');gpu=read(OUT/'gpu-validation.json')
    for validation in [cpu,gpu]:
        assert validation['status']=='passed' and validation['protocol_sha256']==sha(SRC/'protocol.json')
        assert validation['source_sha256']==sha(SRC/'driver.py')
    assert cpu['gradient_max_absolute_error']<1e-10 and gpu['gradient_max_absolute_error']<=1e-5
    data=dict(np.load(OUT/'data.npz'));keys=validate_data(data,p)
    rows=[];perseed=[];strata=[];calibration=[];diagnostics={};evidence={}
    for arm,cfg in p['arms'].items():
        for updates in cfg['snapshots']:
            model,cal=validate_model(arm,updates,data);calibration.append(cal);arrays=[];recs=[]
            for seed in [0,1,2]:
                z,rec,sub,diag=validate_record(arm,updates,seed,data,keys,model)
                arrays.append(z);perseed.append(rec);recs.append(rec);strata.extend(sub);diagnostics[f'{arm}/u{updates}/s{seed}']=diag
            stack={key:np.stack([z[key] for z in arrays]) for key in ['n20','n20_certificate','n1','found']}
            evidence[arm,updates]=stack
            rows.append(dict(arm=arm,updates=updates,unique_inputs=8192,seeds=[0,1,2],
                original_accuracy_pct=100*float(stack['n20'].mean()),certificate_first_accuracy_pct=100*float(stack['n20_certificate'].mean()),
                n1_accuracy_pct=100*float(stack['n1'].mean()),coverage_pct=100*float(stack['found'].sum(1).mean()/arrays[0]['denom'].sum()),
                wrong_stops=sum(r['wrong_stops'] for r in recs),capped_chains=sum(r['capped_chains'] for r in recs),
                mean_moves=float(np.mean([r['mean_moves'] for r in recs])),search_seconds_total=sum(r['search_seconds'] for r in recs)))
    a=evidence['adam-nodecay',250];b=evidence['sgd-pre',4000];denom=np.minimum(data['test_solution_counts'],20)
    stats=A.bootstrap(dict(n20=a['n20_certificate'],found=a['found']),dict(n20=b['n20_certificate'],found=b['found']),denom)
    regressions=(b['n20_certificate']&~a['n20_certificate']).any(0);upper=A.cp_upper(int(regressions.sum()),len(regressions))
    support=upper<.001 and stats['coverage_one_sided_95_lower_pp']>-2
    falsify=stats['accuracy_one_sided_95_upper_pp']<-.1 or stats['coverage_one_sided_95_upper_pp']<-2
    assert not(support and falsify)
    decision='supported' if support else 'falsified' if falsify else 'inconclusive'
    primary=dict(decision=decision,any_regression_inputs=int(regressions.sum()),regression_probability_exact95_upper=upper,
        conservative_accuracy95_lower_pp=-100*upper,**stats)
    sha(__file__);sha(SRC.parent/'analyze.py')
    solution_counts=data['test_solution_counts'].astype(np.float64)
    expected_unique=np.ones_like(solution_counts);many=solution_counts>1
    expected_unique[many]=-solution_counts[many]*np.expm1(20*np.log1p(-1/solution_counts[many]))
    oracle_expected=100*float(expected_unique.sum()/np.minimum(solution_counts,20).sum())
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3e',decision=decision,
        primary=primary,rows=rows,dataset=p['dataset'],calibration_audits=calibration,certificate_diagnostics=diagnostics,
        optimization_validation=gpu,cpu_validation=cpu,independent_saved_boards=len(perseed)*8192*20,
        posthoc_uniform_solution_reference=dict(expected_pooled_coverage_pct=oracle_expected,
            formula='sum_i s_i*(1-(1-1/s_i)^20) / sum_i min(s_i,20)',
            scope='Exact expectation for independent uniform draws from each complete valid solution set; offline oracle reference, not a learned competitor or an upper bound. No influence on settings or decisions.'),
        limits=p['limits'],input_sha256=HASHES,audit_seconds=time.perf_counter()-started,new_training_or_inference=False)
    save(OUT/'summary.json',summary);save(PUBLIC/'summary.json',summary)
    for filename,values in [('per_seed.csv',perseed),('per_clue_seed.csv',strata)]:
        with (PUBLIC/filename).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    names={'adamw-decay':'BP AdamW, decay 0.01','adam-nodecay':'BP Adam, no decay','sgd-pre':'SGD + geometry preconditioner'}
    lines=['# H3e: fixed-recipe transfer to 12-Queens','',
        f"**Frozen primary hypothesis: {decision}.** No-decay Adam at the fixed 250-update prefix was compared with preconditioned SGD at 4,000 updates using certificate-first selection. The n=10 learning rates, 4,000-update cosine horizon, normalized temperature and 20,000-move budget were transferred without n=12 tuning. All arms were fitted from scratch on the same 100,000 unique partial inputs and all their compatible completion labels.",'',
        '| Method | Updates | Original N20 (%) | Certificate-first N20 (%) | Coverage (%) | Wrong stops | Capped chains | Mean moves |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:lines.append(f"| {names[row['arm']]} | {row['updates']:,} | {row['original_accuracy_pct']:.6f} | {row['certificate_first_accuracy_pct']:.6f} | {row['coverage_pct']:.6f} | {row['wrong_stops']:,} | {row['capped_chains']:,} | {row['mean_moves']:.2f} |")
    lines+=['',f"Primary accuracy difference: {stats['accuracy_difference_pp']:+.6f} percentage points, one-sided 95% bootstrap endpoints [{stats['accuracy_one_sided_95_lower_pp']:.6f}, {stats['accuracy_one_sided_95_upper_pp']:.6f}]. Coverage difference: {stats['coverage_difference_pp']:+.6f} points, endpoints [{stats['coverage_one_sided_95_lower_pp']:.6f}, {stats['coverage_one_sided_95_upper_pp']:.6f}]. The exact upper probability of any paired accuracy regression is {100*upper:.6f}% ({int(regressions.sum())} affected inputs). Margins are 0.1 point accuracy and 2 points coverage. Inputs, with all three search repeats, are the resampling units.",'',
        'The 8,192 held-out inputs are disjoint from every fitting input. They contain 6 one-clue, 305 two-clue and 7,881 three-clue inputs, proportionally sampled from the enumerated universe. The full universe has 206,628 completable inputs and 14,200 distinct complete solutions. Two independent enumeration algorithms and direct row/column/diagonal checks agreed on all solutions. Complete labels can repeat across distinct partial inputs, matching the GRAM task; this is not a split by complete solution.', '',
        f"As a post hoc scale reference, independent uniform sampling from each input's valid complete solutions would have expected pooled coverage {oracle_expected:.6f}%, because duplicate samples count once. This is an offline oracle reference, not an upper bound or a learned competitor. It did not affect the protocol or decision.", '',
        'Each selected input contributes every compatible completion, totaling 2,049,002 fitting rows. Exact per-solution free-cell counts reduce the mathematical PL objective to 14,200 solution rows. An independent float64 full-row fixture matched loss and gradients; the full-data GPU check against uncompressed upstream PL passed before training. Float32 reduction order changes, so bitwise equality of complete training trajectories is not claimed.', '',
        f"The bounded optimization check measured median forward/backward times {np.median(gpu['timings_seconds']['expanded_upstream'])*1000:.6f} ms uncompressed and {np.median(gpu['timings_seconds']['aggregated_binary'])*1000:.6f} ms aggregated ({gpu['median_forward_backward_speedup']:.2f}× for this operation only). Loss error was {gpu['loss_absolute_error']:.3g}; maximum gradient error was {gpu['gradient_max_absolute_error']:.3g}. Three measured calls followed one warmup. This is not an end-to-end fit or search speedup; aggregate-path memory figures include still-resident full-data buffers.",'',
        f"The independent audit checked all {len(perseed)*8192*20:,} saved boards against the exact completion set and givens, recomputed original and certificate-first selection identities, reconstructed every training free-cell count, and checked all 8,192 completion counts. Calibration was independently recomputed in float64. Full per-clue counts and certificate-margin diagnostics are retained.",'',
        'This is a new generated n=12 domain and fixed-recipe transfer study, not an official GRAM n=12 benchmark reproduction. There is one deterministic trajectory per optimizer and three inference seeds. No held-out setting selection occurred. Certificate-first is a derivative output policy; original-selection outcomes remain separately visible.', '',
        '[Frozen protocol](protocol.json), [data generator](prepare.py), [runner](driver.py), [auditor](analyze.py), [tracked results](../../../../results/research/loki/h3e/summary.json), [per-seed counts](../../../../results/research/loki/h3e/per_seed.csv), [per-clue counts](../../../../results/research/loki/h3e/per_clue_seed.csv).']
    (SRC/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (PUBLIC/'README.md').write_text('# H3e: fixed-recipe transfer to 12-Queens\n\n'+
        f"The fixed primary hypothesis was **{decision}**. Adam at 250 updates versus SGD at 4,000, with certificate-first output selection: accuracy difference {stats['accuracy_difference_pp']:+.6f} percentage points; coverage difference {stats['coverage_difference_pp']:+.6f} points.\n\n"+
        'All arms use the same 100,000 unique training inputs and all compatible labels. Evaluation uses 8,192 disjoint new n=12 inputs, 20 chains and three repeated search seeds, with no n=12 tuning. This is a generated-domain study rather than an official benchmark reproduction.\n\n'+
        '[Full report and protocol](../../../../research/20261114/loki/h3e/REPORT.md), [summary and hashes](summary.json), [per-seed counts](per_seed.csv), [per-clue counts](per_clue_seed.csv).\n')
    print(json.dumps(dict(status='validated',decision=decision,primary=primary,saved_boards=summary['independent_saved_boards'],
        hashed_files=len(HASHES),audit_seconds=summary['audit_seconds'])))


if __name__=='__main__':main()
