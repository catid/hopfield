"""Independent pattern/input separation, model and saved-output audit for H3g."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3g';OUT=ROOT/'artifacts/research/20261114/loki/h3g';PUBLIC=ROOT/'results/research/loki/h3g'
spec=importlib.util.spec_from_file_location('h3stats',SRC.parent/'analyze.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
HASHES={}


def sha(path):
    path=Path(path)
    with path.open('rb') as handle:value=hashlib.file_digest(handle,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=value;return value
def read(path):sha(path);return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def audit_data(data,p):
    solutions=data['all_solutions'];assert solutions.shape==(14200,144)
    coords=np.indices((12,12));boards=solutions.reshape(-1,12,12)
    assert (boards.sum(-1)==1).all() and (boards.sum(-2)==1).all()
    for d in range(-11,12):assert (boards[:,coords[0]-coords[1]==d].sum(1)<=1).all()
    for d in range(23):assert (boards[:,coords[0]+coords[1]==d].sum(1)<=1).all()
    keys={row.tobytes() for row in solutions};assert len(keys)==14200
    largest=data['largest_bank_indices'];assert len(np.unique(largest))==512
    assert np.array_equal(largest,np.random.default_rng(2026111413).permutation(14200)[:512])
    inputs=data['test_inputs'];input_keys=data['test_keys'];assert len(np.unique(input_keys))==len(inputs)==8192
    assert not np.isin(input_keys,data['source_h3e_test_keys']).any()
    encoded=np.array([sum((int(cell)+1)*145**(2-j) for j,cell in enumerate(np.flatnonzero(row))) for row in inputs])
    assert np.array_equal(encoded,input_keys)
    assert {str(k):int((inputs.sum(1)==k).sum()) for k in [1,2,3]}=={'1':0,'2':16,'3':8176}
    compatible_counts=[];bank_compatible=[]
    for row in inputs:
        compatible=solutions[:,np.flatnonzero(row)].all(1)
        compatible_counts.append(int(compatible.sum()));bank_compatible.append(int(compatible[largest].sum()))
    assert np.array_equal(compatible_counts,data['test_solution_counts']) and min(compatible_counts)>0 and max(bank_compatible)==0
    bank_audits={}
    for bank in [32,128,512]:
        indices=data[f'bank{bank}_indices'];assert np.array_equal(indices,largest[:bank])
        fit_solutions=solutions[indices];expanded=data[f'bank{bank}_expanded_keys'];label_indices=data[f'bank{bank}_expanded_label_indices']
        assert len(expanded)==bank*298 and not np.isin(input_keys,expanded).any()
        counts=np.repeat(np.bincount(label_indices,minlength=bank)[:,None],144,axis=1)
        for power in [2,1,0]:
            cells=expanded//145**power%145;present=cells>0
            assert fit_solutions[label_indices[present],cells[present]-1].all()
            np.add.at(counts,(label_indices[present],cells[present]-1),-1)
        assert np.array_equal(counts,data[f'bank{bank}_free_counts']) and np.array_equal(counts,np.where(fit_solutions,231,298))
        bank_audits[str(bank)]=dict(patterns=bank,expanded_rows=len(expanded),unique_partial_inputs=len(np.unique(expanded)),
            compatible_test_inputs=0,compatible_test_solution_patterns=0,retrieval_accuracy_pct=0.,retrieval_coverage_pct=0.)
    return keys,{row.tobytes() for row in solutions[largest]},bank_audits


def audit_model(cfg,data):
    bank=cfg['bank'];arm=cfg['arm'];folder=OUT/f'models/b{bank}/{arm}';rec=read(folder/'model.json')
    assert rec['status']=='complete' and rec['model_sha256']==sha(folder/'model.pt')
    assert rec['protocol_sha256']==sha(SRC/'protocol.json') and rec['source_sha256']==sha(SRC/'driver.py')
    assert rec['updates']==cfg['updates'] and rec['expanded_rows']==bank*298
    model=torch.load(folder/'model.pt',weights_only=False);S=model['S'].numpy().astype(np.float64);J=S[:,:,1,1];h=model['B'][:,1].numpy().astype(np.float64)
    assert np.array_equal(J,J.T) and not np.count_nonzero(np.diag(J)) and not np.count_nonzero(S[:,:,0,:]) and not np.count_nonzero(S[:,:,:,0])
    offsets=[(dr,dc) for dr in range(12) for dc in range(-11,12) if dr>0 or dc>0];lookup={d:i for i,d in enumerate(offsets)}
    for i in range(144):
        for j in range(i+1,144):assert J[i,j]==model['A'][lookup[j//12-i//12,j%12-i%12]].item()
    assert np.all(h==model['b'][0].item()) and not np.count_nonzero(model['B'][:,0].numpy())
    labels=data['all_solutions'][data[f'bank{bank}_indices']];x=labels.astype(np.float64);fields=x@J+h
    calibration=dict(tau=float((.5*(x*fields).sum(1)+.5*(x*h).sum(1)).max()),delta=float(((1-2*x)*fields).min()),unit=float(np.abs(fields).mean()))
    errors={key:abs(calibration[key]-float(model[key])) for key in calibration};assert max(errors.values())<2e-4,errors
    diagnostic=dict(bank=bank,arm=arm,updates=cfg['updates'],calibration_float64=calibration,absolute_errors=errors,
                    fit_loss=rec['final_loss'],fit_or_construct_seconds=rec['fit_or_construct_seconds'])
    if arm=='exclusion':
        assert rec['support_sha256']==sha(folder/'support.npz');z=dict(np.load(folder/'support.npz'));counts=[]
        for dr,dc in offsets:
            i=[];j=[]
            for r in range(12):
                for c in range(12):
                    rr,cc=r+dr,c+dc
                    if 0<=rr<12 and 0<=cc<12:i.append(r*12+c);j.append(rr*12+cc)
            counts.append(int((labels[:,i]&labels[:,j]).sum()))
        counts=np.array(counts);assert np.array_equal(counts,z['counts']) and np.array_equal(counts==0,z['forbidden'])
        assert np.array_equal(model['A'].numpy(),12*(counts==0).astype(np.float32)-1) and not np.count_nonzero(h)
        attack=np.array([dr==0 or dc==0 or abs(dr)==abs(dc) for dr,dc in offsets]);mask=counts==0
        diag=dict(forbidden=int(mask.sum()),false_forbidden=int((mask&~attack).sum()),missed_forbidden=int((~mask&attack).sum()))
        assert all(diag[key]==rec['posthoc_support'][key] for key in diag)
        diagnostic['posthoc_support']=diag;assert calibration['delta']>=11
    return model,diagnostic


def audit_search(cfg,seed,data,keys,largest_keys,model):
    bank=cfg['bank'];arm=cfg['arm'];folder=OUT/f'search/b{bank}/{arm}/s{seed}';rec=read(folder/'result.json')
    assert rec['status']=='complete' and rec['protocol_sha256']==sha(SRC/'protocol.json') and rec['source_sha256']==sha(SRC/'driver.py')
    assert rec['records_sha256']==sha(folder/'records.npz') and rec['model_sha256']==sha(ROOT/rec['model_path'])
    z=dict(np.load(folder/'records.npz'));boards=np.unpackbits(z['boards'],axis=-1)[...,:144].astype(bool);inputs=data['test_inputs']
    assert np.array_equal(inputs,z['inputs']) and (boards>=inputs[:,None,:]).all()
    pattern_match=np.array([[row.tobytes() in largest_keys for row in samples] for samples in boards]);assert not pattern_match.any()
    valid=np.array([[row.tobytes() in keys and row[given].all() for row in samples] for given,samples in zip(inputs,boards)])
    assert np.array_equal(valid,z['valid']);denom=np.minimum(data['test_solution_counts'],20)
    found=np.array([min(len({row.tobytes() for row in samples[ok]}),d) for samples,ok,d in zip(boards,valid,denom)])
    assert np.array_equal(found,z['found']) and np.array_equal(denom,z['denom']) and np.array_equal(z['solution_counts'],data['test_solution_counts'])
    original=z['energy'].argmin(1);selected=[]
    for energy,stopped in zip(z['energy'],z['stopped']):
        pool=np.flatnonzero(stopped)
        if not len(pool):pool=np.arange(20)
        selected.append(int(pool[energy[pool].argmin()]))
    selected=np.array(selected);ids=np.arange(len(inputs))
    assert np.array_equal(original,z['original_index']) and np.array_equal(selected,z['certificate_index'])
    assert np.array_equal(valid[ids,original],z['n20']) and np.array_equal(valid[ids,selected],z['n20_certificate']) and np.array_equal(valid[:,0],z['n1'])
    assert ((z['moves']>=0)&(z['moves']<=20000)).all() and (z['moves'][~z['stopped']]==20000).all()
    assert np.array_equal(z['flop'],41761+734*z['moves'])
    assert rec['wrong_stops']==int((z['stopped']&~valid).sum()) and rec['capped_chains']==int((~z['stopped']).sum())
    assert rec['original_accuracy_pct']==100*float(z['n20'].mean()) and rec['certificate_accuracy_pct']==100*float(z['n20_certificate'].mean())
    assert rec['coverage_pct']==100*float(found.sum()/denom.sum())
    counts=boards.sum(-1);assert np.array_equal(counts,z['queen_counts']);occupancies={}
    for name,mask in [('stopped_invalid',z['stopped']&~valid),('capped_invalid',~z['stopped']&~valid)]:
        occupancies[name]={label:int((mask&condition).sum()) for label,condition in [('below_q',counts<12),('equal_q',counts==12),('above_q',counts>12)]}
    J=model['S'][:,:,1,1].numpy().astype(np.float64);h=model['B'][:,1].numpy().astype(np.float64);flat=boards.reshape(-1,144);givens=np.repeat(inputs,20,0);gaps=[]
    for lo in range(0,len(flat),4096):
        x=flat[lo:lo+4096].astype(np.float64);m=(1-2*x)*(x@J+h);m[givens[lo:lo+4096]]=np.inf
        gaps.append(m.min(1)-(model['delta']-.001))
    gaps=np.concatenate(gaps)[z['stopped'].ravel()]
    diagnostic=dict(bank=bank,arm=arm,seed=seed,returned_training_pattern_matches=0,invalid_occupancies=occupancies,
        strict_stopped_certificate_deficits=int((gaps<0).sum()),stopped_deficits_beyond_1e3=int((gaps<-.001).sum()),minimum_stopped_gap=float(gaps.min()) if len(gaps) else None)
    row=dict(bank=bank,arm=arm,updates=cfg['updates'],seed=seed,unique_inputs=8192,chains=20,
        original_correct=int(z['n20'].sum()),certificate_first_correct=int(z['n20_certificate'].sum()),n1_correct=int(z['n1'].sum()),
        coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),wrong_stops=rec['wrong_stops'],capped_chains=rec['capped_chains'],
        mean_moves=rec['mean_moves'],search_seconds=rec['search_seconds'],model_sha256=rec['model_sha256'],records_sha256=rec['records_sha256'],result_sha256=sha(folder/'result.json'))
    strata=[]
    for clue in [2,3]:
        mask=inputs.sum(1)==clue;strata.append(dict(bank=bank,arm=arm,seed=seed,clues=clue,unique_inputs=int(mask.sum()),
            original_correct=int(z['n20'][mask].sum()),certificate_first_correct=int(z['n20_certificate'][mask].sum()),
            coverage_numerator=int(found[mask].sum()),coverage_denominator=int(denom[mask].sum())))
    return z,row,strata,diagnostic


def main(models_only=False):
    started=time.perf_counter();torch.set_num_threads(1);p=read(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256','inherited_optimization_validation']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    data=dict(np.load(OUT/'data.npz'));keys,largest_keys,bank_audits=audit_data(data,p)
    models={};model_audits=[]
    for cfg in p['models']:
        model,diagnostic=audit_model(cfg,data);models[cfg['bank'],cfg['arm']]=model;model_audits.append(diagnostic)
    if models_only:
        save(OUT/'preinference-independent-audit.json',dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='Independent immutable data/model checks; no inference results read',bank_audits=bank_audits,model_audits=model_audits,input_sha256=HASHES))
        print(json.dumps(dict(status='passed',scope='data_and_models_only',models=len(models),seconds=time.perf_counter()-started)));return
    PUBLIC.mkdir(parents=True,exist_ok=True);rows=[];perseed=[];strata=[];diagnostics=[];evidence={}
    for cfg in p['models']:
        bank=cfg['bank'];arm=cfg['arm'];zs=[];current=[]
        for seed in [0,1,2]:
            z,row,sub,diagnostic=audit_search(cfg,seed,data,keys,largest_keys,models[bank,arm])
            zs.append(z);current.append(row);perseed.append(row);strata.extend(sub);diagnostics.append(diagnostic)
        arr={key:np.stack([z[key] for z in zs]) for key in ['n20','n20_certificate','n1','found']};evidence[bank,arm]=arr
        rows.append(dict(bank=bank,arm=arm,updates=cfg['updates'],unique_inputs=8192,seeds=[0,1,2],
            original_accuracy_pct=100*float(arr['n20'].mean()),certificate_first_accuracy_pct=100*float(arr['n20_certificate'].mean()),
            n1_accuracy_pct=100*float(arr['n1'].mean()),coverage_pct=100*float(arr['found'].sum(1).mean()/zs[0]['denom'].sum()),
            wrong_stops=sum(r['wrong_stops'] for r in current),capped_chains=sum(r['capped_chains'] for r in current),
            mean_moves=float(np.mean([r['mean_moves'] for r in current])),search_seconds_total=sum(r['search_seconds'] for r in current)))
    comparisons=[];denom=np.minimum(data['test_solution_counts'],20)
    for bank in [512,32,128]:
        a=evidence[bank,'exclusion'];b=evidence[bank,'adam-nodecay']
        stats=A.bootstrap(dict(n20=a['n20_certificate'],found=a['found']),dict(n20=b['n20_certificate'],found=b['found']),denom)
        regression=(b['n20_certificate']&~a['n20_certificate']).any(0);upper=A.cp_upper(int(regression.sum()),len(regression))
        support=upper<.001 and stats['coverage_one_sided_95_lower_pp']>-2
        falsify=stats['accuracy_one_sided_95_upper_pp']<-.1 or stats['coverage_one_sided_95_upper_pp']<-2
        assert not(support and falsify)
        comparisons.append(dict(bank=bank,primary=bank==512,decision='supported' if support else 'falsified' if falsify else 'inconclusive',
            any_regression_inputs=int(regression.sum()),regression_probability_exact95_upper=upper,conservative_accuracy95_lower_pp=-100*upper,**stats))
    primary=comparisons[0];sha(__file__);sha(SRC.parent/'analyze.py')
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3g',decision=primary['decision'],
        primary=primary,rows=rows,comparisons=comparisons,dataset=p['dataset'],training_bank_controls=bank_audits,
        model_audits=model_audits,output_diagnostics=diagnostics,independent_saved_boards=len(perseed)*8192*20,
        every_valid_returned_pattern_unseen=True,input_sha256=HASHES,limits=p['limits'],audit_seconds=time.perf_counter()-started)
    save(OUT/'summary.json',summary);save(PUBLIC/'summary.json',summary)
    for filename,values in [('per_seed.csv',perseed),('per_clue_seed.csv',strata)]:
        with (PUBLIC/filename).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    names={'adamw-decay':'BP AdamW, decay0.01','adam-nodecay':'BP Adam, no decay','sgd-pre':'SGD + geometry preconditioner','exclusion':'Training-count exclusion + fixed energy'}
    lines=['# H3g: held-out complete patterns and bank-incompatible inputs','',
        f"**Frozen 512-pattern primary hypothesis: {primary['decision']}.** Every valid completion of every evaluation input lies outside all 512 patterns in the largest training bank. No evaluation input overlaps any restricted-bank training input or the previous H3e evaluation inputs. A compatible training-bank baseline therefore has 0% accuracy and 0% coverage by construction, independently verified by matching every input.",'',
        '| Training patterns | Method | Updates | Original N20 (%) | Certificate-first N20 (%) | Coverage (%) | Wrong stops | Capped chains |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['bank']} | {names[r['arm']]} | {r['updates']:,} | {r['original_accuracy_pct']:.6f} | {r['certificate_first_accuracy_pct']:.6f} | {r['coverage_pct']:.6f} | {r['wrong_stops']:,} | {r['capped_chains']:,} |")
    lines+=['',f"Primary count-energy versus Adam250 difference: accuracy {primary['accuracy_difference_pp']:+.6f} percentage points (one-sided 95% endpoints [{primary['accuracy_one_sided_95_lower_pp']:.6f}, {primary['accuracy_one_sided_95_upper_pp']:.6f}]); coverage {primary['coverage_difference_pp']:+.6f} points ([{primary['coverage_one_sided_95_lower_pp']:.6f}, {primary['coverage_one_sided_95_upper_pp']:.6f}]). Exact upper probability for any paired input regression: {100*primary['regression_probability_exact95_upper']:.6f}%. Frozen margins were 0.1 point accuracy and 2 points coverage. Search seeds remain clustered inside each resampled input.",'',
        'The conditioned test distribution contains 8,192 unique inputs: 0 with one clue, 16 with two clues and 8,176 with three clues. It is sampled from 111,781 inputs incompatible with the largest training bank and absent from prior H3e evaluation. It is deliberately different from the original GRAM distribution. The nested 32/128/512 banks are prefixes of one fixed random permutation of complete patterns, not independent training replications.', '',
        'Each selected training pattern contributes all 298 one/two/three-clue masks. The banks give 9,536/38,144/152,576 expanded labeled rows and 8,923/31,109/90,239 unique training inputs. Outside patterns never enter fitting, support counts or calibration. All learned models retain the n=10 learning rates and 4,000-step cosine horizon; Adam stops at the fixed 250-update prefix. Count energies retain J=q×forbidden−1, unary0, with q inferred from training cardinality. Allowed-pair reward and offset geometry remain explicit fixed priors.', '',
        f"Independent checks covered all {len(perseed)*8192*20:,} saved boards, preserved givens, selection identities, exact completion counts and model calibration. Every returned valid board was checked to be absent from the entire 512-pattern bank. Stopped/capped invalid boards are separated into below/equal/above 12 queens, with float64 certificate-margin diagnostics. The lower 32/128-bank results and their comparisons are secondary; no setting was selected from them.",'',
        'This study demonstrates only the measured generalization relative to these new restricted banks and this conditioned split. Earlier studies used the full solution universe; no claim is made that these mathematical patterns had never been considered anywhere in the research process. The dataset and all recipes were frozen before the new fits or outcomes.', '',
        '[Frozen protocol](protocol.json), [split builder](prepare.py), [runner](driver.py), [auditor](analyze.py), [tracked summary](../../../../results/research/loki/h3g/summary.json), [per-seed counts](../../../../results/research/loki/h3g/per_seed.csv), [per-clue counts](../../../../results/research/loki/h3g/per_clue_seed.csv).']
    (SRC/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (PUBLIC/'README.md').write_text('# H3g: complete-pattern holdout\n\n'+
        f"Frozen 512-pattern primary result: **{primary['decision']}**. Every valid completion of every one of the 8,192 evaluation inputs lies outside the largest 512-pattern training bank. Compatible-bank retrieval cannot answer any input.\n\n"+
        'This is a conditioned bank-incompatible distribution (0 one-clue, 16 two-clue, 8,176 three-clue inputs), not the original GRAM distribution. The 32/128 banks are secondary nested controls.\n\n'+
        '[Full report and protocol](../../../../research/20261114/loki/h3g/REPORT.md), [summary and hashes](summary.json), [per-seed counts](per_seed.csv), [per-clue counts](per_clue_seed.csv).\n')
    print(json.dumps(dict(status='validated',decision=primary['decision'],primary=primary,saved_boards=summary['independent_saved_boards'],audit_seconds=summary['audit_seconds'])))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--models-only',action='store_true');args=ap.parse_args();main(args.models_only)
