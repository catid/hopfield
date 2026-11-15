"""All-board float64 and unary-aware certificate audit for H3u."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv,datetime as dt,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3u';OUT=ROOT/'artifacts/research/20261114/loki/h3u';PUBLIC=ROOT/'results/research/loki/h3u';HASHES={}
spec=importlib.util.spec_from_file_location('loki_bootstrap',SRC.parent/'analyze.py');BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(BASE)
def sha(p):
    p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();HASHES[str(p.relative_to(ROOT))]=h;return h
def read(p):sha(p);return json.loads(Path(p).read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def csvsave(p,rows):
    with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def capacity(model,data):
    J=model['S'][:,:,1,1].numpy().astype(np.float64);h=model['B'][:,1].numpy().astype(np.float64);y=data['all_solutions'];x=y.astype(np.float64);margin=(1-2*x)*(x@J+h);deficit=margin<model['delta']-.001
    never=(deficit&~y).any(1);total=deficit.sum(1);out=[]
    for given in data['test_inputs']:
        compatible=y[:,given].all(1);clamped=deficit[:,given].sum(1)==total;out.append(int((compatible&clamped&~never).sum()))
    out=np.array(out);info=dict(inputs_with_any_valid_certificate=int((out>0).sum()),inputs_without_valid_certificate=int((out==0).sum()),valid_input_completion_certificates=int(out.sum()),all_compatible_valid_completions=int(data['test_solution_counts'].sum()),
      complete_target_solutions_without_any_deficit=int((total==0).sum()),target_solutions_with_empty_cell_deficits=int(never.sum()),minimum_target_full_solution_margin=float(margin.min()),
      interpretation='Exact posthoc valid-certificate existence for stored float32 coefficients reconstructed in float64, including shared unary and clamped givens. This is not a bound on general fallback-decoder accuracy.')
    # Independent literal certificate check on a deterministic bounded subset of compatible pairs.
    rng=np.random.default_rng(83207);checks=0;i,j=np.triu_indices(169,1)
    for k in rng.choice(len(data['test_inputs']),32,replace=False):
        given=data['test_inputs'][k];solutions=y[y[:,given].all(1)];literal_count=0
        for row in solutions:
            field=np.array([sum(float(J[c,d]) for d in np.flatnonzero(row))+h[c] for c in range(169)])
            costs=(1-2*row.astype(int))*field;literal_count+=bool((costs[~given]>=model['delta']-.001).all());checks+=1
        assert literal_count==out[k]
    info['independent_literal_completion_checks']=checks
    return out,info

def audit_search(arm,seed,m,data,keys,cap):
    folder=OUT/f'search/{arm}/s{seed}';rec=read(folder/'result.json');assert rec['status'] in ['complete','derived_by_exact_identity'];assert rec['protocol_sha256']==sha(SRC/'protocol.json');assert rec['source_sha256']==sha(SRC/'driver.py');assert rec['records_sha256']==sha(folder/'records.npz');assert rec['model_sha256']==sha(OUT/f'models/{arm}/model.pt');assert rec['backend_sha256']==sha(SRC.parent/'h3i/backend.py')
    z=dict(np.load(folder/'records.npz'));boards=np.unpackbits(z['boards'],axis=-1)[...,:169].astype(bool);inputs=data['test_inputs'];assert np.array_equal(z['inputs'],inputs) and (boards>=inputs[:,None,:]).all()
    valid=np.array([[row.tobytes() in keys and row[given].all() for row in samples] for given,samples in zip(inputs,boards)]);assert np.array_equal(valid,z['valid']);denom=np.minimum(data['test_solution_counts'],20)
    found=np.array([min(len({row.tobytes() for row in samples[ok]}),limit) for samples,ok,limit in zip(boards,valid,denom)]);assert np.array_equal(found,z['found']) and np.array_equal(denom,z['denom'])
    original=z['energy'].argmin(1);selected=np.where(z['stopped'].any(1),np.where(z['stopped'],z['energy'],np.inf).argmin(1),original);ids=np.arange(len(inputs))
    assert np.array_equal(original,z['original_index']) and np.array_equal(selected,z['certificate_index']);assert np.array_equal(valid[ids,original],z['n20']) and np.array_equal(valid[ids,selected],z['n20_certificate']) and np.array_equal(valid[:,0],z['n1'])
    assert ((z['moves']>=0)&(z['moves']<=20000)).all() and (z['moves'][~z['stopped']]==20000).all();assert np.array_equal(z['flop'],57461+859*z['moves']);assert np.array_equal(z['cardinality_integer_additions'],168*(z['moves']+1))
    counts=boards.sum(-1);assert np.array_equal(counts,z['queen_counts']) and (counts[z['stopped']]==13).all();wrong=z['stopped']&~valid
    assert rec['wrong_stops']==int(wrong.sum()) and rec['capped_chains']==int((~z['stopped']).sum());assert rec['original_accuracy_pct']==100*float(z['n20'].mean()) and rec['certificate_accuracy_pct']==100*float(z['n20_certificate'].mean());assert rec['coverage_pct']==100*float(found.sum()/denom.sum())
    J=m['S'][:,:,1,1].numpy().astype(np.float64);h=m['B'][:,1].numpy().astype(np.float64);flat=boards.reshape(-1,169);givens=np.repeat(inputs,20,0);gaps=[];energy=[]
    for lo in range(0,len(flat),4096):
        x=flat[lo:lo+4096].astype(np.float64);field=x@J+h;energy.append((.5*x*(field+h)).sum(1));margin=(1-2*x)*field;margin[givens[lo:lo+4096]]=np.inf;gaps.append(margin.min(1)-(m['delta']-.001))
    exact=np.concatenate(energy).reshape(z['energy'].shape);error=np.abs(exact-z['energy']);gap=np.concatenate(gaps).reshape(z['stopped'].shape);stopped_gap=gap[z['stopped']]
    exact_original=exact.argmin(1);exact_selected=np.where(z['stopped'].any(1),np.where(z['stopped'],exact,np.inf).argmin(1),exact_original)
    wrong_rows=flat[wrong.ravel()];pi,pj=np.triu_indices(169,1);dr=pj//13-pi//13;dc=pj%13-pi%13
    attack=(dr==0)|(dc==0)|(abs(dr)==abs(dc))|((dr==1)&(abs(dc)==3))|((dr==3)&(abs(dc)==1))
    conflicts=(wrong_rows[:,pi[attack]]&wrong_rows[:,pj[attack]]).sum(0)
    conflict_offsets={}
    for r,c,v in zip(dr[attack],dc[attack],conflicts):
        key=f'{int(r)},{int(c)}';conflict_offsets[key]=conflict_offsets.get(key,0)+int(v)
    conflict_offsets={k:v for k,v in conflict_offsets.items() if v}
    invalid_conflict_per_row=(wrong_rows[:,pi[attack]]&wrong_rows[:,pj[attack]]).sum(1)
    assert (invalid_conflict_per_row>0).all()
    # Frozen output selection remains based on saved running energies; exact selectors are diagnostics only.
    same_cert=(exact[np.arange(len(inputs)),selected]-exact[np.arange(len(inputs)),exact_selected])
    occupancy={name:{key:int((mask&v).sum()) for key,v in [('below_q',counts<13),('equal_q',counts==13),('above_q',counts>13)]} for name,mask in [('stopped_invalid',wrong),('capped_invalid',~z['stopped']&~valid)]}
    diagnostics=dict(arm=arm,seed=seed,wrong_stops=int(wrong.sum()),stopped_chains=int(z['stopped'].sum()),strict_stopped_certificate_deficits=int((stopped_gap<0).sum()),
      stopped_deficit_counts={str(t):int((stopped_gap < -t).sum()) for t in [1e-6,1e-4,1e-3]},near_stopped_margin_counts={str(t):int((abs(stopped_gap)<=t).sum()) for t in [1e-6,1e-4,1e-3]},
      minimum_stopped_gap=float(stopped_gap.min()) if len(stopped_gap) else None,saved_energy_absolute_error_max=float(error.max()),saved_energy_absolute_error_quantiles={str(q):float(np.quantile(error,q)) for q in [0,.5,.9,.99,1]},
      original_exact_energy_selector_disagreements=int((exact_original!=original).sum()),certificate_exact_energy_selector_disagreements=int((exact_selected!=selected).sum()),
      exact_energy_max_certificate_selected_regret=float(same_cert.max()),exact_energy_certificate_accuracy_pct=100*float(valid[ids,exact_selected].mean()),
      selected_corrected_inputs=int((z['n20_certificate']&~z['n20']).sum()),selected_regressed_inputs=int((z['n20']&~z['n20_certificate']).sum()),invalid_occupancies=occupancy,
      certificate_capacity_partition=dict(inputs_without_valid_certificate=int((cap==0).sum()),failed_without_valid_certificate=int(((cap==0)&~z['n20_certificate']).sum()),succeeded_without_valid_certificate=int(((cap==0)&z['n20_certificate']).sum()),failed_despite_valid_certificate=int(((cap>0)&~z['n20_certificate']).sum())),
      wrong_stop_true_conflict_edge_occurrences=int(invalid_conflict_per_row.sum()),wrong_stop_conflicts_by_offset=conflict_offsets,
      numerical_gate_passed=bool(not (stopped_gap<-.001).any() and error.max()<=.1))
    row=dict(arm=arm,seed=seed,unique_inputs=len(inputs),chains=20,original_correct=int(z['n20'].sum()),certificate_first_correct=int(z['n20_certificate'].sum()),n1_correct=int(z['n1'].sum()),coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),wrong_stops=int(wrong.sum()),capped_chains=int((~z['stopped']).sum()),mean_moves=float(z['moves'].mean()),search_seconds=rec['search_seconds'],model_sha256=rec['model_sha256'],records_sha256=rec['records_sha256'])
    return z,row,diagnostics

def main():
    start=time.perf_counter();torch.set_num_threads(1);PUBLIC.mkdir(parents=True,exist_ok=True);p=read(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for path,h in p[group].items():assert sha(ROOT/path)==h,path
    pre=read(OUT/'independent-preflight.json');assert pre['status']=='passed'
    for path,h in pre['input_sha256'].items():assert sha(ROOT/path)==h,path
    data=dict(np.load(OUT/'data.npz'));keys={r.tobytes() for r in data['all_solutions']};records={};rows=[];diags=[];capacities={};perseed=[];models={};fits={}
    for cfg in p['arms']:
        arm=cfg['arm'];m=torch.load(OUT/f'models/{arm}/model.pt',weights_only=False);models[arm]=m;fit=read(OUT/f'models/{arm}/model.json');fits[arm]=fit;assert fit['model_sha256']==sha(OUT/f'models/{arm}/model.pt')
        if not fit['source_eligible']:continue
        cap,capacity_info=capacity(m,data);capacities[arm]=capacity_info;zz=[]
        for seed in [0,1,2]:
            z,row,diag=audit_search(arm,seed,m,data,keys,cap);zz.append(z);perseed.append(row);diags.append(diag)
        records[arm]=dict(n20=np.stack([z['n20_certificate'] for z in zz]),found=np.stack([z['found'] for z in zz]))
        rows.append(dict(arm=arm,source_patterns=176,target_training_patterns=0,unique_inputs=4096,chains=20,search_seeds=[0,1,2],
          original_accuracy_pct=100*float(np.stack([z['n20'] for z in zz]).mean()),certificate_first_accuracy_pct=100*float(records[arm]['n20'].mean()),n1_accuracy_pct=100*float(np.stack([z['n1'] for z in zz]).mean()),coverage_pct=100*float(records[arm]['found'].sum()/(3*zz[0]['denom'].sum())),wrong_stops=sum(int((z['stopped']&~z['valid']).sum()) for z in zz),capped_chains=sum(int((~z['stopped']).sum()) for z in zz),mean_moves=float(np.stack([z['moves'] for z in zz]).mean()),search_seconds_total=sum(row['search_seconds'] for row in perseed if row['arm']==arm)))
    comparisons=[];denom=np.minimum(data['test_solution_counts'],20)
    for a,b in [('bp-adamw-grouped','exposure-filtered'),('bp-adamw-grouped','count-hybrid'),('bp-adamw-grouped','supplied-rule-oracle'),('exposure-filtered','count-hybrid')]:
        if a not in records:continue
        comparisons.append(dict(arm=a,control=b,primary=len(comparisons)==0,**BASE.bootstrap(records[a],records[b],denom)))
    if not fits['bp-adamw-grouped']['source_eligible']:decision='falsified';primary=dict(decision=decision,reason='source-fit eligibility failure')
    else:
        primary=comparisons[0];wrong=next(r['wrong_stops'] for r in rows if r['arm']=='bp-adamw-grouped');numerical=all(d['numerical_gate_passed'] for d in diags if d['arm'] in ['bp-adamw-grouped','exposure-filtered'])
        if wrong or primary['accuracy_one_sided_95_upper_pp']<5:decision='falsified'
        elif not numerical:decision='inconclusive_numerical_audit'
        elif primary['accuracy_one_sided_95_lower_pp']>=5:decision='supported'
        else:decision='inconclusive'
        primary.update(decision=decision,wrong_stops=wrong,numerical_gate_passed=numerical,required_gain_pp=5)
    source=dict(np.load(OUT/'source-design.npz'));offset_rows=[]
    for k,((r,c),g,count,exposure) in enumerate(zip(source['offsets'],source['groups'],source['counts'],source['exposure'])):
        true=bool(r==0 or c==0 or abs(r)==abs(c) or (r,abs(c)) in [(1,3),(3,1)])
        offset_rows.append(dict(dr=int(r),dc=int(c),group=int(g),source_count=int(count),source_exposure=float(exposure),true_family_attack=true,source_global_extra=bool(count==0 and not true),
          bp_adamw_weight=float(models['bp-adamw-grouped']['A'][g]),count_weight=float(models['count-hybrid']['A'][g]),filtered_weight=float(models['exposure-filtered']['A'][g]),oracle_weight=float(models['supplied-rule-oracle']['A'][g])))
    workers=[read(OUT/f'worker-g{g}.json') for g in [0,1]];assert all(w['status']=='complete' for w in workers);worker_seconds=sum((dt.datetime.fromisoformat(w['completed_utc'])-dt.datetime.fromisoformat(w['started_utc'])).total_seconds() for w in workers)
    identity=read(OUT/'identity-plan.json');mapping=read(OUT/'identity-mapping.json');loss=read(OUT/'loss-validation.json');validations=[read(OUT/f'backend-validation-{d}.json') for d in ['cpu','cuda']]
    for path in [SRC/'analyze.py',SRC.parent/'analyze.py',SRC/'PRELAUNCH_AMENDMENT.md',SRC/'prelaunch-v1/protocol.json',SRC/'prelaunch-v1/preflight.py']:sha(path)
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3u',decision=decision,primary=primary,comparisons=comparisons,rows=rows,dataset=p['dataset'],fits=fits,
       posthoc_valid_certificate_capacity=capacities,output_diagnostics=diags,independent_preflight=pre,loss_and_optimizer_validation=loss,backend_validations=validations,identity_plan=identity,identity_mapping=mapping,
       independent_saved_boards=len(perseed)*4096*20,worker_receipts=workers,search_gpu_worker_seconds=worker_seconds,fit_gpu_seconds=fits['bp-adamw-grouped']['fit_seconds'],
       scope='Matched grouped224+1 BP AdamW, source-calibrated unchanged coefficient transfer; not H3t symbolic J13 or original264+1optimizer architecture.',limits=p['limits'],input_sha256=HASHES,audit_seconds=time.perf_counter()-start)
    save(PUBLIC/'summary.json',summary);csvsave(PUBLIC/'per_seed.csv',perseed);csvsave(PUBLIC/'source_offset_weights.csv',offset_rows)
    print(json.dumps(dict(decision=decision,primary=primary,rows=rows,capacity=capacities,numeric_summary=[{k:d[k] for k in ['arm','seed','strict_stopped_certificate_deficits','saved_energy_absolute_error_max','minimum_stopped_gap','numerical_gate_passed']} for d in diags],search_gpu_worker_seconds=worker_seconds)))
if __name__=='__main__':main()
