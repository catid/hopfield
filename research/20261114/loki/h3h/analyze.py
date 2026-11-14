"""Independent CPU audit of H3h construction, certificates and returned boards."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/'research/20261114/loki/h3h'
OUT=ROOT/'artifacts/research/20261114/loki/h3h'
BASE=ROOT/'artifacts/research/20261114/loki/h3g'
PUBLIC=ROOT/'results/research/loki/h3h'
spec=importlib.util.spec_from_file_location('h3g_audit',SRC.parent/'h3g/analyze.py')
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
HASHES={}


def sha(path):
    path=Path(path)
    with path.open('rb') as handle:value=hashlib.file_digest(handle,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=value
    return value


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def true_graph():
    ids=np.arange(144);r=ids//12;c=ids%12
    attack=(r[:,None]==r)|(c[:,None]==c)|(np.abs(r[:,None]-r)==np.abs(c[:,None]-c))
    np.fill_diagonal(attack,False)
    J=12*attack.astype(np.int64)-1;np.fill_diagonal(J,0)
    return attack,J


def model_audit(bank,data):
    folder=OUT/f'models/b{bank}/pair-count';rec=read(folder/'model.json')
    assert rec['status']=='complete' and rec['model_sha256']==sha(folder/'model.pt')
    assert rec['protocol_sha256']==sha(SRC/'protocol.json') and rec['source_sha256']==sha(SRC/'driver.py')
    assert rec['construction_sha256']==sha(folder/'construction.npz')
    z=dict(np.load(folder/'construction.npz'));m=torch.load(folder/'model.pt',weights_only=False)
    i,j=np.triu_indices(144,1);labels=data['all_solutions'][data[f'bank{bank}_indices']]
    # Literal accumulation by observed occupied pair, without offset lookup or matrix product.
    counts=np.zeros((144,144),dtype=np.int64)
    for row in labels:
        cells=np.flatnonzero(row)
        for a in cells:
            for b in cells:counts[a,b]+=1
    assert np.array_equal(z['pair_i'],i) and np.array_equal(z['pair_j'],j)
    assert np.array_equal(z['cooccurrence'],counts[i,j]) and np.array_equal(z['forbidden'],counts[i,j]==0)
    q=int(np.unique(labels.sum(1)).item());assert q==12
    J=q*(counts==0).astype(np.int64)-1;np.fill_diagonal(J,0)
    S=m['S'].numpy();B=m['B'].numpy()
    assert np.array_equal(S[:,:,1,1],J) and not S[:,:,0,:].any() and not S[:,:,:,0].any() and not B.any()
    assert np.array_equal(m['W'][:,1,1].numpy(),J[i,j]) and np.array_equal(m['A'].numpy(),J[i,j])
    x=labels.astype(np.float64);fields=x@J
    calibration=dict(tau=float((.5*x*fields).sum(1).max()),delta=float(((1-2*x)*fields).min()),unit=float(np.abs(fields).mean()))
    assert all(calibration[k]==m[k] for k in calibration)
    assert calibration['delta']==11 and calibration['tau']==-66 and rec['offset_pooling'] is False
    rng=np.random.default_rng(83000+bank)
    fixtures=np.stack([np.zeros(144,bool),np.ones(144,bool),*rng.integers(0,2,(8,144)).astype(bool),*labels[:3]])
    for row in fixtures:
        literal=sum(int(row[a])*int(row[b])*int(J[a,b]) for a,b in zip(i,j))
        assert literal==.5*row.astype(float)@J@row
        for cell in range(144):
            changed=row.copy();changed[cell]=~changed[cell]
            difference=(changed[i]*changed[j]*J[i,j]).sum()-literal
            assert difference==(1-2*int(row[cell]))*(row@J[:,cell])
    inputs=data['test_inputs'];margin=inputs.astype(np.float64)@J;margin[inputs]=np.inf
    initial=margin.min(1)>=m['delta']-.001
    assert np.array_equal(initial,z['initial_stopped']) and np.array_equal(margin.min(1),z['initial_min_free_margin'])
    attack,trueJ=true_graph();mask=counts[i,j]==0
    diag=dict(bank=bank,arm='pair-count',q=q,individual_pair_terms=len(i),offset_pooling=False,
              forbidden_pairs=int(mask.sum()),false_forbidden=int((mask&~attack[i,j]).sum()),missed_forbidden=int((~mask&attack[i,j]).sum()),
              initial_wrong_certificates=int(initial.sum()),calibration_float64=calibration,
              representation_checks=dict(literal_pair_energies=13,all_single_cell_costs=13*144,exact=True))
    assert diag['forbidden_pairs']==rec['forbidden_pairs'] and diag['false_forbidden']==rec['posthoc_false_forbidden']
    assert diag['missed_forbidden']==rec['posthoc_missed_forbidden']==0
    assert diag['initial_wrong_certificates']==rec['initial_wrong_certificates']
    # This counterfactual removes only false exclusions; it does not rerun search.
    true_margin=inputs.astype(np.float64)@trueJ;true_margin[inputs]=np.inf
    assert not (true_margin.min(1)>=10.999).any()
    diag['initial_false_certificates_rejected_by_true_graph']=int((initial&(true_margin.min(1)<10.999)).sum())
    return m,diag,initial


def feasible_certificates(model,data):
    """Post hoc exact certificate capacity over all compatible valid completions."""
    J=model['S'][:,:,1,1].numpy().astype(np.float64)
    solutions=data['all_solutions'];x=solutions.astype(np.float64)
    deficit=(1-2*x)*(x@J)<model['delta']-.001
    # Empty cells can never be clamped. Given cells are occupied by construction.
    impossible=(deficit&~solutions).any(1)
    total_deficits=deficit.sum(1)
    counts=[]
    for given in data['test_inputs']:
        compatible=solutions[:,given].all(1)
        eligible=~impossible&(deficit[:,given].sum(1)==total_deficits)
        counts.append(int((compatible&eligible).sum()))
    counts=np.array(counts)
    return counts,dict(inputs_with_any_valid_certificate=int((counts>0).sum()),
        inputs_without_valid_certificate=int((counts==0).sum()),valid_input_completion_certificates=int(counts.sum()),
        all_compatible_valid_completions=int(data['test_solution_counts'].sum()),
        certificate_existence_pct=100*float((counts>0).mean()),
        interpretation='Post hoc existence of a local certificate among exact valid completions; not a cap on lowest-energy fallback accuracy and not a search outcome.')


def search_audit(bank,seed,data,keys,largest_keys,model,initial):
    folder=OUT/f'search/b{bank}/pair-count/s{seed}';rec=read(folder/'result.json')
    assert rec['status']=='complete' and rec['protocol_sha256']==sha(SRC/'protocol.json')
    # The frozen H3h worker calls the unchanged H3g search implementation.
    assert rec['source_sha256']==sha(SRC.parent/'h3g/driver.py')
    assert rec['records_sha256']==sha(folder/'records.npz') and rec['model_sha256']==sha(ROOT/rec['model_path'])
    z=dict(np.load(folder/'records.npz'));boards=np.unpackbits(z['boards'],axis=-1)[...,:144].astype(bool);inputs=data['test_inputs']
    assert np.array_equal(inputs,z['inputs']) and (boards>=inputs[:,None,:]).all()
    assert not any(row.tobytes() in largest_keys for samples in boards for row in samples)
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
    wrong=z['stopped']&~valid;identical=(boards==inputs[:,None,:]).all(-1)
    initial_chains=np.repeat(initial[:,None],20,1)
    assert (z['stopped'][initial_chains]&identical[initial_chains]&~valid[initial_chains]).all()
    assert (z['moves'][initial_chains]==0).all()
    assert np.array_equal((z['moves']==0)&z['stopped'],initial_chains)
    assert rec['wrong_stops']==int(wrong.sum()) and rec['capped_chains']==int((~z['stopped']).sum())
    assert rec['original_accuracy_pct']==100*float(z['n20'].mean()) and rec['certificate_accuracy_pct']==100*float(z['n20_certificate'].mean())
    assert rec['coverage_pct']==100*float(found.sum()/denom.sum()) and rec['mean_moves']==float(z['moves'].mean())
    counts=boards.sum(-1);assert np.array_equal(counts,z['queen_counts'])
    occupancy={name:{label:int((mask&condition).sum()) for label,condition in [('below_q',counts<12),('equal_q',counts==12),('above_q',counts>12)]}
               for name,mask in [('stopped_invalid',wrong),('capped_invalid',~z['stopped']&~valid)]}
    J=model['S'][:,:,1,1].numpy().astype(np.float64);_,trueJ=true_graph();flat=boards.reshape(-1,144);givens=np.repeat(inputs,20,0)
    gaps=[];energies=[]
    for lo in range(0,len(flat),4096):
        x=flat[lo:lo+4096].astype(np.float64);fields=x@J;energies.append(.5*(x*fields).sum(1))
        margins=(1-2*x)*fields;margins[givens[lo:lo+4096]]=np.inf;gaps.append(margins.min(1)-(model['delta']-.001))
    assert np.array_equal(np.concatenate(energies).reshape(z['energy'].shape),z['energy'])
    gaps=np.concatenate(gaps)[z['stopped'].ravel()];assert (gaps>=0).all()
    wrong_rows=flat[wrong.ravel()].astype(np.float64);wrong_given=givens[wrong.ravel()]
    counter=(1-2*wrong_rows)*(wrong_rows@trueJ);counter[wrong_given]=np.inf
    counter_reject=counter.min(1)<model['delta']-.001
    assert counter_reject.all()
    diag=dict(bank=bank,arm='pair-count',seed=seed,returned_training_pattern_matches=0,invalid_occupancies=occupancy,
              strict_stopped_certificate_deficits=int((gaps<0).sum()),minimum_stopped_gap=float(gaps.min()) if len(gaps) else None,
              wrong_stops_identical_to_initial_state=int((wrong&identical).sum()),zero_move_wrong_stops=int((wrong&(z['moves']==0)).sum()),
              wrong_stops_rejected_after_removing_only_false_exclusions=int(counter_reject.sum()),
              counterfactual_scope='Evaluate the same stopped boards under the true graph; no causal search rerun or performance claim.')
    row=dict(bank=bank,arm='pair-count',updates=0,seed=seed,unique_inputs=len(inputs),chains=20,
             original_correct=int(z['n20'].sum()),certificate_first_correct=int(z['n20_certificate'].sum()),n1_correct=int(z['n1'].sum()),
             coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),wrong_stops=int(wrong.sum()),capped_chains=rec['capped_chains'],
             mean_moves=rec['mean_moves'],search_seconds=rec['search_seconds'],model_sha256=rec['model_sha256'],records_sha256=rec['records_sha256'],result_sha256=sha(folder/'result.json'))
    return z,row,diag


def main():
    started=time.perf_counter();torch.set_num_threads(1);p=read(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    data=dict(np.load(BASE/'data.npz'));keys,largest_keys,bank_audits=G.audit_data(data,p)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    rows=[];perseed=[];diagnostics=[];models=[];capacity=[];evidence={}
    for bank in [32,128,512]:
        model,diag,initial=model_audit(bank,data);models.append(diag)
        for arm in ['exclusion','pair-count']:
            current=[];zs=[]
            if arm=='exclusion':
                cfg=dict(bank=bank,arm=arm,updates=0);m,_=G.audit_model(cfg,data)
            else:m=model
            counts,capacity_rec=feasible_certificates(m,data);capacity.append(dict(bank=bank,arm=arm,**capacity_rec))
            for seed in [0,1,2]:
                if arm=='exclusion':
                    cached=next(x for x in p['cached_offset_controls'] if x['bank']==bank and x['seed']==seed)
                    assert sha(ROOT/cached['path'])==cached['sha256']
                    z,r,_,diagnostic=G.audit_search(cfg,seed,data,keys,largest_keys,m)
                    assert r['records_sha256']==cached['records_sha256'] and r['model_sha256']==cached['model_sha256']
                else:z,r,diagnostic=search_audit(bank,seed,data,keys,largest_keys,m,initial)
                # A saved valid stopped board must appear in the exact certificate census.
                assert not (z['stopped']&z['valid'])[counts==0].any()
                zs.append(z);current.append(r);perseed.append(r);diagnostics.append(diagnostic)
            arr={key:np.stack([z[key] for z in zs]) for key in ['n20','n20_certificate','n1','found']};evidence[bank,arm]=arr
            rows.append(dict(bank=bank,arm=arm,updates=0,unique_inputs=8192,seeds=[0,1,2],
                original_accuracy_pct=100*float(arr['n20'].mean()),certificate_first_accuracy_pct=100*float(arr['n20_certificate'].mean()),
                n1_accuracy_pct=100*float(arr['n1'].mean()),coverage_pct=100*float(arr['found'].sum(1).mean()/zs[0]['denom'].sum()),
                wrong_stops=sum(r['wrong_stops'] for r in current),capped_chains=sum(r['capped_chains'] for r in current),
                mean_moves=float(np.mean([r['mean_moves'] for r in current])),search_seconds_total=sum(r['search_seconds'] for r in current),
                evidence='cached H3g' if arm=='exclusion' else 'new H3h'))
    comparisons=[];denom=np.minimum(data['test_solution_counts'],20)
    for bank in [32,128,512]:
        a=evidence[bank,'exclusion'];b=evidence[bank,'pair-count']
        stats=G.A.bootstrap(dict(n20=a['n20_certificate'],found=a['found']),dict(n20=b['n20_certificate'],found=b['found']),denom)
        decision='supported' if stats['accuracy_one_sided_95_lower_pp']>=10 else 'falsified' if stats['accuracy_one_sided_95_upper_pp']<10 else 'inconclusive'
        comparisons.append(dict(bank=bank,primary=bank==32,decision=decision if bank==32 else 'secondary',**stats))
    workers=[read(OUT/f'worker-g{gpu}.json') for gpu in [0,1]];assert all(w['status']=='complete' for w in workers)
    gpu_seconds=sum((dt.datetime.fromisoformat(w['completed_utc'])-dt.datetime.fromisoformat(w['started_utc'])).total_seconds() for w in workers)
    assert gpu_seconds<1200
    # Context is explicitly cached, scope-mismatched in bank size to the primary.
    parent_summary=read(ROOT/'results/research/loki/h3g/summary.json')
    adamw=next(r for r in parent_summary['rows'] if r['arm']=='adamw-decay')
    context=dict(adamw,evidence='cached H3g reference at512patterns; not a matched32pattern control')
    sha(__file__);sha(SRC.parent/'h3g/analyze.py');sha(SRC.parent/'analyze.py')
    HASHES.update(G.HASHES)
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3h',decision=comparisons[0]['decision'],
        primary=comparisons[0],comparisons=comparisons,rows=rows,context_bp_adamw=context,dataset=p['dataset'],
        model_audits=models,output_diagnostics=diagnostics,posthoc_valid_certificate_capacity=capacity,
        independent_saved_boards=len(perseed)*8192*20,new_saved_boards=9*8192*20,every_valid_returned_pattern_unseen=True,
        worker_receipts=workers,gpu_worker_seconds=gpu_seconds,input_sha256=HASHES,limits=p['limits'],audit_seconds=time.perf_counter()-started)
    save(OUT/'summary.json',summary);save(PUBLIC/'summary.json',summary)
    with (PUBLIC/'per_seed.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(perseed[0]));writer.writeheader();writer.writerows(perseed)
    print(json.dumps(dict(status='validated',primary=comparisons[0],rows=rows,models=models,capacity=capacity,
        wrong_stop_diagnostics=[x for x in diagnostics if x['arm']=='pair-count'],gpu_worker_seconds=gpu_seconds,audit_seconds=summary['audit_seconds'])))


if __name__=='__main__':main()
