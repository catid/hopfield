"""Independent single-fit replay and exact postfit weighting audit."""
import argparse
from decimal import Decimal,localcontext
from fractions import Fraction
import hashlib
import json
import math
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from confidence_budget_common import paths,load_protocol
from fast_reference_mask import reference_mask


def calibration_audit(d,entry,p,seed):
    cal=json.loads((d/'calibration.json').read_text());fit=cal['fit'];assert cal['estimator_calls']==1 and cal['shared_across_all_inference_arms']
    assert cal['estimator_sha256']==p['estimator_sha256'] and cal['base_estimator_sha256']==p['base_estimator_sha256']
    path=ROOT/entry['calibration_path'];assert sha(path)==entry['calibration_sha256'];clean=np.load(path)
    z=np.load(d/'calibration_observations.npz');assert sha(d/'calibration_observations.npz')==cal['observations_sha256']
    cells=np.random.default_rng(p['streams']['calibration_cell']+seed).integers(0,81,1000)
    rng=np.random.default_rng(p['streams']['calibration_noise']+seed);u=rng.random((1000,2));o=rng.integers(1,9,(1000,2),dtype=np.int8)
    labels=(clean['solutions'][np.arange(1000),cells,None]+(u<.05)*o)%9;ind=labels[:,0]!=labels[:,1]
    for key,value in [('cells',cells),('uniforms',u),('offsets',o),('source_row',clean['source_row']),('labels',labels),('disagreements',ind)]:assert np.array_equal(z[key],value)
    count=int(ind.sum());assert fit['pairs']==1000 and fit['disagreements']==count and fit['calibration_queries']==2000
    eta=Fraction(fit['eta_upper_numerator'],fit['eta_upper_denominator']);assert fit['postfit_covers_actual_eta']==(eta>=Fraction(1,20)) and not fit['diagnostic_used_for_selection']
    def cdf(prob):
        with localcontext() as ctx:
            ctx.prec=80;prob=Decimal(prob.numerator)/prob.denominator
            term=Decimal(math.comb(1000,count))*prob**count*(1-prob)**(1000-count);total=term
            for k in range(count,0,-1):term=term*k/(1001-k)*(1-prob)/prob;total+=term
            return total
    if fit['no_pruning_fallback']:
        assert eta==Fraction(8,9);last=Fraction(888888,1000000);assert cdf(2*last-Fraction(9,8)*last*last)>Decimal('.005')
    else:
        assert (eta*1000000).denominator==1 and 0<eta<Fraction(8,9);prior=eta-Fraction(1,1000000)
        endpoint=cdf(2*eta-Fraction(9,8)*eta*eta);previous=cdf(2*prior-Fraction(9,8)*prior*prior)
        assert endpoint<=Decimal('.005')<previous
        assert abs(endpoint-Decimal(fit['endpoint_cdf_decimal80']))<Decimal('1e-70')
        assert abs(previous-Decimal(fit['previous_grid_cdf_decimal80']))<Decimal('1e-70')
    thresholds=json.loads((d/'thresholds.json').read_text());assert thresholds['eta']==str(eta) and thresholds['boundary']==648000
    threshold=np.array(thresholds['thresholds']);assert len(threshold)==1111
    null=eta/4-9*eta*eta/64;alt=Fraction(1,9);fallback=null>=alt
    a,b=(Fraction(1),Fraction(1)) if fallback else (alt/null,(1-alt)/(1-null))
    with localcontext() as ctx:
        ctx.prec=80;la=(Decimal(a.numerator)/a.denominator).ln();lb=(Decimal(b.numerator)/b.denominator).ln();bound=Decimal(648000).ln()
        for m,c in enumerate(threshold):
            c=int(c)
            if fallback:assert c==m+1
            else:
                if c<=m:assert c*la+(m-c)*lb>=bound
                if c>0:assert (c-1)*la+(m-c+1)*lb<bound
    return eta,a,b,threshold,fit


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--pilot',action='store_true');args=ap.parse_args();start=time.monotonic()
    assert np.__version__=='2.3.4';out,pp=paths(args.pilot);p=load_protocol(args.pilot);d=out/f's{args.seed}';r=json.loads((d/'receipt.json').read_text())
    assert r['protocol_sha256']==sha(pp) and r['wrapper_sha256']==sha(HERE/'confidence_budget_acquire.py') and r['policy_sha256']==p['policy_sha256']
    for name,h in r['artifact_sha256'].items():assert sha(d/name)==h
    manifest=json.loads((ROOT/p['inputs']).read_text());entry=next(e for e in manifest['inputs'] if e['seed']==args.seed)
    source=ROOT/entry['training_path'];assert sha(source)==entry['training_sha256']==r['source_sha256'];clean=np.load(source)['solutions']
    eta,a,b,threshold,fit=calibration_audit(d,entry,p,args.seed);la,lb=math.log(float(a)),math.log(float(b))
    permutation=np.random.default_rng(p['streams']['weight_shuffle']+args.seed).permutation(3240)
    assert np.array_equal(np.load(d/'shuffle_permutation.npy'),permutation)
    noise_file=np.load(d/'potential_noise.npz');noise={k:noise_file[k] for k in noise_file.files}
    for value in noise.values():value.setflags(write=False)
    before_noise={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in noise.items()}
    z=np.load(d/'observations.npz');obs={k:z[k] for k in z.files};events=json.loads((d/'events.json').read_text())
    assert len(events)==1110 and obs['given'].shape==(1110,81) and np.all(obs['given'].sum(1)==25) and np.all(obs['values'][~obs['given']]==-1)
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);excluded=np.zeros((81,81),bool);pair=np.triu_indices(81,1);first=np.full(3240,-1,np.int32)
    ties=np.random.default_rng(p['streams']['ties']+args.seed);nrng=np.random.default_rng(p['streams']['training_noise']+args.seed)
    for row in range(1110):
        order=ties.permutation(81);event=events[row];assert event['row']==row and event['finalized_before_current_noise']
        assert event['past_state_sha256']==hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        mask=reference_mask('exposure_exact',joint,excluded,order);assert np.array_equal(mask,obs['given'][row])
        assert event['mask_sha256']==hashlib.sha256(mask.tobytes()).hexdigest()
        u=nrng.random(81);o=nrng.integers(1,9,81,dtype=np.int8);assert np.array_equal(u,noise['uniforms'][row]) and np.array_equal(o,noise['offsets'][row])
        expected=(clean[row]+(u<.05)*o)%9;values=obs['values'][row];assert np.array_equal(values[mask],expected[mask])
        joint+=(mask[:,None]&mask[None,:]).astype(np.int32)
        for digit in range(9):
            cells=np.flatnonzero(mask&(values==digit));equal[np.ix_(cells,cells)]+=1
        m=joint[pair];c=equal[pair];assert np.allclose(obs['log_lr'][row],m*lb+c*(la-lb),rtol=0,atol=2e-12)
        first[(first<0)&(c>=threshold[m])]=row+1;drop=first>=0;excluded[pair]=drop;excluded[(pair[1],pair[0])]=drop
    archived=np.load(d/'fit.npz');support=first<0
    for key,value in [('support',support),('joint_count',m),('equal_count',c),('first_crossing',first)]:assert np.array_equal(archived[key],value)
    expected_uniform=np.where(support,16,0).astype(np.uint8);expected_evidence=expected_uniform.copy()
    for k in np.flatnonzero(support):
        likelihood=a**int(c[k])*b**int(m[k]-c[k]);expected_evidence[k]=16 if likelihood<=1 else 1
    expected_shuffled=expected_uniform.copy();ordered=[int(k) for k in permutation if support[k]];expected_shuffled[ordered[:int((expected_evidence==1).sum())]]=1
    expected={'uniform':expected_uniform,'evidence':expected_evidence,'shuffled':expected_shuffled};models=[]
    for model in r['models']:
        path=ROOT/model['path'];assert sha(path)==model['sha256'];model_arrays=np.load(path);w=model_arrays['weight16'];arm=model['arm']
        assert np.array_equal(model_arrays['support'],support) and np.array_equal(w,expected[arm]) and np.array_equal(w>0,support)
        assert hashlib.sha256(w.tobytes()).hexdigest()==model['weight16_bytes_sha256']
        models.append({'arm':arm,'weak_weights':int((w==1).sum()),'strong_weights':int((w==16).sum()),'weight16_bytes_sha256':model['weight16_bytes_sha256']})
    assert len(models)==3 and np.array_equal(np.sort(expected_evidence),np.sort(expected_shuffled))
    assert r['one_fit_shared_across_arms'] and r['weight_inputs_readonly_and_unchanged'] and r['shuffle_generated_before_calibration_and_training'] and r['queries_per_shared_fit']==29750
    assert before_noise=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in noise.items()}
    i,j=pair;peer=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    report={'status':'validated','utc':now(),'seed':args.seed,'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),'receipt_sha256':sha(d/'receipt.json'),
        'independent_masks_replayed':1110,'log_lr_values_verified':3596400,'noise_rows_verified':1110,'decimal_threshold_checks':1111,
        'first_crossing_paths_verified':3240,'supports_verified':1,'weight_models_verified':3,'weight_entries_verified':9720,'calibration_fit':fit,
        'models':models,'missing_true_edges':int((peer&~support).sum()),'extra_edges':int((~peer&support).sum()),
        'support_bytes_sha256':hashlib.sha256(support.tobytes()).hexdigest(),'noise_cached_readonly_and_unchanged':True,
        'fast_mask_helper_sha256':sha(HERE/'fast_reference_mask.py'),'elapsed_s':time.monotonic()-start,'numpy':np.__version__}
    save(d/'audit.json',report);print({'seed':args.seed,'masks':1110,'weight_models':3,'elapsed_s':report['elapsed_s']},flush=True)


if __name__=='__main__':main()
