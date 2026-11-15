"""Independent corruption, calibration and complete adaptive-trajectory audit."""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
import time
import numpy as np
from run import ROOT, HERE, sha, save, now
from correlated_noise_common import paths, load_protocol
from fast_reference_mask import reference_mask


def independent_shared_draw(rng):
    uniform=float(rng.random());attempts=0
    while True:
        permutation=rng.permutation(9);attempts+=1
        if all(int(permutation[index])!=index for index in range(9)):
            assert len(set(map(int,permutation)))==9
            return uniform,permutation,attempts


def increment(joint,equal,mask,values):
    joint+=(mask[:,None]&mask[None,:]).astype(np.int32)
    for digit in range(9):
        cells=np.flatnonzero(mask&(values==digit));equal[np.ix_(cells,cells)]+=1


def calibration_audit(d,p,seed,arm,clean):
    z=np.load(d/'calibration_observations.npz');cal=json.loads((d/'calibration.json').read_text());fit=cal['fit']
    assert cal['observations_sha256']==sha(d/'calibration_observations.npz')
    assert cal['estimator_sha256']==sha(HERE/'calibration_estimator.py')
    cells=np.random.default_rng(p['streams']['calibration_cell']+seed).integers(0,81,1000)
    assert np.array_equal(z['cells'],cells);source=clean[np.arange(1000),cells]
    rng=np.random.default_rng(p['streams'][arm+'_calibration_noise']+seed)
    if arm=='iid':
        uniforms=rng.random((1000,2));offsets=rng.integers(1,9,(1000,2),dtype=np.int8)
        labels=(source[:,None]+(uniforms<.05)*offsets)%9
        assert np.array_equal(z['uniforms'],uniforms) and np.array_equal(z['offsets'],offsets)
    else:
        labels=np.empty((1000,2),np.int8)
        for index in range(1000):
            uniform,permutation,attempts=independent_shared_draw(rng)
            assert z['uniforms'][index]==uniform and np.array_equal(z['permutations'][index],permutation)
            assert z['rejection_attempts'][index]==attempts
            label=int(permutation[source[index]]) if uniform<.05 else int(source[index]);labels[index]=[label,label]
    indicators=labels[:,0]!=labels[:,1]
    assert np.array_equal(labels,z['labels']) and np.array_equal(indicators,z['disagreements'])
    count=int(indicators.sum());assert fit['pairs']==1000 and fit['disagreements']==count
    eta=Fraction(fit['eta_upper_numerator'],fit['eta_upper_denominator'])
    assert fit['postfit_covers_actual_eta']==(eta>=Fraction(1,20)) and not fit['diagnostic_used_for_selection']
    def cdf(probability):
        with localcontext() as ctx:
            ctx.prec=80;prob=Decimal(probability.numerator)/probability.denominator
            term=Decimal(math.comb(1000,count))*prob**count*(1-prob)**(1000-count);total=term
            for index in range(count,0,-1):
                term=term*index/(1001-index)*(1-prob)/prob;total+=term
            return total
    if fit['no_pruning_fallback']:
        assert eta==Fraction(8,9)
        previous=Fraction(888888,1000000)
        assert cdf(2*previous-Fraction(9,8)*previous**2)>Decimal('.005')
    else:
        assert (eta*1000000).denominator==1 and 0<eta<Fraction(8,9)
        previous=eta-Fraction(1,1000000)
        endpoint=cdf(2*eta-Fraction(9,8)*eta**2);before=cdf(2*previous-Fraction(9,8)*previous**2)
        assert endpoint<=Decimal('.005')<before
        assert abs(endpoint-Decimal(fit['endpoint_cdf_decimal80']))<Decimal('1e-70')
        assert abs(before-Decimal(fit['previous_grid_cdf_decimal80']))<Decimal('1e-70')
    archived=json.loads((d/'thresholds.json').read_text());threshold=np.array(archived['thresholds'])
    assert archived['eta']==str(eta) and archived['boundary']==648000 and len(threshold)==1111
    null=eta/4-Fraction(9,64)*eta**2;alternative=Fraction(1,9);fallback=null>=alternative
    a,b=(Fraction(1),Fraction(1)) if fallback else (alternative/null,(1-alternative)/(1-null))
    with localcontext() as context:
        context.prec=80;la=(Decimal(a.numerator)/a.denominator).ln();lb=(Decimal(b.numerator)/b.denominator).ln();bound=Decimal(648000).ln()
        for m,c in enumerate(threshold):
            c=int(c)
            if fallback:assert c==m+1
            else:
                if c<=m:assert c*la+(m-c)*lb>=bound
                if c>0:assert (c-1)*la+(m-c+1)*lb<bound
    if arm=='shared':
        assert count==0 and eta<Fraction(1,20)
    return eta,a,b,threshold,fit,int((labels!=source[:,None]).sum())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--arm',choices=['iid','shared'],required=True)
    args=parser.parse_args();start=time.monotonic();assert np.__version__=='2.3.4'
    out,pp=paths();p=load_protocol();seed=args.seed;arm=args.arm;d=out/f's{seed}'/arm
    receipt=json.loads((d/'receipt.json').read_text());assert receipt['protocol_sha256']==sha(pp)
    assert receipt['collector_sha256']==sha(HERE/'correlated_noise_acquire.py') and receipt['queries']==29750
    for name,digest in receipt['artifact_sha256'].items():assert sha(d/name)==digest
    entry=next(e for e in json.loads((ROOT/p['inputs']).read_text())['inputs'] if e['seed']==seed)
    source={}
    for role in ('calibration','training'):
        path=ROOT/entry[role+'_path'];assert sha(path)==entry[role+'_sha256']==receipt[role+'_source_sha256'];source[role]=np.load(path)['solutions']
    eta,a,b,threshold,fit,calibration_errors=calibration_audit(d,p,seed,arm,source['calibration'])
    la,lb=math.log(float(a)),math.log(float(b));z=np.load(d/'observations.npz');obs={key:z[key] for key in z.files}
    z=np.load(d/'potential_noise.npz');noise={key:z[key] for key in z.files}
    for array in noise.values():array.setflags(write=False)
    initial_noise_hash={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in noise.items()}
    events=json.loads((d/'events.json').read_text());assert len(events)==1110 and np.all(obs['given'].sum(1)==25)
    assert np.all(obs['values'][~obs['given']]==-1)
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);excluded=np.zeros((81,81),bool);first=np.full(3240,-1,np.int32);pair=np.triu_indices(81,1)
    clean_joint=joint.copy();clean_equal=equal.copy();clean_excluded=excluded.copy();clean_first=first.copy()
    ties=np.random.default_rng(p['streams']['ties']+seed);nrng=np.random.default_rng(p['streams'][arm+'_training_noise']+seed)
    training_errors=0;invariant_rows=0
    for row in range(1110):
        order=ties.permutation(81);event=events[row]
        assert event['row']==row and event['mask_finalized_before_current_corruption']
        assert event['past_state_sha256']==hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        mask=reference_mask('exposure_exact',joint,excluded,order)
        assert np.array_equal(mask,obs['given'][row]) and event['mask_sha256']==hashlib.sha256(mask.tobytes()).hexdigest()
        if arm=='shared':
            clean_mask=reference_mask('exposure_exact',clean_joint,clean_excluded,order)
            assert np.array_equal(mask,clean_mask)
        if arm=='iid':
            uniform=nrng.random(81);offset=nrng.integers(1,9,81,dtype=np.int8)
            assert np.array_equal(uniform,noise['uniforms'][row]) and np.array_equal(offset,noise['offsets'][row])
            expected=(source['training'][row]+(uniform<.05)*offset)%9
        else:
            uniform,permutation,attempts=independent_shared_draw(nrng)
            assert noise['uniforms'][row]==uniform and np.array_equal(noise['permutations'][row],permutation)
            assert noise['rejection_attempts'][row]==attempts
            expected=np.array([permutation[int(value)] for value in source['training'][row]],np.int8) if uniform<.05 else source['training'][row].copy()
            noisy_values=expected[mask];clean_values=source['training'][row,mask]
            assert np.array_equal(noisy_values[:,None]==noisy_values[None,:],clean_values[:,None]==clean_values[None,:])
        values=obs['values'][row];assert np.array_equal(values[mask],expected[mask])
        training_errors+=int((values[mask]!=source['training'][row,mask]).sum())
        increment(joint,equal,mask,values);m=joint[pair];c=equal[pair]
        assert np.allclose(obs['log_lr'][row],m*lb+c*(la-lb),rtol=0,atol=2e-12)
        first[(first<0)&(c>=threshold[m])]=row+1;drop=first>=0;excluded[pair]=drop;excluded[(pair[1],pair[0])]=drop
        if arm=='shared':
            clean_values=np.full(81,-1,np.int8);clean_values[mask]=source['training'][row,mask]
            increment(clean_joint,clean_equal,mask,clean_values)
            cm=clean_joint[pair];cc=clean_equal[pair]
            clean_first[(clean_first<0)&(cc>=threshold[cm])]=row+1;clean_drop=clean_first>=0
            clean_excluded[pair]=clean_drop;clean_excluded[(pair[1],pair[0])]=clean_drop
            for original,clean_copy in ((joint,clean_joint),(equal,clean_equal),(excluded,clean_excluded),(first,clean_first)):
                assert np.array_equal(original,clean_copy)
            invariant_rows+=1
    archived=np.load(d/'fit.npz');support=first<0
    for key,value in [('support',support),('joint_count',m),('equal_count',c),('first_crossing',first)]:assert np.array_equal(archived[key],value)
    assert initial_noise_hash=={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in noise.items()}
    i,j=pair;true=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    if arm=='shared':assert not equal[pair][true].any() and not (true&~support).any()
    report={'status':'validated','utc':now(),'seed':seed,'arm':arm,'protocol_sha256':sha(pp),'auditor_sha256':sha(__file__),
        'receipt_sha256':sha(d/'receipt.json'),'independent_masks_replayed':1110,'log_lr_values_verified':3596400,
        'first_crossing_paths_verified':3240,'decimal_threshold_checks':1111,'calibration_fit':fit,
        'calibration_error_queries':calibration_errors,'calibration_queries':2000,'training_error_queries':training_errors,'training_queries':27750,
        'clean_corrupted_trajectory_rows_identical':invariant_rows,'clean_reference_only_postfit':True,
        'missing_true_edges':int((true&~support).sum()),'extra_edges':int((~true&support).sum()),
        'support_bytes_sha256':hashlib.sha256(support.tobytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    save(d/'audit.json',report);print(json.dumps({'seed':seed,'arm':arm,'extra_edges':report['extra_edges'],'invariant_rows':invariant_rows}),flush=True)


if __name__=='__main__':main()
