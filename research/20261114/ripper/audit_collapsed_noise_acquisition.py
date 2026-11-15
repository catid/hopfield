"""Independent collapse law, calibration and full predictable-mask replay."""
import argparse
from decimal import Decimal,localcontext
from fractions import Fraction
import hashlib
import json
import math
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from collapsed_noise_common import paths,load_protocol
from fast_reference_mask import reference_mask
from audit_correlated_noise_acquisition import increment


def independent_draw(rng):
    uniform=float(rng.random());digit=int(rng.integers(0,9));event=Fraction(uniform)<Fraction(9,160)
    assert event==(uniform<9/160)
    return uniform,digit,event


def calibration_audit(d,p,seed,clean):
    z=np.load(d/'calibration_observations.npz');cal=json.loads((d/'calibration.json').read_text());fit=cal['fit']
    assert cal['observations_sha256']==sha(d/'calibration_observations.npz')
    assert cal['estimator_sha256']==sha(HERE/'calibration_estimator.py')
    cells=np.random.default_rng(p['streams']['calibration_cell']+seed).integers(0,81,1000)
    assert np.array_equal(z['cells'],cells);source=clean[np.arange(1000),cells]
    rng=np.random.default_rng(p['streams']['calibration_noise']+seed);labels=np.zeros((1000,2),np.int8);events=0
    for index in range(1000):
        uniform,digit,event=independent_draw(rng);events+=int(event)
        assert z['uniforms'][index]==uniform and z['digits'][index]==digit
        labels[index]=[digit,digit] if event else [source[index],source[index]]
    indicators=labels[:,0]!=labels[:,1]
    assert not indicators.any() and np.array_equal(z['labels'],labels) and np.array_equal(z['disagreements'],indicators)
    assert fit['pairs']==1000 and fit['disagreements']==0 and not fit['diagnostic_used_for_selection']
    eta=Fraction(fit['eta_upper_numerator'],fit['eta_upper_denominator'])
    assert 0<eta<Fraction(1,20) and not fit['postfit_covers_actual_eta'] and not fit['no_pruning_fallback']
    assert (eta*1000000).denominator==1
    def zero_cdf(rate):
        with localcontext() as ctx:
            ctx.prec=80;probability=2*rate-Fraction(9,8)*rate**2
            return (1-Decimal(probability.numerator)/probability.denominator)**1000
    endpoint=zero_cdf(eta);before=zero_cdf(eta-Fraction(1,1000000))
    assert endpoint<=Decimal('.005')<before
    assert abs(endpoint-Decimal(fit['endpoint_cdf_decimal80']))<Decimal('1e-70')
    assert abs(before-Decimal(fit['previous_grid_cdf_decimal80']))<Decimal('1e-70')
    archived=json.loads((d/'thresholds.json').read_text());threshold=np.array(archived['thresholds'])
    assert len(threshold)==1111 and archived['eta']==str(eta) and archived['boundary']==648000
    null=eta/4-Fraction(9,64)*eta**2;alternative=Fraction(1,9)
    assert null<Fraction(9,160)<alternative
    a,b=alternative/null,(1-alternative)/(1-null)
    with localcontext() as ctx:
        ctx.prec=80;la=(Decimal(a.numerator)/a.denominator).ln();lb=(Decimal(b.numerator)/b.denominator).ln();bound=Decimal(648000).ln()
        for m,c in enumerate(threshold):
            c=int(c)
            if c<=m:assert c*la+(m-c)*lb>=bound
            if c>0:assert (c-1)*la+(m-c+1)*lb<bound
    return eta,a,b,threshold,fit,int((labels!=source[:,None]).sum()),events


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);args=parser.parse_args();start=time.monotonic()
    assert np.__version__=='2.3.4';out,pp=paths();p=load_protocol();seed=args.seed;d=out/f's{seed}'
    receipt=json.loads((d/'receipt.json').read_text());assert receipt['protocol_sha256']==sha(pp)
    assert receipt['collector_sha256']==sha(HERE/'collapsed_noise_acquire.py') and receipt['queries']==29750
    for name,digest in receipt['artifact_sha256'].items():assert sha(d/name)==digest
    entry=next(e for e in json.loads((ROOT/p['inputs']).read_text())['inputs'] if e['seed']==seed);source={}
    for role in ('calibration','training'):
        path=ROOT/entry[role+'_path'];assert sha(path)==entry[role+'_sha256']==receipt[role+'_source_sha256'];source[role]=np.load(path)['solutions']
    eta,a,b,threshold,fit,cal_errors,cal_events=calibration_audit(d,p,seed,source['calibration'])
    la,lb=math.log(float(a)),math.log(float(b));z=np.load(d/'observations.npz');obs={key:z[key] for key in z.files}
    z=np.load(d/'potential_noise.npz');noise={key:z[key] for key in z.files}
    for array in noise.values():array.setflags(write=False)
    noise_hash={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in noise.items()}
    events=json.loads((d/'events.json').read_text());assert len(events)==1110 and np.all(obs['given'].sum(1)==25)
    assert np.all(obs['values'][~obs['given']]==-1)
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);excluded=np.zeros((81,81),bool);first=np.full(3240,-1,np.int32)
    pair=np.triu_indices(81,1);i,j=pair;true=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    ties=np.random.default_rng(p['streams']['ties']+seed);nrng=np.random.default_rng(p['streams']['training_noise']+seed)
    training_errors=training_events=true_exposures=true_equal=all_equal=predictable_all_equal_num=0
    for row in range(1110):
        order=ties.permutation(81);event=events[row]
        assert event['row']==row and event['mask_finalized_before_current_corruption']
        assert event['past_state_sha256']==hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        mask=reference_mask('exposure_exact',joint,excluded,order)
        assert np.array_equal(mask,obs['given'][row]) and event['mask_sha256']==hashlib.sha256(mask.tobytes()).hexdigest()
        # Only now reconstruct the current independent corruption draw.
        uniform,digit,collapse=independent_draw(nrng);training_events+=int(collapse)
        assert noise['uniforms'][row]==uniform and noise['digits'][row]==digit
        expected=np.full(81,digit,np.int8) if collapse else source['training'][row].copy()
        values=obs['values'][row];assert np.array_equal(values[mask],expected[mask])
        training_errors+=int((values[mask]!=source['training'][row,mask]).sum())
        queried=mask[i]&mask[j];clean_equal=source['training'][row,i]==source['training'][row,j]
        observed_equal=values[i]==values[j];assert not clean_equal[true].any()
        assert np.array_equal(observed_equal[queried],np.ones(int(queried.sum()),bool) if collapse else clean_equal[queried])
        exposures=int((queried&true).sum());equality=int((queried&true&observed_equal).sum())
        assert equality==(exposures if collapse else 0)
        true_exposures+=exposures;true_equal+=equality;all_equal+=int((queried&observed_equal).sum())
        predictable_all_equal_num+=int((queried&clean_equal).sum())*160+int((queried&~clean_equal).sum())*9
        increment(joint,equal,mask,values);m=joint[pair];c=equal[pair]
        assert np.allclose(obs['log_lr'][row],m*lb+c*(la-lb),rtol=0,atol=2e-12)
        first[(first<0)&(c>=threshold[m])]=row+1;drop=first>=0;excluded[pair]=drop;excluded[(pair[1],pair[0])]=drop
    archived=np.load(d/'fit.npz');support=first<0
    for key,value in [('support',support),('joint_count',m),('equal_count',c),('first_crossing',first)]:assert np.array_equal(archived[key],value)
    assert true_exposures==int(m[true].sum()) and true_equal==int(c[true].sum()) and all_equal==int(c.sum())
    assert noise_hash=={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in noise.items()}
    report={'status':'validated','utc':now(),'seed':seed,'arm':'collapse','protocol_sha256':sha(pp),'auditor_sha256':sha(__file__),
        'receipt_sha256':sha(d/'receipt.json'),'independent_masks_replayed':1110,'log_lr_values_verified':3596400,
        'first_crossing_paths_verified':3240,'decimal_threshold_checks':1111,'calibration_fit':fit,
        'calibration_error_queries':cal_errors,'calibration_queries':2000,'calibration_collapse_events':cal_events,
        'training_error_queries':training_errors,'training_queries':27750,'training_collapse_events':training_events,
        'true_pair_exposures':true_exposures,'true_pair_equal_count':true_equal,'observed_true_pair_equal_rate':true_equal/true_exposures,
        'conditional_true_pair_equal_probability':'9/160','predictable_true_equal_sum':str(Fraction(9*true_exposures,160)),
        'all_pair_equal_count':all_equal,'predictable_all_pair_equal_sum':str(Fraction(predictable_all_equal_num,160)),
        'missing_true_edges':int((true&~support).sum()),'extra_edges':int((~true&support).sum()),
        'support_bytes_sha256':hashlib.sha256(support.tobytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    save(d/'audit.json',report);print(json.dumps({'seed':seed,'missing_true':report['missing_true_edges'],'extra_edges':report['extra_edges']}),flush=True)


if __name__=='__main__':main()
