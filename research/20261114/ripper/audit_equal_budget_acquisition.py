"""Independent noisy-mask replay, potential-label audit and log-LR paths."""
import argparse
import hashlib
import json
import math
from fractions import Fraction
import time
import numpy as np
from equal_budget_common import load_protocol,paths
from run import ROOT,HERE,sha,save,now




def reference_mask(arm,joint,excluded,order):
    if arm=='uniform_e':chosen=list(order[:25])
    else:
        w=(8./9.)**joint.astype(np.float64);w[excluded]=0.;w[np.diag_indices(81)]=0.
        chosen=[]
        for step in range(25):
            score=w.sum(1) if not chosen else np.zeros(81)
            for cell in chosen:score=score+w[:,cell]
            score[chosen]=-np.inf;best=score.max()
            chosen.append(next(int(c) for c in order if score[c]==best))
    mask=np.zeros(81,bool);mask[chosen]=True;return mask


def calibration_audit(d,entry,p,seed):
    from decimal import Decimal,localcontext
    cal=json.loads((d/'calibration.json').read_text());r=json.loads((d/'receipt.json').read_text())
    assert sha(d/'calibration.json')==r['calibration_sha256']
    assert cal['estimator_sha256']==p['estimator_sha256'] and cal['base_estimator_sha256']==p['base_estimator_sha256'] and cal['nested_prefixes']
    source=ROOT/entry['calibration_path'];assert sha(source)==entry['calibration_sha256']==r['calibration_source_sha256'];clean=np.load(source)
    archive=d/'calibration_observations.npz';assert sha(archive)==cal['observations_sha256'];z=np.load(archive)
    cells=np.random.default_rng(p['streams']['calibration_cell']+seed).integers(0,81,10000)
    rng=np.random.default_rng(p['streams']['calibration_noise']+seed);flips=rng.random((10000,2))<.01;offsets=rng.integers(1,9,(10000,2),dtype=np.int8)
    labels=(clean['solutions'][np.arange(10000),cells,None]+flips*offsets)%9;indicators=labels[:,0]!=labels[:,1]
    for key,value in [('cells',cells),('flips',flips),('offsets',offsets),('labels',labels),('disagreements',indicators),('source_row',clean['source_row'])]:assert np.array_equal(z[key],value)
    def cdf(prob,count,n):
        # Independent backwards sum from Binomial(n,p)'s count mass, 80 digits.
        with localcontext() as context:
            context.prec=80;prob=Decimal(prob.numerator)/prob.denominator
            if prob==0:return Decimal(1)
            if prob==1:return Decimal(int(count==n))
            term=Decimal(math.comb(n,count))*prob**count*(1-prob)**(n-count);total=term
            for j in range(count,0,-1):term=term*j/(n-j+1)*(1-prob)/prob;total+=term
            return total
    alpha=Decimal('.005');rates={'exact':Fraction(1,100),'fixed_upper':Fraction(1,50)};checked=[]
    for arm in ['learned_small','learned_large']:
        fit=cal['fits'][arm];n=p['arm_calibration_pairs'][arm];count=int(indicators[:n].sum())
        assert fit['pairs']==n and fit['disagreements']==count and fit['calibration_queries']==2*n and not fit['diagnostic_used_for_selection']
        eta=Fraction(fit['eta_upper_numerator'],fit['eta_upper_denominator']);rates[arm]=eta
        assert fit['postfit_covers_actual_eta']==(eta>=Fraction(1,100));endpoint=None;previous=None
        if fit['no_pruning_fallback']:
            assert eta==Fraction(8,9);last=Fraction(888888,1000000);assert cdf(2*last-Fraction(9,8)*last*last,count,n)>alpha
        else:
            assert 0<eta<Fraction(8,9) and (eta*1000000).denominator==1
            before=eta-Fraction(1,1000000);endpoint=cdf(2*eta-Fraction(9,8)*eta*eta,count,n);previous=cdf(2*before-Fraction(9,8)*before*before,count,n)
            assert endpoint<=alpha<previous
            assert abs(endpoint-Decimal(fit['endpoint_cdf_decimal80']))<Decimal('1e-70')
            assert abs(previous-Decimal(fit['previous_grid_cdf_decimal80']))<Decimal('1e-70')
        checked.append({'arm':arm,'pairs':n,'disagreements':count,'eta_upper':float(eta),'eta_fraction':str(eta),'covers_actual_eta':eta>=Fraction(1,100),
            'no_pruning_fallback':fit['no_pruning_fallback'],'endpoint_cdf_decimal80':None if endpoint is None else str(endpoint),
            'previous_cdf_decimal80':None if previous is None else str(previous),'calibration_queries':2*n,'training_queries':25*p['arm_rows'][arm],'total_queries':57500})
    t=json.loads((d/'thresholds.json').read_text());assert sha(d/'thresholds.json')==r['thresholds_sha256'] and t['boundary']==648000
    threshold={};checks=0
    for arm,rate in rates.items():
        assert Fraction(t['rates'][arm])==rate;threshold[arm]=np.array(t['thresholds'][arm]);assert len(threshold[arm])==p['arm_rows'][arm]+1
        q0=rate/4-9*rate*rate/64;q1=Fraction(1,9)
        with localcontext() as context:
            context.prec=80;a=q1/q0;b=(1-q1)/(1-q0);la=(Decimal(a.numerator)/a.denominator).ln();lb=(Decimal(b.numerator)/b.denominator).ln();bound=Decimal(648000).ln()
            for m,c in enumerate(threshold[arm]):
                c=int(c);assert 0<=c<=m+1
                if c<=m:assert c*la+(m-c)*lb>=bound
                if c>0:assert (c-1)*la+(m-c+1)*lb<bound
                checks+=1
    assert checks==8324
    for a in p['arms']:assert p['label_costs'][a]['total']==p['label_costs'][a]['calibration']+25*p['arm_rows'][a]==57500
    return {'nested_pairs':10000,'fits':checked,'label_costs':p['label_costs']},checks,rates,threshold


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',required=True,type=int);ap.add_argument('--pilot',action='store_true');args=ap.parse_args();start=time.monotonic()
    assert np.__version__=='2.3.4';out,pp=paths(args.pilot);p=load_protocol(args.pilot);d=out/f's{args.seed}'
    receipt=json.loads((d/'receipt.json').read_text());assert receipt['protocol_sha256']==sha(pp)
    assert receipt['wrapper_sha256']==sha(HERE/'equal_budget_acquire.py') and receipt['policy_sha256']==sha(HERE/'equal_budget_policy.py')==p['policy_sha256']
    assert receipt['masks_fixed_before_all_reveals']==8320
    manifest=json.loads((ROOT/p['inputs']).read_text());assert sha(ROOT/p['inputs'])==p['input_manifest_sha256']
    entry=next(e for e in manifest['inputs'] if e['seed']==args.seed);source=ROOT/entry['training_path']
    assert sha(source)==entry['training_sha256']==receipt['source_sha256'];clean=np.load(source)['solutions']
    calibration_checks,threshold_checks,rates,threshold=calibration_audit(d,entry,p,args.seed)
    assert sha(d/'potential_noise.npz')==receipt['potential_noise_sha256'];noise=np.load(d/'potential_noise.npz')
    assert sha(d/'events.json')==receipt['events_sha256'];events=json.loads((d/'events.json').read_text());assert len(events)==8320
    archives={}
    for entry in receipt['observations']:
        path=ROOT/entry['path'];assert sha(path)==entry['sha256'];z=np.load(path);arm=entry['arm']
        archives[arm]={k:z[k] for k in z.files};v=archives[arm]
        assert v['given'].shape==(p['arm_rows'][arm],81) and np.all(v['given'].sum(1)==25) and np.all(v['values'][~v['given']]==-1)
    pair=np.triu_indices(81,1);ii,jj=pair
    states={a:{'joint':np.zeros((81,81),np.int32),'equal':np.zeros((81,81),np.int32),'excluded':np.zeros((81,81),bool),
        'first_crossing':np.full(3240,-1,np.int32),'first_excluded':np.full(3240,-1,np.int32)} for a in p['arms']}
    tie_rng=np.random.default_rng(p['streams']['ties']+args.seed);noise_rng=np.random.default_rng(p['streams']['training_noise']+args.seed)
    mask_checks=0;log_checks=0;models=[];event_index=0
    for row in range(2300):
        order=tie_rng.permutation(81)
        masks={}
        active=[a for a in p['arms'] if row<p['arm_rows'][a]]
        for arm in active:
            state=states[arm];event=events[event_index];event_index+=1
            assert event['row']==row and event['arm']==arm and event['finalized_before_all_current_reveals']
            before=hashlib.sha256(state['joint'].tobytes()+state['excluded'].tobytes()).hexdigest();assert before==event['past_state_sha256']
            mask=reference_mask(arm,state['joint'],state['excluded'],order);assert np.array_equal(mask,archives[arm]['given'][row])
            assert hashlib.sha256(mask.tobytes()).hexdigest()==event['mask_sha256'];masks[arm]=mask;mask_checks+=1
        # Independently regenerate current potentials only after all masks match.
        flip=noise_rng.random(81)<.01;offset=noise_rng.integers(1,9,81,dtype=np.int8)
        assert np.array_equal(flip,noise['flips'][row]) and np.array_equal(offset,noise['offsets'][row])
        expected=(clean[row]+flip*offset)%9
        for arm in active:
            state=states[arm];mask=masks[arm];values=archives[arm]['values'][row]
            assert np.array_equal(values[mask],expected[mask]);state['joint']+=(mask[:,None]&mask[None,:]).astype(np.int32)
            for digit in range(9):
                cells=np.flatnonzero(mask&(values==digit));state['equal'][np.ix_(cells,cells)]+=1
            m=state['joint'][pair];c=state['equal'][pair]
            eta=rates[arm];null=eta/4-9*eta**2/64;alternative=Fraction(1,9)
            la=math.log(float(alternative/null));lb=math.log(float((1-alternative)/(1-null)));log=m*lb+c*(la-lb)
            assert np.allclose(log,archives[arm]['log_lr'][row],rtol=0,atol=2e-12);log_checks+=len(log)
            first=state['first_crossing'];first[(first<0)&(c>=threshold[arm][m])]=row+1
            exclude=first>=0
            state['excluded'][pair]=exclude;state['excluded'][(jj,ii)]=exclude
            dropped=state['first_excluded'];dropped[(dropped<0)&exclude]=row+1
            if row+1==p['arm_rows'][arm]:
                entry=next(x for x in receipt['models'] if x['arm']==arm and x['n']==row+1)
                assert sha(ROOT/entry['path'])==entry['sha256'];z=np.load(ROOT/entry['path'])
                assert np.array_equal(z['support'],~exclude) and np.array_equal(z['joint_count'],m) and np.array_equal(z['equal_count'],c)
                assert np.array_equal(z['first_crossing'],first) and np.array_equal(z['first_excluded'],dropped)
                models.append({'arm':arm,'n':row+1,'support':z['support'].copy(),'m':m.copy()})
    assert event_index==8320 and mask_checks==8320 and log_checks==26956800
    for arm,state in states.items():
        assert np.array_equal(state['first_crossing'],archives[arm]['first_crossing'])
        assert np.array_equal(state['first_excluded'],archives[arm]['first_excluded'])
    # Geometry enters only here, after all masks, counts, paths and supports pass.
    peer=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert peer.sum()==810
    rows=[]
    for m in models:
        support=m['support'];missing=int((peer&~support).sum());extra=int((~peer&support).sum())
        rows.append({'seed':args.seed,'arm':m['arm'],'n':m['n'],'missing_true_edges':missing,'extra_edges':extra,
            'exact_graph':missing==extra==0,'joint_min':int(m['m'].min()),'joint_max':int(m['m'].max()),
            'support_bytes_sha256':hashlib.sha256(support.tobytes()).hexdigest()})
    r={'status':'validated','utc':now(),'seed':args.seed,'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),
        'receipt_sha256':sha(d/'receipt.json'),'independent_masks_replayed':mask_checks,'log_lr_values_verified':log_checks,
        'independent_noise_rows_verified':2300,'supports_verified':4,'first_crossing_paths_verified':12960,
        'calibration_audit':calibration_checks,'decimal_threshold_checks':threshold_checks,'rows':rows,'elapsed_s':time.monotonic()-start,'numpy':np.__version__}
    save(d/'audit.json',r);print({'seed':args.seed,'mask_checks':mask_checks,'log_checks':log_checks,'elapsed_s':r['elapsed_s']},flush=True)


if __name__=='__main__':main()
