"""Fixed total-cost nested calibration/training, no query after an arm stops."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from equal_budget_common import paths,load_protocol
from equal_budget_estimator import fit_indicators
from equal_budget_policy import choose_mask,learn_queried,calibration,crossing_thresholds


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--pilot',action='store_true');args=ap.parse_args();start=time.monotonic()
    assert np.__version__=='2.3.4';out,pp=paths(args.pilot);p=load_protocol(args.pilot);seed=args.seed;assert seed in p['seeds']
    manifest=json.loads((ROOT/p['inputs']).read_text());entry=next(e for e in manifest['inputs'] if e['seed']==seed)
    arrays={}
    for role in ['training','calibration']:
        path=ROOT/entry[role+'_path'];assert sha(path)==entry[role+'_sha256'];z=np.load(path);arrays[role]={k:z[k] for k in z.files}
    d=out/f's{seed}';d.mkdir(exist_ok=False);streams=p['streams']
    cells=np.random.default_rng(streams['calibration_cell']+seed).integers(0,81,10000)
    rng=np.random.default_rng(streams['calibration_noise']+seed);flips=rng.random((10000,2))<.01;offsets=rng.integers(1,9,(10000,2),dtype=np.int8)
    labels=(arrays['calibration']['solutions'][np.arange(10000),cells,None]+flips*offsets)%9;indicators=labels[:,0]!=labels[:,1]
    fits={};rates={'exact':Fraction(1,100),'fixed_upper':Fraction(1,50)}
    for arm in ['learned_small','learned_large']:
        count=p['arm_calibration_pairs'][arm]
        # Only this arm's fixed indicator prefix crosses the estimator interface.
        fit=fit_indicators(indicators[:count],count);eta=Fraction(fit['eta_upper_numerator'],fit['eta_upper_denominator'])
        fit.update(estimator_input=f'{count} binary disagreement indicators only',postfit_covers_actual_eta=eta>=Fraction(1,100),diagnostic_used_for_selection=False,calibration_queries=2*count)
        fits[arm]=fit;rates[arm]=eta
    np.savez_compressed(d/'calibration_observations.npz',cells=cells,labels=labels,disagreements=indicators,source_row=arrays['calibration']['source_row'],flips=flips,offsets=offsets)
    save(d/'calibration.json',{'utc':now(),'fits':fits,'estimator_sha256':sha(HERE/'equal_budget_estimator.py'),'base_estimator_sha256':sha(HERE/'calibration_estimator.py'),
        'observations_sha256':sha(d/'calibration_observations.npz'),'nested_prefixes':True})
    ns=p['arm_rows'];thresholds={a:crossing_thresholds(rates[a],ns[a]) for a in p['arms']};logs={}
    for arm in p['arms']:
        q0,ra,rb,fallback=calibration(rates[arm]);logs[arm]=(math.log(float(ra)),math.log(float(rb)))
    save(d/'thresholds.json',{'boundary':648000,'rates':{a:str(v) for a,v in rates.items()},'thresholds':{a:v.tolist() for a,v in thresholds.items()}})
    states={a:{'joint':np.zeros((81,81),np.int32),'equal':np.zeros((81,81),np.int32),'excluded':np.zeros((81,81),bool),
        'given':np.zeros((ns[a],81),bool),'values':np.full((ns[a],81),-1,np.int8),'log_lr':np.zeros((ns[a],3240)),
        'first_crossing':np.full(3240,-1,np.int32),'first_excluded':np.full(3240,-1,np.int32)} for a in p['arms']}
    pair=np.triu_indices(81,1);rng=np.random.default_rng(streams['ties']+seed);nrng=np.random.default_rng(streams['training_noise']+seed)
    noise_flip=np.zeros((2300,81),bool);noise_offset=np.zeros((2300,81),np.int8);events=[];models=[];checks=0
    for row in range(2300):
        order=rng.permutation(81);masks={};active=[a for a in p['arms'] if row<ns[a]]
        for arm in active:
            state=states[arm];joint,excluded=state['joint'],state['excluded'];before=hashlib.sha256(joint.tobytes()+excluded.tobytes()).hexdigest()
            joint.setflags(write=False);excluded.setflags(write=False)
            mask=choose_mask(arm,joint,excluded,25,order);shadow=choose_mask(arm,joint,excluded,25,order)
            assert mask.sum()==25 and np.array_equal(mask,shadow) and before==hashlib.sha256(joint.tobytes()+excluded.tobytes()).hexdigest()
            masks[arm]=mask;events.append({'row':row,'arm':arm,'past_state_sha256':before,'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'finalized_before_all_current_reveals':True})
            joint.setflags(write=True);excluded.setflags(write=True);checks+=1
        flips=nrng.random(81)<.01;offsets=nrng.integers(1,9,81,dtype=np.int8);noise_flip[row]=flips;noise_offset[row]=offsets
        noisy=(arrays['training']['solutions'][row]+flips*offsets)%9
        for arm in active:
            state=states[arm];mask=masks[arm];values=noisy[mask];state['given'][row]=mask;state['values'][row,mask]=values
            learn_queried(state['joint'],state['equal'],mask,values);m=state['joint'][pair];c=state['equal'][pair]
            la,lb=logs[arm];state['log_lr'][row]=c*la+(m-c)*lb
            first=state['first_crossing'];first[(first<0)&(c>=thresholds[arm][m])]=row+1;exclude=first>=0
            state['excluded'][pair]=exclude;state['excluded'][(pair[1],pair[0])]=exclude
            dropped=state['first_excluded'];dropped[(dropped<0)&exclude]=row+1
            if row+1==ns[arm]:
                path=d/f'{arm}-n{ns[arm]}.npz';np.savez_compressed(path,support=~exclude,joint_count=m,equal_count=c,first_crossing=first,first_excluded=dropped)
                models.append({'arm':arm,'n':ns[arm],'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
        if row%250==0:save(d/'progress.json',{'utc':now(),'row':row+1,'elapsed_s':time.monotonic()-start})
    archives=[]
    for arm,state in states.items():
        path=d/f'{arm}-observations.npz';np.savez_compressed(path,**{k:state[k] for k in ['given','values','log_lr','first_crossing','first_excluded']})
        archives.append({'arm':arm,'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    np.savez_compressed(d/'potential_noise.npz',flips=noise_flip,offsets=noise_offset);save(d/'events.json',events)
    r={'status':'complete','scope':'implementation_pilot' if args.pilot else 'fresh_confirmatory','utc':now(),'seed':seed,'protocol_sha256':sha(pp),
        'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'equal_budget_policy.py'),'source_sha256':entry['training_sha256'],'calibration_source_sha256':entry['calibration_sha256'],
        'models':models,'observations':archives,'potential_noise_sha256':sha(d/'potential_noise.npz'),'events_sha256':sha(d/'events.json'),
        'calibration_sha256':sha(d/'calibration.json'),'thresholds_sha256':sha(d/'thresholds.json'),'masks_fixed_before_all_reveals':checks,'label_costs':p['label_costs'],
        'elapsed_s':time.monotonic()-start,'numpy':np.__version__}
    assert checks==8320
    save(d/'models.json',{'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'equal_budget_policy.py'),
        'models':[dict(x,seed=seed,support_bytes_sha256=hashlib.sha256(np.load(ROOT/x['path'])['support'].tobytes()).hexdigest()) for x in models]})
    save(d/'receipt.json',r);print({'seed':seed,'eta_upper':{a:fits[a]['eta_upper'] for a in fits},'masks':checks,'elapsed_s':r['elapsed_s']},flush=True)


if __name__=='__main__':main()
