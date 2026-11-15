"""Repeated-query calibration and predictable-mask acquisition, with full receipts."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from calibrated_common import paths,load_protocol
from calibration_estimator import fit_disagreements
from calibrated_policy import choose_mask,learn_queried,calibration,crossing_thresholds


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--pilot',action='store_true');args=ap.parse_args();start=time.monotonic()
    assert np.__version__=='2.3.4';out,pp=paths(args.pilot);p=load_protocol(args.pilot);assert args.seed in p['seeds']
    m=json.loads((ROOT/p['inputs']).read_text());entry=next(e for e in m['inputs'] if e['seed']==args.seed)
    arrays={}
    for role in ['training','calibration']:
        path=ROOT/entry[role+'_path'];assert sha(path)==entry[role+'_sha256'];z=np.load(path);arrays[role]={k:z[k] for k in z.files}
    d=out/f's{args.seed}';d.mkdir(exist_ok=False);streams=p['streams'];seed=args.seed
    cellrng=np.random.default_rng(streams['calibration_cell']+seed);nrng=np.random.default_rng(streams['calibration_noise']+seed)
    cells=cellrng.integers(0,81,10000);flip=nrng.random((10000,2))<.01;offset=nrng.integers(1,9,(10000,2),dtype=np.int8)
    clean_cal=arrays['calibration']['solutions'][np.arange(10000),cells];observed=(clean_cal[:,None]+flip*offset)%9
    disagreements=observed[:,0]!=observed[:,1]
    # The estimator receives only indicators, never clean labels, cells or source identities.
    fitted=fit_disagreements(disagreements)
    np.savez_compressed(d/'calibration_observations.npz',cells=cells,labels=observed,disagreements=disagreements,
        source_row=arrays['calibration']['source_row'],flips=flip,offsets=offset)
    eta=Fraction(fitted['eta_upper_numerator'],fitted['eta_upper_denominator'])
    fitted.update(utc=now(),estimator_sha256=sha(HERE/'calibration_estimator.py'),observations_sha256=sha(d/'calibration_observations.npz'),
        postfit_covers_actual_eta=eta>=Fraction(1,100),diagnostic_used_for_selection=False,estimator_input='10000 binary disagreement indicators only',calibration_queries=20000)
    save(d/'calibration.json',fitted)
    rates={'exact':Fraction(1,100),'learned':eta,'fixed_upper':Fraction(1,50)}
    thresholds={a:crossing_thresholds(rates[a],1500) for a in p['arms']};logs={}
    for a in p['arms']:
        q0,ra,rb,fallback=calibration(rates[a]);logs[a]=(math.log(float(ra)),math.log(float(rb)))
    save(d/'thresholds.json',{'boundary':648000,'rates':{a:str(v) for a,v in rates.items()},'thresholds':{a:v.tolist() for a,v in thresholds.items()}})
    states={a:{'joint':np.zeros((81,81),np.int32),'equal':np.zeros((81,81),np.int32),'excluded':np.zeros((81,81),bool),
        'given':np.zeros((1500,81),bool),'values':np.full((1500,81),-1,np.int8),'log_lr':np.zeros((1500,3240)),
        'first_crossing':np.full(3240,-1,np.int32),'first_excluded':np.full(3240,-1,np.int32)} for a in p['arms']}
    pair=np.triu_indices(81,1);rng=np.random.default_rng(streams['ties']+seed);nrng=np.random.default_rng(streams['training_noise']+seed)
    noise_flip=np.zeros((1500,81),bool);noise_offset=np.zeros((1500,81),np.int8);events=[];models=[];checks=0
    for row in range(1500):
        order=rng.permutation(81);masks={}
        for arm,state in states.items():
            joint,excluded=state['joint'],state['excluded'];before=hashlib.sha256(joint.tobytes()+excluded.tobytes()).hexdigest()
            joint.setflags(write=False);excluded.setflags(write=False)
            mask=choose_mask(arm,joint,excluded,25,order);shadow=choose_mask(arm,joint,excluded,25,order)
            assert mask.sum()==25 and np.array_equal(mask,shadow) and before==hashlib.sha256(joint.tobytes()+excluded.tobytes()).hexdigest()
            masks[arm]=mask;events.append({'row':row,'arm':arm,'past_state_sha256':before,'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'finalized_before_all_current_reveals':True})
            joint.setflags(write=True);excluded.setflags(write=True);checks+=1
        flip=nrng.random(81)<.01;offset=nrng.integers(1,9,81,dtype=np.int8);noise_flip[row]=flip;noise_offset[row]=offset
        noisy=(arrays['training']['solutions'][row]+flip*offset)%9
        for arm,state in states.items():
            mask=masks[arm];values=noisy[mask];state['given'][row]=mask;state['values'][row,mask]=values
            learn_queried(state['joint'],state['equal'],mask,values);m=state['joint'][pair];c=state['equal'][pair]
            la,lb=logs[arm];state['log_lr'][row]=c*la+(m-c)*lb
            first=state['first_crossing'];first[(first<0)&(c>=thresholds[arm][m])]=row+1;exclude=first>=0
            state['excluded'][pair]=exclude;state['excluded'][(pair[1],pair[0])]=exclude
            dropped=state['first_excluded'];dropped[(dropped<0)&exclude]=row+1
            if row==1499:
                path=d/f'{arm}-n1500.npz';np.savez_compressed(path,support=~exclude,joint_count=m,equal_count=c,first_crossing=first,first_excluded=dropped)
                models.append({'arm':arm,'n':1500,'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
        if row%250==0:save(d/'progress.json',{'utc':now(),'row':row+1,'elapsed_s':time.monotonic()-start})
    archives=[]
    for arm,state in states.items():
        path=d/f'{arm}-observations.npz';np.savez_compressed(path,**{k:state[k] for k in ['given','values','log_lr','first_crossing','first_excluded']})
        archives.append({'arm':arm,'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    np.savez_compressed(d/'potential_noise.npz',flips=noise_flip,offsets=noise_offset);save(d/'events.json',events)
    r={'status':'complete','scope':'implementation_pilot' if args.pilot else 'fresh_confirmatory','utc':now(),'seed':seed,'protocol_sha256':sha(pp),
        'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'calibrated_policy.py'),'source_sha256':entry['training_sha256'],'calibration_source_sha256':entry['calibration_sha256'],
        'models':models,'observations':archives,'potential_noise_sha256':sha(d/'potential_noise.npz'),'events_sha256':sha(d/'events.json'),
        'calibration_sha256':sha(d/'calibration.json'),'thresholds_sha256':sha(d/'thresholds.json'),'masks_fixed_before_all_reveals':checks,'elapsed_s':time.monotonic()-start,'numpy':np.__version__}
    save(d/'models-n1500.json',{'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'calibrated_policy.py'),
        'models':[dict(x,seed=seed,support_bytes_sha256=hashlib.sha256(np.load(ROOT/x['path'])['support'].tobytes()).hexdigest()) for x in models]})
    save(d/'receipt.json',r);print({'seed':seed,'eta_upper':float(eta),'disagreements':fitted['disagreements'],'masks':checks,'elapsed_s':r['elapsed_s']},flush=True)


if __name__=='__main__':main()
