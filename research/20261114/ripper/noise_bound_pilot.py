"""Bounded old-bank exploration, masks fixed before noisy labels are revealed."""
import argparse
import hashlib
import json
import math
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from noise_bound_policy import choose_mask,learn_queried,calibration

OUT=ROOT/'artifacts/research/20261114/ripper/noise_bound_pilot'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);args=ap.parse_args();start=time.monotonic()
    assert np.__version__=='2.3.4';pp=HERE/'protocol-noise-bound-pilot.json';p=json.loads(pp.read_text())
    assert sha(HERE/'noise_bound_policy.py')==p['policy_sha256']
    preflight=json.loads((ROOT/'artifacts/research/20261114/ripper/noise_bound_preflight.json').read_text())
    assert preflight['status']=='validated' and preflight['policy_sha256']==p['policy_sha256']
    threshold={a:np.array(v) for a,v in preflight['thresholds'].items()};mp=ROOT/p['manifest'];assert sha(mp)==p['manifest_sha256']
    manifest=json.loads(mp.read_text());entry=next(e for e in manifest['inputs'] if e['seed']==args.seed)
    path=ROOT/entry['completion_path'];assert sha(path)==entry['completion_sha256'];z=np.load(path);clean=z['solutions']
    d=OUT/f's{args.seed}';d.mkdir(parents=True,exist_ok=False)
    arms=p['arms'];states={}
    for arm in arms:states[arm]={'joint':np.zeros((81,81),np.int32),'equal':np.zeros((81,81),np.int32),
        'excluded':np.zeros((81,81),bool),'given':np.zeros((1500,81),bool),'values':np.full((1500,81),-1,np.int8),
        'log_lr':np.zeros((1500,3240)),'first_crossing':np.full(3240,-1,np.int32),'first_excluded':np.full(3240,-1,np.int32)}
    pair=np.triu_indices(81,1);rng=np.random.default_rng(20261125+args.seed);noise_rng=np.random.default_rng(20265000+args.seed)
    noise_flip=np.zeros((1500,81),bool);noise_offset=np.zeros((1500,81),np.int8);events=[];models=[];mask_checks=0
    for row in range(1500):
        order=rng.permutation(81);masks={}
        # Every mask is fixed before ANY arm receives current-row noisy labels.
        for arm,state in states.items():
            joint,excluded=state['joint'],state['excluded'];before=hashlib.sha256(joint.tobytes()+excluded.tobytes()).hexdigest()
            joint.setflags(write=False);excluded.setflags(write=False)
            mask=choose_mask(arm,joint,excluded,25,order);shadow=choose_mask(arm,joint,excluded,25,order)
            assert mask.sum()==25 and np.array_equal(mask,shadow)
            assert before==hashlib.sha256(joint.tobytes()+excluded.tobytes()).hexdigest()
            masks[arm]=mask;events.append({'row':row,'arm':arm,'past_state_sha256':before,'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'finalized_before_all_current_reveals':True})
            joint.setflags(write=True);excluded.setflags(write=True);mask_checks+=1
        flip=noise_rng.random(81)<.01;offset=noise_rng.integers(1,9,81,dtype=np.int8)
        noise_flip[row]=flip;noise_offset[row]=offset;noisy=(clean[row]+flip*offset)%9
        for arm,state in states.items():
            mask=masks[arm];values=noisy[mask];state['given'][row]=mask;state['values'][row,mask]=values
            learn_queried(state['joint'],state['equal'],mask,values)
            m=state['joint'][pair];c=state['equal'][pair]
            q0,ratio_a,ratio_b=calibration(arm)
            state['log_lr'][row]=c*math.log(float(ratio_a))+(m-c)*math.log(float(ratio_b))
            crossed=c>=threshold[arm][m];first=state['first_crossing'];first[(first<0)&crossed]=row+1
            exclude=first>=0
            state['excluded'][pair]=exclude;state['excluded'][(pair[1],pair[0])]=exclude
            dropped=state['first_excluded'];dropped[(dropped<0)&exclude]=row+1
            if row+1 in p['sizes']:
                out=d/f'{arm}-n{row+1}.npz';np.savez_compressed(out,support=~exclude,joint_count=m,equal_count=c,first_crossing=first,first_excluded=dropped)
                models.append({'arm':arm,'n':row+1,'path':str(out.relative_to(ROOT)),'sha256':sha(out)})
        if row%250==0:save(d/'progress.json',{'utc':now(),'row':row+1,'elapsed_s':time.monotonic()-start})
    archives=[]
    for arm,state in states.items():
        out=d/f'{arm}-observations.npz';np.savez_compressed(out,given=state['given'],values=state['values'],log_lr=state['log_lr'],first_crossing=state['first_crossing'],first_excluded=state['first_excluded'])
        archives.append({'arm':arm,'path':str(out.relative_to(ROOT)),'sha256':sha(out)})
    noise=d/'potential_noise.npz';np.savez_compressed(noise,flips=noise_flip,offsets=noise_offset)
    save(d/'events.json',events)
    r={'status':'complete','scope':'exploratory_old_banks','utc':now(),'seed':args.seed,'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),
        'policy_sha256':sha(HERE/'noise_bound_policy.py'),'preflight_sha256':sha(ROOT/'artifacts/research/20261114/ripper/noise_bound_preflight.json'),
        'source_sha256':sha(path),'models':models,'observations':archives,'potential_noise_sha256':sha(noise),
        'events_sha256':sha(d/'events.json'),'masks_fixed_before_all_reveals':mask_checks,'elapsed_s':time.monotonic()-start,'numpy':np.__version__}
    save(d/'receipt.json',r);print({'seed':args.seed,'masks':mask_checks,'elapsed_s':r['elapsed_s']},flush=True)


if __name__=='__main__':main()
