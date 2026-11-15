"""Collapse-corrupted query acquisition, with mask fixed before the board latent."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import time
import numpy as np
from run import ROOT, HERE, sha, save, now
from collapsed_noise_common import paths, load_protocol
from collapsed_noise_policy import collapse_draw, apply_collapse
from calibration_estimator import fit_count
from calibrated_policy import calibration, crossing_thresholds, learn_queried
from noisy_policy import choose_mask


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    start=time.monotonic();assert np.__version__=='2.3.4';out,pp=paths();p=load_protocol();seed=args.seed;assert seed in p['seeds']
    entry=next(e for e in json.loads((ROOT/p['inputs']).read_text())['inputs'] if e['seed']==seed);source={}
    for role in ('calibration','training'):
        path=ROOT/entry[role+'_path'];assert sha(path)==entry[role+'_sha256'];source[role]=np.load(path)['solutions']
    d=out/f's{seed}';d.mkdir(exist_ok=False);streams=p['streams']
    cells=np.random.default_rng(streams['calibration_cell']+seed).integers(0,81,1000)
    clean=source['calibration'][np.arange(1000),cells];rng=np.random.default_rng(streams['calibration_noise']+seed)
    uniforms=np.zeros(1000);digits=np.zeros(1000,np.int8);labels=np.zeros((1000,2),np.int8)
    for index in range(1000):
        uniform,digit=collapse_draw(rng);uniforms[index]=uniform;digits[index]=digit
        labels[index]=apply_collapse(clean[index:index+1],uniform,digit)[0]
    indicators=labels[:,0]!=labels[:,1];fit=fit_count(int(indicators.sum()),1000)
    eta=Fraction(fit['eta_upper_numerator'],fit['eta_upper_denominator'])
    fit.update(actual_eta='1/20',postfit_covers_actual_eta=eta>=Fraction(1,20),diagnostic_used_for_selection=False,calibration_queries=2000)
    np.savez_compressed(d/'calibration_observations.npz',cells=cells,labels=labels,disagreements=indicators,uniforms=uniforms,digits=digits)
    save(d/'calibration.json',{'fit':fit,'observations_sha256':sha(d/'calibration_observations.npz'),'estimator_sha256':sha(HERE/'calibration_estimator.py')})
    threshold=crossing_thresholds(eta,1110);_,a,b,_=calibration(eta);la,lb=math.log(float(a)),math.log(float(b))
    save(d/'thresholds.json',{'eta':str(eta),'boundary':648000,'thresholds':threshold.tolist()})
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);excluded=np.zeros((81,81),bool);pair=np.triu_indices(81,1)
    first=np.full(3240,-1,np.int32);given=np.zeros((1110,81),bool);values=np.full((1110,81),-1,np.int8);logs=np.zeros((1110,3240));events=[]
    noise=np.random.default_rng(streams['training_noise']+seed);ties=np.random.default_rng(streams['ties']+seed)
    uniforms=np.zeros(1110);digits=np.zeros(1110,np.int8)
    for row in range(1110):
        order=ties.permutation(81);before=hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        for array in (joint,equal,excluded):array.setflags(write=False)
        mask=choose_mask('exponential_e',joint,excluded,25,order)
        assert mask.sum()==25 and before==hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        events.append({'row':row,'past_state_sha256':before,'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'mask_finalized_before_current_corruption':True})
        for array in (joint,equal,excluded):array.setflags(write=True)
        uniform,digit=collapse_draw(noise);uniforms[row]=uniform;digits[row]=digit
        noisy=apply_collapse(source['training'][row],uniform,digit);given[row]=mask;values[row,mask]=noisy[mask]
        learn_queried(joint,equal,mask,noisy[mask]);m=joint[pair];c=equal[pair];logs[row]=m*lb+c*(la-lb)
        first[(first<0)&(c>=threshold[m])]=row+1;drop=first>=0;excluded[pair]=drop;excluded[(pair[1],pair[0])]=drop
    np.savez_compressed(d/'fit.npz',support=first<0,joint_count=m,equal_count=c,first_crossing=first)
    np.savez_compressed(d/'observations.npz',given=given,values=values,log_lr=logs)
    np.savez_compressed(d/'potential_noise.npz',uniforms=uniforms,digits=digits);save(d/'events.json',events)
    receipt={'status':'complete','utc':now(),'seed':seed,'arm':'collapse','protocol_sha256':sha(pp),'collector_sha256':sha(__file__),
        'calibration_source_sha256':entry['calibration_sha256'],'training_source_sha256':entry['training_sha256'],'queries':29750,
        'mask_finalized_before_current_noise':1110,'elapsed_s':time.monotonic()-start,
        'artifact_sha256':{name:sha(d/name) for name in ['fit.npz','observations.npz','potential_noise.npz','events.json','calibration.json','calibration_observations.npz','thresholds.json']}}
    save(d/'receipt.json',receipt);print(json.dumps({'seed':seed,'eta_upper':fit['eta_upper'],'elapsed_s':receipt['elapsed_s']}),flush=True)


if __name__=='__main__':main()
