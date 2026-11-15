"""One learned exposure fit, shared across three frozen inference weightings."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from confidence_weight_common import paths,load_protocol
from confidence_weight_estimator import fit_indicators
from confidence_weight_policy import weights
from calibrated_policy import calibration,crossing_thresholds,learn_queried
from noisy_policy import choose_mask


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--pilot',action='store_true');args=ap.parse_args();start=time.monotonic()
    assert np.__version__=='2.3.4';out,pp=paths(args.pilot);p=load_protocol(args.pilot);seed=args.seed;assert seed in p['seeds']
    manifest=json.loads((ROOT/p['inputs']).read_text());entry=next(e for e in manifest['inputs'] if e['seed']==seed);arrays={}
    for role in ('calibration','training'):
        path=ROOT/entry[role+'_path'];assert sha(path)==entry[role+'_sha256'];z=np.load(path);arrays[role]={k:z[k] for k in z.files}
    d=out/f's{seed}';d.mkdir(exist_ok=False);streams=p['streams']
    permutation=np.random.default_rng(streams['weight_shuffle']+seed).permutation(3240)
    np.save(d/'shuffle_permutation.npy',permutation)
    cells=np.random.default_rng(streams['calibration_cell']+seed).integers(0,81,1000)
    rng=np.random.default_rng(streams['calibration_noise']+seed);uniforms=rng.random((1000,2));offsets=rng.integers(1,9,(1000,2),dtype=np.int8)
    labels=(arrays['calibration']['solutions'][np.arange(1000),cells,None]+(uniforms<.05)*offsets)%9;indicators=labels[:,0]!=labels[:,1]
    fit=fit_indicators(indicators);eta=Fraction(fit['eta_upper_numerator'],fit['eta_upper_denominator'])
    fit.update(actual_eta='1/20',postfit_covers_actual_eta=eta>=Fraction(1,20),diagnostic_used_for_selection=False,calibration_queries=2000)
    np.savez_compressed(d/'calibration_observations.npz',cells=cells,uniforms=uniforms,offsets=offsets,source_row=arrays['calibration']['source_row'],labels=labels,disagreements=indicators)
    save(d/'calibration.json',{'utc':now(),'fit':fit,'estimator_sha256':sha(HERE/'confidence_weight_estimator.py'),'base_estimator_sha256':sha(HERE/'calibration_estimator.py'),
        'observations_sha256':sha(d/'calibration_observations.npz'),'shared_across_all_inference_arms':True,'estimator_calls':1})
    threshold=crossing_thresholds(eta,2220);q0,a,b,fallback=calibration(eta);la,lb=math.log(float(a)),math.log(float(b))
    save(d/'thresholds.json',{'boundary':648000,'eta':str(eta),'thresholds':threshold.tolist()})
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);excluded=np.zeros((81,81),bool);pair=np.triu_indices(81,1)
    given=np.zeros((2220,81),bool);values=np.full((2220,81),-1,np.int8);logs=np.zeros((2220,3240));first=np.full(3240,-1,np.int32)
    noise_uniform=np.zeros((2220,81));noise_offset=np.zeros((2220,81),np.int8);events=[]
    ties=np.random.default_rng(streams['ties']+seed);noise=np.random.default_rng(streams['training_noise']+seed)
    for row in range(2220):
        order=ties.permutation(81);before=hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        for value in (joint,equal,excluded):value.setflags(write=False)
        mask=choose_mask('exponential_e',joint,excluded,25,order)
        assert mask.sum()==25 and before==hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        events.append({'row':row,'past_state_sha256':before,'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'finalized_before_current_noise':True})
        for value in (joint,equal,excluded):value.setflags(write=True)
        u=noise.random(81);o=noise.integers(1,9,81,dtype=np.int8);noise_uniform[row]=u;noise_offset[row]=o
        noisy=(arrays['training']['solutions'][row]+(u<.05)*o)%9;given[row]=mask;values[row,mask]=noisy[mask]
        learn_queried(joint,equal,mask,noisy[mask]);m=joint[pair];c=equal[pair];logs[row]=m*lb+c*(la-lb)
        first[(first<0)&(c>=threshold[m])]=row+1;drop=first>=0;excluded[pair]=drop;excluded[(pair[1],pair[0])]=drop
        if row%250==0:save(d/'progress.json',{'utc':now(),'row':row+1,'elapsed_s':time.monotonic()-start})
    fit_path=d/'fit.npz';support=first<0
    np.savez_compressed(fit_path,support=support,joint_count=m,equal_count=c,first_crossing=first)
    np.savez_compressed(d/'observations.npz',given=given,values=values,log_lr=logs)
    np.savez_compressed(d/'potential_noise.npz',uniforms=noise_uniform,offsets=noise_offset);save(d/'events.json',events)
    for value in (support,m,c,permutation):value.setflags(write=False)
    before=hashlib.sha256(support.tobytes()+m.tobytes()+c.tobytes()+permutation.tobytes()).hexdigest();assigned=weights(support,m,c,eta,permutation)
    assert before==hashlib.sha256(support.tobytes()+m.tobytes()+c.tobytes()+permutation.tobytes()).hexdigest();models=[]
    for arm,weight16 in assigned.items():
        path=d/f'{arm}-n2220.npz';np.savez_compressed(path,support=support,weight16=weight16)
        models.append({'arm':arm,'n':2220,'seed':seed,'path':str(path.relative_to(ROOT)),'sha256':sha(path),
            'support_bytes_sha256':hashlib.sha256(support.tobytes()).hexdigest(),'weight16_bytes_sha256':hashlib.sha256(weight16.tobytes()).hexdigest()})
    receipt={'status':'complete','utc':now(),'seed':seed,'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'confidence_weight_policy.py'),
        'source_sha256':entry['training_sha256'],'calibration_source_sha256':entry['calibration_sha256'],'models':models,'fit_path':str(fit_path.relative_to(ROOT)),
        'artifact_sha256':{name:sha(d/name) for name in ['fit.npz','observations.npz','potential_noise.npz','events.json','calibration.json','thresholds.json','shuffle_permutation.npy']},
        'masks_fixed_before_noise':2220,'read_only_history_fields':['joint','equal','excluded'],'one_fit_shared_across_arms':True,'weight_inputs_readonly_and_unchanged':True,
        'shuffle_generated_before_calibration_and_training':True,'queries_per_shared_fit':57500,'elapsed_s':time.monotonic()-start,'numpy':np.__version__}
    save(d/'models.json',{'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),'policy_sha256':receipt['policy_sha256'],'models':models})
    save(d/'receipt.json',receipt);print({'seed':seed,'eta_upper':fit['eta_upper'],'weak_weights':int((assigned['evidence']==1).sum()),'elapsed_s':receipt['elapsed_s']},flush=True)


if __name__=='__main__':main()
