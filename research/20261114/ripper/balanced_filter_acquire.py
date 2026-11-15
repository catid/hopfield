"""Consume every query; rejected boards contribute no pair evidence."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from balanced_filter_common import paths,load_protocol
from balanced_filter_policy import accept_observed
from collapsed_noise_policy import collapse_draw,apply_collapse
from calibration_estimator import fit_count
from calibrated_policy import calibration,crossing_thresholds,learn_queried
from noisy_policy import choose_mask


def shared_calibration(bank,p,seed,clean,source_hash,protocol_hash):
    if (bank/'calibration_receipt.json').exists():
        receipt=json.loads((bank/'calibration_receipt.json').read_text())
        assert receipt['protocol_sha256']==protocol_hash and receipt['source_sha256']==source_hash
        for name,digest in receipt['artifact_sha256'].items():assert sha(bank/name)==digest
        fit=json.loads((bank/'calibration.json').read_text())['fit']
        return fit,np.array(json.loads((bank/'thresholds.json').read_text())['thresholds'])
    cells=np.random.default_rng(p['streams']['calibration_cell']+seed).integers(0,81,1000)
    selected=clean[np.arange(1000),cells];rng=np.random.default_rng(p['streams']['calibration_noise']+seed)
    uniforms=np.zeros(1000);digits=np.zeros(1000,np.int8);labels=np.zeros((1000,2),np.int8)
    for index in range(1000):
        uniform,digit=collapse_draw(rng);uniforms[index]=uniform;digits[index]=digit
        labels[index]=apply_collapse(selected[index:index+1],uniform,digit)[0]
    indicators=labels[:,0]!=labels[:,1];fit=fit_count(int(indicators.sum()),1000)
    eta=Fraction(fit['eta_upper_numerator'],fit['eta_upper_denominator'])
    fit.update(actual_eta='1/20',postfit_covers_actual_eta=eta>=Fraction(1,20),diagnostic_used_for_selection=False,calibration_queries=2000)
    np.savez_compressed(bank/'calibration_observations.npz',cells=cells,labels=labels,disagreements=indicators,uniforms=uniforms,digits=digits)
    save(bank/'calibration.json',{'fit':fit,'observations_sha256':sha(bank/'calibration_observations.npz'),'estimator_sha256':sha(HERE/'calibration_estimator.py')})
    threshold=crossing_thresholds(eta,1110);save(bank/'thresholds.json',{'eta':str(eta),'boundary':648000,'thresholds':threshold.tolist()})
    save(bank/'calibration_receipt.json',{'protocol_sha256':protocol_hash,'source_sha256':source_hash,'estimator_calls':1,
        'shared_across_both_arms':True,'artifact_sha256':{name:sha(bank/name) for name in ['calibration_observations.npz','calibration.json','thresholds.json']}})
    return fit,threshold


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--arm',choices=['unfiltered','filtered'],required=True)
    args=parser.parse_args();start=time.monotonic();assert np.__version__=='2.3.4';out,pp=paths();p=load_protocol();seed=args.seed;arm=args.arm
    assert seed in p['seeds'];entry=next(e for e in json.loads((ROOT/p['inputs']).read_text())['inputs'] if e['seed']==seed);source={}
    for role in ('calibration','training'):
        path=ROOT/entry[role+'_path'];assert sha(path)==entry[role+'_sha256'];source[role]=np.load(path)['solutions']
    bank=out/f's{seed}';bank.mkdir(exist_ok=True);d=bank/arm;d.mkdir(exist_ok=False)
    fit,threshold=shared_calibration(bank,p,seed,source['calibration'],entry['calibration_sha256'],sha(pp))
    eta=Fraction(fit['eta_upper_numerator'],fit['eta_upper_denominator']);_,a,b,_=calibration(eta);la,lb=math.log(float(a)),math.log(float(b))
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);excluded=np.zeros((81,81),bool);pair=np.triu_indices(81,1)
    first=np.full(3240,-1,np.int32);given=np.zeros((1110,81),bool);values=np.full((1110,81),-1,np.int8);logs=np.zeros((1110,3240));events=[]
    accepted=np.zeros(1110,bool);violations=np.zeros(1110,bool);uniforms=np.zeros(1110);digits=np.zeros(1110,np.int8)
    noise=np.random.default_rng(p['streams']['training_noise']+seed);ties=np.random.default_rng(p['streams']['ties']+seed)
    for row in range(1110):
        order=ties.permutation(81);before=hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        for array in (joint,equal,excluded):array.setflags(write=False)
        mask=choose_mask('exponential_e',joint,excluded,25,order)
        assert mask.sum()==25 and before==hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        for array in (joint,equal,excluded):array.setflags(write=True)
        uniform,digit=collapse_draw(noise);uniforms[row]=uniform;digits[row]=digit
        noisy=apply_collapse(source['training'][row],uniform,digit);observed=noisy[mask]
        given[row]=mask;values[row,mask]=observed
        violations[row]=not accept_observed(observed)
        accepted[row]=arm=='unfiltered' or not violations[row]
        if accepted[row]:learn_queried(joint,equal,mask,observed)
        else:assert before==hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        m=joint[pair];c=equal[pair];logs[row]=m*lb+c*(la-lb)
        first[(first<0)&(c>=threshold[m])]=row+1;drop=first>=0;excluded[pair]=drop;excluded[(pair[1],pair[0])]=drop
        events.append({'row':row,'past_state_sha256':before,'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),
            'mask_finalized_before_current_corruption':True,'filter_uses_only_observed_labels':True,'accepted':bool(accepted[row])})
    np.savez_compressed(d/'fit.npz',support=first<0,joint_count=m,equal_count=c,first_crossing=first)
    np.savez_compressed(d/'observations.npz',given=given,values=values,log_lr=logs,accepted=accepted,violates_digit_balance=violations)
    np.savez_compressed(d/'potential_noise.npz',uniforms=uniforms,digits=digits);save(d/'events.json',events)
    receipt={'status':'complete','utc':now(),'seed':seed,'arm':arm,'protocol_sha256':sha(pp),'collector_sha256':sha(__file__),
        'calibration_source_sha256':entry['calibration_sha256'],'training_source_sha256':entry['training_sha256'],
        'shared_calibration_receipt_sha256':sha(bank/'calibration_receipt.json'),'queries_per_arm':29750,'training_queries_spent':27750,
        'accepted_boards':int(accepted.sum()),'rejected_boards':int((~accepted).sum()),'elapsed_s':time.monotonic()-start,
        'artifact_sha256':{name:sha(d/name) for name in ['fit.npz','observations.npz','potential_noise.npz','events.json']}}
    save(d/'receipt.json',receipt);print(json.dumps({'seed':seed,'arm':arm,'accepted_boards':receipt['accepted_boards']}),flush=True)


if __name__=='__main__':main()
