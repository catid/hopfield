"""Known-law null governs adaptive masks from the first observation."""
import argparse,hashlib,json,math,time
from fractions import Fraction
import numpy as np
from run import ROOT,HERE,sha,save,now
from adaptive_null_common import paths,load_protocol
from permuted_noise_policy import permutation_draw,apply_permutation
from balanced_filter_policy import accept_observed
from noisy_policy import choose_mask
from online_policy import learn_queried

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);seed=parser.parse_args().seed;started=time.monotonic()
    out,pp=paths();p=load_protocol();assert np.__version__=='2.3.4'and seed in p['seeds']
    entry=next(e for e in json.loads((ROOT/p['inputs']).read_text())['inputs']if e['seed']==seed);path=ROOT/entry['training_path'];assert sha(path)==entry['training_sha256']
    clean=np.load(path)['solutions'];d=out/f's{seed}';d.mkdir(exist_ok=False)
    threshold=np.load(out/'thresholds.npz')['threshold'];q0=Fraction(9,1600);q1=Fraction(1,9);a,b=q1/q0,(1-q1)/(1-q0);la,lb=math.log(float(a)),math.log(float(b))
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);excluded=np.zeros((81,81),bool);pair=np.triu_indices(81,1)
    first=np.full(3240,-1,np.int32);given=np.zeros((1110,81),bool);values=np.full((1110,81),-1,np.int8);logs=np.zeros((1110,3240));events=[]
    noise=np.random.default_rng(p['streams']['training_noise']+seed);ties=np.random.default_rng(p['streams']['ties']+seed)
    uniforms=np.zeros(1110);permutations=np.zeros((1110,81),np.uint8)
    for row in range(1110):
        order=ties.permutation(81);before=hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        for array in(joint,equal,excluded):array.setflags(write=False)
        mask=choose_mask('exponential_e',joint,excluded,25,order)
        assert mask.sum()==25 and before==hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        for array in(joint,equal,excluded):array.setflags(write=True)
        uniform,permutation=permutation_draw(noise);uniforms[row]=uniform;permutations[row]=permutation
        noisy=apply_permutation(clean[row],uniform,permutation);assert accept_observed(noisy[mask]);given[row]=mask;values[row,mask]=noisy[mask]
        learn_queried(joint,equal,mask,noisy[mask]);m,c=joint[pair],equal[pair];logs[row]=m*lb+c*(la-lb)
        first[(first<0)&(c>=threshold[m])]=row+1;drop=first>=0;excluded[pair]=drop;excluded[(pair[1],pair[0])]=drop
        events.append(dict(row=row,past_state_sha256=before,mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),mask_finalized_before_current_corruption=True,filter_accepted=True))
    np.savez_compressed(d/'fit.npz',support=first<0,joint_count=m,equal_count=c,first_crossing=first)
    np.savez_compressed(d/'observations.npz',given=given,values=values,log_lr=logs)
    np.savez_compressed(d/'potential_noise.npz',uniforms=uniforms,permutations=permutations);save(d/'events.json',events)
    receipt=dict(status='complete',utc=now(),seed=seed,arm='adaptive_exact_law',protocol_sha256=sha(pp),collector_sha256=sha(__file__),training_source_sha256=entry['training_sha256'],queries=27750,new_calibration_queries=0,mask_finalized_before_current_noise=1110,elapsed_s=time.monotonic()-started,
        artifact_sha256={n:sha(d/n)for n in ['fit.npz','observations.npz','potential_noise.npz','events.json']})
    save(d/'receipt.json',receipt);print(json.dumps(dict(seed=seed,status='complete')),flush=True)
if __name__=='__main__':main()
