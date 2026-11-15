"""Every new adaptive mask, latent draw, cumulative LR and first crossing."""
import argparse,hashlib,json,math,time
from fractions import Fraction
import numpy as np
from run import ROOT,HERE,sha,save,now
from adaptive_null_common import paths,load_protocol
from fast_reference_mask import reference_mask
from audit_correlated_noise_acquisition import increment
from audit_permuted_noise_acquisition import independent_draw

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);seed=parser.parse_args().seed;started=time.monotonic()
    out,pp=paths();p=load_protocol();d=out/f's{seed}';r=json.loads((d/'receipt.json').read_text());assert r['status']=='complete'and r['protocol_sha256']==sha(pp)and r['collector_sha256']==sha(HERE/'adaptive_null_acquire.py')and r['queries']==27750 and r['new_calibration_queries']==0
    for name,digest in r['artifact_sha256'].items():assert sha(d/name)==digest
    entry=next(e for e in json.loads((ROOT/p['inputs']).read_text())['inputs']if e['seed']==seed);path=ROOT/entry['training_path'];assert sha(path)==entry['training_sha256']==r['training_source_sha256'];clean=np.load(path)['solutions']
    reg=json.loads((out/'input_registration.json').read_text());assert reg['threshold_sha256']==sha(out/'thresholds.npz')and reg['decimal_threshold_checks']==1111
    threshold=np.load(out/'thresholds.npz')['threshold'];q0=Fraction(9,1600);q1=Fraction(1,9);a,b=q1/q0,(1-q1)/(1-q0);la,lb=math.log(float(a)),math.log(float(b))
    obs=np.load(d/'observations.npz');noise=np.load(d/'potential_noise.npz');events=json.loads((d/'events.json').read_text())
    olddir=ROOT/'artifacts/research/20261114/ripper/permuted_noise'/f's{seed}';oldr=json.loads((olddir/'receipt.json').read_text());parent=json.loads((ROOT/'artifacts/research/20261114/ripper/permuted_noise/audit.json').read_text());assert sha(olddir/'receipt.json')==parent['artifact_sha256'][str((olddir/'receipt.json').relative_to(ROOT))];assert sha(olddir/'observations.npz')==oldr['artifact_sha256']['observations.npz'];assert sha(olddir/'potential_noise.npz')==oldr['artifact_sha256']['potential_noise.npz'];oldnoise=np.load(olddir/'potential_noise.npz');oldobs=np.load(olddir/'observations.npz')
    for key in ['uniforms','permutations']:assert np.array_equal(noise[key],oldnoise[key])
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);excluded=np.zeros((81,81),bool);first=np.full(3240,-1,np.int32)
    pair=np.triu_indices(81,1);i,j=pair;truth=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    ties=np.random.default_rng(p['streams']['ties']+seed);rng=np.random.default_rng(p['streams']['training_noise']+seed);errors=corrupt=changed=0
    for row in range(1110):
        order=ties.permutation(81);e=events[row];assert e['row']==row and e['mask_finalized_before_current_corruption']and e['filter_accepted']
        assert e['past_state_sha256']==hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
        mask=reference_mask('exposure_exact',joint,excluded,order);assert np.array_equal(mask,obs['given'][row])and e['mask_sha256']==hashlib.sha256(mask.tobytes()).hexdigest();changed+=int(not np.array_equal(mask,oldobs['given'][row]))
        uniform,permutation,event=independent_draw(rng);corrupt+=int(event)
        assert noise['uniforms'][row]==uniform and np.array_equal(noise['permutations'][row],permutation)
        assert all(np.count_nonzero(clean[row]==digit)==9 for digit in range(9))
        expected=clean[row,permutation]if event else clean[row];values=obs['values'][row]
        assert np.array_equal(values[mask],expected[mask])and(values[~mask]==-1).all()and mask.sum()==25
        assert max(np.count_nonzero(values[mask]==digit)for digit in range(9))<=9
        errors+=int((values[mask]!=clean[row,mask]).sum());increment(joint,equal,mask,values)
        m,c=joint[pair],equal[pair];assert np.allclose(obs['log_lr'][row],m*lb+c*(la-lb),rtol=0,atol=2e-12)
        first[(first<0)&(c>=threshold[m])]=row+1;drop=first>=0;excluded[pair]=drop;excluded[(pair[1],pair[0])]=drop
    model=np.load(d/'fit.npz');support=first<0
    for key,value in [('support',support),('joint_count',m),('equal_count',c),('first_crossing',first)]:assert np.array_equal(model[key],value)
    missing=int((truth&~support).sum());extras=int((~truth&support).sum())
    result=dict(status='validated',seed=seed,arm='adaptive_exact_law',protocol_sha256=sha(pp),auditor_sha256=sha(__file__),receipt_sha256=sha(d/'receipt.json'),independent_masks_replayed=1110,cumulative_log_lr_values_verified=3596400,first_crossing_paths_verified=3240,queries_spent=27750,new_calibration_queries=0,training_error_queries=errors,permutation_events=corrupt,changed_masks_vs_archived=changed,missing_true_edges=missing,extra_edges=extras,all_true_retained=missing==0,exact_graph=missing==extras==0,support_bytes_sha256=hashlib.sha256(support.tobytes()).hexdigest(),elapsed_s=time.monotonic()-started)
    save(d/'audit.json',result);print(json.dumps(dict(seed=seed,missing=missing,extras=extras)),flush=True)
if __name__=='__main__':main()
