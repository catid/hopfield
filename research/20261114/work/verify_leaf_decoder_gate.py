"""Public-only H2ah source pairing, scaling, graph and final-state audit."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import flip,LD
from verify_hard_negative_feasibility import valid_shortest


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory,training,baseline):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());assert sha(training/'training-arrays.npz')==p['training_manifest']['arrays_sha256'];assert sha(baseline/'mining-audit.json')==p['retained_baseline']['mining_audit_sha256'];cast_gate=json.loads((directory/'cast-gate.json').read_text());rows=[]
    with np.load(training/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
    assert tokens.shape==labels.shape==(1000,30,30) and all(valid_shortest(t,y) for t,y in zip(tokens,labels))
    for seed in range(3):
        receipt=json.loads((directory/f'mined-seed{seed}.json').read_text());old_receipt=json.loads((baseline/f'mined-seed{seed}.json').read_text());assert sha(directory/f'mined-seed{seed}.npz')==receipt['outputs_sha256'] and receipt['protocol_sha256']==sha(directory/'protocol.json');assert sha(baseline/f'mined-seed{seed}.npz')==old_receipt['outputs_sha256']
        with np.load(directory/f'mined-seed{seed}.npz') as z:pred=z['prediction'];stopped=z['stopped'];finite=z['finite'];saved=z['minimum_flip_margin'];sweeps=z['sweeps']
        with np.load(baseline/f'mined-seed{seed}.npz') as z:old=z['prediction'];old_stopped=z['stopped'];old_finite=z['finite']
        assert pred.shape==old.shape==(1000,30,30) and pred.dtype==bool and np.array_equal(pred[tokens!=1],(tokens>=2)[tokens!=1]);assert stopped.all() and finite.all() and old_stopped.all() and old_finite.all()
        initial=directory/f'initial-seed{seed}.npz';assert sha(initial)==p['initial_models'][seed]['sha256']
        with np.load(initial) as z:original=np.concatenate([z['T'].ravel(),z['b'],z['kappa']]);mu=z['mu_offsets'];ao=z['aoffs']
        candidate=directory/f'parent-theta-seed{seed}.npy';assert sha(candidate)==cast_gate['records'][seed]['source_candidate_sha256'];theta=np.load(candidate);scale=float(abs(original).max())/float(abs(theta).max());cast=np.asarray(theta*scale,np.float32);cp=directory/f'leaf-scaled-seed{seed}.npy';assert sha(cp)==p['coefficient_arrays'][str(seed)]['sha256'] and np.array_equal(cast,np.load(cp));theta=cast.astype(LD)
        ap=training/f'train-readout-seed{seed}.npz';assert sha(ap)==p['readouts'][str(seed)]['readout_sha256']
        with np.load(ap) as z:attention=z['couplings']
        maxerror=0.;minimum=np.inf;label_min=np.inf
        for lo in range(0,1000,64):
            t=tokens[lo:lo+64];a=attention[lo:lo+64].astype(LD);delta=flip(t,pred[lo:lo+64],a,theta,mu,ao);m=np.where(t==1,delta,np.inf).reshape(len(t),-1).min(1);assert (m>=-1e-8).all();minimum=min(minimum,float(m.min()));maxerror=max(maxerror,float(abs(m-saved[lo:lo+64]).max()));label_min=min(label_min,float(flip(t,labels[lo:lo+64],a,theta,mu,ao)[t==1].min()))
        assert maxerror<1e-10 and label_min>0
        new_valid=np.array([valid_shortest(t,b) for t,b in zip(tokens,pred)]);old_valid=np.array([valid_shortest(t,b) for t,b in zip(tokens,old)])
        with np.load(directory/f'negatives-seed{seed}.npz') as z:assert np.array_equal(z['indices'],np.flatnonzero(~new_valid)) and np.array_equal(z['boards'],pred[~new_valid])
        assert int((~old_valid).sum())==p['retained_baseline']['wrong_by_model'][seed]
        rows.append({'seed':seed,'inputs':1000,'baseline_wrong':int((~old_valid).sum()),'new_wrong':int((~new_valid).sum()),'paired_gains':int((new_valid&~old_valid).sum()),'paired_losses':int((~new_valid&old_valid).sum()),'wrong_indices':np.flatnonzero(~new_valid).tolist(),'timeouts':int((~stopped).sum()),'nonfinite':int((~finite).sum()),'minimum_final_flip_margin':minimum,'minimum_label_flip_margin':label_min,'maximum_direct_vs_saved_margin_error':maxerror,'maximum_sweeps':int(sweeps.max()),'exact_norm_match_and_cast':True})
    old=sum(r['baseline_wrong'] for r in rows);new=sum(r['new_wrong'] for r in rows);assert old==580
    return {'status':'passed','hypothesis':'H2ah','decision':'descriptive_target_met' if new<=290 else 'descriptive_target_not_met','rows':rows,'source_inputs_per_model':1000,'source_model_input_trajectories':3000,'baseline_wrong':old,'new_wrong':new,'wrong_reduction_percent':100*(old-new)/old,'paired_gains':sum(r['paired_gains'] for r in rows),'paired_losses':sum(r['paired_losses'] for r in rows),'new_independent_test_inputs':0,'source_graph_states_checked':6000,'new_final_states_independently_checked':3000,'seconds':time.perf_counter()-started,'protocol_sha256':sha(directory/'protocol.json'),'verifier_sha256':sha(Path(__file__)),'scope':'Fixed training inputs, three models, all cases retained. Independent graph labels/outcomes, original norm match and single cast, all final flip certificates. Source wrong-count target descriptive; no generalization or independent test claim.'}


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);a.add_argument('training',type=Path);a.add_argument('baseline',type=Path);v=a.parse_args();print(json.dumps(verify(v.directory,v.training,v.baseline),indent=2))
