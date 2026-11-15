"""Norm-match H2ab hard-negative witnesses, cast once, verify allsource constraints; no freshtest."""
import argparse
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import larger_relaxation as L
import verify_training_feasibility as V

DEST=S.OUT/'hard_negative_transfer'
TRAIN=S.OUT/'contrastive_training'
NORM=S.OUT/'hard_negative_feasibility'


def prepare():
    DEST.mkdir(exist_ok=True);torch,_,_=S.imports();proof=json.loads((NORM/'independent-verification.json').read_text());assert proof['status']=='passed';records=[]
    with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
    for seed in range(3):
        model=S.model_for(seed,30,'original','cpu');original=np.concatenate([model.T.numpy().ravel(),model.b.numpy(),model.kappa.numpy()]);path=NORM/f'seed{seed}/candidate-theta.npy';theta=np.load(path);scale=float(np.max(abs(original)).astype(np.float64)/np.max(abs(theta)));cast=np.asarray(theta*scale,np.float32);assert cast.shape==(23,) and np.isfinite(cast).all()
        with torch.no_grad():model.T.copy_(torch.tensor(cast[:16].reshape(4,2,2)));model.b.copy_(torch.tensor(cast[16:18]));model.kappa.copy_(torch.tensor(cast[18:]));
        target=DEST/f'hard-negative-scaled-seed{seed}.pt'
        if target.exists():raise FileExistsError(target)
        torch.save(model.state_dict(),target);np.save(DEST/f'hard-negative-scaled-seed{seed}.npy',cast)
        with np.load(TRAIN/f'train-readout-seed{seed}.npz') as z:attention=z['couplings']
        t64=cast.astype(V.LD);muoffs=model.mu_offsets;aoffs=model.aoffs;minima=np.full(4,np.inf,V.LD);counts=np.zeros(4,np.int64)
        for lo in range(0,1000,64):
            t=tokens[lo:lo+64];y=labels[lo:lo+64];a=attention[lo:lo+64].astype(V.LD);free=t==1;on=(free&y).sum((1,2));off=(free&~y).sum((1,2));base=V.energy(t,y,a,t64,muoffs,aoffs);values=[V.flip(t,y,a,t64,muoffs,aoffs)[free],(V.energy(t,np.zeros_like(y),a,t64,muoffs,aoffs)-base)/on,(V.energy(t,np.ones_like(y),a,t64,muoffs,aoffs)-base)/off]
            for k,value in enumerate(values):minima[k]=min(minima[k],value.min());counts[k]+=value.size
        with np.load(NORM/f'negatives-seed{seed}.npz') as z:indices=z['indices'];boards=z['boards'];hamming=z['hamming']
        for lo in range(0,len(indices),64):
            ix=indices[lo:lo+64];t=tokens[ix];y=labels[ix];a=attention[ix].astype(V.LD);z=boards[lo:lo+64];value=(V.energy(t,z,a,t64,muoffs,aoffs)-V.energy(t,y,a,t64,muoffs,aoffs))/hamming[lo:lo+64];assert np.isfinite(value).all();minima[3]=min(minima[3],value.min());counts[3]+=len(ix)
        m=V.LD(256*900+1000);u=V.LD(2)**-53;gamma=m*u/(1-m*u);allowance=4*gamma*(10000*max(V.LD(1),abs(t64).max())+1);assert np.all(minima-allowance>0)
        records.append({'seed':seed,'positive_scale':scale,'original_linf':float(abs(original).max()),'source_feasibility_primal_linf':float(abs(theta).max()),'cast_linf':float(abs(cast).max()),'maximum_cast_error':float(np.max(abs(cast.astype(np.float64)-theta*scale))),'minimum_normalized_source_margins':list(map(float,minima)),'conservative_absolute_allowance':float(allowance),'minimum_certified_positive_margin':float(minima.min()-allowance),'source_constraints_checked':int(counts.sum()),'source_constraints_by_kind':counts.tolist(),'source_candidate_sha256':S.sha(path),'cast_array_sha256':S.sha(DEST/f'hard-negative-scaled-seed{seed}.npy'),'checkpoint_sha256':S.sha(target),'original_checkpoint_sha256':S.sha(S.OUT/f'models/seed{seed}.pt'),'continued_pl_checkpoint_sha256':S.sha(TRAIN/f'fits/s{seed}-pl/checkpoint.pt')})
    S.save(DEST/'cast-gate.json',{'status':'passed','utc':S.utc(),'records':records,'new_optimizer_updates':0,'fresh_test_inputs_used':0,'norm_verification_sha256':S.sha(NORM/'independent-verification.json'),'script_sha256':S.sha(__file__),'primal_verifier_sha256':S.sha(V.__file__),'scope':'Onepositive uniformscale to originalanswerparameterLinf followed by onefloat32cast. Everyoriginaltraininglocalflip and bothrankings andallminedhardnegative inequalities verified positive with independentdirectbinaryenergy/flip formulas. No targettuning.'});S.emit('lp_transfer_cast_gate',records=records)



if __name__=='__main__':prepare()
