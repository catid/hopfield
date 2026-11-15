"""Norm-match LPwitnesses, cast once, verify allsource constraints; no freshtest."""
import argparse
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import larger_relaxation as L
import verify_training_feasibility as V

DEST=S.OUT/'lp_transfer'
TRAIN=S.OUT/'contrastive_training'
NORM=S.OUT/'training_min_norm'


def prepare():
    DEST.mkdir(exist_ok=True);torch,_,_=S.imports();proof=json.loads((NORM/'independent-verification.json').read_text());assert proof['status']=='passed';records=[]
    with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
    for seed in range(3):
        model=S.model_for(seed,30,'original','cpu');original=np.concatenate([model.T.numpy().ravel(),model.b.numpy(),model.kappa.numpy()]);path=NORM/f'seed{seed}/candidate-theta.npy';theta=np.load(path);scale=float(np.max(abs(original)).astype(np.float64)/np.max(abs(theta)));cast=np.asarray(theta*scale,np.float32);assert cast.shape==(23,) and np.isfinite(cast).all()
        with torch.no_grad():model.T.copy_(torch.tensor(cast[:16].reshape(4,2,2)));model.b.copy_(torch.tensor(cast[16:18]));model.kappa.copy_(torch.tensor(cast[18:]));
        target=DEST/f'lp-scaled-seed{seed}.pt'
        if target.exists():raise FileExistsError(target)
        torch.save(model.state_dict(),target);np.save(DEST/f'lp-scaled-seed{seed}.npy',cast)
        with np.load(TRAIN/f'train-readout-seed{seed}.npz') as z:attention=z['couplings']
        t64=cast.astype(V.LD);muoffs=model.mu_offsets;aoffs=model.aoffs;minima=np.full(3,np.inf,V.LD);counts=np.zeros(3,np.int64)
        for lo in range(0,1000,64):
            t=tokens[lo:lo+64];y=labels[lo:lo+64];a=attention[lo:lo+64].astype(V.LD);free=t==1;on=(free&y).sum((1,2));off=(free&~y).sum((1,2));base=V.energy(t,y,a,t64,muoffs,aoffs);values=[V.flip(t,y,a,t64,muoffs,aoffs)[free],(V.energy(t,np.zeros_like(y),a,t64,muoffs,aoffs)-base)/on,(V.energy(t,np.ones_like(y),a,t64,muoffs,aoffs)-base)/off]
            for k,value in enumerate(values):minima[k]=min(minima[k],value.min());counts[k]+=value.size
        m=V.LD(256*900+1000);u=V.LD(2)**-53;gamma=m*u/(1-m*u);allowance=4*gamma*(10000*max(V.LD(1),abs(t64).max())+1);assert np.all(minima-allowance>0)
        records.append({'seed':seed,'positive_scale':scale,'original_linf':float(abs(original).max()),'source_min_norm_primal_linf':float(abs(theta).max()),'cast_linf':float(abs(cast).max()),'maximum_cast_error':float(np.max(abs(cast.astype(np.float64)-theta*scale))),'minimum_normalized_source_margins':list(map(float,minima)),'conservative_absolute_allowance':float(allowance),'minimum_certified_positive_margin':float(minima.min()-allowance),'source_constraints_checked':int(counts.sum()),'source_candidate_sha256':S.sha(path),'cast_array_sha256':S.sha(DEST/f'lp-scaled-seed{seed}.npy'),'checkpoint_sha256':S.sha(target),'original_checkpoint_sha256':S.sha(S.OUT/f'models/seed{seed}.pt'),'continued_pl_checkpoint_sha256':S.sha(TRAIN/f'fits/s{seed}-pl/checkpoint.pt')})
    S.save(DEST/'cast-gate.json',{'status':'passed','utc':S.utc(),'records':records,'new_optimizer_updates':0,'fresh_test_inputs_used':0,'norm_verification_sha256':S.sha(NORM/'independent-verification.json'),'script_sha256':S.sha(__file__),'primal_verifier_sha256':S.sha(V.__file__),'scope':'Onepositive uniformscale to originalanswerparameterLinf followed by onefloat32cast. Everyoriginaltraininglocalflip and bothrankings verified positive with independentdirectbinaryenergy/flip formulas. No targettuning.'});S.emit('lp_transfer_cast_gate',records=records)


def performance(gpu):
    lock,inventory=F.acquire(gpu);torch,_,M=S.imports();gate=json.loads((DEST/'cast-gate.json').read_text());assert gate['status']=='passed';records=[]
    with np.load(TRAIN/'training-arrays.npz') as z:t=torch.tensor(z['tokens'][:64],device='cuda')
    for seed in range(3):
        if seed%2!=gpu:continue
        model=S.model_for(seed,30,'original','cuda');initial={k:v.cpu().clone() for k,v in model.state_dict().items()};model.load_state_dict(torch.load(DEST/f'lp-scaled-seed{seed}.pt',map_location='cuda',weights_only=True));assert all(torch.equal(initial[k],v.cpu()) for k,v in model.state_dict().items() if k not in ['T','b','kappa'])
        with np.load(TRAIN/f'train-readout-seed{seed}.npz') as z:a=torch.tensor(z['couplings'][:64],device='cuda')
        with torch.no_grad():
            psi,_=model.settle(t,tol=1e-6,max_steps=4000,check=8);assert torch.equal(a,model.couplings(t,psi));L.solve(model,t,a,torch.full_like(a[:,0],3.),32,0.);torch.cuda.synchronize();start=time.perf_counter();out=L.solve(model,t,a,torch.full_like(a[:,0],3.),8192,0.);torch.cuda.synchronize();elapsed=time.perf_counter()-start;native=M.descend(model,t,a,steps=8192,eta=.5,margin=3.,certify_every=16);assert all(torch.equal(native[k],out[k]) for k in range(3))
        records.append({'seed':seed,'inputs':64,'source_training_inputs_only':True,'unchanged_hidden_tensors':True,'cache_exact':True,'native_board_step_flag_parity':True,'answer_seconds':elapsed,'maximum_steps':int(out[1].max()),'minimum_steps':int(out[1].min()),'all_finite':bool(out[3].all())})
    S.save(DEST/f'performance-gate-gpu{gpu}.json',{'status':'passed','utc':S.utc(),'records':records,'inventory':inventory,'script_sha256':S.sha(__file__),'scope':'Originaltrainingfirst64only; no fresh120cohort, no qualityselection, no solver/scale/cap changes.'});S.emit('lp_transfer_performance_gate',gpu=gpu,records=records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','performance']);p.add_argument('--gpu',type=int,choices=[0,1]);args=p.parse_args()
    if args.command=='prepare':prepare()
    else:performance(args.gpu)
