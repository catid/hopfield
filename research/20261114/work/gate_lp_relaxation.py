"""H2z source-only binary invariance, gradient and fixed-throughput gates."""
import argparse
import itertools
import json
import time
import numpy as np
import study as S
import binary_preserving as B
import larger_relaxation as L
import selective_fallback as F
from lp_transfer import PP as PARENT, protocol as parent_protocol
from prepare_lp_transfer import TRAIN

DEST=S.OUT/'lp_relaxation'
STRENGTH=4.


def model_for(seed,device,dtype=None):
    torch,_,_=S.imports();p=parent_protocol();model=S.model_for(seed,30,'original',device,dtype)
    model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['lp_scaled']['path'],map_location=device,weights_only=True))
    return model


def numerical():
    torch,_,_=S.imports();records=[];started=time.perf_counter();DEST.mkdir(exist_ok=True)
    fixture=np.array([[2,1,0],[1,1,3],[0,1,1]],np.int64);free=fixture==1
    boards=np.zeros((2**int(free.sum()),3,3),np.float64)
    for i,bits in enumerate(itertools.product([0.,1.],repeat=int(free.sum()))):boards[i][free]=bits
    for seed in range(3):
        model=model_for(seed,'cpu',torch.float64);assert float(model.kappa[0])>0
        t=torch.tensor(np.repeat(fixture[None],len(boards),axis=0));y=torch.tensor(boards)
        rng=np.random.default_rng(2026111510+seed);raw=rng.random((1,5,3,3));raw/=raw.sum(1,keepdims=True)
        a=torch.tensor(np.repeat(raw,len(boards),axis=0));old=model.energy(t,y,a);new=B.energy(model,t,y,a,STRENGTH)
        assert torch.equal(old,new)
        native=model.flip_delta(t,y,a);errors=[]
        for rr,cc in np.argwhere(free):
            changed=y.clone();changed[:,rr,cc]=1-changed[:,rr,cc]
            brute=B.energy(model,t,changed,a,STRENGTH)-new
            errors.append(float((brute-native[:,rr,cc]).abs().max()))
        assert max(errors)<1e-9
        soft=torch.tensor(rng.uniform(.1,.9,size=(1,3,3)),dtype=torch.float64,requires_grad=True)
        tt=t[:1];aa=a[:1];reg=(STRENGTH*(soft*(1-soft))*(tt==1)).sum()
        gradient=torch.autograd.grad(reg,soft)[0]
        field_change=B.field(model,tt,soft,aa,STRENGTH)-model.field(tt,soft,aa)
        gradient_error=float((field_change+gradient).abs().max());assert gradient_error<1e-12
        # Curvature bound holds for all nonnegative attention, as kappa0>0.
        bound=1/(.45*.55)-2*STRENGTH;assert bound<0
        records.append({'seed':seed,'binary_boards':len(boards),'binary_flips':len(boards)*int(free.sum()),'binary_energies_exactly_unchanged':True,'maximum_independent_binary_flip_error':max(errors),'independent_regularizer_gradient_error':gradient_error,'kappa0_positive':float(model.kappa[0]),'coordinate_curvature_upper_for_p_45_55':bound})
    S.save(DEST/'numerical-gate.json',{'status':'passed','utc':S.utc(),'lambda_fixed':STRENGTH,'records':records,'seconds':time.perf_counter()-started,'script_sha256':S.sha(__file__),'binary_helper_sha256':S.sha(B.__file__),'parent_protocol_sha256':S.sha(PARENT),'scope':'Source-only random finite fixtures; no target data. All32binaryboards and160singleflips permodel. Independent autograd derivative of explicit regularizer. Positive kappa0 and attention imply p in(.45,.55) has negative coordinate curvature atlambda4, excluding such states from continuous localminima; no convergence/rounding guarantee.'})


def performance(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();records=[];DEST.mkdir(exist_ok=True);started=time.perf_counter()
    with np.load(TRAIN/'training-arrays.npz') as x:tokens=x['tokens'][:64]
    t=torch.tensor(tokens.astype(np.int64),device='cuda')
    for seed in range(3):
        if seed%2!=gpu:continue
        model=model_for(seed,'cuda')
        with np.load(TRAIN/f'train-readout-seed{seed}.npz') as x:a=torch.tensor(x['couplings'][:64],device='cuda')
        arms=[]
        for strength in [0.,STRENGTH]:
            torch.cuda.synchronize();before=time.perf_counter();out=L.solve(model,t,a,torch.full_like(a[:,0],3.),8192,strength);torch.cuda.synchronize()
            arms.append({'lambda':strength,'seconds':time.perf_counter()-before,'certified':int(out[2].sum()),'finite':int(out[3].sum()),'timeouts':int((~out[2]&out[3]).sum()),'steps_min':int(out[1].min()),'steps_max':int(out[1].max())})
            assert out[3].all()
        records.append({'seed':seed,'source_inputs':64,'arms':arms})
    S.save(DEST/f'performance-gate-gpu{gpu}.json',{'status':'passed','utc':S.utc(),'inventory':inventory,'lambda_fixed':STRENGTH,'records':records,'seconds':time.perf_counter()-started,'script_sha256':S.sha(__file__),'scope':'Actual first64originaltraininginputs; source-only finite/throughput check. No targets, no graph outcome selection, no lambda/eta/cap changes after results.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['numerical','performance']);p.add_argument('--gpu',type=int,choices=[0,1]);args=p.parse_args()
    if args.command=='numerical':numerical()
    else:performance(args.gpu)
