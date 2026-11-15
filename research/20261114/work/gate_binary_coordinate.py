"""H2aa source gates: exact additivity, descent, clamping and throughput."""
import argparse
from fractions import Fraction as Q
import itertools
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
from gate_lp_relaxation import model_for
from prepare_lp_transfer import TRAIN

DEST=S.OUT/'binary_coordinate'


def exact_energy(t,board,a,theta,muoffs,aoffs):
    p=np.where(t==1,board,t>=2).astype(int);h,w=t.shape;value=Q(0)
    for r in range(h):
        for c in range(w):
            if t[r,c]==1:value-=Q(float(theta[16+p[r,c]]))
            for k,(dr,dc) in enumerate(muoffs):
                rr,cc=r+dr,c+dc
                if 0<=rr<h and 0<=cc<w:value+=Q(float(theta[4*k+2*p[r,c]+p[rr,cc]]))
            for k,(dr,dc) in enumerate(aoffs):
                rr,cc=r+dr,c+dc
                if 0<=rr<h and 0<=cc<w:value-=Q(float(theta[18+k]))*Q(float(a[k,r,c]))*int(p[r,c])*int(p[rr,cc])
    return value


def numerical():
    torch,_,_=S.imports();DEST.mkdir(exist_ok=True);started=time.perf_counter();rows=[]
    fixture=np.array([[2,1,0],[1,1,3],[0,1,1]],np.int64);positions=np.argwhere(fixture==1)
    boards=np.repeat((fixture>0)[None],2**len(positions),axis=0).astype(np.float64)
    for i,bits in enumerate(itertools.product([0,1],repeat=len(positions))):
        for (r,c),bit in zip(positions,bits):boards[i,r,c]=bit
    # Offsets never connect two different cells of the same parity color.
    assert all(dr%2 or dc%2 for dr,dc in C.OFFSETS)
    for seed in range(3):
        model=model_for(seed,'cpu',torch.float64);theta=np.concatenate([model.T.numpy().ravel(),model.b.numpy(),model.kappa.numpy()]);rng=np.random.default_rng(2026111511+seed)
        raw=rng.random((5,3,3)).astype(np.float32);raw/=raw.sum(0,keepdims=True)
        energies={tuple(b[fixture==1].astype(int)):exact_energy(fixture,b,raw,theta,model.mu_offsets,model.aoffs) for b in boards}
        def energy(board):return energies[tuple(board[fixture==1].astype(int))]
        t=torch.tensor(np.repeat(fixture[None],len(boards),axis=0));p=torch.tensor(boards);a=torch.tensor(np.repeat(raw[None],len(boards),axis=0),dtype=torch.float64);cache=C.BinaryQuadratic(model,t,a)
        delta=cache.delta(p).numpy();upstream=model.flip_delta(t,p,a).numpy();max_error=0.;additive_checks=0
        for i,b in enumerate(boards):
            for r,c in positions:
                flipped=b.copy();flipped[r,c]=1-flipped[r,c];exact=energy(flipped)-energy(b)
                max_error=max(max_error,abs(delta[i,r,c]-float(exact)),abs(upstream[i,r,c]-float(exact)))
            for color in C.COLORS:
                selected=[(r,c) for r,c in positions if (r%2,c%2)==color]
                for mask in itertools.product([0,1],repeat=len(selected)):
                    combined=b.copy();total=Q(0)
                    for (r,c),bit in zip(selected,mask):
                        if bit:
                            single=b.copy();single[r,c]=1-single[r,c];total+=energy(single)-energy(b);combined[r,c]=1-combined[r,c]
                    assert energy(combined)-energy(b)==total;additive_checks+=1
        assert max_error<1e-10
        out=C.solve(cache,initial=p,trace=True);phase_checks=changed_checks=0
        for step in out['history']:
            before=step['before'].numpy();change=step['flip'].numpy();after=np.where(change,1-before,before)
            for b,z,mask in zip(before,after,change):
                difference=energy(z)-energy(b);assert difference<0 if mask.any() else difference==0
                assert np.array_equal(z[fixture!=1],(fixture>0)[fixture!=1]);phase_checks+=1;changed_checks+=int(mask.any())
        assert out['stopped'].all() and out['finite'].all()
        for i,b in enumerate(out['prediction'].numpy()):
            for r,c in positions:
                z=b.copy();z[r,c]=1-z[r,c];assert energy(z)-energy(b)>=Q(-1,100000000)
        for i in [0,7,15,31]:
            single=C.BinaryQuadratic(model,t[i:i+1],a[i:i+1]);other=C.solve(single,initial=p[i:i+1])
            for key in ['prediction','sweeps','stopped','finite','flip_count']:assert torch.equal(other[key],out[key][i:i+1])
        bad=p.clone();bad[:,0,0]=0
        try:C.solve(cache,initial=bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid clamp accepted')
        rows.append({'seed':seed,'all_binary_boards':32,'single_flip_checks':160,'exact_samecolor_additivity_checks':additive_checks,'exact_phase_energy_checks':phase_checks,'strictly_decreasing_changed_phases':changed_checks,'maximum_float64_vs_exact_flip_error':max_error,'certified_final_boards':32,'batch_single_stopmask_parity_cases':4,'invalid_clamp_rejected':True,'maximum_sweeps':int(out['sweeps'].max())})
    S.save(DEST/'numerical-gate.json',{'status':'passed','utc':S.utc(),'rows':rows,'script_sha256':S.sha(__file__),'helper_sha256':S.sha(C.__file__),'seconds':time.perf_counter()-started,'scope':'Source-only exhaustive5freecell fixtures with exact rational binaryenergies. Pairindependence, phaseadditivity/decrease, originalfloat64flip parity, clamping and stopmask batchparity. No targetinput or labelscore.'})


def performance(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();DEST.mkdir(exist_ok=True);started=time.perf_counter();rows=[]
    with np.load(TRAIN/'training-arrays.npz') as x:tokens=x['tokens'][:64]
    t=torch.tensor(tokens.astype(np.int64),device='cuda')
    for seed in range(3):
        if seed%2!=gpu:continue
        model=model_for(seed,'cuda',torch.float64)
        with np.load(TRAIN/f'train-readout-seed{seed}.npz') as x:a=torch.tensor(x['couplings'][:64],device='cuda',dtype=torch.float64)
        torch.cuda.synchronize();before=time.perf_counter();cache=C.BinaryQuadratic(model,t,a);torch.cuda.synchronize();setup=time.perf_counter()-before
        class Native:
            free=cache.free;colors=cache.colors
            def initial(self):return cache.initial()
            def delta(self,p):return model.flip_delta(t,p,a)
        # Warmup no parameter updates; then alternate2complete timingreplays.
        C.solve(cache,cap=1);C.solve(Native(),cap=1);timings={'cached':[],'reference':[]};saved={}
        for rep in range(2):
            for name,obj in ([('cached',cache),('reference',Native())] if rep==0 else [('reference',Native()),('cached',cache)]):
                torch.cuda.synchronize();before=time.perf_counter();out=C.solve(obj);torch.cuda.synchronize();timings[name].append(time.perf_counter()-before)
                if name in saved:
                    for key in ['prediction','sweeps','stopped','finite','flip_count']:assert torch.equal(out[key],saved[name][key])
                saved[name]=out
        for key in ['prediction','sweeps','stopped','finite','flip_count']:assert torch.equal(saved['cached'][key],saved['reference'][key]),key
        out=saved['cached'];actual=model.flip_delta(t,out['prediction'],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1)
        error=float((actual-out['minimum_flip_margin']).abs().max());assert error<1e-10 and out['finite'].all()
        rows.append({'seed':seed,'source_inputs':64,'cache_setup_seconds':setup,'timings_seconds':timings,'reference_over_cached_median':float(np.median(timings['reference'])/np.median(timings['cached'])),'exact_reference_boards_sweeps_flags_flips':True,'maximum_margin_error':error,'certified':int(out['stopped'].sum()),'timeouts':int((~out['stopped']).sum()),'maximum_sweeps':int(out['sweeps'].max()),'mean_sweeps':float(out['sweeps'].double().mean()),'maximum_flips':int(out['flip_count'].max())})
    S.save(DEST/f'performance-gate-gpu{gpu}.json',{'status':'passed','utc':S.utc(),'inventory':inventory,'records':rows,'seconds':time.perf_counter()-started,'script_sha256':S.sha(__file__),'helper_sha256':S.sha(C.__file__),'scope':'Originaltrainingfirst64/model. Twoalternating completecached/reference timingreplays plusone-sweepwarmup; no labels/graphoutcomes and no scientificindependentinputs. Fixedcap512/threshold-1e-8, no retuning.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['numerical','performance']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=='numerical':numerical()
    else:performance(args.gpu)
