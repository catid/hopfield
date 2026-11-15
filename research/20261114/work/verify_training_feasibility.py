"""Public-only NumPy verifier of every training inequality; no solver or Torch."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

LD=np.longdouble


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def offsets(shape,dr,dc):
    h,w=shape
    return (slice(max(0,-dr),min(h,h-dr)),slice(max(0,-dc),min(w,w-dc))), (slice(max(0,dr),min(h,h+dr)),slice(max(0,dc),min(w,w+dc)))


def energy(t,board,a,theta,muoffs,aoffs):
    p=np.where(t==1,board,t>=2).astype(np.int8);n=len(t);answer=np.zeros(n,LD);table=theta[:16].reshape(4,2,2);bias=theta[16:18];kappa=theta[18:]
    answer-=np.where(t==1,bias[p],LD(0)).sum((1,2),dtype=LD)
    for k,(dr,dc) in enumerate(muoffs):
        l,r=offsets(t.shape[-2:],int(dr),int(dc));answer+=table[k,p[:,l[0],l[1]],p[:,r[0],r[1]]].sum((1,2),dtype=LD)
    for k,(dr,dc) in enumerate(aoffs):
        l,r=offsets(t.shape[-2:],int(dr),int(dc));answer-=(kappa[k]*a[:,k,l[0],l[1]]*p[:,l[0],l[1]]*p[:,r[0],r[1]]).sum((1,2),dtype=LD)
    return answer


def flip(t,board,a,theta,muoffs,aoffs):
    p=np.where(t==1,board,t>=2).astype(np.int8);change=(1-2*p).astype(LD);table=theta[:16].reshape(4,2,2);bias=theta[16:18];kappa=theta[18:];answer=change*(bias[0]-bias[1])
    for k,(dr,dc) in enumerate(muoffs):
        l,r=offsets(t.shape[-2:],int(dr),int(dc));left=p[:,l[0],l[1]];right=p[:,r[0],r[1]];answer[:,l[0],l[1]]+=table[k,1-left,right]-table[k,left,right];answer[:,r[0],r[1]]+=table[k,left,1-right]-table[k,left,right]
    for k,(dr,dc) in enumerate(aoffs):
        dr,dc=int(dr),int(dc)
        if dr==dc==0:answer-=kappa[k]*a[:,k]*change;continue
        l,r=offsets(t.shape[-2:],dr,dc);edge=kappa[k]*a[:,k,l[0],l[1]];answer[:,l[0],l[1]]-=edge*p[:,r[0],r[1]]*change[:,l[0],l[1]];answer[:,r[0],r[1]]-=edge*p[:,l[0],l[1]]*change[:,r[0],r[1]]
    return answer


def verify(directory,training):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());summary=json.loads((directory/'solver-summary.json').read_text());assert summary['status']=='complete' and summary['protocol_sha256']==sha(directory/'protocol.json');assert sha(training/'training-arrays.npz')==p['training_manifest']['arrays_sha256']
    with np.load(training/'training-arrays.npz',allow_pickle=False) as x:tokens=x['tokens'];labels=x['labels']
    assert tokens.shape==labels.shape==(1000,30,30);rows=[]
    for r in summary['records']:
        seed=r['seed'];assert r['status']=='verified_feasible_primal';candidate=directory/f'seed{seed}/candidate-theta.npy';assert sha(candidate)==r['candidate_sha256'];theta=np.load(candidate,allow_pickle=False).astype(LD);assert theta.shape==(23,) and np.isfinite(theta).all()
        cp=directory/f'initial-seed{seed}.npz';assert sha(cp)==p['initial_models'][seed]['sha256']
        with np.load(cp,allow_pickle=False) as z:muoffs=z['mu_offsets'];aoffs=z['aoffs'];initial=np.concatenate([z['T'].ravel(),z['b'],z['kappa']]).astype(LD)
        ap=training/f'train-readout-seed{seed}.npz';assert sha(ap)==p['readouts'][str(seed)]['readout_sha256']
        with np.load(ap,allow_pickle=False) as z:attention=z['couplings']
        minima=np.full(3,np.inf,LD);counts=np.zeros(3,np.int64)
        for lo in range(0,1000,64):
            t=tokens[lo:lo+64];y=labels[lo:lo+64];a=attention[lo:lo+64].astype(LD);free=t==1;on=(free&y).sum((1,2));off=(free&~y).sum((1,2));assert np.all(on>0) and np.all(off>0)
            canonical=energy(t,y,a,theta,muoffs,aoffs);margins=[flip(t,y,a,theta,muoffs,aoffs)[free],(energy(t,np.zeros_like(y),a,theta,muoffs,aoffs)-canonical)/on,(energy(t,np.ones_like(y),a,theta,muoffs,aoffs)-canonical)/off]
            for k,value in enumerate(margins):assert np.isfinite(value).all();minima[k]=min(minima[k],value.min());counts[k]+=value.size
        # Use the declared, deliberately looser float64 error bound even when
        # NumPy provides80-bitlongdouble. This covers64-bitfallback platforms.
        norm=abs(theta).max();m=LD(256*900+1000);u=LD(2)**-53;gamma=m*u/(1-m*u);allowance=4*gamma*(10000*max(LD(1),norm)+1);assert np.all(minima-allowance>1),(seed,minima,allowance)
        assert counts.tolist()==[564329,1000,1000]
        rows.append({'seed':seed,'status':'all_inequalities_verified','constraints_by_kind':counts.tolist(),'minimum_normalized_margins':list(map(float,minima)),'conservative_absolute_allowance':float(allowance),'minimum_certified_margin_after_allowance':float(minima.min()-allowance),'parameter_linf':float(norm),'parameter_l1':float(abs(theta).sum()),'parameter_l2':float(np.sqrt((theta*theta).sum())),'change_from_initial_linf':float(abs(theta-initial).max()),'candidate_sha256':sha(candidate)})
    assert len(rows)==3
    return {'status':'passed','hypothesis':'H2w','decision':'supported_for_stated_training_constraints','rows':rows,'total_verified_constraints':sum(sum(r['constraints_by_kind']) for r in rows),'numpy_version':np.__version__,'longdouble_mantissa_bits':int(np.finfo(LD).nmant),'seconds':time.perf_counter()-started,'verifier_sha256':sha(Path(__file__)),'protocol_sha256':sha(directory/'protocol.json'),'scope':'Three independentlyverified feasibleprimals for unrestrictedreal23parameter answerenergy, on actualoriginal1000trainingarrays and frozenreadouts only. Alllocalflip and sparse/densenegativeunitmargin constraints pass. No testgeneralization, decoderstability, globaloptimality or small-weight conclusion.','method':'Independentdirect pairtable and directedlink evaluation, no imported featurebuilder/Torch/SciPy/solver. Every normalizedconstraint exceeds1 plus conservativelybounded arithmeticerror; exactfloat64candidate and float32readout constants promoted losslessly.'}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[3];parser=argparse.ArgumentParser();parser.add_argument('directory',nargs='?',type=Path,default=root/'results/research/work/training-feasibility');parser.add_argument('training',nargs='?',type=Path,default=root/'results/research/work/contrastive-training');args=parser.parse_args();print(json.dumps(verify(args.directory,args.training),indent=2))
