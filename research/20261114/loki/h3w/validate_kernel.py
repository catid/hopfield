"""Independent exhaustive permutation transition and source-only GPU feasibility gates."""
import argparse,datetime as dt,hashlib,itertools,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3w';OUT=ROOT/'artifacts/research/20261114/loki/h3w';OLD=ROOT/'artifacts/research/20261114/loki/h3u'
sys.path.insert(0,str(SRC));import backend
sys.path.insert(0,str(ROOT/'upstream'));from cram import cram_nqueens as Q

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(device):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;rng=np.random.default_rng(83301);n=4;c=16;raw=rng.integers(-8,9,(c,c));J=np.triu(raw,1)/8;J+=J.T;h=np.arange(c)/8-.5;F=(abs(J)>.5).astype(np.int32);np.fill_diagonal(F,0)
    checks=0
    for perm in itertools.permutations(range(n)):
        x=np.zeros(c,np.int64);x[np.arange(n)*n+perm]=1;field=x@J+h;E=.5*x@J@x+x@h
        for clues in [0,1,2]:
            for r,s in itertools.combinations(range(clues,n),2):
                cols=np.array(perm);cols[r],cols[s]=cols[s],cols[r];y=np.zeros(c,np.int64);y[np.arange(n)*n+cols]=1;i=r*n+perm[r];j=s*n+perm[s];k=r*n+perm[s];l=s*n+perm[r]
                direct=.5*y@J@y+y@h-E;delta=field[k]+field[l]-field[i]-field[j]+J[i,j]+J[k,l]-J[i,k]-J[i,l]-J[j,k]-J[j,l];assert abs(direct-delta)<1e-12
                before=.5*x@F@x;counts=x@F;cd=counts[k]+counts[l]-counts[i]-counts[j]+F[i,j]+F[k,l]-F[i,k]-F[i,l]-F[j,k]-F[j,l];assert .5*y@F@y-before==cd
                forward=1/mathcomb(n-clues,2);reverse=1/mathcomb(n-clues,2);assert forward==reverse
                for T in [.1,1.,3.]:assert np.isclose(np.exp(-E/T)*forward*min(1.,np.exp(-direct/T)),np.exp(-(.5*y@J@y+y@h)/T)*reverse*min(1.,np.exp(direct/T)),atol=1e-10,rtol=1e-12)
                checks+=1
    # Exhaustive initialization support: each free-row/free-column permutation maps bijectively.
    init_counts=[]
    for clues in [0,1,2]:
        states=set()
        for perm in itertools.permutations(range(clues,n)):
            full=tuple(range(clues))+perm;assert len(set(full))==n;states.add(full)
        assert len(states)==mathfactorial(n-clues);init_counts.append(len(states))
    givens=np.zeros((8,c),bool)
    for k in range(8):givens[k,np.arange(k%3)*(n+1)]=True
    z=backend.solve(J,h,F,givens,q=n,unit=1.,chains=3,cap=64,seed=23,device=device,trace=True)
    for t in z['trace']:
        x=t['before'].astype(float);y=t['after'].astype(float);g=np.repeat(givens,3,0)[t['ids']];assert(y.reshape(-1,n,n).sum(1)==1).all()and(y.reshape(-1,n,n).sum(2)==1).all()and(y>=g).all()
        for a,cells in enumerate(t['cells']):
            i,j,k,l=cells;assert not g[a,i]and not g[a,j];trial=x[a].copy();trial[[i,j]]=0;trial[[k,l]]=1;expected=.5*trial@J@trial+trial@h-(.5*x[a]@J@x[a]+x[a]@h);assert abs(expected-t['delta'][a])<1e-6
        assert np.array_equal(t['counts'],y@F)and np.array_equal(t['conflicts'],.5*(y*(y@F)).sum(1));assert np.allclose(t['energy'],(.5*y*(y@J)).sum(1)+y@h,atol=1e-6,rtol=0);assert np.allclose(t['field'],y@J+h,atol=1e-6,rtol=0)
    d=np.load(OLD/'source-design.npz');m=torch.load(OLD/'models/bp-adamw-grouped/model.pt',weights_only=False);offsets=d['offsets'];mask=(d['counts']==0)&(d['exposure']>=np.log(264/.01));mask|=(offsets@d['lines'].T==0).any(1);coef=m['A'].numpy()[d['groups']];unit=float(coef[mask].mean());ii,jj=np.triu_indices(144,1);lookup={tuple(o):k for k,o in enumerate(offsets)};idx=np.array([lookup[j//12-i//12,j%12-i%12]for i,j in zip(ii,jj)])
    f=np.zeros((144,144),np.int32);f[ii,jj]=mask[idx];f[jj,ii]=f[ii,jj];sourceJ=np.zeros((144,144),np.float32);sourceJ[ii,jj]=coef[idx]*mask[idx];sourceJ[jj,ii]=sourceJ[ii,jj];W=torch.zeros(len(ii),2,2);W[:,1,1]=torch.tensor(sourceJ[ii,jj]);S=Q.full_S(W,torch.tensor(np.stack([ii,jj])),144)
    fixtures=np.concatenate([d['labels'][:16],rng.integers(0,2,(16,144)).astype(bool)]);native=Q.board_energy(S.double(),torch.zeros(144,2,dtype=torch.float64),torch.tensor(fixtures,dtype=torch.long)).numpy();literal=(fixtures[:,ii]*fixtures[:,jj]*sourceJ[ii,jj]).sum(1,dtype=np.float64);assert np.array_equal(native,literal)
    inputs=np.zeros((128,144),bool)
    for k,row in enumerate(d['labels'][:128]):inputs[k,rng.choice(np.flatnonzero(row),3,replace=False)]=True
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter();pilot=backend.solve(sourceJ,np.zeros(144),f,inputs,q=12,unit=unit,chains=20,cap=2000,seed=0,device=device)
    if device=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-start;y=pilot['boards'];assert(y.reshape(-1,12,12).sum(1)==1).all()and(y.reshape(-1,12,12).sum(2)==1).all()and(y>=inputs[:,None,:]).all();assert np.array_equal(pilot['conflicts'],(.5*y*(y@f)).sum(-1));assert(pilot['conflicts'][pilot['stopped']]==0).all()
    rec=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),device=device,exhaustive_permutations=24,exhaustive_transpositions=checks,uniform_initialization_support_sizes=init_counts,detailed_balance_temperatures=[.1,1.,3.],every_trace_q_rows_columns_givens_fields_energy_counts_verified=True,native_source_dense_energy_fixture_boards=32,
       source_only_pilot=dict(inputs=128,chains=20,cap=2000,seconds=seconds,proposed=int(pilot['moves'].sum()),accepted=int(pilot['accepted'].sum()),stopped=int(pilot['stopped'].sum()),no_target_outcomes=True),
       input_sha256={str(p.relative_to(ROOT)):sha(p)for p in [SRC/'backend.py',Path(__file__),OLD/'source-design.npz',OLD/'models/bp-adamw-grouped/model.pt',ROOT/'upstream/cram/cram_nqueens.py']})
    (OUT/f'kernel-validation-{device}.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))
def mathcomb(n,k):
    import math
    return math.comb(n,k)
def mathfactorial(n):
    import math
    return math.factorial(n)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args();validate(a.device)
