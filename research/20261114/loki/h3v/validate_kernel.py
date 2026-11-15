"""Exhaustive toy and source-only parity/throughput gates for the new swap kernel."""
import argparse,datetime as dt,hashlib,itertools,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3v';OUT=ROOT/'artifacts/research/20261114/loki/h3v';OLD=ROOT/'artifacts/research/20261114/loki/h3u'
sys.path.insert(0,str(SRC));import backend
sys.path.insert(0,str(ROOT/'upstream'));from cram import cram_nqueens as Q

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(device):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;rng=np.random.default_rng(83220);c=9;q=3;raw=rng.integers(-8,9,(c,c));J=np.triu(raw,1)/8;J=J+J.T;h=np.linspace(-.5,.5,c);F=(abs(J)>.5).astype(np.int32);np.fill_diagonal(F,0)
    states=[]
    for cells in itertools.combinations(range(c),q):
        x=np.zeros(c,dtype=np.int64);x[list(cells)]=1;states.append(x)
    states=np.array(states);energies=.5*(states*(states@J)).sum(1)+states@h;flip_checks=0;balance=[]
    for x,E in zip(states,energies):
        for clue_count in [0,1,2]:
            givens=np.flatnonzero(x)[:clue_count];free_occupied=np.setdiff1d(np.flatnonzero(x),givens)
            pproposal=1/(len(free_occupied)*(c-q));assert len(free_occupied)==q-clue_count
            for i in free_occupied:
                for j in np.flatnonzero(~x.astype(bool)):
                    y=x.copy();y[i]=0;y[j]=1;assert y[givens].all();direct=.5*y@J@y+y@h-E;field=x@J+h;formula=field[j]-field[i]-J[i,j];assert abs(direct-formula)<1e-12
                    reverse=(y@J+h)[i]-(y@J+h)[j]-J[j,i];assert abs(reverse+formula)<1e-12
                    assert pproposal==1/((int(y.sum())-clue_count)*(c-q))
                    for T in [.1,1.,3.]:
                        a=min(1.,np.exp(-direct/T));b=min(1.,np.exp(direct/T));left=np.exp(-E/T)*pproposal*a;right=np.exp(-(.5*y@J@y+y@h)/T)*pproposal*b;assert np.isclose(left,right,atol=1e-10,rtol=1e-12)
                    flip_checks+=1
    # Trace verification is independent on each device; CPU/GPU RNG algorithms differ.
    givens=np.zeros((8,c),bool)
    for k in range(8):givens[k,:k%3]=True
    trace=backend.solve(J,h,F,givens,q=q,unit=1.,chains=3,cap=64,seed=19,device=device,trace=True)
    for row in trace['trace']:
        x=row['before'].astype(float);y=row['after'].astype(float);ids=row['ids'];g=np.repeat(givens,3,0)[ids];i=row['i'];j=row['j'];a=np.arange(len(x))
        assert (x.sum(1)==q).all() and (y.sum(1)==q).all() and (y>=g).all();assert not g[a,i].any() and x[a,i].all() and not x[a,j].any()
        assert np.allclose(row['delta'],(.5*y*(y@J)).sum(1)+y@h-(.5*x*(x@J)).sum(1)-x@h,atol=1e-6,rtol=0) if row['accepted'].all() else True
        original_delta=(x@J+h)[a,j]-(x@J+h)[a,i]-J[i,j];assert np.allclose(row['delta'],original_delta,atol=1e-6,rtol=0)
        assert np.array_equal(row['counts'],y@F) and np.array_equal(row['conflicts'],.5*(y*(y@F)).sum(1));assert np.allclose(row['field'],y@J+h,atol=1e-6,rtol=0);assert np.allclose(row['energy'],(.5*y*(y@J)).sum(1)+y@h,atol=1e-6,rtol=0)
    boards=trace['boards'];assert (boards.sum(-1)==q).all() and (boards>=givens[:,None,:]).all();assert np.array_equal(trace['conflicts'],(.5*boards*(boards@F)).sum(-1));assert (trace['conflicts'][trace['stopped']]==0).all()
    # Use source parameters/source boards only for native dense energy parity and performance.
    d=np.load(OLD/'source-design.npz');m=torch.load(OLD/'models/bp-adamw-grouped/model.pt',weights_only=False);offsets=d['offsets'];mask=(d['counts']==0)&(d['exposure']>=np.log(264/.01));mask|=(offsets@d['lines'].T==0).any(1);coeff=m['A'].numpy()[d['groups']];assert (coeff[mask]>0).all();unit=float(coeff[mask].mean())
    ii,jj=np.triu_indices(144,1);lookup={tuple(o):k for k,o in enumerate(offsets)};index=np.array([lookup[j//12-i//12,j%12-i%12] for i,j in zip(ii,jj)]);f=np.zeros((144,144),np.int32);f[ii,jj]=mask[index];f[jj,ii]=f[ii,jj];Jsource=np.zeros((144,144),np.float32);Jsource[ii,jj]=coeff[index]*mask[index];Jsource[jj,ii]=Jsource[ii,jj]
    W=torch.zeros(len(ii),2,2);W[:,1,1]=torch.tensor(Jsource[ii,jj]);S=Q.full_S(W,torch.tensor(np.stack([ii,jj])),144);B=torch.zeros(144,2);fixtures=np.concatenate([d['labels'][:16],rng.integers(0,2,(16,144)).astype(bool)])
    native=Q.board_energy(S.double(),B.double(),torch.tensor(fixtures,dtype=torch.int64)).numpy();literal=(fixtures[:,ii]*fixtures[:,jj]*Jsource[ii,jj]).sum(1,dtype=np.float64);assert np.allclose(native,literal,atol=1e-10,rtol=0)
    pilot_inputs=np.zeros((128,144),bool)
    for k,row in enumerate(d['labels'][:128]):pilot_inputs[k,rng.choice(np.flatnonzero(row),3,replace=False)]=True
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter();pilot=backend.solve(Jsource,np.zeros(144),f,pilot_inputs,q=12,unit=unit,chains=20,cap=2000,seed=0,device=device)
    if device=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-start;boards=pilot['boards'];assert (boards.sum(-1)==12).all() and (boards>=pilot_inputs[:,None,:]).all();direct=(.5*boards*(boards@f)).sum(-1);assert np.array_equal(direct,pilot['conflicts']) and (direct[pilot['stopped']]==0).all()
    rec=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),device=device,toy_states=len(states),exhaustive_swap_checks=flip_checks,detailed_balance_temperatures=[.1,1.,3.],all_trace_q_givens_field_energy_conflicts_verified=True,
       native_source_dense_energy_fixture_boards=len(fixtures),source_minimum_retained_positive_coefficient=float(coeff[mask].min()),source_temperature_unit=unit,
       source_only_pilot=dict(inputs=128,chains=20,cap=2000,seconds=seconds,proposals=int(pilot['moves'].sum()),accepted=int(pilot['accepted'].sum()),stopped=int(pilot['stopped'].sum()),no_target_outcomes=True),
       input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SRC/'backend.py',Path(__file__),OLD/'source-design.npz',OLD/'models/bp-adamw-grouped/model.pt',ROOT/'upstream/cram/cram_nqueens.py']})
    (OUT/f'kernel-validation-{device}.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args();validate(a.device)
