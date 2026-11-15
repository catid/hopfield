"""Source-only checks of explicit starts and unchanged frozen transition loops."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse,ast,datetime as dt,hashlib,importlib.util,json
from pathlib import Path
import time
import numpy as np
import torch
import arbitrary_backend as arbitrary
import permutation_backend as permutation
from initialization import make
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3aa';OUT=ROOT/'artifacts/research/20261114/loki/h3aa';BASE=ROOT/'artifacts/research/20261114/loki'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def loop_ast(path):
    tree=ast.parse(Path(path).read_text());loops=[n for n in ast.walk(tree)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='step'];assert len(loops)==1;return ast.dump(loops[0],include_attributes=False)
def main(device):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;p=json.loads((SRC/'protocol.json').read_text())
    for group in['source_sha256','data_sha256']:
        for rel,h in p[group].items():assert sha(ROOT/rel)==h,rel
    legacy={'arbitrary':load(SRC.parent/'h3v/backend.py','legacy_arbitrary'),'permutation':load(SRC.parent/'h3w/backend.py','legacy_permutation')};current={'arbitrary':arbitrary,'permutation':permutation}
    for name,old in[('arbitrary','h3v'),('permutation','h3w')]:assert loop_ast(SRC/f'{name}_backend.py')==loop_ast(SRC.parent/old/'backend.py')
    rng=np.random.default_rng(83411);n=4;c=16;raw=rng.integers(-8,9,(c,c));J=np.triu(raw,1)/8;J+=J.T;h=np.arange(c)/8-.5;F=(abs(J)>.5).astype(np.int32);np.fill_diagonal(F,0)
    givens=np.zeros((8,c),bool)
    for k in range(8):givens[k,np.arange(k%3)*(n+1)]=True
    parity=[]
    for name in current:
        kwargs=dict(q=n,unit=1.,chains=3,cap=64,seed=23,device=device,block=7,trace=True)
        old=legacy[name].solve(J,h,F,givens,**kwargs);new=current[name].solve(J,h,F,givens,**kwargs)
        for key in old:
            if key=='trace':
                assert len(old[key])==len(new[key])
                for a,b in zip(old[key],new[key]):
                    for k in a:assert np.array_equal(a[k],b[k]),(name,k)
            else:assert np.array_equal(old[key],new[key]),(name,key)
        parity.append(dict(kernel=name,all_outputs_and_every_trace_byte_equal=True,multiple_blocks=True))
    starts={kind:make(givens,q=4,chains=3,seed=0,kind=kind)for kind in['arbitrary','permutation']};checks=0;starts_checked=0
    for kind,kernel in[('arbitrary','arbitrary'),('permutation','arbitrary'),('permutation','permutation')]:
        init=starts[kind];z=current[kernel].solve(J,h,F,givens,q=4,unit=1.,chains=3,cap=64,seed=0,device=device,block=7,trace=True,initial=init)
        flat=init.reshape(-1,16)
        for t in z['trace']:
            x=t['before'].astype(float);y=t['after'].astype(float);given=np.repeat(givens,3,0)[t['ids']];assert(y.sum(1)==4).all()and(y>=given).all()
            if t['step']==0:assert np.array_equal(x,flat[t['ids']]);starts_checked+=len(x)
            if kernel=='permutation':assert(y.reshape(-1,4,4).sum(1)==1).all()and(y.reshape(-1,4,4).sum(2)==1).all()
            expected_field=y@J+h;expected_e=(.5*y*(y@J)).sum(1)+y@h;assert np.allclose(t['field'],expected_field,atol=1e-6,rtol=0)
            assert np.allclose(t['energy'],expected_e,atol=1e-6,rtol=0);assert np.array_equal(t['counts'],y@F)and np.array_equal(t['conflicts'],.5*(y*(y@F)).sum(1))
            for k,row in enumerate(x):
                trial=row.copy()
                if kernel=='arbitrary':trial[t['i'][k]]=0;trial[t['j'][k]]=1
                else:ii,jj,kk,ll=t['cells'][k];trial[[ii,jj]]=0;trial[[kk,ll]]=1
                delta=.5*trial@J@trial+trial@h-(.5*row@J@row+row@h);assert abs(delta-t['delta'][k])<1e-6
            checks+=len(x)
        done=(.5*(flat*(flat@F)).sum(1))==0
        assert z['stopped'].reshape(-1)[done].all()and np.array_equal(z['boards'].reshape(-1,16)[done],flat[done])
    d=np.load(BASE/'h3u/source-design.npz');m=torch.load(BASE/'h3u/models/bp-adamw-grouped/model.pt',weights_only=False);offsets=d['offsets'];mask=(d['counts']==0)&(d['exposure']>=np.log(264/.01));mask|=(offsets@d['lines'].T==0).any(1);coef=m['A'].numpy()[d['groups']];unit=float(coef[mask].mean());ii,jj=np.triu_indices(144,1);lookup={tuple(o):k for k,o in enumerate(offsets)};idx=np.array([lookup[j//12-i//12,j%12-i%12]for i,j in zip(ii,jj)])
    f=np.zeros((144,144),np.int32);f[ii,jj]=mask[idx];f[jj,ii]=f[ii,jj];sourceJ=np.zeros((144,144),np.float32);sourceJ[ii,jj]=coef[idx]*mask[idx];sourceJ[jj,ii]=sourceJ[ii,jj];inputs=np.zeros((128,144),bool)
    for k,row in enumerate(d['labels'][:128]):inputs[k,rng.choice(np.flatnonzero(row),3,replace=False)]=True
    initial={kind:make(inputs,q=12,chains=20,seed=0,kind=kind)for kind in['arbitrary','permutation']};pilots=[]
    for kind,kernel in[('arbitrary','arbitrary'),('permutation','arbitrary'),('permutation','permutation')]:
        if device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();z=current[kernel].solve(sourceJ,np.zeros(144),f,inputs,q=12,unit=unit,chains=20,cap=2000,seed=0,device=device,initial=initial[kind])
        if device=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-start;y=z['boards'];assert(y.sum(-1)==12).all()and(y>=inputs[:,None,:]).all()
        if kernel=='permutation':assert(y.reshape(-1,12,12).sum(1)==1).all()and(y.reshape(-1,12,12).sum(2)==1).all()
        counts=(.5*y*(y@f)).sum(-1);assert np.array_equal(counts,z['conflicts'])and(counts[z['stopped']]==0).all()
        pilots.append(dict(initialization=kind,kernel=kernel,seconds=elapsed,proposals=int(z['moves'].sum()),accepted=int(z['accepted'].sum()),initial_bytes_sha256=hashlib.sha256(initial[kind].tobytes()).hexdigest()))
    assert pilots[1]['initial_bytes_sha256']==pilots[2]['initial_bytes_sha256']
    record=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),device=device,transition_loop_AST_identical_to_frozen_legacy=True,legacy_parity=parity,
        explicit_start_states_checked=starts_checked,trace_transitions_checked=checks,shared_permutation_initial_bytes=True,all_q_givens_fields_energies_conflicts_verified=True,
        source_only_pilots=pilots,source_pilot_inputs=128,source_pilot_chains=20,source_pilot_cap=2000,no_target_data_or_outcomes=True,
        protocol_sha256=sha(SRC/'protocol.json'),input_sha256={str(path.relative_to(ROOT)):sha(path)for path in[SRC/'gate.py',SRC/'initialization.py',SRC/'arbitrary_backend.py',SRC/'permutation_backend.py',SRC.parent/'h3v/backend.py',SRC.parent/'h3w/backend.py']})
    (OUT/f'gate-{device}.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--device',choices=['cpu','cuda'],required=True);args=a.parse_args();main(args.device)
