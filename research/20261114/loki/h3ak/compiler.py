"""Complete repaired zero sets after source-only repair is sealed."""
import hashlib
import itertools
import subprocess
import time
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now
from enumerators import forward,reverse_literal,eligibility
FLAGS=['-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror']
def reverse(F,q,tag):
    folder=OUT/'verifier-io';folder.mkdir(exist_ok=True);inp=folder/f'{tag}.F.bin';out=folder/f'{tag}.columns.bin';F.astype(np.uint8).tofile(inp);r=subprocess.run([str(OUT/'reverse-literal'),str(q),str(inp),str(out)],text=True,capture_output=True,check=True);a=np.fromfile(out,np.uint8).reshape(-1,q);assert int(r.stdout)==len(a);return a

def build_validate(p,cpu):
    wall=time.perf_counter();start=cpu();version=subprocess.check_output(['g++','--version'],text=True).splitlines()[0];assert version==p['build']['compiler_version']
    for src,dest in [('reverse_literal','reverse-literal'),('truth_camel','truth-camel')]:subprocess.run(['g++',*FLAGS,str(SRC/f'{src}.cpp'),'-o',str(OUT/dest)],check=True)
    rows=[];rng=np.random.default_rng(2026113109)
    for q in range(2,8):
        n=q*q;ii,jj=np.indices((n,n));cliques=((ii//q==jj//q)|(ii%q==jj%q))&(ii!=jj);upper=np.triu(~cliques&~np.eye(n,dtype=bool),1);cases=[]
        for density in [0,.1,.5,.9,1]:
            edges=(rng.random((n,n))<density)&upper;cases.append((f'p{density}',(cliques|edges|edges.T).astype(np.int32)))
        for action in ['remove','add']:
            F=cases[-1 if action=='remove'else 0][1].copy();i,j=np.argwhere(upper)[0];F[i,j]=F[j,i]=int(action=='add');cases.append((action,F))
        for tag,F in cases:
            a=forward(F,q);b=reverse_literal(F,q);c=reverse(F,q,f'q{q}-{tag}');literal=[]
            for perm in itertools.permutations(range(q)):
                cells=q*np.arange(q)+perm
                if all(F[cells[i],cells[j]]==0 for i in range(q)for j in range(i+1,q)):literal.append(perm)
            d=np.array(literal,np.uint8).reshape(-1,q);assert np.array_equal(a,b)and np.array_equal(a,c)and np.array_equal(a,d);rows.append(dict(q=q,case=tag,states=len(a),all_four_equal=True))
    r=dict(status='passed',utc=now(),compiler_version=version,flags=FLAGS,cases=rows,cpu_seconds=cpu()-start,wall_seconds=time.perf_counter()-wall,binary_sha256={name:sha(OUT/name)for name in ['reverse-literal','truth-camel']});save(OUT/'compiler-validation.json',r);return r

def compile_supports(repair):
    results=[]
    for support in repair['supports']:
        group=support['group'];bank=support['representative_bank'];dose=support['representative_dose'];path=OUT/f'b{bank}/d{dose}/model.npz';m=np.load(path);proof=eligibility(m);F=m['F'];folder=OUT/f'support{group}';folder.mkdir(exist_ok=True);save(OUT/'progress.json',dict(stage='learned_enumeration',group=group,utc=now()));start=time.perf_counter();a=forward(F,15);np.save(folder/'unverified-forward-columns.npy',a);b=reverse(F,15,f'N15-support{group}');assert np.array_equal(a,b)and len(np.unique(a,axis=0))==len(a);cells=15*np.arange(15)[None,:]+a;i,j=np.triu_indices(15,1);assert not F[cells[:,i],cells[:,j]].sum(1,dtype=np.int64).any();boards=np.zeros((len(a),225),bool);boards[np.arange(len(a))[:,None],cells]=True;np.savez_compressed(folder/'cache.npz',columns=a,boards=boards,index=np.packbits(boards.T,axis=1,bitorder='little'));record=dict(support,status='complete_verified',states=len(a),wall_seconds=time.perf_counter()-start,cache_sha256=sha(folder/'cache.npz'),model_sha256=sha(path),proof=proof,no_target_truth_input=True);save(folder/'compilation.json',record);results.append(record);print('COMPILED',group,len(a),record['wall_seconds'],flush=True)
    save(OUT/'learned-enumeration-seal.json',dict(status='all_complete',utc=now(),supports=results,protocol_sha256=sha(SRC/'protocol.json')));return results
