"""Fresh compiled validators and independent supplied-rule reference."""
import hashlib
import itertools
import subprocess
import time
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now
from enumerators import forward,reverse_literal
FLAGS=['-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror']
def support_hash(F):return hashlib.sha256(np.asarray(F,np.int32).tobytes()).hexdigest()
def graph(q):
    n=q*q;i,j=np.indices((n,n));dr=abs(i//q-j//q);dc=abs(i%q-j%q);F=(((dr==0)|(dc==0)|(dr==dc)|((dr==1)&(dc==3))|((dr==3)&(dc==1)))&(i!=j)).astype(np.int32)
    # Separately express the supplied relation over coordinate pairs.
    for a in range(n):
        for b in range(a+1,n):
            ar,ac=divmod(a,q);br,bc=divmod(b,q);literal=(ar==br or ac==bc or abs(ar-br)==abs(ac-bc)or (abs(ar-br),abs(ac-bc))in[(1,3),(3,1)]);assert F[a,b]==F[b,a]==literal
    return F

def reverse(F,q,tag):
    folder=OUT/'verifier-io';folder.mkdir(exist_ok=True);inp=folder/f'{tag}.F.bin';out=folder/f'{tag}.columns.bin';np.asarray(F,np.uint8).tofile(inp);r=subprocess.run([str(OUT/'reverse-literal'),str(q),str(inp),str(out)],text=True,capture_output=True,check=True);a=np.fromfile(out,np.uint8).reshape(-1,q);assert len(a)==int(r.stdout);return a

def truth(q,tag):
    folder=OUT/'verifier-io';folder.mkdir(exist_ok=True);out=folder/f'{tag}.truth-columns.bin';r=subprocess.run([str(OUT/'truth-camel'),str(q),str(out)],text=True,capture_output=True,check=True);a=np.fromfile(out,np.uint8).reshape(-1,q);assert len(a)==int(r.stdout);return a

def counts(columns,F):
    q=columns.shape[1];cells=q*np.arange(q)[None,:]+columns;i,j=np.triu_indices(q,1);return F[cells[:,i],cells[:,j]].sum(1,dtype=np.int64)

def prepare(p,cpu):
    wall=time.perf_counter();start_cpu=cpu();version=subprocess.check_output(['g++','--version'],text=True).splitlines()[0];assert version==p['build']['compiler_version'];build=time.perf_counter()
    for name,binary in [('reverse_literal','reverse-literal'),('truth_camel','truth-camel')]:subprocess.run(['g++',*FLAGS,str(SRC/f'{name}.cpp'),'-o',str(OUT/binary)],check=True)
    build_seconds=time.perf_counter()-build;rows=[];rng=np.random.default_rng(2026113104)
    for q in range(2,8):
        n=q*q;ii,jj=np.indices((n,n));cliques=((ii//q==jj//q)|(ii%q==jj%q))&(ii!=jj);upper=np.triu(~cliques&~np.eye(n,dtype=bool),1);cases=[]
        for density in [0,.1,.5,.9,1]:
            edges=(rng.random((n,n))<density)&upper;cases.append((f'p{density}',(cliques|edges|edges.T).astype(np.int32)))
        for action in ['remove','add']:
            F=cases[-1 if action=='remove'else 0][1].copy();i,j=np.argwhere(upper)[0];F[i,j]=F[j,i]=int(action=='add');cases.append((action,F))
        for tag,F in cases:
            a=forward(F,q);b=reverse_literal(F,q);c=reverse(F,q,f'q{q}-{tag}');exhaustive=[]
            for perm in itertools.permutations(range(q)):
                cells=q*np.arange(q)+perm
                if all(F[cells[i],cells[j]]==0 for i in range(q)for j in range(i+1,q)):exhaustive.append(perm)
            d=np.array(exhaustive,np.uint8).reshape(-1,q);assert np.array_equal(a,b)and np.array_equal(a,c)and np.array_equal(a,d);rows.append(dict(q=q,case=tag,states=len(a),all_four_equal=True))
        F=graph(q);a=forward(F,q);b=reverse(F,q,f'truth-q{q}');c=truth(q,f'truth-q{q}');assert np.array_equal(a,b)and np.array_equal(a,c);rows.append(dict(q=q,case='supplied_camel',states=len(a),all_three_equal=True))
    for q in [12,13]:
        F=graph(q);a=forward(F,q);b=reverse(F,q,f'truth-q{q}');c=truth(q,f'truth-q{q}');old=np.load(ROOT/p['prior_true_solutions'][str(q)]);cells=q*np.arange(q)[None,:]+a;boards=np.zeros((len(a),q*q),bool);boards[np.arange(len(a))[:,None],cells]=True;assert np.array_equal(a,b)and np.array_equal(a,c)and np.array_equal(boards,old);rows.append(dict(q=q,case='prior_true_camel',states=len(a),all_three_and_prior_equal=True))
    seen={};learned=[]
    for bank,cfg in enumerate(p['models']):
        F=np.load(ROOT/cfg['inherited_n13_model_path'])['F'];key=support_hash(F)
        if key in seen:assert np.array_equal(F,seen[key][0]);learned.append(dict(bank=bank,deduplicated_to=seen[key][1],full_F_byte_equal=True));continue
        a=forward(F,13);b=reverse_literal(F,13);c=reverse(F,13,f'learned-N13-b{bank}');old=np.load(ROOT/p['prior_true_solutions']['13']);cols=np.flatnonzero(old).reshape(-1,13)%169%13;filtered=cols[counts(cols,F)==0];assert np.array_equal(a,b)and np.array_equal(a,c)and np.array_equal(a,filtered);seen[key]=(F,bank);learned.append(dict(bank=bank,states=len(a),original_python_compiled_and_filtered_truth_equal=True))
    result=dict(status='passed',utc=now(),compiler_version=version,flags=FLAGS,build_seconds=build_seconds,wall_seconds=time.perf_counter()-wall,cpu_seconds=cpu()-start_cpu,synthetic_and_truth_cases=rows,learned_N13_cases=learned,binary_sha256={name:sha(OUT/name)for name in ['reverse-literal','truth-camel']},protocol_sha256=sha(SRC/'protocol.json'));save(OUT/'verification.json',result);return result
