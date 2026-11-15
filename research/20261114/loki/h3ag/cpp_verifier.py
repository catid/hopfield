"""Compiled literal-F wrapper and prospective model-only equivalence gate."""
import itertools
import subprocess
import sys
import time
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now
FLAGS=['-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror']

def run(F,q,tag):
    folder=OUT/'verifier-io';folder.mkdir(exist_ok=True)
    path=folder/f'{tag}.F.bin';dest=folder/f'{tag}.columns.bin'
    np.asarray(F,np.uint8).tofile(path)
    r=subprocess.run([str(OUT/'reverse-literal'),str(q),str(path),str(dest)],text=True,capture_output=True,check=True)
    a=np.fromfile(dest,np.uint8).reshape(-1,q);assert len(a)==int(r.stdout)
    return a

def prerequisites(reverse_literal,forward,cpu):
    start=time.perf_counter();start_cpu=cpu();version=subprocess.check_output(['g++','--version'],text=True).splitlines()[0]
    p=read(SRC/'protocol.json');assert version==p['compiled_verifier']['compiler_version']
    build=time.perf_counter();subprocess.run(['g++',*FLAGS,str(SRC/'reverse_literal.cpp'),'-o',str(OUT/'reverse-literal')],check=True);build_seconds=time.perf_counter()-build
    subprocess.run([sys.executable,str(SRC/'validate_query.py')],check=True)
    rows=[];rng=np.random.default_rng(2026113101)
    for q in range(2,8):
        n=q*q;ii,jj=np.indices((n,n));cliques=((ii//q==jj//q)|(ii%q==jj%q))&(ii!=jj);off=~cliques&~np.eye(n,dtype=bool);upper=np.triu(off,1)
        cases=[]
        for density in [0.,.1,.5,.9,1.]:
            edges=(rng.random((n,n))<density)&upper;F=(cliques|edges|edges.T).astype(np.int32);cases.append((f'p{density}',F))
        base=cases[2][1]
        for action in ['remove','add']:
            F=cases[-1 if action=='remove' else 0][1].copy();eligible=np.argwhere(upper&(F==int(action=='remove')))
            assert len(eligible)>0
            i,j=eligible[0];F[i,j]=F[j,i]=int(action=='add');cases.append((action,F))
        for tag,F in cases:
            stamp=time.perf_counter();original=reverse_literal(F,q);compiled=run(F,q,f'q{q}-{tag}');primary=forward(F,q)
            literal=[]
            for perm in itertools.permutations(range(q)):
                cells=q*np.arange(q)+perm
                if all(F[cells[i],cells[j]]==0 for i in range(q) for j in range(i+1,q)):literal.append(perm)
            exhaustive=np.array(literal,np.uint8).reshape(-1,q)
            assert np.array_equal(original,compiled)and np.array_equal(original,primary)and np.array_equal(original,exhaustive)
            rows.append(dict(q=q,case=tag,solutions=len(compiled),wall_seconds=time.perf_counter()-stamp,full_solution_set_parity=True))
    for bank in range(2):
        stamp=time.perf_counter();model=ROOT/p['models'][bank]['inherited_n13_model_path'];F=np.load(model)['F'];original=reverse_literal(F,13);compiled=run(F,13,f'N13-b{bank}');primary=forward(F,13);old=np.load(ROOT/f'artifacts/research/20261114/loki/h3ae/b{bank}/cache.npz')['columns']
        assert np.array_equal(original,compiled)and np.array_equal(original,primary)and np.array_equal(original,old)
        rows.append(dict(q=13,bank=bank,solutions=len(compiled),wall_seconds=time.perf_counter()-stamp,full_solution_set_parity=True))
    result=dict(status='passed',utc=now(),compiler_version=version,flags=FLAGS,cpp_source_sha256=sha(SRC/'reverse_literal.cpp'),binary_sha256=sha(OUT/'reverse-literal'),build_wall_seconds=build_seconds,all_prerequisite_wall_seconds=time.perf_counter()-start,all_prerequisite_cpu_seconds=cpu()-start_cpu,cases=rows,small_grid_seed=2026113101,independent_exhaustive_permutation_reference_q2_through_q7=True,includes_two_complete_N13_python_parity_cases=True,protocol_sha256=sha(SRC/'protocol.json'));save(OUT/'verifier-validation.json',result);return result
