"""H4x: frozen post-hoc coefficient-magnitude intervention on H4w models."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import ast
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
RIPPER=HERE.parent/'ripper'
PARENT=ROOT/'results/research/foureyes/canonical_study'
OUT=ROOT/'artifacts/research/20261114/foureyes/canonical_weights'
PUBLIC=ROOT/'results/research/foureyes/canonical_weights'
PROTOCOL=HERE/'canonical_weights.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


def capture_equivalence():
    """Exact AST identity after deleting only branches known dead at restart=0."""
    classes=[]
    for path,restart in [(RIPPER/'run.py',False),(HERE/'restart_capture.py',True)]:
        tree=ast.parse(path.read_text())
        cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Captured')
        step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step_check')
        if restart:
            index=next(i for i,n in enumerate(step.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.restart and self.t >= self.next_restart')
        else:
            index=next(i for i,n in enumerate(step.body) if isinstance(n,ast.Assert) and ast.unparse(n.test)=='self.restart == 0')
        del step.body[index]
        classes.append(ast.dump(cls,include_attributes=False))
    assert classes[0]==classes[1]
    classes=[]
    for path in [RIPPER/'sparse_backend.py',HERE/'restart_backend.py']:
        tree=ast.parse(path.read_text())
        cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='DerivedSparse')
        classes.append(ast.dump(cls,include_attributes=False))
    assert classes[0]==classes[1]
    return dict(capture_ast_equal_at_restart_zero=True,sparse_class_ast_equal=True)


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    files=[Path(__file__),HERE/'audit_canonical_weights.py',RIPPER/'confidence_weight_search.py',RIPPER/'confidence_weight_common.py',
        RIPPER/'confidence_integer_audit.py',RIPPER/'sparse_backend.py',RIPPER/'run.py',RIPPER/'bootstrap_two_way.py',
        HERE/'restart_capture.py',HERE/'restart_backend.py',HERE/'canonical_study.json',
        ROOT/'upstream/cram/cram_sudoku_general.py',ROOT/'upstream/cram/cram_sudoku_search.py',ROOT/'upstream/cram/cram_sudoku_sparse.py']
    inputs=[PARENT/'models.npz',PARENT/'summary.json',PARENT/'audited-outcome-bits.npz',PARENT/'certificates.json',ROOT/'upstream/data/full-test/full-all.npz']
    p=dict(id='H4x-coefficient-magnitude-weights',frozen_utc=datetime.now(timezone.utc).isoformat(),
        scope='Adaptive post-hoc mechanism study on H4w64 frozen weak-penalty fits and the same already-seen10k inputs. No fresh generalization claim.',
        arms=['uniform','magnitude','shuffled'],banks=64,rows_half_open=[390000,400000],search_seed=0,
        rule='For each canonical_weak J: threshold=np.median(J[J>0])/4 inFP64; J>=threshold gets16,0<J<threshold gets1,else0; divideby16 forinference. No geometry or labels inassignment.',
        shuffle='Permute magnitude weight16 values only among positive-support positions with NumPy default_rng(202611330000+bank). Same multiset and support.',
        shuffle_seed_base=202611330000,primary='Magnitude minus uniform >=10pp AND magnitude minus shuffled >=10pp, withzero invalid certified stops across allarms.',
        interval='5000paired two-waybank/input bootstrapdraws; seed20261116011 foruniform and20261116012 forshuffled. Marginal95 intervals, no simultaneouscoverageclaim.',
        secondary='Absoluteaccuracy andcertified/capped decomposition descriptive; no accuracyfloor target.',
        numerical_gate='Every newsearch uses150general/sparse,132hidden-target and6scaled-integer checks, including127unrefreshedmoves. ExactASTequivalence validates archivedrestart0backend reuse.',
        inference='Unchanged unit1,temp.4->.31,theta.45,C4,tau0,ebar0,tol.001,cap262144moves,anneal524288,check4096,seed0. Positiveweights preservezero set.',
        budget=dict(search_phase_wall_seconds=450,total_gpu_minutes_upper_bound=30),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},
        baseline_equivalence=capture_equivalence())
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def verify(p):
    for name,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/name)==digest,name
    assert capture_equivalence()==p['baseline_equivalence']


def prepare():
    import numpy as np
    p=json.loads(PROTOCOL.read_text());verify(p);assert not OUT.exists();OUT.mkdir(parents=True)
    with np.load(PARENT/'models.npz',allow_pickle=False) as data:
        coefficients=data['canonical_weak_coefficients'];supports=data['canonical_weak_support']
    parent=json.loads((PARENT/'summary.json').read_text());known={};rows=[];queue=[]
    for bank in range(64):
        j=coefficients[bank];support=supports[bank];assert np.array_equal(support,j>0)
        threshold=float(np.median(j[support])/4)
        magnitude=np.where(support,np.where(j>=threshold,16,1),0).astype(np.uint8)
        shuffled=magnitude.copy();shuffled[support]=np.random.default_rng(p['shuffle_seed_base']+bank).permutation(magnitude[support])
        uniform=(support*16).astype(np.uint8)
        prior=next(r for r in parent['model_mappings'] if r['bank']==bank and r['arm']=='canonical_weak')
        key=uniform.tobytes();known.setdefault(key,dict(kind='archived',job=prior['actual']))
        for arm,weights in [('uniform',uniform),('magnitude',magnitude),('shuffled',shuffled)]:
            path=OUT/'models'/f'b{bank:02d}-{arm}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,support=support,weight16=weights)
            key=weights.tobytes()
            if key not in known:
                job=f'b{bank:02d}-{arm}';known[key]=dict(kind='new',job=job)
                queue.append(dict(job_id=job,seed=0,n=600,arm=arm,status='pending',model=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))))
            rows.append(dict(bank=bank,arm=arm,threshold=threshold,weight16_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),model_path=str(path.relative_to(ROOT)),model_sha256=sha(path),**known[key]))
    save(OUT/'mapping.json',dict(protocol_sha256=sha(PROTOCOL),rows=rows,unique_new_searches=len(queue)))
    save(OUT/'queue.json',queue);print(json.dumps(dict(logical_cells=len(rows),new_searches=len(queue))))


def worker(gpu):
    assert platform.node()=='foureyes'
    lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),CRAM_DATA=str(ROOT/'upstream/data'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.path[:0]=[str(RIPPER),str(ROOT/'upstream')]
    import torch
    import numpy as np
    from cram import cram_sudoku_general as GS,cram_sudoku_search as SS,cram_sudoku_sparse as SP
    import confidence_weight_search as backend
    p=json.loads(PROTOCOL.read_text());verify(p)
    source=json.loads((OUT/'source-audit.json').read_text());assert source['status']=='validated' and source['protocol_sha256']==sha(PROTOCOL)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    backend.OUT=OUT;backend.PROTOCOL=PROTOCOL
    lo,hi=p['rows_half_open'];dataset=SS.load_puzzles('full-all',lo,hi-lo)
    while (job:=backend.claim(gpu)) is not None:
        try:backend.run(job,gpu,p,torch,np,GS,SS,SP,dataset)
        except BaseException:backend.claim(gpu,job['job_id'],'failed');raise
        backend.claim(gpu,job['job_id'],'complete')


def launch():
    p=json.loads(PROTOCOL.read_text());verify(p)
    start=time.monotonic();workers=[];logs=[];samples=[]
    try:
        for gpu in range(4):
            log=(OUT/f'gpu{gpu}.log').open('x');logs.append(log)
            workers.append(subprocess.Popen([str(ROOT/'upstream/.venv/bin/python'),str(Path(__file__)),'worker','--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(w.poll() is None for w in workers):
            assert time.monotonic()-start<p['budget']['search_phase_wall_seconds'] and all(w.poll() in [None,0] for w in workers)
            q=json.loads((OUT/'queue.json').read_text());samples.append(dict(elapsed=time.monotonic()-start,complete=sum(r['status']=='complete'for r in q),running=sum(r['status']=='running'for r in q),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)))
            time.sleep(2)
        assert all(w.returncode==0 for w in workers)
        result=dict(status='complete',wall_seconds=time.monotonic()-start,samples=samples,returncodes=[w.returncode for w in workers]);save(OUT/'launch.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
    finally:
        for w in workers:
            if w.poll() is None:
                os.killpg(w.pid,signal.SIGTERM)
                try:w.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
        for log in logs:log.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','worker','launch']);ap.add_argument('--gpu',type=int);a=ap.parse_args()
    if a.action=='worker':worker(a.gpu)
    else:globals()[a.action]()
