"""H4z: weaken low-confidence penalties while preserving the learned zero set."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RIPPER = HERE.parent/'ripper'
PARENT = ROOT/'results/research/foureyes/canonical_confirm'
OUT = ROOT/'artifacts/research/20261114/foureyes/weak_weight_study'
PUBLIC = ROOT/'results/research/foureyes/weak_weight_study'
PROTOCOL = HERE/'weak_weight_study.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def verify(p):
    for path, digest in (p['source_sha256'] | p['input_sha256']).items():
        assert sha(ROOT/path) == digest, path


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    source = [HERE/n for n in ['weak_weight_study.py','weak_weight_search.py','weak_weight_math.py','weak_weight_gate.py','audit_weak_weight_study.py']]
    source += [RIPPER/n for n in ['run.py','sparse_backend.py','bootstrap_two_way.py']]
    source += [RIPPER/'confidence_weight_search.py']
    source += [ROOT/'upstream/cram'/n for n in ['cram_sudoku_general.py','cram_sudoku_search.py','cram_sudoku_sparse.py']]
    inputs = [PARENT/n for n in ['models.npz','summary.json','audited-outcome-bits.npz']]
    inputs += [ROOT/'upstream/data/full-test/full-all.npz', HERE/'canonical_confirm.json']
    p = dict(id='H4z-positive-weak-penalty-strength', frozen_utc=datetime.now(timezone.utc).isoformat(),
        scope='Post-hoc mechanism study on all64 H4y canonical fits and the same already-seen5000 puzzles; no fresh-generalization claim.',
        arms=['weak16','weak64','weak256'], banks=64, rows_half_open=[410000,415000], search_seed=0,
        rule='Same H4y support and median-positive-J/4 threshold. Strong coefficient stays1. Weak coefficient is1/16,1/64 or1/256. No geometry/label in assignment.',
        representation='uint16 weight256 numerators{0,16,256},{0,4,256},{0,1,256}; inferencefloat32 dividedby256. Minimumpositive1/256>.001 stop tolerance.',
        primary='weak256 minus weak16 >=5percentage points withzero weak256 invalidcertifiedstops.',
        uncertainty='5000paired two-waybank/input bootstrapdraws,primaryseed20261116031. weak64 contrast descriptive,seed20261116032.',
        decision='Supported if primary95lower>=5 andzero primarywrongstops; falsified ifupper<5 oranyprimarywrongstop; otherwiseinconclusive.',
        secondary='Absoluteaccuracy,weak64dose response andcertified/capped decomposition descriptive. BPAdamW/counts are inherited separately fitted H4y controls.',
        inference='C4,seed0,unit1,temp.4->.31,theta.45,tau0,ebar0,tol.001,anneal524288,check4096,262144moves/chain. No temperature renormalization.',
        numerical_gate='Four prior-source model prefixes comparegeneral/sparse/hidden-target states and scaled-integer fields; toy unavoidable single-weak-conflict cases may not stop, whilezero-energy cases must stop. Per-new-search gates also retained.',
        budget=dict(gate_phase_wall_seconds=45,search_phase_wall_seconds=255,total_gpu_minutes_upper_bound=20),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source},
        input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs})
    save(PROTOCOL,p); print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    import numpy as np
    p=json.loads(PROTOCOL.read_text());verify(p);assert not OUT.exists();OUT.mkdir(parents=True)
    parent=json.loads((PARENT/'summary.json').read_text())
    with np.load(PARENT/'models.npz',allow_pickle=False)as z:coeff=z['canonical_weak_coefficients']
    rows=[];queue=[];known={}
    for bank in range(64):
        j=coeff[bank];positive=j>0;cut=float(np.median(j[positive])/4)
        baseline=np.where(positive,np.where(j>=cut,256,16),0).astype(np.uint16)
        prior=next(r for r in parent['rows'] if r['bank']==bank and r['arm']=='magnitude')
        key=(baseline.astype(np.float32)/256).tobytes()
        known.setdefault(key,dict(kind='archived',job=prior['job']))
        for arm,weak in [('weak16',16),('weak64',4),('weak256',1)]:
            w=np.where(baseline==16,weak,baseline).astype(np.uint16)
            path=OUT/'models'/f'b{bank:02d}-{arm}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,support=w>0,weight256=w)
            key=(w.astype(np.float32)/256).tobytes()
            if key not in known:
                job=f'b{bank:02d}-{arm}';known[key]=dict(kind='new',job=job)
                queue.append(dict(job_id=job,seed=0,n=600,arm=arm,status='pending',model=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))))
            rows.append(dict(bank=bank,arm=arm,model_path=str(path.relative_to(ROOT)),model_sha256=sha(path),weight256_sha256=hashlib.sha256(w.tobytes()).hexdigest(),**known[key]))
    save(OUT/'mapping.json',dict(protocol_sha256=sha(PROTOCOL),rows=rows,unique_new_searches=len(queue)))
    save(OUT/'queue.json',queue);print(json.dumps(dict(logical_cells=len(rows),new_searches=len(queue))))


def setup(gpu):
    assert platform.node()=='foureyes'
    lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),CRAM_DATA=str(ROOT/'upstream/data'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.path[:0]=[str(RIPPER),str(ROOT/'upstream')]
    import torch
    import numpy as np
    from cram import cram_sudoku_general as GS,cram_sudoku_search as SS,cram_sudoku_sparse as SP
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    return lock,torch,np,GS,SS,SP


def search(gpu):
    lock,torch,np,GS,SS,SP=setup(gpu)
    p=json.loads(PROTOCOL.read_text());verify(p)
    source=json.loads((OUT/'source-audit.json').read_text());assert source['status']=='validated'and source['protocol_sha256']==sha(PROTOCOL)
    for bank in range(4):assert json.loads((OUT/f'gate-bank{bank}.json').read_text())['status']=='passed'
    import confidence_weight_search as queue
    import weak_weight_search as runner
    queue.OUT=OUT
    lo,hi=p['rows_half_open'];data=SS.load_puzzles('full-all',lo,hi-lo)
    while (job:=queue.claim(gpu)) is not None:
        try:runner.run(job,gpu,p,torch,np,GS,SS,SP,data)
        except BaseException:queue.claim(gpu,job['job_id'],'failed');raise
        queue.claim(gpu,job['job_id'],'complete')


def launch(phase):
    p=json.loads(PROTOCOL.read_text());verify(p);start=time.monotonic();workers=[];logs=[];samples=[]
    try:
        for gpu in range(4):
            log=(OUT/f'{phase}-gpu{gpu}.log').open('x');logs.append(log)
            workers.append(subprocess.Popen([str(ROOT/'upstream/.venv/bin/python'),str(Path(__file__)),phase,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(w.poll()is None for w in workers):
            assert time.monotonic()-start<p['budget'][phase+'_phase_wall_seconds']and all(w.poll()in[None,0]for w in workers)
            q=json.loads((OUT/'queue.json').read_text())
            samples.append(dict(elapsed=time.monotonic()-start,complete=sum(r['status']=='complete'for r in q),running=sum(r['status']=='running'for r in q),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)))
            time.sleep(2)
        assert all(w.returncode==0 for w in workers)
        result=dict(status='complete',wall_seconds=time.monotonic()-start,samples=samples,returncodes=[w.returncode for w in workers]);save(OUT/(phase+'-launch.json'),result);print(json.dumps({k:v for k,v in result.items()if k!='samples'}))
    finally:
        for w in workers:
            if w.poll()is None:
                os.killpg(w.pid,signal.SIGTERM)
                try:w.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
        for log in logs:log.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','gate','search','launch-gate','launch-search']);ap.add_argument('--gpu',type=int);a=ap.parse_args()
    if a.action=='gate':
        from weak_weight_gate import run
        run(a.gpu)
    elif a.action=='search':search(a.gpu)
    elif a.action.startswith('launch-'):launch(a.action.split('-')[1])
    else:globals()[a.action]()
