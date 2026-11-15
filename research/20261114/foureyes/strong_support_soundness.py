"""H4ab: exact global soundness audit of all H4aa strong-only supports."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
RIPPER=HERE.parent/'ripper'
PUBLIC=ROOT/'results/research/foureyes/strong_support_soundness'
OUT=ROOT/'artifacts/research/20261114/foureyes/strong_support_soundness'
PROTOCOL=HERE/'strong_support_soundness.json'
PARENT=ROOT/'results/research/foureyes/strong_support_study/summary.json'
MODELS=ROOT/'results/research/foureyes/canonical_confirm/models.npz'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def verify():
    p=json.loads(PROTOCOL.read_text())
    for path,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/path)==digest,path
    return p


def freeze():
    assert not PROTOCOL.exists()and not OUT.exists()and not PUBLIC.exists()
    code=[Path(__file__),HERE/'verify_strong_support_soundness.py',HERE/'certify_sudoku_soundness.py',HERE/'verify_sudoku_soundness.py',RIPPER/'counterexample_coloring.py']
    inputs=[PARENT,MODELS,HERE/'strong_support_study.json']
    p=dict(id='H4ab-global-soundness-after-weak-edge-deletion',frozen_utc=datetime.now(timezone.utc).isoformat(),
           banks=64,scope='All64 unchanged H4aa strong-only supports. Geometry-supplied exact audit, no fitting, source/test rows or GPU work.',
           hypothesis='Every one of64 strong-only supports implies ordinary Sudoku validity for every zero-energy coloring over9 symbols.',
           reconstruction='Rebuild H4aa support from public FP64 coefficients: J>0 and J>=median(positiveJ)/4. Check every original weight256 hash before proof. No graph-based bank selection.',
           proof='Unchanged clique/count-identity unit closure; only27 independently replayed certified units prove soundness.',
           counterexample='For closure-unresolved graphs, unchanged deterministic DSATUR forces each missing true pair equal in lexicographic order;2CPUseconds/100000nodes shared per full support. Only explicit independently validated zero-energy invalid boards prove unsoundness.',
           decision='Supported only if all64 have complete proofs; falsified if any explicit invalid zero-energy witness; otherwise inconclusive. Incomplete closure, timeout, complete search without portable UNSAT proof and no observed wrong stops are never soundness proofs.',
           deduplication='Only byte-identical full Boolean supports share proof/search work. All64 bank outcomes remain reported.',
           budget=dict(aggregate_cpu_seconds=180,wall_seconds=240,cpu_workers=1,per_unresolved_graph_cpu_seconds=2,node_limit=100000,gpu_seconds=0),
           source_sha256={str(f.relative_to(ROOT)):sha(f)for f in code},input_sha256={str(f.relative_to(ROOT)):sha(f)for f in inputs})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def compute():
    import numpy as np
    p=verify();assert not OUT.exists();OUT.mkdir(parents=True)
    from certify_sudoku_soundness import close
    sys.path.insert(0,str(RIPPER));from counterexample_coloring import search
    parent=json.loads(PARENT.read_text());assert parent['status']=='validated'
    with np.load(MODELS,allow_pickle=False)as z:coeff=z['canonical_weak_coefficients'].copy()
    assert coeff.shape==(64,3240)and coeff.dtype==np.float64 and np.isfinite(coeff).all()
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
    rows=[];supports=[];cache={}
    for bank,j in enumerate(coeff):
        cut=float(np.median(j[j>0])/4);support=(j>0)&(j>=cut);w=support.astype(np.uint16)*256
        prior=next(r for r in parent['rows']if r['bank']==bank and r['arm']=='strongonly')
        assert hashlib.sha256(w.tobytes()).hexdigest()==prior['weight256_sha256']
        ident=hashlib.sha256(support.tobytes()).hexdigest();supports.append(support)
        if ident not in cache:
            trace,known=close(support);counter=None
            if len(known)<27:
                missing=[(int(i),int(k))for i,k in zip(ii[truth&~support],jj[truth&~support])]
                counter=search(support,missing,cpu_seconds=2,node_limit=100000)
            cache[ident]=dict(trace=trace,certified_units=known,sound=len(known)==27,counterexample_search=counter)
        row=dict(bank=bank,threshold=cut,support_sha256=ident,weight256_sha256=prior['weight256_sha256'],
                 missing_true=int((truth&~support).sum()),extra=int((~truth&support).sum()),observed_wrong_stops=prior['wrong_stops'],**cache[ident])
        row['classification']='sound_proof'if row['sound']else'unsound_witness'if row['counterexample_search']['status']=='explicit_counterexample'else'unresolved'
        rows.append(row);save(OUT/'progress.json',dict(completed_banks=bank+1,unique_supports=len(cache)))
    PUBLIC.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(PUBLIC/'supports.npz',banks=np.arange(64),supports=np.stack(supports))
    save(PUBLIC/'certificates.json',dict(status='produced_pending_independent_verification',protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),parent_sha256=sha(PARENT),supports_sha256=sha(PUBLIC/'supports.npz'),unique_supports=len(cache),rows=rows))
    from verify_strong_support_soundness import audit
    result=audit();save(PUBLIC/'summary.json',result);save(OUT/'audit.json',result)
    print(json.dumps(dict(status=result['status'],primary=result['primary'],counts=result['counts'])))


def launch():
    p=verify();log=OUT.parent/'strong-support-soundness-execution.log';assert not log.exists()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic();samples=[];ticks=os.sysconf('SC_CLK_TCK')
    with log.open('x')as f:
        child=subprocess.Popen([sys.executable,str(Path(__file__)),'compute'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:
            while child.poll()is None:
                elapsed=time.monotonic()-start;own=resource.getrusage(resource.RUSAGE_SELF);done=resource.getrusage(resource.RUSAGE_CHILDREN)
                cpu=own.ru_utime+own.ru_stime+done.ru_utime+done.ru_stime
                try:
                    fields=Path(f'/proc/{child.pid}/stat').read_text().split();cpu+=(int(fields[13])+int(fields[14]))/ticks
                except FileNotFoundError:pass
                samples.append(dict(wall_s=elapsed,aggregate_cpu_s=cpu,pid=child.pid))
                assert elapsed<p['budget']['wall_seconds']and cpu<p['budget']['aggregate_cpu_seconds'];time.sleep(.1)
            assert child.returncode==0,log.read_text()
        finally:
            if child.poll()is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    own=resource.getrusage(resource.RUSAGE_SELF);done=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=own.ru_utime+own.ru_stime+done.ru_utime+done.ru_stime;wall=time.monotonic()-start
    assert cpu<180 and wall<240
    runtime=dict(status='validated',wall_s=wall,aggregate_cpu_s=cpu,gpu_seconds=0,cpu_workers=1,samples=samples,wrapper_sha256=sha(__file__),accounting='Parent self plus child user/system CPU; active child polled from/proc. One-thread BLAS, no GPU.')
    save(PUBLIC/'runtime.json',runtime);save(OUT/'runtime.json',runtime)
    print(json.dumps({k:v for k,v in runtime.items()if k!='samples'}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','compute','launch']);globals()[ap.parse_args().action]()
