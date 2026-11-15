"""H1aq: global zero-energy soundness of every fixed H1ap learned support."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,resource,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;FOUREYES=HERE.parent/'foureyes'
PARENT=ROOT/'artifacts/research/20261114/ripper/permuted_noise'
OUT=ROOT/'artifacts/research/20261114/ripper/permuted_soundness'
PUBLIC=ROOT/'results/research/ripper';PROTOCOL=HERE/'protocol-permuted-soundness.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def verify():
    p=json.loads(PROTOCOL.read_text())
    for path,digest in (p['code_sha256']|p['input_sha256']).items():assert sha(ROOT/path)==digest,path
    return p

def freeze():
    assert not PROTOCOL.exists()and not OUT.exists()
    code=[HERE/n for n in ['permuted_soundness.py','verify_permuted_soundness.py','counterexample_coloring.py']]+[FOUREYES/n for n in ['certify_sudoku_soundness.py','verify_sudoku_soundness.py']]
    inputs=[PARENT/'audit.json',HERE/'protocol-permuted-noise.json']
    for b in range(32):inputs += [PARENT/f's{b}'/n for n in ['fit.npz','receipt.json','audit.json']]
    p=dict(id='H1aq-global-zero-set-soundness-after-true-edge-loss',frozen_utc=datetime.now(timezone.utc).isoformat(),banks=32,
        scope='All32 fixed H1ap permutation-noise supports; post-hoc exact graph audit. No fit modification, source rows, queries, puzzles or GPU work. Geometry is supplied only to the audit.',
        hypothesis='At least29/32 graphs imply ordinary Sudoku validity for every zero-energy coloring over9 symbols.',
        proof='Unchanged frozen Foureyes clique/count-identity unit closure. A9-clique uses all9 colors; certified partition count identities imply a final missing unit. Certify sound only when all27 units are proved; independently replay each proof with the frozen Counter-based verifier.',
        counterexample_search='For closure-unresolved graphs, enumerate missing true pairs in lexicographic order, force pair equal with color0, and exact deterministic DSATUR: highest saturation, then total degree, then lowest vertex index; ascending canonical colors0..min(maxused+1,8). Shared per-graph2CPU-second/100000-node ceiling. Only explicit proper-support9-colorings violating a Sudoku unit prove unsound. Complete no-witness searches also remain unproved for the primary, because no portable unsatisfiability proof is exported.',
        decision='Supported if at least29 independently replayed27-unit proofs; falsified if at least4 independently checked explicit invalid zero-energy witnesses; otherwise inconclusive. Timeout, missing edges, no observed wrong stop and incomplete closure are never counted as unsound.',
        audit='Preserve all32 banks and graph identities; independently verify frozen fit hashes, support shape/dtype, all27-unit traces, every supplied invalid board, pair inequality on every learned edge, and outcome classification. Reject mutated count proofs. Export supports and proof/witness certificates.',
        budget=dict(aggregate_cpu_seconds=120,wall_seconds=180,cpu_workers=1,per_unresolved_graph_cpu_seconds=2,node_limit=100000,gpu_seconds=0),
        code_sha256={str(f.relative_to(ROOT)):sha(f)for f in code},input_sha256={str(f.relative_to(ROOT)):sha(f)for f in inputs})
    save(PROTOCOL,p);print(json.dumps({'protocol_sha256':sha(PROTOCOL)}))

def compute():
    import numpy as np
    p=verify();assert not OUT.exists();OUT.mkdir(parents=True);sys.path.insert(0,str(FOUREYES))
    from certify_sudoku_soundness import close
    from counterexample_coloring import search
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
    parent=json.loads((PARENT/'audit.json').read_text());rows=[];supports=[];cache={}
    for b in range(32):
        path=PARENT/f's{b}'/'fit.npz';r=json.loads((PARENT/f's{b}'/'receipt.json').read_text());assert r['artifact_sha256']['fit.npz']==sha(path)
        with np.load(path,allow_pickle=False)as z:support=z['support'].copy()
        assert support.dtype==bool and support.shape==(3240,);ident=sha(path);rawid=hashlib.sha256(support.tobytes()).hexdigest();supports.append(support)
        if rawid not in cache:
            trace,known=close(support)
            counter=None
            if len(known)<27:
                pairs=[(int(i),int(j))for i,j in zip(ii[truth&~support],jj[truth&~support])]
                counter=search(support,pairs,cpu_seconds=2,node_limit=100000)
            cache[rawid]=dict(trace=trace,certified_units=known,sound=len(known)==27,counterexample_search=counter)
        row=dict(bank=b,fit_sha256=ident,support_sha256=rawid,missing_true_edges=int((truth&~support).sum()),extra_edges=int((~truth&support).sum()),**cache[rawid])
        row['classification']='sound_proof'if row['sound']else'unsound_witness'if row['counterexample_search']['status']=='explicit_counterexample'else'unresolved'
        rows.append(row);save(OUT/'progress.json',dict(completed_banks=b+1,unique_supports=len(cache)))
    PUBLIC.mkdir(parents=True,exist_ok=True);np.savez_compressed(PUBLIC/'permuted_soundness_supports.npz',banks=np.arange(32),supports=np.stack(supports))
    certificate=dict(status='produced_pending_independent_verification',protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),parent_audit_sha256=sha(PARENT/'audit.json'),supports_sha256=sha(PUBLIC/'permuted_soundness_supports.npz'),unique_supports=len(cache),rows=rows)
    save(PUBLIC/'permuted_soundness_certificates.json',certificate)
    from verify_permuted_soundness import audit
    audit()

def launch():
    verify();log=OUT.parent/'permuted_soundness-execution.log';assert not log.exists();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');start=time.monotonic();samples=[];ticks=os.sysconf('SC_CLK_TCK')
    with log.open('x')as f:
        process=subprocess.Popen([sys.executable,str(Path(__file__)), 'compute'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:
            while process.poll()is None:
                elapsed=time.monotonic()-start
                try:
                    fields=Path(f'/proc/{process.pid}/stat').read_text().split();cpu=(int(fields[13])+int(fields[14]))/ticks
                except FileNotFoundError:cpu=0
                samples.append(dict(wall_s=elapsed,cpu_s=cpu,pid=process.pid));assert elapsed<180 and cpu<120;time.sleep(.1)
            assert process.returncode==0,log.read_text()
        finally:
            if process.poll()is None:process.terminate();process.wait(timeout=5)
    usage=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=usage.ru_utime+usage.ru_stime;runtime=dict(status='validated',wall_s=time.monotonic()-start,aggregate_cpu_s=cpu,wall_ceiling_s=180,aggregate_cpu_ceiling_s=120,cpu_workers=1,gpu_seconds=0,samples=samples,wrapper_sha256=sha(__file__))
    assert cpu<120 and runtime['wall_s']<180
    result=json.loads((OUT/'audit.json').read_text());result['runtime']=runtime;save(OUT/'runtime.json',runtime);save(PUBLIC/'permuted_soundness_runtime.json',runtime);save(OUT/'audit.json',result);save(PUBLIC/'permuted_soundness.json',result)
    with (HERE/'REPORT-permuted-soundness.md').open('a')as f:f.write(f'\nExecution used {cpu:.4f} CPU-seconds and {runtime["wall_s"]:.4f} wall seconds, within the frozen 120 CPU-second / 180-second ceilings.\n')
    print(json.dumps(dict(primary=result['primary'],counts=result['counts'],runtime={k:v for k,v in runtime.items()if k!='samples'})))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','compute','launch']);globals()[parser.parse_args().action]()
