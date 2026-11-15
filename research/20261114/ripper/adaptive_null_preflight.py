"""Bounded loader-equivalence gate and one complete corrected bank audit."""
from pathlib import Path
import argparse,ast,hashlib,json,os,resource,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
RAW=ROOT/'artifacts/research/20261114/ripper/adaptive_null';OUT=ROOT/'artifacts/research/20261114/ripper/adaptive_null_preflight'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def worker():
    import numpy as np
    old=(HERE/'audit_adaptive_null.py').read_text();new=(HERE/'audit_adaptive_null_eager.py').read_text()
    helper='\ndef eager_npz(path):\n    with np.load(path,allow_pickle=False) as archive:\n        return {name:archive[name] for name in archive.files}\n\n'
    expected=old.replace('\ndef main():',helper+'\ndef main():',1)
    for a,b in [("obs=np.load(d/'observations.npz')","obs=eager_npz(d/'observations.npz')"),("noise=np.load(d/'potential_noise.npz')","noise=eager_npz(d/'potential_noise.npz')"),("oldnoise=np.load(olddir/'potential_noise.npz')","oldnoise=eager_npz(olddir/'potential_noise.npz')"),("oldobs=np.load(olddir/'observations.npz')","oldobs=eager_npz(olddir/'observations.npz')")]:assert expected.count(a)==1;expected=expected.replace(a,b)
    assert expected==new and ast.dump(ast.parse(expected))==ast.dump(ast.parse(new))
    original=json.loads((HERE/'adaptive-null-original-artifacts.json').read_text())
    for p,d in original['artifact_sha256'].items():assert sha(ROOT/p)==d
    import audit_adaptive_null_eager as audit
    specs=[]
    for bank in range(8):
        for base in [RAW,ROOT/'artifacts/research/20261114/ripper/permuted_noise']:
            for name in ['observations.npz','potential_noise.npz']:
                path=base/f's{bank}'/name;eager=audit.eager_npz(path)
                with np.load(path,allow_pickle=False)as lazy:
                    assert set(eager)==set(lazy.files)
                    for key in lazy.files:
                        a,b=lazy[key],eager[key];assert a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
                        digest=hashlib.sha256(a.tobytes()).hexdigest();assert digest==hashlib.sha256(b.tobytes()).hexdigest()
                        specs.append(dict(path=str(path.relative_to(ROOT)),array=key,dtype=str(a.dtype),shape=list(a.shape),bytes_sha256=digest))
    sys.argv=[str(HERE/'audit_adaptive_null_eager.py'),'--seed','0'];audit.main()
    for p,d in original['artifact_sha256'].items():assert sha(ROOT/p)==d
    result=dict(status='passed',exact_source_and_ast_eager_only_change=True,original_auditor_sha256=sha(HERE/'audit_adaptive_null.py'),corrected_auditor_sha256=sha(HERE/'audit_adaptive_null_eager.py'),diff_sha256=sha(HERE/'adaptive-null-eager-audit.diff'),original_manifest_sha256=sha(HERE/'adaptive-null-original-artifacts.json'),arrays_checked=specs,complete_corrected_bank=0,corrected_bank_audit_sha256=sha(RAW/'s0/audit.json'),preserved_original_files=len(original['artifact_sha256']),aggregate_endpoint_computed=False)
    (OUT/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
def main():
    assert not OUT.exists();OUT.mkdir();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');started=time.monotonic();ticks=os.sysconf('SC_CLK_TCK');samples=[]
    with (OUT/'worker.log').open('x')as f:
        p=subprocess.Popen([sys.executable,str(Path(__file__)),'worker'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:
            while p.poll()is None:
                own=resource.getrusage(resource.RUSAGE_SELF);done=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=own.ru_utime+own.ru_stime+done.ru_utime+done.ru_stime
                try:s=Path(f'/proc/{p.pid}/stat').read_text().split();cpu+=(int(s[13])+int(s[14]))/ticks
                except FileNotFoundError:pass
                elapsed=time.monotonic()-started;samples.append(dict(cpu_s=cpu,wall_s=elapsed));assert cpu<10 and elapsed<10;time.sleep(.05)
            assert p.returncode==0,(OUT/'worker.log').read_text()
        finally:
            if p.poll()is None:
                p.terminate()
                try:p.wait(timeout=2)
                except subprocess.TimeoutExpired:p.kill();p.wait()
    own=resource.getrusage(resource.RUSAGE_SELF);done=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=own.ru_utime+own.ru_stime+done.ru_utime+done.ru_stime;wall=time.monotonic()-started
    assert cpu<10 and wall<10
    r=dict(status='validated',full_parent_child_cpu_s=cpu,wall_s=wall,cpu_ceiling_s=10,wall_ceiling_s=10,charged_to_additional_continuation_budget=True,wrapper_sha256=sha(__file__),checks_sha256=sha(OUT/'checks.json'),samples=samples)
    (OUT/'runtime.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='samples'}))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','worker']);a=parser.parse_args();worker()if a.action=='worker'else main()
