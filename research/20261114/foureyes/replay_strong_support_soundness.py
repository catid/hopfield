"""Timed isolated replay after the H4ab metadata-only verifier correction.

Preserves original certificates and reruns the unchanged producer with the
corrected hash helper, then requires byte-identical proof and outcome files.
"""
from pathlib import Path
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/20261114/foureyes/strong_support_soundness'
PUBLIC=ROOT/'results/research/foureyes/strong_support_soundness'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    replay=OUT/'corrected_replay';assert not replay.exists()
    original={n:sha(PUBLIC/n)for n in ['supports.npz','certificates.json','summary.json']}
    note=json.loads((HERE/'strong-soundness-audit-correction.json').read_text())
    assert note['original_verifier_sha256']==sha(HERE/'verify_strong_support_soundness.py')
    assert note['corrected_verifier_sha256']==sha(HERE/'verify_strong_support_soundness_corrected.py')
    driver="""
import sys
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'research/20261114/foureyes'))
import strong_support_soundness as producer
import verify_strong_support_soundness_corrected as verifier
producer.OUT=root/'artifacts/research/20261114/foureyes/strong_support_soundness/corrected_replay/work'
producer.PUBLIC=root/'artifacts/research/20261114/foureyes/strong_support_soundness/corrected_replay/public'
verifier.PUBLIC=producer.PUBLIC
sys.modules['verify_strong_support_soundness']=verifier
producer.compute()
"""
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=time.monotonic();samples=[];ticks=os.sysconf('SC_CLK_TCK')
    with (OUT/'corrected-replay.log').open('x')as log:
        child=subprocess.Popen([str(ROOT/'upstream/.venv/bin/python'),'-c',driver],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll()is None:
                own=resource.getrusage(resource.RUSAGE_SELF);done=resource.getrusage(resource.RUSAGE_CHILDREN)
                cpu=own.ru_utime+own.ru_stime+done.ru_utime+done.ru_stime
                try:
                    fields=Path(f'/proc/{child.pid}/stat').read_text().split();cpu+=(int(fields[13])+int(fields[14]))/ticks
                except FileNotFoundError:pass
                wall=time.monotonic()-started;samples.append(dict(wall_seconds=wall,cpu_seconds=cpu))
                assert cpu<180 and wall<240;time.sleep(.1)
            assert child.returncode==0,(OUT/'corrected-replay.log').read_text()
        finally:
            if child.poll()is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    own=resource.getrusage(resource.RUSAGE_SELF);done=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=own.ru_utime+own.ru_stime+done.ru_utime+done.ru_stime;wall=time.monotonic()-started
    assert cpu<180 and wall<240
    for n,h in original.items():assert sha(PUBLIC/n)==h==sha(replay/'public'/n),n
    receipt=dict(status='validated',source_sha256=sha(__file__),correction_sha256=sha(HERE/'strong-soundness-audit-correction.json'),
                 aggregate_cpu_seconds=cpu,wall_seconds=wall,original_artifacts_sha256=original,byte_identical_replay=True,
                 cpu_limit_seconds=180,wall_limit_seconds=240,gpu_seconds=0,samples=samples,
                 scope='Separate complete post-correction proof/verification replay, including process startup, imports and parent monitor. Original production artifacts unchanged. Not a solver experiment or speed claim.')
    (PUBLIC/'replay-runtime.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items()if k!='samples'}))


if __name__=='__main__':main()
