"""H2br full-study zero-draw lifecycle, source-boundary and audit structure gate."""
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import study as S
from freeze_endpoint_dispatch import modules


def check_worker(tmp):
    env={**os.environ,'PYTHONPATH':str(S.HERE),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    code='import sys;from pathlib import Path;import endpoint_dispatch_study as M;M.DEST=Path(sys.argv[1]);M.PP=Path(sys.argv[2]);M.worker("timing",0)'
    out=tmp/'worker';args=[sys.executable,'-c',code,str(out),str(tmp/'missing')]
    first=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10);assert first.returncode!=0
    file=out/'failure-timing-gpu0.json';before=S.sha(file);assert json.loads(file.read_text())['status']=='incomplete'
    second=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10);assert second.returncode!=0 and 'FileExistsError' in second.stderr and S.sha(file)==before
    fifo=tmp/'blocked';os.mkfifo(fifo);out=tmp/'signal';child=subprocess.Popen([sys.executable,'-c',code,str(out),str(fifo)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+5
        while not (out/'timing-worker0.json').exists():
            assert child.poll() is None and time.monotonic()<deadline;time.sleep(.01)
        child.send_signal(signal.SIGALRM);child.communicate(timeout=5)
        r=json.loads((out/'failure-timing-gpu0.json').read_text());assert r['status']=='incomplete' and 'TimeoutError' in r['error']
    finally:
        if child.poll() is None:child.kill();child.communicate()
    return env


def check_cpu(tmp,env):
    code='''
import json,sys
from pathlib import Path
import study as S
import prepare_endpoint_dispatch as G
import audit_endpoint_dispatch_inputs as I
base=Path(sys.argv[1]);base.mkdir();G.DEST=base;G.PP=base/'protocol.json';G.PP.write_text('{}');packet=base/'source-compatibility-packet.json';packet.write_text('{}');gates=base/'source-compatibility-attempt1';gates.mkdir()
G.protocol=lambda:{'cohort':{'base_seed':2026111532,'n':4096},'budgets':{'fresh_generation_wall_seconds':300,'fresh_generation_processes':8},'compatibility_packet_sha256':S.sha(packet)}
for gpu in [0,1]:S.save(gates/f'gate-gpu{gpu}.json',{'status':'passed','protocol_sha256':S.sha(packet),'source_model_inputs_per_arm':384})
class FailedPool:
 def __init__(self,**kwargs):pass
 def map(self,*args,**kwargs):raise RuntimeError('Synthetic failure before any draw')
 def shutdown(self,**kwargs):pass
G.ProcessPoolExecutor=FailedPool
for i in range(2):
 try:G.main()
 except (RuntimeError,FileExistsError) as e:assert isinstance(e,RuntimeError if i==0 else FileExistsError)
 else:raise AssertionError('Generation attempt not refused')
r=json.loads((base/'generation-worker.json').read_text());assert r['status']=='incomplete' and r['inputs_drawn']==0
I.DEST=base;I.PP=base/'missing-input-protocol'
for i in range(2):
 try:I.main()
 except FileNotFoundError:assert i==0
 except FileExistsError:assert i==1
 else:raise AssertionError('Input audit attempt accepted')
assert json.loads((base/'input-audit-worker.json').read_text())['status']=='incomplete'
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp/'cpu')],env=env,capture_output=True,text=True,timeout=10);assert result.returncode==0,result.stderr
    out=tmp/'audit';out.mkdir();command=[sys.executable,str(S.HERE/'audit_endpoint_dispatch.py'),str(S.ROOT),str(out),'--workers','16']
    first=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10);assert first.returncode!=0
    file=out/'audit-worker.json';before=S.sha(file);assert json.loads(file.read_text())['status']=='incomplete'
    second=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10);assert second.returncode!=0 and 'FileExistsError' in second.stderr and S.sha(file)==before


def main():
    started=time.perf_counter();tree=ast.parse((S.HERE/'endpoint_dispatch_study.py').read_text())
    reads=[n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='archive' and isinstance(n.slice,ast.Constant)];assert reads==['tokens','tokens']
    original=ast.parse((S.HERE/'audit_bp_first_fresh_states.py').read_text());adapted=(S.HERE/'audit_endpoint_dispatch_states.py').read_text().replace("'block0-LP96'","'pair0-LP96'").replace("'block0-BP_geometry_LP96'","'pair0-BP_geometry_LP96'");adapted=ast.parse(adapted)
    assert ast.dump(ast.Module(body=original.body[1:],type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=adapted.body[1:],type_ignores=[]),include_attributes=False)
    packet=json.loads((S.OUT/'bp_first_endpoint_dispatch/source-compatibility-packet.json').read_text())
    for key in ['source_sha256','source_evidence_sha256']:
        for path,digest in packet[key].items():assert S.sha(S.ROOT/path)==digest,path
    with tempfile.TemporaryDirectory(prefix='h2br-study-lifecycle-') as tmp:
        tmp=Path(tmp);env=check_worker(tmp);check_cpu(tmp,env)
    names=modules()
    for name in names:compile((S.HERE/name).read_text(),name,'exec')
    result={'status':'passed','synthetic_only':True,'fresh_worker_archive_reads':reads,'direct_original_factor_auditor_AST_unchanged_except_block_directory_names':True,'source_compatibility_bindings_unchanged':True,'timing_worker_failure_overwrite_and_owned_alarm_passed':True,'generation_failure_zero_draw_and_overwrite_passed':True,'input_audit_failure_and_overwrite_passed':True,'full_audit_failure_and_overwrite_passed':True,'fresh_inputs_drawn':0,'seconds':time.perf_counter()-started,'source_sha256':{name:S.sha(S.HERE/name) for name in names}}
    S.save(S.OUT/'bp_first_endpoint_dispatch/study-synthetic-gate.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
