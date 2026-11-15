"""H2br source-gate lifecycle, control delegation and archive-read checks."""
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


def main():
    started=time.perf_counter();tree=ast.parse((S.HERE/'endpoint_dispatch_batch.py').read_text());run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
    expected=ast.parse("if mode in ['LP96','BP_geometry_LP96']:\n return unchanged(mode,host_tokens,models,doubles,watchdog,after_answer)\n").body[0]
    assert ast.dump(run.body[0],include_attributes=False)==ast.dump(expected,include_attributes=False)
    source=ast.parse((S.HERE/'endpoint_dispatch_compatibility.py').read_text())
    reads=[n.slice.value for n in ast.walk(source) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='z' and isinstance(n.slice,ast.Constant)]
    assert reads==['tokens']
    env={**os.environ,'PYTHONPATH':str(S.HERE),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    command='import sys;from pathlib import Path;import endpoint_dispatch_compatibility as M;M.DEST=Path(sys.argv[1]);M.PP=Path(sys.argv[2]);M.worker("gate",0)'
    with tempfile.TemporaryDirectory(prefix='h2br-source-lifecycle-') as tmp:
        tmp=Path(tmp);out=tmp/'failed';args=[sys.executable,'-c',command,str(out),str(tmp/'missing-packet')]
        first=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10);assert first.returncode!=0
        failure=out/'failure-gate-gpu0.json';digest=S.sha(failure);assert json.loads(failure.read_text())['status']=='incomplete'
        second=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10);assert second.returncode!=0 and 'FileExistsError' in second.stderr and S.sha(failure)==digest
        fifo=tmp/'blocked-packet';os.mkfifo(fifo);out=tmp/'signal';child=subprocess.Popen([sys.executable,'-c',command,str(out),str(fifo)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+5
            while not (out/'gate-worker0.json').exists():
                assert child.poll() is None and time.monotonic()<deadline;time.sleep(.01)
            child.send_signal(signal.SIGALRM);child.communicate(timeout=5)
            receipt=json.loads((out/'failure-gate-gpu0.json').read_text());assert receipt['status']=='incomplete' and 'TimeoutError' in receipt['error']
        finally:
            if child.poll() is None:child.kill();child.communicate()
    names=['endpoint_dispatch_batch.py','endpoint_dispatch_compatibility.py','endpoint_dispatch_pipeline.py','endpoint_dispatch_source_parity.py','monitor_endpoint_dispatch_compatibility.py','preflight_endpoint_dispatch_compatibility.py','bp_first_fresh_batch.py','bp_first_geometry_barrier.py','bounded_archive_writer.py']
    for name in names:compile((S.HERE/name).read_text(),name,'exec')
    result={'status':'passed','synthetic_only':True,'controls_delegate_to_frozen_unchanged_batch':True,'worker_archive_reads':reads,'source_worker_failure_receipt_retained':True,'existing_attempt_overwrite_refused':True,'owned_alarm_failure_receipt_retained':True,'fresh_inputs_drawn':0,'seconds':time.perf_counter()-started,'source_sha256':{name:S.sha(S.HERE/name) for name in names}}
    S.save(S.OUT/'bp_first_endpoint_dispatch/compatibility-synthetic-gate.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
