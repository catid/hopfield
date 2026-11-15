"""Eight-worker CPU pipeline with live aggregate-CPU and wall ceilings."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
from run import ROOT, HERE, sha, save, now


def main():
    protocol=json.loads((HERE/'protocol-adaptive-null.json').read_text())
    for name,expected in protocol['code_sha256'].items():assert sha(HERE/name)==expected,name
    assert protocol['source_manifest_sha256']==sha(ROOT/protocol['inputs'])
    started=time.monotonic();started_utc=now();stop=threading.Event();finished=threading.Event();lock=threading.Lock();active={};completed=[]
    execution=ROOT/'artifacts/research/20261114/ripper/adaptive_null_execution';execution.mkdir(exist_ok=False)
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    ticks=os.sysconf('SC_CLK_TCK')
    def cpu_seconds():
        own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        total=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime
        with lock:pids=list(active)
        for pid in pids:
            try:
                fields=Path(f'/proc/{pid}/stat').read_text().split();total+=(int(fields[13])+int(fields[14]))/ticks
            except FileNotFoundError:pass
        return total
    def monitor():
        while not finished.is_set():
            cpu=cpu_seconds();elapsed=time.monotonic()-started
            with lock:processes=list(active.values());count=len(completed)
            record={'utc':now(),'elapsed_s':elapsed,'aggregate_cpu_s':cpu,'active_pids':[p.pid for p in processes],'completed_commands':count}
            save(execution/'progress.json',record)
            if elapsed>=60 or cpu>=120:
                stop.set();save(execution/'budget_exhausted.json',record)
                for process in processes:
                    if process.poll() is None:process.terminate()
                return
            finished.wait(.25)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    def run(script,*arguments):
        assert not stop.is_set()
        label=script.removesuffix('.py')+'-'+'-'.join(str(x).replace('--','') for x in arguments)
        with (execution/(label+'.log')).open('w') as log:
            process=subprocess.Popen([sys.executable,str(HERE/script),*map(str,arguments)],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
            with lock:active[process.pid]=process
            code=process.wait()
            with lock:active.pop(process.pid,None);completed.append({'script':script,'arguments':list(arguments),'returncode':code})
            assert code==0 and not stop.is_set(),(script,arguments,code)
    try:
        for script in ['prepare_adaptive_null.py']:
            run(script)
        def bank(seed):
            run('adaptive_null_acquire.py','--seed',seed)
            run('audit_adaptive_null.py','--seed',seed)
        with ThreadPoolExecutor(max_workers=8) as workers:list(workers.map(bank,range(32)))
        run('summarize_adaptive_null.py')
        elapsed=time.monotonic()-started;cpu=cpu_seconds();assert elapsed<60 and cpu<120 and not stop.is_set()
        runtime={'status':'validated','started_utc':started_utc,'completed_utc':now(),'wall_s':elapsed,'aggregate_cpu_s':cpu,
            'wall_ceiling_s':60,'aggregate_cpu_ceiling_s':120,'max_cpu_workers':8,'gpu_seconds':0,
            'wrapper_sha256':sha(__file__),'commands':completed,'accounting':'Process user+system CPU for this orchestrator and all reaped children; active child CPU polled from /proc. OMP/OpenBLAS/MKL threads fixed to1.'}
        out=ROOT/'artifacts/research/20261114/ripper/adaptive_null';public=ROOT/'results/research/ripper'
        save(out/'runtime.json',runtime);save(public/'adaptive_null_runtime.json',runtime)
        result=json.loads((out/'audit.json').read_text());result['runtime']=runtime
        save(out/'audit.json',result);save(public/'adaptive_null.json',result)
        with (HERE/'REPORT-adaptive-null.md').open('a') as report:
            report.write(f'\nExecution used {cpu/60:.4f} aggregate CPU-min and {elapsed:.4f} seconds wall time, within the frozen2 CPU-min/60-second ceilings. Up to eight CPU workers ran; no GPUs were allocated.\n')
        print(json.dumps({'status':'validated','wall_s':elapsed,'aggregate_cpu_s':cpu,'commands':len(completed),'primary':result['primary']}),flush=True)
    finally:
        finished.set();watcher.join(timeout=1)
        with lock:processes=list(active.values())
        for process in processes:
            if process.poll() is None:process.terminate()


if __name__=='__main__':main()
