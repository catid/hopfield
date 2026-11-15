"""Bounded, fail-fast H1bf execution on Ripper; no retries or filtered banks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
for _key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[_key]='1'
WALL_ORIGIN=time.monotonic()
from transfer_feedback_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,sha,save,verify


def main():
    assert platform.node()=='ripper';p=verify();budget=p['budget']
    execution=OUT.parent/'transfer_feedback_execution';execution.mkdir(exist_ok=False)
    start=WALL_ORIGIN;active={};lock=threading.Lock();failure_lock=threading.Lock();finished=threading.Event();stop=threading.Event();completed=[]
    ticks=os.sysconf('SC_CLK_TCK');env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

    def cpu_seconds():
        a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
        total=a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime
        with lock:pids=list(active)
        for pid in pids:
            try:
                fields=Path(f'/proc/{pid}/stat').read_text().split();total+=(int(fields[13])+int(fields[14]))/ticks
            except FileNotFoundError:pass
        return total

    def halt(reason):
        stop.set()
        with lock:processes=list(active.values())
        with failure_lock:
            if not (execution/'failure.json').exists():
                save(execution/'failure.json',dict(reason=reason,wall_s=time.monotonic()-start,cpu_s=cpu_seconds(),owned_pids=[v.pid for v in processes]))
        for process in processes:
            if process.poll() is None:process.terminate()

    def monitor():
        while not finished.is_set():
            wall=time.monotonic()-start;cpu=cpu_seconds()
            with lock:pids=list(active);count=len(completed)
            save(execution/'progress.json',dict(wall_s=wall,cpu_s=cpu,active_pids=pids,completed_commands=count))
            if wall>=budget['wall_seconds'] or cpu>=budget['aggregate_cpu_seconds']:
                halt('Frozen runtime ceiling reached');return
            finished.wait(.1)

    def run(action,seed=None):
        assert not stop.is_set()
        args=[sys.executable,str(HERE/'transfer_feedback_study.py'),action]
        if seed is not None:args+=['--seed',str(seed)]
        with (execution/f'{action}-{seed}.log').open('x') as log:
            with lock:
                assert not stop.is_set()
                proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[proc.pid]=proc
            code=proc.wait()
            with lock:
                active.pop(proc.pid,None);completed.append(dict(action=action,seed=seed,returncode=code))
            if code!=0:
                halt(f'{action} seed={seed} returncode={code}')
            assert code==0 and not stop.is_set(),(action,seed,code)

    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        run('prepare')
        with ThreadPoolExecutor(max_workers=budget['max_cpu_workers']) as pool:
            list(pool.map(lambda seed:run('bank',seed),p['seeds']))
        run('summary');wall=time.monotonic()-start;cpu=cpu_seconds()
        assert wall<budget['wall_seconds'] and cpu<budget['aggregate_cpu_seconds'] and not stop.is_set()
        runtime=dict(status='validated',wall_s=wall,aggregate_cpu_s=cpu,
                     wall_ceiling_s=budget['wall_seconds'],aggregate_cpu_ceiling_s=budget['aggregate_cpu_seconds'],
                     max_cpu_workers=budget['max_cpu_workers'],gpu_seconds=0,commands=completed,
                     wrapper_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),all_banks_retained=True,wall_timer_scope='From before numerical imports through scientific execution; CPU from process lifetime and reaped children.')
        save(execution/'progress.json',dict(wall_s=wall,cpu_s=cpu,active_pids=[],completed_commands=len(completed),status='complete'))
        save(OUT/'runtime.json',runtime);save(PUBLIC/'transfer_feedback_runtime.json',runtime)
        result=json.loads((OUT/'audit.json').read_text());result['runtime']=runtime
        save(OUT/'audit.json',result);save(PUBLIC/'transfer_feedback.json',result)
        print(json.dumps(dict(primary=result['primary'],grouped=result['grouped'],runtime=runtime)),flush=True)
    finally:
        finished.set();watcher.join(timeout=2)
        with lock:processes=list(active.values())
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__=='__main__':main()
