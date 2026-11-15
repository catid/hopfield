"""Bounded, fail-fast H1ch execution on Ripper; no retries or filtered banks."""
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
from correlated_repeat_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,sha,save,verify


def command(action, seed=None):
    assert action in ['select','predraw-all','materialize','source-audit','prepare','worker','summary']
    assert (action=='worker' and type(seed) is int and 0<=seed<8) or (action!='worker' and seed is None)
    args=[sys.executable,str(HERE/'correlated_repeat_study.py'),action]
    if seed is not None:args+=['--seed',str(seed)]
    return args


def discard_primary():
    # Only this attempt's aggregate decisions are removed; bank evidence stays.
    for path in [OUT/'audit.json',PUBLIC/'summary.json']:
        path.unlink(missing_ok=True)


def main():
    assert platform.node()=='ripper';p=verify();budget=p['budget']
    assert not OUT.exists()
    execution=OUT.parent/'correlated_repeat_execution';execution.mkdir(exist_ok=False)
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
            per_pid={}
            for pid in pids:
                try:
                    fields=Path(f'/proc/{pid}/stat').read_text().split();per_pid[str(pid)]=(int(fields[13])+int(fields[14]))/ticks
                except FileNotFoundError:pass
            sample=dict(wall_s=wall,cpu_s=cpu,active_pids=pids,per_pid_cpu_s=per_pid,completed_commands=count,
                        completed_LPs=sum(json.loads(path.read_text())['completed_LPs'] for path in OUT.glob('s*/progress.json')),audited_banks=len(list(OUT.glob('s*/audit.json'))))
            save(execution/'progress.json',sample)
            with (execution/'samples.jsonl').open('a') as log:log.write(json.dumps(sample)+'\n')
            if wall>=budget['wall_seconds'] or cpu>=budget['aggregate_cpu_seconds']:
                halt('Frozen runtime ceiling reached');return
            finished.wait(.1)

    def run(action,seed=None):
        assert not stop.is_set()
        args=command(action,seed)
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

    complete=False;error=None
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        for action in ['select','predraw-all','materialize','source-audit','prepare']:run(action)
        with ThreadPoolExecutor(max_workers=budget['workers']) as pool:
            list(pool.map(lambda seed:run('worker',seed),range(8)))
        run('summary');wall=time.monotonic()-start;cpu=cpu_seconds()
        assert wall<budget['wall_seconds'] and cpu<budget['aggregate_cpu_seconds'] and not stop.is_set()
        runtime=dict(status='validated',wall_s=wall,aggregate_cpu_s=cpu,
                     wall_ceiling_s=budget['wall_seconds'],aggregate_cpu_ceiling_s=budget['aggregate_cpu_seconds'],
                     max_cpu_workers=budget['workers'],gpu_seconds=0,commands=completed,
                     wrapper_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),all_banks_retained=True,wall_timer_scope='From before numerical imports through scientific execution; CPU from process lifetime and reaped children.')
        finished.set();watcher.join(timeout=2)
        result=json.loads((OUT/'audit.json').read_text())
        save(execution/'progress.json',dict(wall_s=wall,cpu_s=cpu,active_pids=[],completed_commands=len(completed),completed_LPs=result['attempted_LPs'],audited_banks=72,status='complete'))
        save(OUT/'runtime.json',runtime);save(PUBLIC/'runtime.json',runtime)
        result=json.loads((OUT/'audit.json').read_text());result['runtime']=runtime
        save(OUT/'audit.json',result);save(PUBLIC/'summary.json',result)
        print(json.dumps(dict(endpoints=result['endpoints'],grouped=result['grouped'],runtime=runtime)),flush=True)
        complete=True
    except BaseException as exc:
        error=repr(exc)
        raise
    finally:
        finished.set();watcher.join(timeout=2)
        with lock:processes=list(active.values())
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if not complete:
            discard_primary()
            record=dict(status='incomplete',error=error,wall_s=time.monotonic()-start,aggregate_cpu_s=cpu_seconds(),
                        commands=completed,wrapper_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),
                        primary_decision_committed=False)
            save(execution/'incomplete-runtime.json',record)
            if OUT.exists():save(OUT/'runtime.json',record)
            if PUBLIC.exists():save(PUBLIC/'runtime.json',record)


if __name__=='__main__':main()
