"""Complete the fixed32-bank study under a separately frozen additional budget."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,resource,subprocess,sys,threading,time
from run import ROOT,HERE,sha,save,now
from adaptive_null_continuation import verify,PROTOCOL

def main():
    p,scientific,pre=verify();out=ROOT/'artifacts/research/20261114/ripper/adaptive_null';execution=ROOT/'artifacts/research/20261114/ripper/adaptive_null_continuation_execution';execution.mkdir(exist_ok=False)
    assert sorted(int(f.parent.name[1:])for f in out.glob('s*/receipt.json'))==list(range(8))
    assert sorted(int(f.parent.name[1:])for f in out.glob('s*/audit.json'))==[0]
    pre_cpu=pre['full_parent_child_cpu_s'];pre_wall=pre['wall_s'];started=time.monotonic();stop=threading.Event();finished=threading.Event();lock=threading.Lock();active={};completed=[];ticks=os.sysconf('SC_CLK_TCK')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def cpu_seconds():
        a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);total=pre_cpu+a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime
        with lock:pids=list(active)
        for pid in pids:
            try:fields=Path(f'/proc/{pid}/stat').read_text().split();total+=(int(fields[13])+int(fields[14]))/ticks
            except FileNotFoundError:pass
        return total
    def monitor():
        while not finished.is_set():
            cpu=cpu_seconds();wall=time.monotonic()-started+pre_wall
            with lock:processes=list(active.values());count=len(completed)
            record=dict(utc=now(),additional_cpu_including_preflight_s=cpu,additional_execution_wall_including_preflight_s=wall,active_pids=[x.pid for x in processes],completed_commands=count)
            save(execution/'progress.json',record)
            if cpu>=120 or wall>=60:
                stop.set();save(execution/'budget_exhausted.json',record)
                for proc in processes:
                    if proc.poll()is None:proc.terminate()
                return
            finished.wait(.1)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    def run(script,*args):
        assert not stop.is_set();name=script.removesuffix('.py')+'-'+'-'.join(map(str,args))
        with(execution/(name+'.log')).open('w')as f:
            proc=subprocess.Popen([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            with lock:active[proc.pid]=proc
            code=proc.wait()
            with lock:active.pop(proc.pid,None);completed.append(dict(script=script,args=list(args),returncode=code))
            assert code==0 and not stop.is_set(),(script,args,code)
    try:
        def bank(seed):
            if seed>=8:run('adaptive_null_acquire.py','--seed',seed)
            if seed!=0:run('audit_adaptive_null_eager.py','--seed',seed)
        with ThreadPoolExecutor(max_workers=8)as workers:list(workers.map(bank,range(32)))
        verify();assert len(list(out.glob('s*/audit.json')))==32
        run('summarize_adaptive_null_continuation.py')
        cpu=cpu_seconds();wall=time.monotonic()-started+pre_wall;assert cpu<120 and wall<60 and not stop.is_set()
        runtime=dict(status='validated_additional_budget',continuation_protocol_sha256=sha(PROTOCOL),additional_aggregate_cpu_s=cpu,additional_execution_wall_s=wall,continuation_cpu_s=cpu-pre_cpu,continuation_wall_s=wall-pre_wall,preflight_cpu_s=pre_cpu,preflight_wall_s=pre_wall,additional_cpu_ceiling_s=120,additional_wall_ceiling_s=60,max_cpu_workers=8,gpu_seconds=0,commands=completed,wrapper_sha256=sha(__file__),original_gate_status='failed_budget_limit',original_trigger_cpu_s=120.004219,original_trigger_wall_s=15.055995583999902,original_complete_final_cpu='unavailable',reused_acquisition_banks=list(range(8)),new_acquisition_banks=list(range(8,32)),reused_preflight_audits=[0])
        public=ROOT/'results/research/ripper';save(out/'continuation_runtime.json',runtime);save(public/'adaptive_null_runtime.json',runtime)
        result=json.loads((out/'audit.json').read_text());result['runtime']=runtime;save(out/'audit.json',result);save(public/'adaptive_null.json',result)
        with(HERE/'REPORT-adaptive-null.md').open('a')as f:f.write(f'\nThe separately authorized corrected continuation used {cpu:.6f} additional CPU-seconds and {wall:.6f} execution wall seconds, including its {pre_cpu:.6f} CPU-second / {pre_wall:.6f}-second preflight. This cost is reported separately from the failed original attempt; no combined exact CPU total is available.\n')
        print(json.dumps(dict(status=runtime['status'],primary=result['primary'],additional_cpu_s=cpu,additional_execution_wall_s=wall,new_acquisitions=24,reused_acquisitions=8)),flush=True)
    finally:
        finished.set();watcher.join(timeout=1)
        with lock:processes=list(active.values())
        for proc in processes:
            if proc.poll()is None:
                proc.terminate()
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
if __name__=='__main__':main()
