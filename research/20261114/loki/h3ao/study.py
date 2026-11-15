"""Source preparation, two useful GPU fit queues, exact repair, then target census."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import resource
import signal
import subprocess
import sys
import time
from common import ROOT,SRC,OUT,frozen,read,save,sha,now

def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime
def main():
    p=frozen();assert not(OUT/'study-result.json').exists();OUT.mkdir(parents=True,exist_ok=True);wall=time.perf_counter();start=cpu();workers=[];gpu_seconds=0.;utilization=[]
    def expired(*args):raise TimeoutError('H3ao600CPU/120GPU-worker/900wall gate')
    signal.signal(signal.SIGUSR1,expired);save(OUT/'study-start.json',dict(utc=now(),deadline_unix=time.time()+900,protocol_sha256=sha(SRC/'protocol.json')))
    try:
        subprocess.run([sys.executable,str(SRC/'source.py'),'prepare'],check=True)
        for gpu in(0,1):
            log=(OUT/f'fit-worker-g{gpu}.log').open('w');started=time.perf_counter();process=subprocess.Popen([sys.executable,str(SRC/'source.py'),'worker','--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT);workers.append(dict(gpu=gpu,process=process,started=started,ended=None,log=log))
        last_util=0.
        while any(w['process'].poll()is None for w in workers):
            current=time.perf_counter()
            for w in workers:
                if w['ended']is None and w['process'].poll()is not None:w['ended']=current
            gpu_seconds=sum((w['ended']or current)-w['started']for w in workers)
            assert gpu_seconds<=120,'Aggregate GPU-worker120second gate'
            if current-last_util>=.5:
                status=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True,timeout=10).strip();progress={str(w['gpu']):read(OUT/f"fit-worker-g{w['gpu']}.json").get('progress',{})if(OUT/f"fit-worker-g{w['gpu']}.json").exists()else{'stage':'imports'}for w in workers};utilization.append(dict(utc=now(),aggregate_GPU_worker_seconds=gpu_seconds,gpu_status=status,counters=progress));save(OUT/'live-utilization.json',dict(samples=utilization));last_util=current
            time.sleep(.02)
        current=time.perf_counter()
        for w in workers:
            if w['ended']is None:w['ended']=current
            w['log'].close();assert w['process'].returncode==0,f"GPU{w['gpu']}fit worker failure"
        gpu_seconds=sum(w['ended']-w['started']for w in workers);assert gpu_seconds<=120
        fits=[read(OUT/f'b{index}/model.json')for index in range(8)]
        save(OUT/'source-fits-seal.json',dict(utc=now(),all_eight_fits_retained=True,source_eligible_fits=sum(r['source_eligible']for r in fits),models={str(index):sha(OUT/f'b{index}/model.npz')for index in range(8)},aggregate_GPU_worker_seconds=gpu_seconds,worker_wall_seconds={str(w['gpu']):w['ended']-w['started']for w in workers},fit_update_counts={str(index):fits[index]['updates']for index in range(8)}))
        subprocess.run([sys.executable,str(SRC/'source_outcomes.py')],check=True)
        subprocess.run([sys.executable,str(SRC/'evaluate.py')],check=True)
        result=read(OUT/'evaluation.json');save(OUT/'study-result.json',dict(result,utc=now(),aggregate_GPU_worker_seconds=gpu_seconds,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,source_fits_seal_sha256=sha(OUT/'source-fits-seal.json'),study_source_sha256=sha(__file__)))
    except BaseException as exc:
        for w in workers:
            if w['process'].poll()is None:w['process'].terminate()
        for w in workers:
            try:w['process'].wait(timeout=2)
            except subprocess.TimeoutExpired:w['process'].kill();w['process'].wait()
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),aggregate_GPU_worker_seconds=gpu_seconds,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),partial_results_usable=False));raise
if __name__=='__main__':main()
