"""Inclusive whole-study600CPU-second/900wall-second monitor."""
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from common import SRC,OUT,frozen,read,save,sha,now
TICKS=os.sysconf('SC_CLK_TCK')
def own():r=resource.getrusage(resource.RUSAGE_SELF);return r.ru_utime+r.ru_stime
def children():r=resource.getrusage(resource.RUSAGE_CHILDREN);return r.ru_utime+r.ru_stime
def tree(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();kids=Path(f'/proc/{pid}/task/{pid}/children').read_text().split();return sum(int(stat[k])for k in(11,12,13,14))/TICKS+sum(tree(int(child))for child in kids)
    except FileNotFoundError:return 0.
def main():
    frozen();assert not(OUT/'capacity-gate.json').exists();OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();begin=own();old=children();process=subprocess.Popen([sys.executable,str(SRC/'study.py')],start_new_session=True);peak=0.;expired=False
    while process.poll()is None:
        used=own()-begin+tree(process.pid);peak=max(peak,used)
        if used>=599.5 or time.perf_counter()-start>=899.5:
            expired=True;os.kill(process.pid,signal.SIGUSR1)
            try:process.wait(timeout=.3)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            break
        time.sleep(.02)
    total=own()-begin+children()-old;elapsed=time.perf_counter()-start;raw=read(OUT/'study-result.json')if(OUT/'study-result.json').exists()else dict(status='failed_or_incomplete',error='No final stage receipt');passed=process.returncode==0 and not expired and raw['status']=='complete_verified'and total<=600 and elapsed<=900 and raw['aggregate_GPU_worker_seconds']<=120
    save(OUT/'capacity-gate.json',dict(raw,status='complete_verified'if passed else'failed_or_incomplete',utc=now(),aggregate_cpu_seconds=total,wall_seconds=elapsed,monitor_peak_live_cpu_seconds=peak,monitor_expired=expired,aggregate_cpu_budget=600,aggregate_GPU_worker_budget=120,wall_budget=900,protocol_sha256=sha(SRC/'protocol.json')));print('GATE',passed,total,elapsed,raw.get('aggregate_GPU_worker_seconds'),flush=True);assert passed,'Frozen whole-study gate failed'
if __name__=='__main__':main()
