"""Subprocess-inclusive CPU monitor for preparation, parity and the online policy."""
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from common import SRC,OUT,read,save,sha,now,frozen
TICKS=os.sysconf('SC_CLK_TCK')
def own_cpu():
    r=resource.getrusage(resource.RUSAGE_SELF);return r.ru_utime+r.ru_stime
def child_cpu():
    r=resource.getrusage(resource.RUSAGE_CHILDREN);return r.ru_utime+r.ru_stime

def tree_cpu(pid):
    # Waited children are included in cutime/cstime; live children recurse.
    # A process transition can race one sample; the final rusage is authoritative.
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        value=sum(int(stat[k])for k in [11,12,13,14])/TICKS
        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        return value+sum(tree_cpu(int(child))for child in children)
    except FileNotFoundError:return 0.

def main():
    frozen();assert not(OUT/'capacity-gate.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    wall=time.perf_counter();cpu=own_cpu();previous_children=child_cpu();process=subprocess.Popen([sys.executable,str(SRC/'study.py')],start_new_session=True);peak=0.;expired=False;reason=None
    while process.poll() is None:
        used=own_cpu()-cpu+tree_cpu(process.pid);peak=max(peak,used)
        if used>=599.8 or time.perf_counter()-wall>=899.8:
            expired=True;reason='Aggregate600CPU-second ceiling or 900wall-second guard';os.kill(process.pid,signal.SIGUSR1)
            try:process.wait(timeout=.15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            break
        time.sleep(.02)
    total=own_cpu()-cpu+child_cpu()-previous_children;elapsed=time.perf_counter()-wall
    source=read(OUT/'study-result.json')if(OUT/'study-result.json').exists()else dict(status='failed_or_incomplete',error='Compiler stopped without final receipt')
    passed=process.returncode==0 and not expired and source['status']=='complete_verified' and total<=600 and elapsed<=900
    # Charge full monitor/import/common overhead together with build/parity to each family.
    family_wall=source.get('stage_wall_seconds',0.)
    result=dict(source,status='complete_verified'if passed else'failed_or_incomplete',utc=now(),aggregate_cpu_seconds=total,wall_seconds=elapsed,cpu_seconds_ceiling=600,wall_seconds_guard=900,monitor_peak_live_cpu_seconds=peak,monitor_cpu_seconds=own_cpu()-cpu,subprocess_inclusive_CPU_accounting=True,monitor_and_import_wall_seconds=elapsed-family_wall,partial_cache_usable=passed,monitor_expired=expired,monitor_reason=reason,protocol_sha256=sha(SRC/'protocol.json'))
    save(OUT/'capacity-gate.json',result);print('QUERY_GATE',result['status'],total,elapsed,flush=True);assert passed,'Global build/parity/compilation gate failed'
if __name__=='__main__':main()
