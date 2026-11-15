"""Subprocess-inclusive CPU monitor for build, parity and both compilations."""
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
    frozen();assert not(OUT/'compilation-gate.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    wall=time.perf_counter();cpu=own_cpu();previous_children=child_cpu();process=subprocess.Popen([sys.executable,str(SRC/'compiler.py')],start_new_session=True);peak=0.;expired=False;reason=None
    while process.poll() is None:
        used=own_cpu()-cpu+tree_cpu(process.pid);peak=max(peak,used)
        if used>=119.8 or time.perf_counter()-wall>=119.8:
            expired=True;reason='Aggregate120CPU-second ceiling or conservative120wall-second guard';os.kill(process.pid,signal.SIGUSR1)
            try:process.wait(timeout=.15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            break
        time.sleep(.02)
    total=own_cpu()-cpu+child_cpu()-previous_children;elapsed=time.perf_counter()-wall
    source=read(OUT/'compiler-result.json')if(OUT/'compiler-result.json').exists()else dict(status='failed_or_incomplete',error='Compiler stopped without final receipt')
    passed=process.returncode==0 and not expired and source['status']=='complete_verified' and total<=120 and elapsed<=120
    # Charge full monitor/import/common overhead together with build/parity to each family.
    family_wall=sum(r['compile_wall_seconds']for r in source.get('families',source.get('completed_families',[])))
    result=dict(source,status='complete_verified'if passed else'failed_or_incomplete',utc=now(),aggregate_cpu_seconds=total,wall_seconds=elapsed,cpu_seconds_ceiling=120,wall_seconds_guard=120,monitor_peak_live_cpu_seconds=peak,monitor_cpu_seconds=own_cpu()-cpu,subprocess_inclusive_CPU_accounting=True,shared_prerequisite_wall_seconds=elapsed-family_wall,partial_cache_usable=passed,monitor_expired=expired,monitor_reason=reason,protocol_sha256=sha(SRC/'protocol.json'))
    save(OUT/'compilation-gate.json',result);print('COMPILATION_GATE',result['status'],total,elapsed,flush=True);assert passed,'Global build/parity/compilation gate failed'
if __name__=='__main__':main()
