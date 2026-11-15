"""Process-lifetime CPU and early wall accounting, including source hashing."""
import time
WALL_ORIGIN=time.perf_counter()
import os
import resource
import signal
import subprocess
import sys
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

def main(cpu_limit=600.,wall_limit=900.):
    frozen();assert not(OUT/'capacity-gate.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    # Full RUSAGE_SELF includes Python startup and imports before this function.
    # WALL_ORIGIN is set before common import and the initial frozen() hash pass.
    wall=WALL_ORIGIN;process=None;peak=own_cpu()+child_cpu();expired=False;reason=None
    margin=min(.2,cpu_limit*.1,wall_limit*.1)
    if peak>=cpu_limit-margin or time.perf_counter()-wall>=wall_limit-margin:
        expired=True;reason='Initial imports/source verification exhausted the aggregate budget'
    else:process=subprocess.Popen([sys.executable,str(SRC/'study.py')],start_new_session=True)
    while process is not None and process.poll() is None:
        used=own_cpu()+child_cpu()+tree_cpu(process.pid);peak=max(peak,used)
        if used>=cpu_limit-margin or time.perf_counter()-wall>=wall_limit-margin:
            expired=True;reason='Aggregate CPU ceiling or wall guard';os.kill(process.pid,signal.SIGUSR1)
            try:process.wait(timeout=.15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            break
        time.sleep(.02)
    source=read(OUT/'study-result.json')if(OUT/'study-result.json').exists()else dict(status='failed_or_incomplete',error='Compiler stopped without final receipt')
    source_digest=sha(OUT/'study-result.json')if(OUT/'study-result.json').exists()else None
    total=own_cpu()+child_cpu();elapsed=time.perf_counter()-wall
    passed=process is not None and process.returncode==0 and not expired and source['status']=='complete_verified' and total<=cpu_limit and elapsed<=wall_limit
    final_budget_exceeded=total>cpu_limit or elapsed>wall_limit
    if not passed and reason is None:
        reason='Final aggregate CPU/wall check exceeded budget' if final_budget_exceeded else 'Child exit or source receipt did not complete'
    # Charge full monitor/import/common overhead together with build/parity to each family.
    family_wall=source.get('stage_wall_seconds',0.)
    result=dict(source,status='complete_verified'if passed else'failed_or_incomplete',utc=now(),aggregate_cpu_seconds=total,wall_seconds=elapsed,cpu_seconds_ceiling=cpu_limit,wall_seconds_guard=wall_limit,monitor_peak_live_cpu_seconds=peak,monitor_cpu_seconds=own_cpu(),subprocess_inclusive_CPU_accounting=True,full_monitor_process_lifetime_CPU_charged=True,initial_source_hashing_inside_wall_and_CPU_budget=True,monitor_and_import_wall_seconds=elapsed-family_wall,partial_cache_usable=passed,monitor_expired=expired,monitor_reason=reason,protocol_sha256=sha(SRC/'protocol.json'))
    result.update(primary_decision_emitted=passed,study_result_sha256=source_digest,
                  internally_complete_but_not_accepted=source['status']=='complete_verified' and not passed,
                  final_budget_exceeded=final_budget_exceeded)
    if not passed:
        # An internally complete source receipt is preserved as unaccepted
        # evidence. Only this final capacity receipt can accept an endpoint.
        result.pop('decision',None);result.pop('primary',None)
    save(OUT/'capacity-gate.json',result);print('QUERY_GATE',result['status'],total,elapsed,flush=True);assert passed,'Global build/parity/compilation gate failed'
if __name__=='__main__':main()
