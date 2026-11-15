"""One bounded H4be source/fit/audit run with globally sealed phases."""
import time
START=time.monotonic()
import argparse
import os
import resource
import shutil
import signal
import subprocess
from repeat_diversity_common import *


def launch():
    p=protocol()
    assert not OUT.exists() and not PUBLIC.exists()
    OUT.mkdir(parents=True)
    for name in ["observed","noise","source_audit","fits","audits","progress","logs"]:(OUT/name).mkdir()
    children,streams,samples=[],[],[]
    ticks=os.sysconf("SC_CLK_TCK")
    status,error="incomplete",None

    def cpu():
        total=sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)])
        for child in children:
            if child.poll() is None:
                try:
                    fields=Path(f"/proc/{child.pid}/stat").read_text().split()
                    total+=(int(fields[13])+int(fields[14]))/ticks
                except FileNotFoundError:pass
        return total

    def monitor(phase):
        row=dict(phase=phase,wall_seconds=time.monotonic()-START,aggregate_CPU_seconds=cpu(),
            active_pids=[c.pid for c in children if c.poll() is None],
            counters={name:sum(read(f)["completed_banks"] for f in (OUT/"progress").glob(name+"*.json")) for name in ["source","fit","audit"]})
        samples.append(row);save(OUT/"progress.json",row)
        assert row["wall_seconds"]<p["budget"]["complete_wall_seconds"],"wall_budget"
        assert row["aggregate_CPU_seconds"]<p["budget"]["aggregate_CPU_seconds"],"CPU_budget"
        assert all(c.poll() in [None,0] for c in children),"owned_worker_failure"

    def phase(name,script):
        batch=[]
        for i in range(8):
            stream=(OUT/"logs"/f"{name}{i}.log").open("x");streams.append(stream)
            child=subprocess.Popen([str(CPU_PYTHON),str(HERE/script),"--worker",str(i)],
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            batch.append(child);children.append(child)
        while any(c.poll() is None for c in batch):monitor(name);time.sleep(.2)
        monitor(name)

    try:
        monitor("bindings")
        phase("source","repeat_diversity_source.py");seal("source")
        phase("fit","repeat_diversity_fit.py");seal("fit")
        phase("audit","audit_repeat_diversity.py")
        rows=[read(OUT/"audits"/f"b{i:03d}.json") for i in range(N)]
        result=dict(status="validated",protocol_sha256=sha(PROTOCOL),rows=rows,**aggregate(rows),
                    accounting=p["accounting"],new_LPs=N,new_BP_fits=0)
        PUBLIC.mkdir(parents=True)
        for name in ["observed","source_audit","fits","audits"]:
            shutil.copytree(OUT/name,PUBLIC/name);monitor("export")
        for name in ["source-seal.json","fit-seal.json"]:shutil.copyfile(OUT/name,PUBLIC/name)
        monitor("export_complete")
        save(OUT/"summary.json",result);save(PUBLIC/"summary.json",result)
        monitor("summaries_written")
        status="complete"
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        if status!="complete":
            for folder in [OUT,PUBLIC]:(folder/"summary.json").unlink(missing_ok=True)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for stream in streams:stream.close()
        record=dict(status=status,error=error,protocol_sha256=sha(PROTOCOL),wall_seconds=time.monotonic()-START,
                    aggregate_CPU_seconds=cpu(),returncodes=[c.returncode for c in children],samples=samples,
                    budget_scope="Full source/provenance, fits, independent audit, monitor and export; parent plus all child CPU.")
        save(OUT/"runtime.json",record)
        if PUBLIC.exists():save(PUBLIC/"runtime.json",record)
        print(json.dumps({k:v for k,v in record.items() if k!="samples"}),flush=True)


if __name__=="__main__":launch()
