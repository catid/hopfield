"""H4bc bounded source, overlapping CPU/GPU fits, final-seal audit and export."""
import time
START = time.monotonic()
import argparse
import os
import resource
import shutil
import signal
import subprocess
import threading
from fresh_pooled_donor_common import *


def enforce_BP_deadline(directory, gpu, child):
    path = directory/f"bp-active{gpu}.json"
    if child.poll() is not None or not path.exists():
        return False
    record = read(path)
    assert record["pid"] == child.pid and record["gpu"] == gpu
    if record["status"] != "running":
        return False
    elapsed = (time.monotonic_ns()-record["job_started_monotonic_ns"])/1e9
    if elapsed <= 30.25:
        return False
    save(directory/f"bp-watchdog-failure{gpu}.json", dict(status="incomplete", gpu=gpu,
         pid=child.pid, index=record["index"], actual_job_wall_seconds=elapsed,
         frozen_cap_seconds=30, polling_and_termination_overrun_seconds=elapsed-30,
         scope="Owned BP worker only; whole-study incomplete, no primary decision."))
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return True


def launch():
    p = protocol()
    assert not OUT.exists() and not PUBLIC.exists()
    OUT.mkdir(parents=True)
    for name in ["raw", "observed", "source_audit", "fits", "audits", "progress", "logs"]:
        (OUT/name).mkdir()
    children, streams, samples = [], [], []
    bp_children = {}
    guard_stop = threading.Event()
    status, error = "incomplete", None
    phase_started = time.monotonic()
    lp_started = None
    lp_cpu_started = None
    ticks = os.sysconf("SC_CLK_TCK")
    def BP_watchdog():
        while not guard_stop.wait(.1):
            for gpu, child in list(bp_children.items()):
                enforce_BP_deadline(OUT, gpu, child)
    guard = threading.Thread(target=BP_watchdog, daemon=True)
    guard.start()
    def cpu():
        total = sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)])
        for c in children:
            if c.poll() is None:
                try:
                    v = Path(f"/proc/{c.pid}/stat").read_text().split()
                    total += (int(v[13])+int(v[14]))/ticks
                except FileNotFoundError:
                    pass
        return total
    def monitor(phase):
        now = time.monotonic()
        counters = {}
        for name in ["source", "grid", "donor", "masked", "bp", "audit"]:
            counters[name] = sum(read(f)["completed_banks"] for f in (OUT/"progress").glob(name+"*.json"))
        row = dict(phase=phase, wall_seconds=now-START, aggregate_CPU_seconds=cpu(),
                   active_pids=[c.pid for c in children if c.poll() is None], counters=counters)
        if len(samples) % 4 == 0:
            row["GPU"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"], text=True)
        samples.append(row)
        save(OUT/"progress.json", row)
        assert row["wall_seconds"] < 900 and row["aggregate_CPU_seconds"] < 7200, "complete_budget"
        assert all(c.poll() in [None, 0] for c in children), "owned_worker_failure"
        if phase == "source":
            assert now-phase_started < 120, "source_budget"
        if lp_started is not None:
            # Includes concurrent BP-host CPU conservatively; never subtract work.
            assert now-lp_started < 600 and cpu()-lp_cpu_started < 3600, "LP_and_audit_budget"
    def start_workers(name, count, script, args, python):
        batch = []
        for i in range(count):
            stream = (OUT/"logs"/f"{name}{i}.log").open("x")
            streams.append(stream)
            child = subprocess.Popen([str(python), str(HERE/script), *args, "--worker", str(i)],
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            if name == "bp":
                bp_children[i] = child
            batch.append(child)
        return batch
    def wait(batch, phase):
        while any(c.poll() is None for c in batch):
            monitor(phase)
            time.sleep(.2)
        monitor(phase)
    try:
        from fresh_pooled_donor_source import orders
        orders(p)
        source = start_workers("source", 8, "fresh_pooled_donor_source.py", [], CPU_PYTHON)
        wait(source, "source")
        seal("source")
        lp_started, lp_cpu_started = time.monotonic(), cpu()
        bp = start_workers("bp", 4, "fresh_pooled_donor_fit.py", ["bp"], GPU_PYTHON)
        grid = start_workers("grid", 8, "fresh_pooled_donor_fit.py", ["grid"], CPU_PYTHON)
        wait(grid, "grid")
        seal("grid")
        donor = start_workers("donor", 8, "fresh_pooled_donor_fit.py", ["donor"], CPU_PYTHON)
        wait(donor, "donor")
        seal("donor")
        masked = start_workers("masked", 8, "fresh_pooled_donor_fit.py", ["masked"], CPU_PYTHON)
        wait(masked, "masked")
        wait(bp, "bp_finish")
        seal("final")
        audit = start_workers("audit", 8, "audit_fresh_pooled_donor.py", [], CPU_PYTHON)
        wait(audit, "audit")
        from audit_fresh_pooled_donor import aggregate
        rows = [read(OUT/"audits"/f"b{i:03d}.json") for i in range(N)]
        result = dict(status="validated", protocol_sha256=sha(PROTOCOL), rows=rows, **aggregate(rows))
        assert result["attempted_LPs"] <= 11008
        save(OUT/"summary.json", result)
        PUBLIC.mkdir(parents=True)
        for name in ["observed", "source_audit", "fits", "audits"]:
            shutil.copytree(OUT/name, PUBLIC/name)
        for name in ["orders.npz", "orders-seal.json", "source-seal.json", "grid-seal.json", "donor-seal.json", "final-seal.json", "summary.json"]:
            shutil.copyfile(OUT/name, PUBLIC/name)
        # Raw generation receipts remain available publicly; full latent arrays stay outside Git.
        (PUBLIC/"raw").mkdir()
        for file in (OUT/"raw").glob("*.json"):
            shutil.copyfile(file, PUBLIC/"raw"/file.name)
        monitor("export")
        status = "complete"
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        guard_stop.set()
        guard.join(timeout=1)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for stream in streams:
            stream.close()
        record = dict(status=status, error=error, protocol_sha256=sha(PROTOCOL), wall_seconds=time.monotonic()-START,
                      aggregate_CPU_seconds=cpu(), returncodes=[c.returncode for c in children], samples=samples,
                      resource_scope="Complete parent and children CPU; LP/audit subcap conservatively includes overlapping BP-host CPU.")
        save(OUT/"runtime.json", record)
        if PUBLIC.exists():
            save(PUBLIC/"runtime.json", record)
        print(json.dumps({k: v for k, v in record.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    launch()
