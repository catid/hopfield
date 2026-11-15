"""H4bd correction-only audit; never changes H4bc or executes new fits."""
import argparse
import datetime
import os
import resource
import shutil
import signal
import subprocess
import time
from fresh_pooled_donor_common import *

PARENT = OUT
OUT = ROOT / "artifacts/research/20261114/foureyes/fresh_pooled_donor_corrected"
PUBLIC = ROOT / "results/research/foureyes/fresh_pooled_donor_corrected"
CORRECTED = HERE / "fresh-pooled-donor-corrected.json"
DESIGN = HERE / "fresh-pooled-donor-corrected-proposal.json"


def validate_sources():
    old = (HERE/"audit_fresh_pooled_donor.py").read_text()
    before = 'assert env["numpy"] == "2.3.4" and env["physical_GPU"]'
    after = 'assert env["numpy"] == "2.5.4" and env["physical_GPU"]'
    assert old.count(before) == 1
    assert old.replace(before, after) == (HERE/"audit_fresh_pooled_donor_corrected.py").read_text()
    protocol()  # All original scientific bindings remain unchanged.
    for phase in ["source", "grid", "donor", "final"]:
        check_seal(phase, PARENT)


def freeze():
    assert not CORRECTED.exists() and not OUT.exists() and not PUBLIC.exists()
    validate_sources()
    p = read(DESIGN)
    for rel, digest in p["parent_sha256"].items():
        assert sha(ROOT/rel) == digest
    original = read(PARENT/"runtime.json")
    assert original["status"] == "incomplete" and original["error"] == "AssertionError('owned_worker_failure')"
    assert original["returncodes"] == [0]*36+[1]*8
    assert not list((PARENT/"audits").glob("*.json"))
    assert not (PARENT/"summary.json").exists()
    for index in range(N):
        bp = read(PARENT/"fits"/f"b{index:03d}"/"bp.json")
        assert bp["environment"]["numpy"] == "2.5.4"
        assert bp["environment"]["physical_GPU"] == bp["gpu"] == index % 4
    gate = HERE/"fresh-pooled-donor-corrected-preflight.json"
    receipt = read(gate)
    assert receipt["status"] == "passed" and receipt["cohort_reads"] == 0
    assert receipt["source_sha256"] == {str(f.relative_to(ROOT)): sha(f) for f in
        [Path(__file__), HERE/"audit_fresh_pooled_donor_corrected.py", HERE/"preflight_fresh_pooled_donor_corrected.py"]}
    p.update(status="frozen_before_corrected_outcome_audit",
             frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             source_sha256={str(f.relative_to(ROOT)): sha(f) for f in
                 [Path(__file__), HERE/"audit_fresh_pooled_donor_corrected.py", DESIGN,
                  HERE/"preflight_fresh_pooled_donor_corrected.py", gate]})
    save(CORRECTED, p)
    print("FROZEN", sha(CORRECTED), flush=True)


def corrected_protocol():
    p = read(CORRECTED)
    assert p["status"] == "frozen_before_corrected_outcome_audit"
    for rel, digest in (p["source_sha256"] | p["parent_sha256"]).items():
        assert sha(ROOT/rel) == digest, rel
    return p


def worker(worker_id):
    corrected_protocol()
    p = protocol()
    import audit_fresh_pooled_donor_corrected as audit
    for index in range(worker_id, N, 8):
        row = audit.audit_bank(index, p)
        save(OUT/"audits"/f"b{index:03d}.json", row)
        save(OUT/"progress"/f"worker{worker_id}.json", dict(completed_banks=(index-worker_id)//8+1))


def launch():
    started = time.monotonic()
    p = corrected_protocol()
    assert not OUT.exists() and not PUBLIC.exists()
    OUT.mkdir(parents=True)
    for name in ["logs", "audits", "progress"]:
        (OUT/name).mkdir()
    children, streams, samples = [], [], []
    ticks = os.sysconf("SC_CLK_TCK")
    status, error = "incomplete", None

    def cpu():
        total = sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)])
        for child in children:
            if child.poll() is None:
                try:
                    values = Path(f"/proc/{child.pid}/stat").read_text().split()
                    total += (int(values[13])+int(values[14]))/ticks
                except FileNotFoundError:
                    pass
        return total

    def monitor(phase):
        row = dict(phase=phase, wall_seconds=time.monotonic()-started,
                   aggregate_CPU_seconds=cpu(),
                   audited_banks=sum(read(f)["completed_banks"] for f in (OUT/"progress").glob("*.json")),
                   active_pids=[c.pid for c in children if c.poll() is None])
        samples.append(row)
        save(OUT/"progress.json", row)
        assert row["wall_seconds"] < p["budget"]["wall_seconds"], "wall_budget"
        assert row["aggregate_CPU_seconds"] < p["budget"]["aggregate_CPU_seconds"], "CPU_budget"
        assert all(c.poll() in [None, 0] for c in children), "owned_audit_failure"

    try:
        validate_sources()
        for i in range(8):
            stream = (OUT/"logs"/f"audit{i}.log").open("x")
            streams.append(stream)
            children.append(subprocess.Popen([str(CPU_PYTHON), __file__, "--worker", str(i)],
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
        while any(c.poll() is None for c in children):
            monitor("audit")
            time.sleep(.2)
        monitor("audits_complete")
        validate_sources()
        from audit_fresh_pooled_donor_corrected import aggregate
        rows = [read(OUT/"audits"/f"b{i:03d}.json") for i in range(N)]
        original = read(PARENT/"runtime.json")
        summary = dict(status="validated_corrected_audit", protocol_sha256=sha(CORRECTED),
                       original_protocol_sha256=sha(PROTOCOL), original_run_status=original["status"],
                       rows=rows, **aggregate(rows))
        PUBLIC.mkdir(parents=True)
        for name in ["observed", "source_audit", "fits"]:
            shutil.copytree(PARENT/name, PUBLIC/name)
            monitor("export")
        shutil.copytree(OUT/"audits", PUBLIC/"audits")
        (PUBLIC/"raw").mkdir()
        for file in (PARENT/"raw").glob("*.json"):
            shutil.copyfile(file, PUBLIC/"raw"/file.name)
        for name in ["orders.npz", "orders-seal.json", "source-seal.json", "grid-seal.json", "donor-seal.json", "final-seal.json"]:
            shutil.copyfile(PARENT/name, PUBLIC/name)
        shutil.copyfile(PARENT/"runtime.json", PUBLIC/"original-incomplete-runtime.json")
        monitor("export_complete")
        save(OUT/"summary.json", summary)
        save(PUBLIC/"summary.json", summary)
        status = "complete"
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if status != "complete":
            # These two files are owned by this new attempt. Every bank audit
            # and partial export stays retained, but no primary is committed.
            for folder in [OUT, PUBLIC]:
                (folder/"summary.json").unlink(missing_ok=True)
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
        record = dict(status=status, error=error, protocol_sha256=sha(CORRECTED),
                      wall_seconds=time.monotonic()-started, aggregate_CPU_seconds=cpu(),
                      returncodes=[c.returncode for c in children], samples=samples,
                      new_observations=0, new_LPs=0, new_BP_fits=0)
        save(OUT/"runtime.json", record)
        if PUBLIC.exists():
            save(PUBLIC/"runtime.json", record)
        print(json.dumps({k: v for k, v in record.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--worker", type=int, choices=range(8))
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.worker is not None:
        worker(args.worker)
    else:
        launch()
