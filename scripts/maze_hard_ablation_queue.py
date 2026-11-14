"""Run assigned development ablations only after a verified long-BPTT test finishes."""
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/maze_hard_ablations"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--marker-seed", type=int, required=True)
    p.add_argument("--jobs", nargs="+", required=True, help="variant:seed entries")
    a = p.parse_args()
    jobs = []
    for item in a.jobs:
        variant, seed = item.split(":")
        assert variant in ("token-source", "edge-gates") and int(seed) in range(3)
        assert variant != "edge-gates" or int(seed) == 0
        jobs.append((variant, int(seed)))
    OUT.mkdir(parents=True, exist_ok=True)
    queue_path = OUT / f"queue-{platform.node()}-gpu{a.gpu}.json"
    record = {"hostname": platform.node(), "gpu": a.gpu, "pid": os.getpid(), "jobs": jobs,
              "marker_seed": a.marker_seed, "started_utc": now(), "completed": [],
              "wrapper_sha256": sha(__file__)}

    def save(status, **kw):
        record.update(status=status, updated_utc=now(), **kw)
        tmp = queue_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(queue_path)

    marker = ROOT / f"artifacts/long_bptt/seed{a.marker_seed}.json"
    save("waiting_for_complete_long_bptt_test", marker=str(marker))
    print(json.dumps(record), flush=True)
    while True:
        if marker.exists():
            dependency = read(marker)
            if dependency.get("status") == "complete":
                assert dependency["test"]["n"] == 50000
                assert dependency["test_exit_code"] == 0
                assert dependency["source_commit"] == "62bbe830c7d6f87f2ba18f30a28039258b986454"
                final = Path(dependency["run"]) / "final.json"
                test = Path(dependency["run"]) / "test.json"
                assert sha(final) == dependency["final_sha256"] and sha(test) == dependency["test_sha256"]
                break
            save("waiting_for_complete_long_bptt_test", dependency_status=dependency.get("status"))
        time.sleep(30)
    record["dependency_sha256"] = sha(marker)
    lock = (ROOT / f"artifacts/gpu{a.gpu}.lock").open("a")
    save("waiting_for_gpu_lock")
    fcntl.flock(lock, fcntl.LOCK_EX)
    gpu_uuid = subprocess.check_output(["nvidia-smi", f"--id={a.gpu}", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    while True:
        owners = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
        if gpu_uuid not in owners:
            break
        save("waiting_for_free_gpu", gpu_uuid=gpu_uuid)
        time.sleep(30)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for variant, seed in jobs:
        argv = [str(ROOT / "upstream/.venv/bin/python"), "-u", str(ROOT / "scripts/run_maze_hard_ablations.py"),
                "--variant", variant, "--seed", str(seed)]
        log = OUT / f"{variant}-seed{seed}.log"
        save("running", current_job={"variant": variant, "seed": seed}, argv=argv, log=str(log), gpu_uuid=gpu_uuid)
        with log.open("a") as stream:
            proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                try:
                    inventory = subprocess.check_output(["nvidia-smi", f"--id={a.gpu}",
                        "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], text=True).strip()
                    curve = OUT / variant / f"log-s{seed}.jsonl"
                    latest = curve.read_text().splitlines()[-1] if curve.exists() and curve.stat().st_size else None
                    save("running", child_pid=proc.pid, live_gpu=inventory,
                         latest_training_counter=json.loads(latest) if latest else None)
                except (OSError, subprocess.SubprocessError, ValueError) as e:
                    save("running", monitoring_error=repr(e), child_pid=proc.pid)
                time.sleep(30)
        if proc.returncode:
            save("failed", exit_code=proc.returncode)
            raise RuntimeError(f"Ablation {variant}:{seed} failed with exit {proc.returncode}")
        result = read(OUT / variant / f"provenance-s{seed}.json")
        assert result["status"] == "complete" and result["result"]["n"] == 100
        record["completed"].append({"variant": variant, "seed": seed, "result": result["result"]})
        print(json.dumps(record["completed"][-1]), flush=True)
    save("complete")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
