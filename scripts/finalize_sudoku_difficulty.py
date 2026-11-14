"""Wait for all three full Figure 3a searches, then validate and export on CPU."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/difficulty"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "finalizer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    target = OUT / "finalization.json"
    previous = json.loads(target.read_text()) if target.exists() else {}
    if previous.get("status") == "complete":
        for relative, expected in previous["outputs"].items():
            assert sha(ROOT / relative) == expected
        print(json.dumps(previous))
        return
    while True:
        seeds = []
        for seed in range(3):
            path = OUT / f"full/provenance-s{seed}.json"
            try:
                value = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                # Existing runners write their small marker in-place. A read
                # during that write is pending evidence, never completion.
                value = {"status": "not_ready"}
            seeds.append({"seed": seed, "status": value["status"], "path": str(path.relative_to(ROOT))})
            if value["status"] == "complete":
                assert value["seed"] == seed and value["stop"] == "zero"
                assert value["test_size"] == "full" and value["test_summary"]["n"] == 412786
                assert value["test_rows"] == [10000, 422786]
                assert value["model_sha256"] == sha(OUT / f"full/full-aug-s{seed}/model.pt")
        status = "dependency_failed" if any(s["status"] == "failed" for s in seeds) else "waiting_for_full_searches"
        state = dict(status=status, seeds=seeds, pid=os.getpid(),
                     updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write(target, state)
        if status == "dependency_failed":
            raise RuntimeError("A full difficulty search failed; inspect its retained provenance and logs")
        if all(s["status"] == "complete" for s in seeds):
            break
        if args.once:
            print(json.dumps(state))
            return
        time.sleep(30)
    python = ROOT / "artifacts/plotting-venv/bin/python"
    assert python.exists(), "The isolated plotting environment is required"
    argv = [str(python), str(ROOT / "scripts/export_sudoku_difficulty.py"), "--plot"]
    state.update(status="exporting", argv=argv)
    write(target, state)
    with (OUT / "export.log").open("a") as log:
        process = subprocess.run(argv, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
                                 stdout=log, stderr=subprocess.STDOUT)
    state["exit_code"] = process.returncode
    if process.returncode:
        state["status"] = "export_failed"
        write(target, state)
        process.check_returncode()
    result = json.loads((OUT / "figure3a.json").read_text())
    assert len(result["source_records"]) == 3 and result["subset"]["n"] == 10000
    state.update(status="complete", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 outputs={str((OUT / f"figure3a.{ext}").relative_to(ROOT)): sha(OUT / f"figure3a.{ext}")
                          for ext in ("json", "png", "pdf")})
    write(target, state)
    print(json.dumps(state))


if __name__ == "__main__":
    main()
