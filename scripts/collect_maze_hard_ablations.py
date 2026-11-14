"""Collect task-owned ablation queues and validate complete development records."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/maze_hard_ablations"
JOBS = [("token-source", s) for s in range(3)] + [("edge-gates", 0)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize():
    rows = []
    for variant, seed in JOBS:
        provenance_path = OUT / variant / f"provenance-s{seed}.json"
        provenance = json.loads(provenance_path.read_text())
        if provenance["status"] != "complete":
            return None
        path = OUT / variant / f"dev-s{seed}.json"
        rec = json.loads(path.read_text())
        assert rec["seed"] == seed and rec["recipe"] == provenance["recipe"]
        assert len(provenance["fit_rows"]) == 900 and len(provenance["development_rows"]) == 100
        assert not set(provenance["fit_rows"]) & set(provenance["development_rows"])
        assert not provenance["test_data_used"] and not provenance["tuning"]
        assert rec["curve"][-1]["u"] == 4000
        assert all(math.isfinite(c["pl"]) for c in rec["curve"])
        assert sha(OUT / variant / f"model-s{seed}.pt") == provenance["model_sha256"]
        flags = rec["dev100_per_maze"]
        exact, stopped = flags["exact"], flags["stopped"]
        assert len(exact) == len(stopped) == rec["dev100"]["n"] == 100
        assert sum(exact) == rec["dev100"]["solved"]
        assert sum(s and not e for s, e in zip(stopped, exact)) == rec["dev100"]["wrong_stops"]
        assert sum(not s for s in stopped) == rec["dev100"]["budget_failures"]
        rows.append({"variant": variant, "seed": seed, "solved": sum(exact), "n": 100,
                     "wrong_stops": rec["dev100"]["wrong_stops"], "budget_failures": rec["dev100"]["budget_failures"],
                     "fit_seconds": rec["times"]["total_s"], "updates_per_train_second": 4000 / rec["times"]["train_s"],
                     "final_loss": rec["curve"][-1]["pl"], "params": rec["params"],
                     "hardware": rec["hardware"], "hostname": provenance["hostname"],
                     "result_path": str(path.relative_to(ROOT)), "result_sha256": sha(path),
                     "provenance_sha256": sha(provenance_path), "model_sha256": provenance["model_sha256"]})
    groups = {}
    for variant, expected_n in [("token-source", 300), ("edge-gates", 100)]:
        subset = [r for r in rows if r["variant"] == variant]
        total = sum(r["solved"] for r in subset)
        n = sum(r["n"] for r in subset)
        assert n == expected_n
        groups[variant] = {"fresh_reconstruction": {"solved": total, "n": n},
                           "published": {"solved": 0, "n": expected_n},
                           "matches_published_count": total == 0, "per_seed": subset}
    result = {"status": "complete", "scope": "Controlled single-field reconstructions; original failed-ablation recipes and seed identities were not released",
              "test_data_used": False, "groups": groups}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    while True:
        transfer = subprocess.run(["rsync", "-a", "ripper:/home/catid/hopfield/artifacts/maze_hard_ablations/", str(OUT) + "/"],
                                  capture_output=True, text=True, timeout=120)
        snapshot = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "rsync_exit": transfer.returncode,
                    "rsync_error": transfer.stderr.strip(), "queues": []}
        for path in sorted(OUT.glob("queue-*.json")):
            snapshot["queues"].append(json.loads(path.read_text()))
        with (OUT / "monitor.jsonl").open("a") as f:
            f.write(json.dumps(snapshot) + "\n")
        failed = [q for q in snapshot["queues"] if q["status"] == "failed"]
        if failed:
            print(json.dumps({"status": "failed", "queues": failed}), flush=True)
            raise RuntimeError("Ablation queue failed; retained partial artifacts require inspection")
        if transfer.returncode == 0:
            result = summarize()
            if result is not None:
                print(json.dumps(result), flush=True)
                return
        if a.once:
            print(json.dumps(snapshot), flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
