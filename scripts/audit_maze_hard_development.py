"""Validate the saved three-seed successful Maze-Hard development control on CPU."""
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from run_maze_reproduction import path_audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/maze_hard_development"


def main():
    started = time.monotonic()
    evidence = {}
    def sha(path):
        path = Path(path)
        relative = str(path.relative_to(ROOT))
        if relative not in evidence:
            before = path.stat()
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    h.update(block)
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            evidence[relative] = h.hexdigest()
        return evidence[relative]
    def read(path):
        sha(path)
        return json.loads(path.read_text())
    recipe_path = ROOT / "upstream/experiments/recipes/maze-hard-latent.json"
    recipe = read(recipe_path)["recipe"]
    train_path = ROOT / "upstream/data/maze-hard/train.csv"
    sha(train_path)
    with train_path.open() as handle:
        rows = list(csv.reader(handle))[1:]
    assert len(rows) == 1000
    # Independent character decoder checks the retained arrays against original CSV.
    mapping = {"#": 0, " ": 1, "S": 2, "G": 3}
    tokens = np.asarray([[mapping[char] for char in row[1]] for row in rows]).reshape(1000, 30, 30)
    labels = np.asarray([[char in "oSG" for char in row[2]] for row in rows]).reshape(1000, 30, 30)
    order = np.random.default_rng(0).permutation(1000)
    fit_rows, development_rows = np.sort(order[100:]), np.sort(order[:100])
    runs = []
    for seed in range(3):
        directory = OUT / f"seed{seed}"
        provenance = read(directory / "provenance.json")
        assert provenance["status"] == "complete" and provenance["seed"] == seed
        assert provenance["source_commit"] == "62bbe830c7d6f87f2ba18f30a28039258b986454"
        assert provenance["recipe"] == recipe and recipe["updates"] == 4000
        assert provenance["recipe_sha256"] == sha(recipe_path)
        assert provenance["train_csv_sha256"] == sha(train_path)
        assert provenance["test_data_used"] is False and provenance["tuning"] is False
        assert provenance["data_csv_reads"] == [str(train_path.relative_to(ROOT))]
        assert np.array_equal(provenance["fit_rows"], fit_rows)
        assert np.array_equal(provenance["development_rows"], development_rows)
        assert provenance["wrapper_sha256"] == sha(ROOT / "scripts/run_maze_hard_development.py")
        assert provenance["independent_audit_source_sha256"] == sha(ROOT / "scripts/run_maze_reproduction.py")
        for path, digest in provenance["source_sha256"].items():
            assert sha(ROOT / path) == digest
        assert set(provenance["output_sha256"]) == {"model.pt", "train.json", "train.jsonl", "development.json", "development_predictions.npz", "independent_development_audit.json"}
        for name, digest in provenance["output_sha256"].items():
            assert sha(directory / name) == digest
        fit = read(directory / "train.json")
        log = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        assert fit["curve"] == log and [row["u"] for row in log] == list(range(50, 4001, 50))
        assert all(math.isfinite(row["pl"]) and row["ptr"] is None for row in log)
        assert fit["parameters"]["params"] == 29
        assert fit["times"] == provenance["times"]
        result = read(directory / "development.json")
        arrays = np.load(directory / "development_predictions.npz")
        assert np.array_equal(arrays["development_rows"], development_rows)
        assert np.array_equal(arrays["inputs"], tokens[development_rows])
        assert np.array_equal(arrays["labels"], labels[development_rows])
        prediction = arrays["predictions"]
        assert prediction.shape == (100, 30, 30) and np.isin(prediction, [0, 1]).all()
        exact = np.all(prediction.astype(bool) == labels[development_rows], axis=(1, 2))
        stopped, steps = arrays["stopped"], arrays["steps"]
        assert stopped.dtype == bool and stopped.shape == steps.shape == (100,)
        assert ((steps >= 16) & (steps <= 2048) & (steps % 16 == 0)).all()
        assert np.array_equal(result["per_maze"]["exact"], exact & stopped)
        assert np.array_equal(result["per_maze"]["stopped"], stopped)
        counts = {"n": 100, "solved": int((exact & stopped).sum()),
                  "wrong_stops": int((~exact & stopped).sum()), "budget_failures": int((~stopped).sum())}
        assert counts == provenance["result"]
        assert all(result[k] == value for k, value in counts.items())
        flop = np.asarray(result["per_maze"]["flop"])
        assert flop.shape == (100,) and np.isfinite(flop).all() and (flop > 0).all()
        for name, actual in (("flop_mean", flop.mean()), ("flop_p99", np.quantile(flop, .99)), ("flop_max", flop.max())):
            assert math.isclose(result[name], actual, rel_tol=1e-12, abs_tol=1e-6)
        independent = path_audit(arrays["inputs"], prediction, arrays["labels"])
        assert independent == read(directory / "independent_development_audit.json") == provenance["independent_audit"]
        assert independent["labels_match_independent_BFS"] == 100
        runs.append({"seed": seed, "status": "complete", **counts, "updates": 4000,
                     "independent_audit": independent, "cold_fit_s": fit["times"]["total_s"],
                     "provenance_path": str((directory / "provenance.json").relative_to(ROOT)),
                     "model_sha256": provenance["output_sha256"]["model.pt"],
                     "prediction_sha256": provenance["output_sha256"]["development_predictions.npz"]})
    sha(OUT / "monitor.jsonl")
    sha(Path(__file__).resolve())
    receipt = {"status": "validated", "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "complete_seeds": [0, 1, 2], "runs": runs, "total_seed_evaluations": 300,
               "distinct_development_mazes": 100, "total_solved": sum(r["solved"] for r in runs),
               "test_data_used": False, "tuning": False, "new_training_or_inference": False,
               "validation_scope": "Hash and schedule validation, independent training-CSV decoding, saved-board BFS, exact labels, fixed-point status and legacy FLOP accounting; CPU only",
               "evidence_sha256": evidence, "hashed_files": len(evidence),
               "audit_seconds": time.monotonic() - started}
    (OUT / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "complete_seeds", "total_solved", "hashed_files", "audit_seconds")}))


if __name__ == "__main__":
    main()
