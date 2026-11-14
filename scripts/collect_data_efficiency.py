"""Audit all 102 frozen data-efficiency jobs, preserving missing and skipped rows.

CPU only. --watch repeats the audit and renders complete six-seed points as
they arrive. File hashes are reused only while inode/size/mtime/ctime agree;
validated outputs are reused only while every dependency digest agrees.
"""
from pathlib import Path
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/data_efficiency"
CONFIG = ROOT / "configs/data_efficiency_jobs.json"
DATA = ROOT / "reproduction/baselines/experiments/runs/trm/data"
KS = [1, 3, 10, 12, 14, 17, 19, 21, 23, 25, 27, 30, 100, 300, 1000, 3000, 10000]
SEEDS = list(range(6))
# A long-lived watcher must identify the code it actually loaded. Reading a
# changed on-disk script hash during a later iteration could otherwise label
# old validation logic as the new version and incorrectly reuse its cache.
LOADED_SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class Hashes:
    def __init__(self):
        self.path = OUT / "collection-hash-cache.json"
        self.cache = read(self.path) if self.path.exists() else {}
        self.bytes_read = 0
        self.hits = 0

    def sha(self, path):
        path = Path(path).resolve()
        st = path.stat()
        fingerprint = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
        previous = self.cache.get(str(path))
        if previous and previous["stat"] == fingerprint:
            self.hits += 1
            return previous["sha256"]
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
                self.bytes_read += len(chunk)
        after = path.stat()
        assert fingerprint == [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns], f"File changed during hashing: {path}"
        value = h.hexdigest()
        self.cache[str(path)] = {"stat": fingerprint, "sha256": value}
        return value

    def save(self):
        write(self.path, self.cache)


def dataset(job, hashes, reference):
    k, seed = job["k"], job["seed"]
    directory = DATA / ("sudoku-extreme-1k-aug-1000" if k == 1000 else f"sudoku-smalldata/s{seed}-n{k}")
    marker = directory / ("build_provenance.json" if k == 1000 else "subset.json")
    manifest = read(marker)
    if k != 1000:
        assert manifest["seed"] == seed and manifest["size"] == k
        assert manifest["unique_puzzles"] == k and manifest["augmented_examples"] == k * 1001
        assert manifest["num_aug"] == 1000
        assert manifest["script_sha256"] == hashes.sha(ROOT / "upstream/baselines/trm/trm_smalldata.py")
        preparation = read(OUT / f"preparation-seed{seed}.json")
        rel = str(marker.relative_to(DATA))
        assert preparation["subset_manifest_sha256"][rel] == hashes.sha(marker)
        assert preparation["reference_provenance_sha256"] == hashes.sha(DATA / "sudoku-extreme-1k-aug-1000/build_provenance.json")
    expected = reference["arrays"] if k == 1000 else manifest["sha256"]
    actual = {}
    for name in ("inputs", "labels", "group_indices"):
        path = directory / "train" / f"all__{name}.npy"
        value = hashes.sha(path)
        assert value == expected[f"train/all__{name}.npy" if k == 1000 else name], f"Dataset hash mismatch: {path}"
        actual[name] = value
    groups = np.load(directory / "train/all__group_indices.npy", mmap_mode="r")
    x = np.load(directory / "train/all__inputs.npy", mmap_mode="r")
    y = np.load(directory / "train/all__labels.npy", mmap_mode="r")
    assert groups.shape == (k + 1,) and np.array_equal(groups, np.arange(k + 1) * 1001)
    assert x.shape == y.shape == (k * 1001, 81)
    assert np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.integer)
    return {"path": str(directory), "manifest_sha256": hashes.sha(marker), "array_sha256": actual,
            "stored_rows": k * 1001, "selected_rows": k * 100,
            "input_dtype": str(x.dtype), "label_dtype": str(y.dtype)}


def migration_evidence(job, meta, run, hashes):
    """Validate saved continuation provenance even when the result row is cached."""
    migration = meta.get("search_migration")
    if migration is None:
        return {"execution_backend": "original single-GPU move kernel"}
    records = {}
    paths = {}
    for kind in ("receipt", "handoff", "proof"):
        path = (ROOT / migration[f"{kind}_path"]).resolve()
        assert path.is_relative_to(ROOT.resolve()), "Migration evidence escapes workspace"
        assert hashes.sha(path) == migration[f"{kind}_sha256"], f"Changed migration {kind}"
        paths[kind] = path
        records[kind] = read(path)
    receipt, handoff, proof = (records[k] for k in ("receipt", "handoff", "proof"))
    result = read(run / "test10k.json")
    assert receipt["status"] == "complete" and receipt["exit_code"] == 0
    assert receipt["seed"] == handoff["seed"] == job["seed"]
    assert receipt["k"] == handoff["k"] == job["k"]
    assert receipt["handoff_parity"]["equal"] is True
    assert receipt["handoff_parity"]["label"] == "actual_resume/pre_refresh"
    assert receipt["result_sha256"] == hashes.sha(run / "test10k.json")
    assert receipt["original_signature"] == result["signature"]
    assert receipt["final_moves"] == result["per_puzzle"]["moves"]
    assert receipt["summary"] == result["summary"]
    assert receipt["search_seconds"] == result["per_puzzle"]["seconds"]
    assert receipt["config_sha256"] == handoff["config_sha256"] == meta["config_sha256"]
    assert receipt["handoff_sha256"] == migration["handoff_sha256"]
    assert receipt["proof_sha256"] == migration["proof_sha256"]
    assert receipt["input_sha256"] == handoff["input_sha256"]
    assert handoff["original_writer_stopped"] is True
    checkpoint_archive = (ROOT / handoff["checkpoint_archive_path"]).resolve()
    assert checkpoint_archive.is_relative_to((ROOT / "artifacts/data_efficiency/migrations").resolve())
    assert hashes.sha(checkpoint_archive) == handoff["input_sha256"]["test10k-state/ckpt.pt"]
    for name in ("model.pt", "fit.json", "train.npz"):
        assert handoff["input_sha256"][name] == hashes.sha(run / name)
    assert handoff["test_data_sha256"] == hashes.sha(ROOT / "upstream/data/full-test/full-all.npz")
    assert proof["status"] == "validated" and proof["exact_state"] is True and proof["speedup"] > 1.15
    assert receipt["devices"] == proof["devices"] and receipt["host"] == proof["host"]
    assert proof["comparisons"] and all(c["equal"] is True for c in proof["comparisons"])
    assert set(receipt["execution_source_sha256"]) == {
        "resume_sharded_data_efficiency.py", "sharded_sudoku_moves.py", "benchmark_sharded_sudoku.py"}
    for name, digest in receipt["execution_source_sha256"].items():
        assert hashes.sha(paths["receipt"].parent / "execution_source" / name) == digest
    assert proof["sharding_sha256"] == receipt["execution_source_sha256"]["sharded_sudoku_moves.py"]
    assert proof["script_sha256"] == receipt["execution_source_sha256"]["benchmark_sharded_sudoku.py"]
    if handoff["source_host"] != receipt["host"]:
        native_path = (ROOT / handoff["native_resume_proof_path"]).resolve()
        assert native_path.is_relative_to(ROOT.resolve())
        assert hashes.sha(native_path) == handoff["native_resume_proof_sha256"]
        native = read(native_path)
        assert native["status"] == "passed" and native["seed"] == job["seed"]
        assert native["from_host"] == handoff["source_host"] and native["to_host"] == receipt["host"]
        assert native["static_model_and_signature_exact"] is True
        assert len(native["all_state_fields_except_elapsed_exact"]) == 24
        assert all(v is True for v in native["all_state_fields_except_elapsed_exact"].values())
    return {"execution_backend": "validated exact-state GPU move sharding",
            "migration_evidence": migration, "devices": receipt["devices"],
            "resume_host": receipt["host"], "starting_move": receipt["starting_move"],
            "actual_resume_state_parity": receipt["handoff_parity"],
            "benchmark_speedup": proof["speedup"]}


def validate_completed(job, meta, run, data, hashes, execution=None):
    import torch
    torch.set_num_threads(1)
    k, seed = job["k"], job["seed"]
    for phase in ("fit", "search"):
        phases = [p for p in meta["phases"] if p["phase"] == phase]
        assert phases and phases[-1]["exit_code"] == 0
    fit, result = read(run / "fit.json"), read(run / "test10k.json")
    expected = dict(aug=False, lam=3e-3 / 9, lr=540., momentum=.5, updates=32, split=seed, seed=seed)
    assert fit["k"] == k and fit["seed"] == seed and fit["config"] == expected
    assert fit["variants"] == 99 and fit["per_update_relabeling"] is False
    assert fit["examples"] == k * 100 and fit["params"] == 263169
    assert len(fit["fit_gpu_warm_s"]) == 7 and fit["warm_bitwise_equal"] is True
    assert all(math.isfinite(t) and t > 0 for t in fit["fit_gpu_warm_s"])
    assert statistics.median(fit["fit_gpu_warm_s"]) == fit["fit_gpu_warm_median_s"]
    assert fit["model_sha256"] == hashes.sha(run / "model.pt")
    model = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    W, B = model["W"], model["B"]
    assert W.shape == (3240, 9, 9) and B.shape == (81, 9)
    assert bool(torch.isfinite(W).all()) and bool(torch.isfinite(B).all())
    assert fit["positive_W"] == int((W > 0).sum()) and fit["negative_W"] == int((W < 0).sum())
    assert fit["nonzero_params"] == int((W != 0).sum() + (B != 0).sum())
    train = np.load(run / "train.npz")
    solutions, free = train["solutions"], train["free"]
    assert solutions.shape == free.shape == (k * 100, 81)
    directory = Path(data["path"]) / "train"
    original_x = np.load(directory / "all__inputs.npy", mmap_mode="r")
    original_y = np.load(directory / "all__labels.npy", mmap_mode="r")
    # Compare every selected row, in small blocks, against the original and
    # first 99 stored variants of each 1001-row group.
    for lo in range(0, k, 256):
        hi = min(k, lo + 256)
        ids = (np.arange(lo, hi)[:, None] * 1001 + np.arange(100)).reshape(-1)
        x, y = original_x[ids], original_y[ids]
        assert np.array_equal(solutions[lo * 100:hi * 100], y.astype(np.int16) - 2)
        assert np.array_equal(free[lo * 100:hi * 100], x == 1)
        assert np.all((x == 1) | (x == y)) and y.min() >= 2 and y.max() <= 10
    row = {"k": k, "seed": seed, "job_id": job["id"], "training_examples": k * 100,
           "variants": 99, "config": expected, "model_sha256": fit["model_sha256"],
           "data": data, "output_sha256": meta["output_sha256"], "train_npz_sha256": hashes.sha(run / "train.npz"),
           "fit_gpu_cold_s": fit["fit_gpu_cold_s"], "fit_gpu_warm_median_s": fit["fit_gpu_warm_median_s"],
           "params": fit["params"], "nonzero_params": fit["nonzero_params"],
           "tau": fit.get("tau"), "hardware": fit["gpu"], "hostname": fit["node"],
           "complete_fit_and_search_phase": True}
    if "skipped" in result:
        assert fit["tau"] is None and fit["positive_W"] == 0
        assert result["skipped"] == fit["degenerate"]
        row.update(status="degenerate", reason=result["skipped"], test_n=None, accuracy_pct=None,
                   interpretation="Search undefined and skipped by released protocol; this is not a measured 0% result")
        return row
    assert fit["tau"] is not None and fit["positive_W"] > 0
    sig, summary, pp = result["signature"], result["summary"], result["per_puzzle"]
    assert sig["k"] == k and sig["seed"] == seed and sig["data"] == "test10k" and sig["n"] == 10000
    assert sig["model_sha256"] == fit["model_sha256"] and sig["tau"] == fit["tau"]
    assert sig["hbar"] == fit["hbar"] and sig["D"] == fit["D"]
    assert sig["rel"] == [.4, .31, .45] and sig["C"] == 4 and sig["cap"] == 4e11
    assert sig["check"] == 4096 and sig["tol"] == 1e-3
    assert np.array_equal(pp["puzzle"], np.arange(10000, 20000))
    solved, stopped, flops = np.asarray(pp["solved"], bool), np.asarray(pp["stopped"], bool), np.asarray(pp["flops"], float)
    assert solved.shape == stopped.shape == flops.shape == (10000,)
    assert np.isfinite(flops).all() and (flops >= 0).all()
    counts = dict(n=10000, solved=int(solved.sum()), stopped=int(stopped.sum()),
                  wrong_stops=int((stopped & ~solved).sum()), budget_failures=int((~stopped).sum()),
                  unstopped_but_right=int((~stopped & solved).sum()))
    cap_moves = int(sig["cap"] // (sig["fl_move"] * sig["C"]))
    assert pp["T_cap"] == cap_moves
    assert isinstance(pp["moves"], int) and pp["moves"] > 0 and pp["moves"] % sig["check"] == 0
    assert pp["moves"] < cap_moves + sig["check"]
    assert pp["moves"] >= cap_moves or counts["budget_failures"] == 0, "Partial search is not a completed result"
    assert all(summary[key] == value for key, value in counts.items())
    metrics = {"mean_flop": float(flops.mean()), "median_flop": float(np.median(flops)),
               "p99_flop": float(np.quantile(flops, .99)), "p999_flop": float(np.quantile(flops, .999)), "max_flop": float(flops.max())}
    assert all(np.isclose(summary[key], value, rtol=1e-12, atol=1e-6) for key, value in metrics.items())
    steps = flops / (sig["C"] * sig["fl_move"])
    assert np.allclose(steps, np.rint(steps), rtol=0, atol=1e-6)
    assert np.all((steps >= 1) & (steps <= pp["moves"]))
    assert np.allclose(steps[~stopped], pp["moves"], rtol=0, atol=1e-6)
    if meta.get("search_migration"):
        migration = meta["search_migration"]
        receipt = read(ROOT / migration["receipt_path"])
        handoff = read(ROOT / migration["handoff_path"])
        checkpoint = torch.load(ROOT / handoff["checkpoint_archive_path"], map_location="cpu", weights_only=False)
        assert checkpoint["signature"] == sig
        assert receipt["starting_move"] == handoff["checkpoint_move"] == checkpoint["state"]["t"]
        assert receipt["prior_search_seconds"] == checkpoint["state"]["elapsed"]
        assert receipt["starting_active_puzzles"] == len(checkpoint["state"]["act"])
        assert result["resumes"] == checkpoint["resumes"] + 1
        assert pp["moves"] >= checkpoint["state"]["t"]
        # The upstream final elapsed field is rounded to one decimal place.
        assert pp["seconds"] + .05 >= checkpoint["state"]["elapsed"]
    row.update(status="complete", test_n=10000, accuracy_pct=100 * counts["solved"] / 10000,
               summary=summary, legacy_operation_counts=metrics, search_signature=sig,
               search_seconds=pp["seconds"], resumes=result["resumes"],
               search_host=result.get("node"), search_hardware=result.get("gpu"),
               search_execution=execution or {"execution_backend": "original single-GPU move kernel"})
    return row


def published_references(hashes):
    baseline_path = ROOT / "upstream/results/sudoku-data-eff-baselines-v99.json"
    cram_path = ROOT / "upstream/results/sudoku-data-eff-shared-v99.json"
    base, cram = read(baseline_path), read(cram_path)
    assert base["variants"] == cram["variants"] == 99 and not base["partial"] and not base["missing"]
    rows = []
    for r in base["rows"]:
        test = r["test10k"]["gtan0.2-cap64-N5"] if r["method"] == "EBR" else r["test10k"]
        assert test["n"] == 10000
        rows.append({"method": r["method"], "k": r["k"], "seed": r["seed"], "status": "published only",
                     "test_n": 10000, "accuracy_pct": test["solved"] / 100})
    missing = []
    for r in cram["rows"]:
        test = r.get("test10k")
        if test is None:
            missing.append({"method": "CRAM", "k": r["k"], "seed": r["seed"], "reason": "published export omits test record"})
        rows.append({"method": "CRAM", "k": r["k"], "seed": r["seed"], "status": "published only" if test else "missing published test",
                     "test_n": 10000 if test else None, "accuracy_pct": test["solved"] / 100 if test else None})
    points = []
    for method in ("TRM", "EBR", "FRM", "CRAM"):
        for k in sorted({r["k"] for r in rows if r["method"] == method}):
            group = [r for r in rows if r["method"] == method and r["k"] == k]
            assert sorted(r["seed"] for r in group) == SEEDS
            values = [r["accuracy_pct"] for r in group]
            complete = all(v is not None for v in values)
            points.append({"method": method, "k": k, "n_seeds": 6 if complete else 0,
                           "accuracy_pct_per_seed": values,
                           "accuracy_pct_median": statistics.median(values) if complete else None})
    return {"status": "Published references; no baseline trained or evaluated locally",
            "training_data_caveat": "Published curves use unavailable original unseeded arrays; fresh CRAM uses the seed42 recreation",
            "sources": {str(p.relative_to(ROOT)): hashes.sha(p) for p in (baseline_path, cram_path)},
            "rows": rows, "points": points, "missing": missing}


def collect():
    start = time.perf_counter()
    hashes = Hashes()
    config = read(CONFIG)
    config_sha = hashes.sha(CONFIG)
    source = Path(config["source"])
    assert sorted((j["k"], j["seed"]) for j in config["jobs"]) == [(k, s) for k in KS for s in SEEDS]
    assert config["protocol"]["variants"] == 99 and config["protocol"]["test_n"] == 10000
    assert config["protocol"]["updates"] == 32 and config["protocol"]["warm_fit_repeats"] == 7
    for rel, expected in config["source_sha256"].items():
        assert hashes.sha(source / rel) == expected, f"Staged source changed: {rel}"
    reference_path = DATA / "sudoku-extreme-1k-aug-1000/build_provenance.json"
    assert hashes.sha(reference_path) == config["reference_provenance_sha256"]
    reference = read(reference_path)
    assert reference["seed"] == 42 and len(reference["selected_train_rows"]) == 1000
    deployment_manifest = ROOT / "artifacts/deployment/data_manifest.json"
    test_pin = next(r for r in read(deployment_manifest) if r["path"] == "full-test/full-all.npz")
    test_sha = hashes.sha(ROOT / "upstream/data/full-test/full-all.npz")
    assert test_sha == test_pin["sha256"], "Test dataset changed since validated deployment"
    cache_path = OUT / "collection-validation-cache.json"
    validation_cache = read(cache_path) if cache_path.exists() else {}
    rows, data_cache = [], {}
    for job in sorted(config["jobs"], key=lambda j: (j["k"], j["seed"])):
        row = {"k": job["k"], "seed": job["seed"], "job_id": job["id"], "status": "pending", "accuracy_pct": None}
        try:
            data_key = (job["k"], job["seed"] if job["k"] != 1000 else 0)
            if data_key not in data_cache:
                data_cache[data_key] = dataset(job, hashes, reference)
            data = data_cache[data_key]
            row["data"] = data
            meta_path = OUT / "jobs" / f'{job["id"]}.json'
            if not meta_path.exists():
                row["reason"] = "Job has not written provenance"
            else:
                meta = read(meta_path)
                assert meta["config_sha256"] == config_sha
                assert meta["data_sha256"] == data["array_sha256"] and meta["data_manifest_sha256"] == data["manifest_sha256"]
                assert meta["protocol"] == config["protocol"]
                row["worker_status"] = meta["status"]
                if meta["status"] != "complete":
                    interrupted = [p for p in meta.get("phases", []) if p["exit_code"] != 0
                                   and p.get("interrupted_for_checkpoint_migration")]
                    failed = [p for p in meta.get("phases", []) if p["exit_code"] != 0
                              and not p.get("interrupted_for_checkpoint_migration")]
                    row.update(status="failed" if failed else f'running_{meta["status"]}', phase_failures=failed)
                    if interrupted:
                        row["checkpoint_migration_interruptions"] = interrupted
                else:
                    run = source / "experiments" / config["run_root"] / f'k{job["k"]}-s{job["seed"]}'
                    for name, expected in meta["output_sha256"].items():
                        assert hashes.sha(run / name) == expected, f"Output hash mismatch: {run/name}"
                    assert set(meta["output_sha256"]) == {"model.pt", "fit.json", "test10k.json"}
                    execution = migration_evidence(job, meta, run, hashes)
                    key = hashlib.sha256(json.dumps({"script": LOADED_SCRIPT_SHA, "config": config_sha,
                        "meta": hashes.sha(meta_path), "outputs": meta["output_sha256"],
                        "train": hashes.sha(run / "train.npz"), "data": data,
                        "test_data": test_sha, "search_execution": execution}, sort_keys=True).encode()).hexdigest()
                    old = validation_cache.get(job["id"])
                    if old and old["key"] == key:
                        row = old["row"]
                    else:
                        row = validate_completed(job, meta, run, data, hashes, execution)
                        validation_cache[job["id"]] = {"key": key, "row": row}
            rows.append(row)
        except (AssertionError, KeyError, ValueError, OSError, RuntimeError) as e:
            frame = traceback.extract_tb(e.__traceback__)[-1]
            row.update(status="invalid", reason=f"{type(e).__name__}: {e}",
                       validation_location=f"{Path(frame.filename).name}:{frame.lineno}")
            rows.append(row)
    measured = [r for r in rows if r["status"] == "complete"]
    degenerate = [r for r in rows if r["status"] == "degenerate"]
    missing = [r for r in rows if r["status"] not in ("complete", "degenerate")]
    points = []
    for k in KS:
        group = [r for r in rows if r["k"] == k]
        values = [r.get("accuracy_pct") for r in group]
        complete = all(r["status"] == "complete" for r in group)
        points.append({"k": k, "expected_seeds": SEEDS, "n_measured": sum(r["status"] == "complete" for r in group),
                       "statuses": [r["status"] for r in group], "accuracy_pct_per_seed": values,
                       "accuracy_pct_median": statistics.median(values) if complete else None,
                       "accuracy_pct_min": min(values) if complete else None,
                       "accuracy_pct_max": max(values) if complete else None,
                       "status": "complete_six_seed_point" if complete else "missing_measured_seeds"})
    result = {"updated_utc": now(), "status": "complete" if len(measured) == 102 else
              ("complete_with_degenerate_runs" if not missing else "incomplete"),
              "figure_complete": len(measured) == 102, "expected_jobs": 102, "measured_jobs": len(measured),
              "degenerate_jobs": len(degenerate), "missing_jobs": len(missing), "protocol": config["protocol"],
              "status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
              "scope": "Seed42 training-reference recreation; published unseeded arrays were unavailable",
              "test_rows": [10000, 20000], "rows": rows, "points": points,
              "missing": [{k: r.get(k) for k in ("job_id", "k", "seed", "status", "reason", "validation_location")} for r in missing],
              "degenerate": [{k: r.get(k) for k in ("job_id", "k", "seed", "reason")} for r in degenerate],
              "published_reference": published_references(hashes),
              "provenance": {"config_sha256": config_sha, "source_commit": config["source_commit"],
                             "source_sha256": config["source_sha256"], "reference_provenance_sha256": hashes.sha(reference_path),
                             "test_data_sha256": test_sha,
                             "deployment_manifest_sha256": hashes.sha(deployment_manifest),
                             "collector_sha256": hashes.sha(__file__)}}
    result["validation_performance"] = {"seconds": time.perf_counter() - start, "bytes_hashed": hashes.bytes_read,
                                         "reused_hashes": hashes.hits, "fully_validated_rows_cached": len(validation_cache)}
    hashes.save()
    write(cache_path, validation_cache)
    write(OUT / "collection.json", result)
    with (OUT / "rows.csv").open("w") as f:
        fields = ["job_id", "k", "seed", "status", "accuracy_pct", "test_n", "fit_gpu_cold_s", "fit_gpu_warm_median_s", "reason"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    while True:
        try:
            result = collect()
        except Exception as e:
            write(OUT / "collection-error.json", {"status": "invalid", "utc": now(), "error": repr(e)})
            raise
        report = {k: result[k] for k in ("updated_utc", "status", "figure_complete", "measured_jobs", "degenerate_jobs", "missing_jobs", "status_counts", "validation_performance")}
        print(json.dumps(report), flush=True)
        with (OUT / "collection-monitor.jsonl").open("a") as f:
            f.write(json.dumps(report) + "\n")
        if a.plot and any(p["n_measured"] == 6 for p in result["points"]):
            state = OUT / "collection-plot-state.json"
            key = hashlib.sha256(json.dumps({"points": result["points"], "complete": result["figure_complete"],
                                            "plot_script": (ROOT / "scripts/plot_data_efficiency.py").read_text()}, sort_keys=True).encode()).hexdigest()
            if not state.exists() or read(state)["key"] != key:
                subprocess.run([str(ROOT / "artifacts/plotting-venv/bin/python"),
                                str(ROOT / "scripts/plot_data_efficiency.py")], check=True)
                write(state, {"key": key, "updated_utc": now()})
        if not a.watch or result["missing_jobs"] == 0:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
