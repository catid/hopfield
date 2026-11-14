"""Independently audit saved Figure 3b results after its collector exits.

CPU only: no collector invocation, fitting, inference, or plot regeneration.
The required PNG digest acknowledges a separate human/agent visual inspection
of that exact artifact; numerical checks cannot replace visual review.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/data_efficiency"
KS = [1, 3, 10, 12, 14, 17, 19, 21, 23, 25, 27, 30, 100, 300, 1000, 3000, 10000]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-reviewed-png", required=True, help="SHA256 of the visually reviewed final PNG")
    args = parser.parse_args()
    start = time.perf_counter()
    inputs, fingerprints = {}, {}

    def sha(path):
        path = (ROOT / path).resolve()
        relative = str(path.relative_to(ROOT))
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if relative not in inputs:
            with path.open("rb") as handle:
                inputs[relative] = hashlib.file_digest(handle, "sha256").hexdigest()
            fingerprints[relative] = identity
        assert fingerprints[relative] == identity, f"Input changed: {relative}"
        return inputs[relative]

    def read(path):
        sha(path)
        return json.loads((ROOT / path).read_text())

    collector = read("artifacts/data_efficiency/collector-process.json")
    process = Path(f'/proc/{collector["pid"]}')
    if process.exists():
        command = (process / "cmdline").read_bytes()
        state = (process / "stat").read_text().rsplit(") ", 1)[1].split()[0]
        assert state == "Z" or b"collect_data_efficiency.py" not in command, "Collector is still running"
    collection_path = "artifacts/data_efficiency/collection.json"
    data = read(collection_path)
    assert data["status"] == "complete" and data["figure_complete"] is True
    assert data["measured_jobs"] == data["expected_jobs"] == 102
    assert data["missing_jobs"] == data["degenerate_jobs"] == 0
    assert data["status_counts"] == {"complete": 102} and not data["missing"] and not data["degenerate"]
    assert len(data["rows"]) == 102 and len(data["points"]) == 17
    assert sorted((r["k"], r["seed"]) for r in data["rows"]) == [(k, s) for k in KS for s in range(6)]
    config_path = "configs/data_efficiency_jobs.json"
    config = read(config_path)
    assert sha(config_path) == data["provenance"]["config_sha256"]
    assert config["source_commit"] == data["provenance"]["source_commit"] == "62bbe830c7d6f87f2ba18f30a28039258b986454"
    source = Path(config["source"])
    for relative, expected in config["source_sha256"].items():
        assert sha(source / relative) == expected
    assert len(config["source_sha256"]) == 46
    assert sha("upstream/data/full-test/full-all.npz") == data["provenance"]["test_data_sha256"]
    reference = "reproduction/baselines/experiments/runs/trm/data/sudoku-extreme-1k-aug-1000/build_provenance.json"
    assert sha(reference) == config["reference_provenance_sha256"] == data["provenance"]["reference_provenance_sha256"]
    rows = []
    for row in data["rows"]:
        k, seed = row["k"], row["seed"]
        run = source / "experiments" / config["run_root"] / f"k{k}-s{seed}"
        assert row["status"] == "complete" and row["test_n"] == 10000
        for name, expected in row["output_sha256"].items():
            assert sha(run / name) == expected
        assert sha(run / "train.npz") == row["train_npz_sha256"]
        fit, raw = read(run / "fit.json"), read(run / "test10k.json")
        assert fit["config"] == row["config"] and fit["model_sha256"] == row["model_sha256"]
        assert fit["warm_bitwise_equal"] is True and len(fit["fit_gpu_warm_s"]) == 7
        assert fit["variants"] == 99 and fit["examples"] == k * 100
        assert fit["config"]["updates"] == 32 and fit["per_update_relabeling"] is False
        dataset = Path(row["data"]["path"])
        for name, expected in row["data"]["array_sha256"].items():
            assert sha(dataset / "train" / f"all__{name}.npy") == expected
        marker = "build_provenance.json" if k == 1000 else "subset.json"
        assert sha(dataset / marker) == row["data"]["manifest_sha256"]
        signature, pp = raw["signature"], raw["per_puzzle"]
        assert signature == row["search_signature"]
        assert signature["k"] == k and signature["seed"] == seed
        assert signature["rel"] == [.4, .31, .45] and signature["C"] == 4
        assert signature["cap"] == 4e11 and signature["check"] == 4096 and signature["tol"] == 1e-3
        assert np.array_equal(pp["puzzle"], np.arange(10000, 20000))
        solved, stopped = np.asarray(pp["solved"]), np.asarray(pp["stopped"])
        assert solved.dtype == stopped.dtype == np.dtype(bool) and solved.shape == stopped.shape == (10000,)
        flops = np.asarray(pp["flops"], dtype=float)
        assert flops.shape == (10000,) and np.isfinite(flops).all() and (flops >= 0).all()
        counts = dict(n=10000, solved=int(solved.sum()), stopped=int(stopped.sum()),
                      wrong_stops=int((stopped & ~solved).sum()), budget_failures=int((~stopped).sum()),
                      unstopped_but_right=int((~stopped & solved).sum()))
        assert all(raw["summary"][name] == row["summary"][name] == value for name, value in counts.items())
        cap = int(signature["cap"] // (signature["fl_move"] * 4))
        moves = pp["moves"]
        assert pp["T_cap"] == cap and type(moves) is int and moves > 0 and moves % 4096 == 0
        assert moves < cap + 4096 and (moves >= cap or counts["budget_failures"] == 0)
        steps = flops / (signature["fl_move"] * 4)
        assert np.allclose(steps, np.rint(steps), rtol=0, atol=1e-6)
        assert ((steps >= 1) & (steps <= moves)).all()
        assert np.allclose(steps[~stopped], moves, rtol=0, atol=1e-6)
        metrics = dict(mean_flop=flops.mean(), median_flop=np.median(flops), p99_flop=np.quantile(flops, .99),
                       p999_flop=np.quantile(flops, .999), max_flop=flops.max())
        assert all(np.isclose(value, raw["summary"][name], rtol=1e-12, atol=1e-6) for name, value in metrics.items())
        accuracy = counts["solved"] / 100
        assert accuracy == row["accuracy_pct"]
        rows.append(dict(k=k, seed=seed, **counts, accuracy_pct=accuracy, moves=moves, move_cap=cap))
    points = []
    for k in KS:
        values = [row["accuracy_pct"] for row in sorted(rows, key=lambda r: r["seed"]) if row["k"] == k]
        point = dict(k=k, n_measured=6, accuracy_pct_per_seed=values, accuracy_pct_median=statistics.median(values),
                     accuracy_pct_min=min(values), accuracy_pct_max=max(values))
        saved = next(p for p in data["points"] if p["k"] == k)
        assert all(saved[key] == value for key, value in point.items())
        points.append(point)
    published = defaultdict(dict)
    for path, expected in data["published_reference"]["sources"].items():
        assert sha(path) == expected
        raw = read(path)
        assert raw["variants"] == 99
        for row in raw["rows"]:
            method = row.get("method", "CRAM")
            test = row.get("test10k")
            if method == "EBR":
                test = test["gtan0.2-cap64-N5"]
            if test is not None:
                assert test["n"] == 10000
            published[(method, row["k"])][row["seed"]] = test["solved"] / 100 if test is not None else None
    pub_points = []
    for (method, k), by_seed in sorted(published.items()):
        assert sorted(by_seed) == list(range(6))
        values = [by_seed[s] for s in range(6)]
        complete = all(v is not None for v in values)
        median = statistics.median(values) if complete else None
        saved = next(p for p in data["published_reference"]["points"] if p["method"] == method and p["k"] == k)
        assert saved["accuracy_pct_per_seed"] == values
        assert saved["n_seeds"] == (6 if complete else 0)
        assert median is None and saved["accuracy_pct_median"] is None or np.isclose(median, saved["accuracy_pct_median"], rtol=0, atol=1e-12)
        pub_points.append(dict(method=method, k=k, n_seeds=6 if complete else 0, accuracy_pct_median=median))
    assert published[("CRAM", 1)] == dict.fromkeys(range(6))
    migration = []
    for seed in (3, 4, 5):
        receipt_path = f"artifacts/orchestration_review/resumed-data-efficiency-seed{seed}-audit.json"
        receipt = read(receipt_path)
        assert receipt["status"] == "validated" and receipt["seed"] == seed
        for path, expected in receipt["sha256"].items():
            assert sha(path) == expected
        assert receipt["archived_stop_decisions_and_flops_preserved"]
        assert receipt["all_new_stop_moves_after_handoff"] and receipt["all_unstopped_puzzles_exhausted_budget"]
        migration.append(dict(seed=seed, receipt=receipt_path, receipt_sha256=sha(receipt_path), status="validated"))
    figure = read("artifacts/data_efficiency/figure3b.json")
    assert figure["status"] == "complete" and figure["figure_complete"] is True
    assert figure["measured_jobs"] == figure["expected_jobs"] == 102
    assert figure["source_sha256"] == sha(collection_path)
    assert figure["plotted_k"] == KS and not figure["missing_k"]
    assert figure["baseline_status"] == "published only" and "seed42" in figure["scope"].lower()
    png_sha = sha("artifacts/data_efficiency/figure3b.png")
    assert png_sha == args.visual_reviewed_png
    sha("artifacts/data_efficiency/figure3b.pdf")
    sha("scripts/plot_data_efficiency.py")
    sha(Path(__file__).resolve())
    for path in list(inputs):
        sha(path)
    receipt = {
        "status": "validated", "utc": datetime.now(timezone.utc).isoformat(),
        "collection_sha256": sha(collection_path), "figure_png_sha256": png_sha,
        "measured_rows": 102, "six_seed_points": 17, "test_evaluations": 1020000,
        "distinct_test_puzzles_per_run": 10000, "source_files_verified": 46,
        "independent_rows": rows, "independent_points": points, "published_points": pub_points,
        "migration_audits": migration, "input_sha256": inputs, "hashed_files": len(inputs),
        "visual_review": {"status": "passed", "artifact": "artifacts/data_efficiency/figure3b.png",
                          "sha256": png_sha, "checks": ["Final title, no incomplete badge", "All 17 local points including k=1",
                          "Readable axes, legend and six-seed range", "Published curves and seed42 recreation explicitly labeled",
                          "Published k=1 CRAM absence preserved"]},
        "scope": "Fresh seed42 reference recreation; original unseeded training arrays unavailable. Published baselines were not locally trained or evaluated.",
        "limitations": ["Recomputed saved Sudoku correctness flags; final boards were not retained for independent Sudoku-rule validation.",
                        "Different sampled training arrays prevent a matched local-baseline comparison.",
                        "Resumed search_seconds excludes staging and discarded post-checkpoint work; no end-to-end or matched-hardware speedup inferred."],
        "new_training_or_inference": False, "audit_seconds": time.perf_counter() - start,
    }
    path = OUT / "final-audit.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
    print(json.dumps({k: receipt[k] for k in ("status", "measured_rows", "six_seed_points", "hashed_files", "audit_seconds", "collection_sha256", "figure_png_sha256")}))


if __name__ == "__main__":
    main()
