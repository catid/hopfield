"""Independent CPU audit of one completed checkpoint continuation; no inference."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(3, 4, 5))
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    evidence = {}

    def sha(path):
        path = Path(path).resolve()
        if str(path) not in evidence:
            before = path.stat()
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 << 20), b""):
                    h.update(block)
            after = path.stat()
            assert (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), path
            evidence[str(path)] = h.hexdigest()
        return evidence[str(path)]

    config_path = ROOT / "configs/data_efficiency_jobs.json"
    config = read(config_path)
    job = next(j for j in config["jobs"] if j["k"] == 1 and j["seed"] == args.seed)
    meta_path = ROOT / "artifacts/data_efficiency/jobs" / (job["id"] + ".json")
    meta = read(meta_path)
    assert meta["status"] == "complete" and meta["config_sha256"] == sha(config_path)
    sha(meta_path)
    run = ROOT / "dataeff_source/experiments" / config["run_root"] / f"k1-s{args.seed}"
    migration = meta["search_migration"]
    paths = {name: ROOT / migration[name + "_path"] for name in ("receipt", "handoff", "proof")}
    records = {}
    for name, path in paths.items():
        assert sha(path) == migration[name + "_sha256"]
        records[name] = read(path)
    receipt, handoff, proof = (records[name] for name in ("receipt", "handoff", "proof"))
    base = paths["receipt"].parent
    original_meta = read(base / "original-job-meta.json")
    sha(base / "original-job-meta.json")
    assert meta["phases"][:len(original_meta["phases"])] == original_meta["phases"]
    added = meta["phases"][len(original_meta["phases"]):]
    assert len(added) == 2 and added[0]["exit_code"] == -15
    assert added[0]["interrupted_for_checkpoint_migration"] is True
    assert added[0]["original_parent_retired_first"] is True
    assert added[1]["exit_code"] == 0 and added[1]["resumed_from_checkpoint"] is True
    assert added[1]["migration_receipt_sha256"] == sha(paths["receipt"])
    assert receipt["status"] == "complete" and receipt["exit_code"] == 0
    assert receipt["handoff_parity"] == {"label": "actual_resume/pre_refresh", "equal": True, "state_fields": 24}
    assert receipt["host"] == proof["host"] and receipt["devices"] == proof["devices"]
    assert proof["status"] == "validated" and proof["exact_state"] is True and proof["speedup"] > 1.15
    assert proof["comparisons"] and all(c["equal"] is True and c["state_fields"] == 24 for c in proof["comparisons"])
    assert receipt["handoff_sha256"] == sha(paths["handoff"])
    assert receipt["proof_sha256"] == sha(paths["proof"])
    assert receipt["k"] == handoff["k"] == 1 and receipt["seed"] == handoff["seed"] == args.seed
    assert receipt["config_sha256"] == handoff["config_sha256"] == sha(config_path)
    assert receipt["input_sha256"] == handoff["input_sha256"]
    assert handoff["original_writer_stopped"] is True
    for name, digest in config["source_sha256"].items():
        assert sha(ROOT / "dataeff_source" / name) == digest
    assert len(config["source_sha256"]) == 46
    for name, digest in receipt["execution_source_sha256"].items():
        assert sha(base / "execution_source" / name) == digest
    assert set(receipt["execution_source_sha256"]) == {
        "resume_sharded_data_efficiency.py", "sharded_sudoku_moves.py", "benchmark_sharded_sudoku.py"}
    assert proof["sharding_sha256"] == receipt["execution_source_sha256"]["sharded_sudoku_moves.py"]
    assert proof["script_sha256"] == receipt["execution_source_sha256"]["benchmark_sharded_sudoku.py"]
    for name in ("model.pt", "fit.json", "train.npz"):
        assert sha(run / name) == sha(base / f"k1-s{args.seed}" / name) == handoff["input_sha256"][name]
    for name, digest in meta["output_sha256"].items():
        assert sha(run / name) == digest
    assert sha(run / "train.npz") == meta["training_archive_sha256"]
    assert sha(ROOT / "upstream/data/full-test/full-all.npz") == handoff["test_data_sha256"]
    archive = ROOT / handoff["checkpoint_archive_path"]
    assert sha(archive) == handoff["input_sha256"]["test10k-state/ckpt.pt"] == added[0]["checkpoint_sha256"]
    checkpoint = torch.load(archive, map_location="cpu", weights_only=False)
    state = checkpoint["state"]
    result = read(run / "test10k.json")
    assert sha(run / "test10k.json") == sha(base / f"k1-s{args.seed}/test10k.json") == receipt["result_sha256"]
    signature, pp = result["signature"], result["per_puzzle"]
    assert signature == checkpoint["signature"] == receipt["original_signature"]
    assert signature["k"] == 1 and signature["seed"] == args.seed
    assert signature["n"] == 10000 and signature["data"] == "test10k"
    assert signature["C"] == 4 and signature["check"] == 4096 and signature["cap"] == 4e11
    assert signature["rel"] == [.4, .31, .45] and signature["tol"] == .001
    assert receipt["starting_move"] == handoff["checkpoint_move"] == added[0]["checkpoint_move"] == state["t"]
    assert receipt["prior_search_seconds"] == state["elapsed"]
    assert receipt["starting_active_puzzles"] == state["act"].numel()
    assert result["resumes"] == checkpoint["resumes"] + 1
    assert receipt["final_moves"] == pp["moves"] and receipt["search_seconds"] == pp["seconds"]
    assert pp["seconds"] + .05 >= state["elapsed"] and pp["moves"] >= state["t"]
    assert np.array_equal(pp["puzzle"], np.arange(10000, 20000))
    assert all(type(v) is bool for name in ("solved", "stopped") for v in pp[name])
    solved, stopped = np.asarray(pp["solved"], bool), np.asarray(pp["stopped"], bool)
    flops = np.asarray(pp["flops"], np.float64)
    assert solved.shape == stopped.shape == flops.shape == (10000,)
    assert np.isfinite(flops).all()
    cap = int(signature["cap"] // (signature["fl_move"] * signature["C"]))
    assert pp["T_cap"] == cap
    assert type(pp["moves"]) is int and pp["moves"] % 4096 == 0
    assert pp["moves"] < cap + 4096 and (pp["moves"] >= cap or stopped.all())
    steps = flops / (signature["fl_move"] * signature["C"])
    assert np.array_equal(steps, np.rint(steps)) and ((steps >= 1) & (steps <= pp["moves"])).all()
    assert (steps[~stopped] == pp["moves"]).all()
    before_stopped = state["stopped"].numpy()
    assert np.all(stopped[before_stopped])
    assert np.array_equal(solved[before_stopped], state["ok"].numpy()[before_stopped])
    assert np.array_equal(steps[before_stopped], state["stop_t"].numpy()[before_stopped])
    assert not bool(state["restarts"].any()), "This audited protocol uses no restart charges"
    new_stops = stopped & ~before_stopped
    assert (steps[new_stops] > state["t"]).all()
    counts = {"n": 10000, "solved": int(solved.sum()), "stopped": int(stopped.sum()),
              "wrong_stops": int((stopped & ~solved).sum()), "budget_failures": int((~stopped).sum()),
              "unstopped_but_right": int((~stopped & solved).sum())}
    summary = result["summary"]
    assert all(summary[k] == v for k, v in counts.items())
    stats = {"mean_flop": float(flops.mean()), "median_flop": float(np.median(flops)),
             "p99_flop": float(np.quantile(flops, .99)), "p999_flop": float(np.quantile(flops, .999)),
             "max_flop": float(flops.max())}
    assert all(math.isclose(summary[k], v, rel_tol=1e-12, abs_tol=1e-6) for k, v in stats.items())
    assert summary == receipt["summary"] == meta["summary"]
    if handoff["source_host"] != receipt["host"]:
        native_path = ROOT / handoff["native_resume_proof_path"]
        assert sha(native_path) == handoff["native_resume_proof_sha256"]
        native = read(native_path)
        assert native["status"] == "passed" and native["seed"] == args.seed
        assert native["from_host"] == handoff["source_host"] and native["to_host"] == receipt["host"]
        assert native["static_model_and_signature_exact"] is True
        assert set(native["all_state_fields_except_elapsed_exact"]) == set(state) - {"elapsed"}
        assert all(v is True for v in native["all_state_fields_except_elapsed_exact"].values())
    assert not torch.cuda.is_initialized()
    out = {"status": "validated", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "seed": args.seed, "k": 1, "summary": summary, "terminal_moves": pp["moves"],
           "native_move_cap": cap, "resumes": result["resumes"], "archive_move": state["t"],
           "archive_stopped": int(before_stopped.sum()), "new_stops_after_handoff": int(new_stops.sum()),
           "archived_stop_decisions_and_flops_preserved": True,
           "all_new_stop_moves_after_handoff": True, "all_unstopped_puzzles_exhausted_budget": True,
           "accuracy_pct": 100 * counts["solved"] / counts["n"],
           "benchmark_speedup": proof["speedup"], "benchmark_checks": proof["checks"],
           "benchmark_timing_checks": proof["timing_checks"], "actual_handoff_parity": receipt["handoff_parity"],
           "checkpoint_age_seconds_at_capture": added[0]["checkpoint_age_seconds_at_capture"],
           "archived_elapsed_seconds": state["elapsed"], "final_search_seconds": pp["seconds"],
           "continuation_wall_seconds": receipt["elapsed_wall_s"],
           "hashed_files": len(evidence), "sha256": evidence,
           "scope": "Fresh independent CPU artifact audit; no inference. Final boards were not retained, so correctness flags were not independently checked against Sudoku rules.",
           "audit_seconds": time.monotonic() - started}
    output = ROOT / f"artifacts/orchestration_review/resumed-data-efficiency-seed{args.seed}-audit.json"
    output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("status", "seed", "summary", "archive_stopped", "new_stops_after_handoff", "hashed_files", "audit_seconds")}))


if __name__ == "__main__":
    main()
