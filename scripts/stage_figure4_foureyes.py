"""Stage only frozen Figure 4 prerequisites; never allocate or launch a GPU job."""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/figure4_jobs.json"
OUT = ROOT / "artifacts/figure4/future_capacity"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_bytes(target, content, expected):
    assert hashlib.sha256(content).hexdigest() == expected
    if target.exists():
        assert sha(target) == expected, f"Refusing to overwrite different existing file: {target}"
        return "existing_identical"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves any file another process creates first.
    with target.open("xb") as stream:
        stream.write(content)
    assert sha(target) == expected
    return "created"


def main():
    start = time.perf_counter()
    assert socket.gethostname().split(".")[0] == "foureyes"
    assert not torch.cuda.is_initialized()
    config = json.loads(CONFIG.read_text())
    config_sha = sha(CONFIG)
    source, data = Path(config["source"]), Path(config["data"])
    assert source == ROOT / "objective_source"
    assert len(config["source_sha256"]) == 46
    assert config["source_commit"] == subprocess.check_output(
        ["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True).strip()
    created, preserved = [], []
    for relative, expected in {**config["source_sha256"], **config["fresh_hbar_prerequisite_sha256"]}.items():
        origin = ROOT / "upstream" / relative
        assert sha(origin) == expected, f"Pinned local source mismatch; do not stage: {relative}"
        target = source / relative
        status = stage_bytes(target, origin.read_bytes(), expected)
        (created if status == "created" else preserved).append(relative)
    recipe_hashes = {}
    for recipe in config["recipe_materializations"]:
        content = recipe["text"].encode()
        digest = hashlib.sha256(content).hexdigest()
        for relative in recipe["to"]:
            status = stage_bytes(source / relative, content, digest)
            (created if status == "created" else preserved).append(relative)
            recipe_hashes[relative] = digest

    checked_data = []
    for item in config["data_manifest"]:
        path = data / item["path"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
        with np.load(path) as arrays:
            shape = list(arrays["solutions"].shape)
            assert arrays["clue_mask"].shape == arrays["solutions"].shape
            assert shape == ([10000, 81] if item["path"] == "frm/dev10k/dev.npz" else [1000, 81])
        checked_data.append(dict(item, solutions_shape=shape))
    assert {x["path"] for x in checked_data} == {
        "frm/dev10k/dev.npz", *(f"frm/split/train-s{s}.npz" for s in range(20))}

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CRAM_DATA=str(data), V1_DATA=str(data),
               FRM_DATA=str(data / "frm/split"), PYTHONPATH=str(source), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # The pinned split checker reads identities only; it does not evaluate models.
    split_check = subprocess.run([config["python"], "training_curves.py", "splits"],
        cwd=source / "experiments", env=env, text=True, capture_output=True, check=True)
    sys.path[:0] = [str(ROOT / "scripts"), str(source), str(source / "experiments")]
    import objective_queue
    import training_curves
    validation = objective_queue.verify(config)
    recipes = {}
    for method in config["protocol"]["methods"]:
        recipe = training_curves.recipe("sudoku", method)
        for job in (j for j in config["jobs"] if j["method"] == method):
            for key, expected in job["frozen_choice"].items():
                assert recipe.get(key) == expected, (method, key, recipe.get(key), expected)
            assert job["seed"] in range(20) and job["expected_probe_n"] == 512
            assert job["updates_resolved"] == (60000 if method == "bptt" else 32 if method == "pl" else 2000)
        assert recipe["probe_per_octave"] == 8 and recipe["probe_every_until"] == 64
        recipes[method] = recipe
    models = {}
    for relative, expected in config["fresh_hbar_prerequisite_sha256"].items():
        model = torch.load(source / relative, map_location="cpu", weights_only=True)
        assert set(model) == {"W", "B", "Wq", "Bq"}
        assert model["W"].shape == (3240, 9, 9) and model["B"].shape == (81, 9)
        assert all(torch.isfinite(t).all() for t in model.values()) and (model["W"] > 0).any()
        models[relative] = {"sha256": expected, "cpu_finite_tensor_check": True,
                            "positive_weight_mean": float(model["W"][model["W"] > 0].mean())}
    assert not torch.cuda.is_initialized()
    current = json.loads(CONFIG.read_text())
    dependency_keys = ("source_sha256", "data_manifest", "fresh_hbar_prerequisite_sha256", "recipe_materializations")
    assert all(current[key] == config[key] for key in dependency_keys)
    disk = shutil.disk_usage(ROOT)
    predicates = []
    for lane in range(3):
        predicates.append({"lane": f"foureyes:{lane}", "physical_gpu_lock": f"artifacts/gpu{lane}.lock",
            "main_marker": f"artifacts/main_followups/seed{lane}.json",
            "main_require": {"status": "complete", "data_efficiency_exit_code": 0},
            "dataeff_marker": f"artifacts/data_efficiency/gpu{lane}-worker.json",
            "dataeff_require": {"status": "complete"},
            "dataeff_job_check": "All 34 frozen jobs on this lane have complete provenance and validated model/fit/test hashes; degenerate rows retain their explicit skip status.",
            "pid_check": "Both recorded main_followups and data-efficiency worker PIDs are absent before dispatch."})
    for host, lane, seed, assigned in [("foureyes", 3, 0, [["token-source", 0], ["edge-gates", 0]]),
                                      ("ripper", 2, 1, [["token-source", 1]]),
                                      ("ripper", 3, 2, [["token-source", 2]])]:
        predicates.append({"lane": f"{host}:{lane}", "check_on_owning_host": host,
            "physical_gpu_lock": f"artifacts/gpu{lane}.lock",
            "long_bptt_marker": f"artifacts/long_bptt/seed{seed}.json",
            "long_bptt_require": {"status": "complete", "test_exit_code": 0, "test.n": 50000},
            "long_bptt_artifact_check": "Verify full 3840000 updates, frozen lr=2, knob=10, l1=0.1 config, selected update, final_sha256, test_sha256 and selected_checkpoint_sha256 against owning-host final.json/test.json/model-uK.pt.",
            "maze_queue_marker": f"artifacts/maze_hard_ablations/queue-{host}-gpu{lane}.json",
            "maze_require": {"status": "complete", "jobs": assigned, "completed_variant_seed_pairs": assigned},
            "maze_artifact_check": "For every assigned variant/seed, provenance status complete; model_sha256 matches model-sS.pt; dev-sS.json curve[-1].u=4000, dev100.n=100, frozen recipe and seed match; 900 fit and 100 dev inputs disjoint; no test/tuning.",
            "pid_check": "Recorded long-fit, final-test and MazeHard queue/child PIDs must be absent before dispatch."})
    result = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "stage_ready": True, "gpu_dispatch_authorized_now": False, "gpu_work_launched": False,
        "scope": "Source/data/recipe readiness only; active dispatch config and all pre-existing files preserved.",
        "config_sha256_at_start": config_sha, "config_sha256_at_finish": sha(CONFIG),
        "frozen_dependency_identity": hashlib.sha256(json.dumps({k:config[k] for k in dependency_keys},sort_keys=True).encode()).hexdigest(),
        "source": str(source), "python": config["python"], "created": created, "preserved_identical": preserved,
        "source_sha256": config["source_sha256"], "recipe_sha256": recipe_hashes, "recipes_resolved": recipes,
        "fresh_hbar_models": models, "data_checked": checked_data, "official_verifier": validation,
        "split_identity_validation": {"command": [config["python"], "training_curves.py", "splits"],
            "stdout": split_check.stdout, "all20_training_sets_unique_and_disjoint_from_dev_and_test": True,
            "note": "Reads held-out puzzle identities for leakage checks only; no model scoring."},
        "dense_clock_dependency": "New lanes may accept only untouched deferred jobs by default. Dense seed0-2 jobs also require their matching original final-run clock/checkpoints and comparison; these run outputs were deliberately not staged.",
        "future_release_predicates": predicates,
        "mandatory_dispatch_recheck": [
            "Evaluate every applicable completion/artifact/PID predicate on the owning host; a ready source manifest is not a release marker.",
            "Recheck staged source, recipes, hbar models and data hashes against the live frozen dependencies.",
            "Under global dispatch coordination, migrate only unclaimed jobs; verify no execution receipt or figure4_provenance exists on the old or new owner. Preserve exact command, seed, budget, outputs and identity.",
            "Acquire the same owning-host artifacts/gpuN.lock used by main_followups, data-efficiency, BPTT and MazeHard queues; recheck NVIDIA UUID, live compute holders and process ownership under the lock.",
            "Launch only after the GPU has no owner. No existing process may be stopped to create capacity. MPS is not implied by this staging manifest."],
        "disk_free_bytes": disk.free, "elapsed_seconds": time.perf_counter()-start,
        "script_sha256": sha(Path(__file__))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "foureyes-ready.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("stage_ready", "gpu_work_launched", "official_verifier", "disk_free_bytes", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
