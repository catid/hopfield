"""CPU-only provenance guards for exact-state search continuation evidence."""
import hashlib
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_data_efficiency as C


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path
    monkeypatch.setattr(C, "ROOT", root)
    monkeypatch.setattr(C, "OUT", root / "artifacts/data_efficiency")
    base = root / "artifacts/data_efficiency/migrations/handoff-1"
    run = root / "frozen-source/experiments/runs/k1-s4"
    run.mkdir(parents=True)
    inputs = {}
    for name in ("model.pt", "fit.json", "train.npz"):
        (run / name).write_bytes(("immutable " + name).encode())
        inputs[name] = digest(run / name)
    result = {"signature": {"k": 1, "seed": 4}, "summary": {"n": 10000, "solved": 123},
              "per_puzzle": {"moves": 102400, "seconds": 210.5}}
    put(run / "test10k.json", result)
    archive = base / "checkpoint-archive.pt"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"original complete checkpoint")
    inputs["test10k-state/ckpt.pt"] = digest(archive)
    data = root / "upstream/data/full-test/full-all.npz"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"frozen full test data")
    source = {}
    for name in ("resume_sharded_data_efficiency.py", "sharded_sudoku_moves.py", "benchmark_sharded_sudoku.py"):
        path = base / "execution_source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# preserved execution source: " + name + "\n")
        source[name] = digest(path)
    paths = {kind: base / f"{kind}.json" for kind in ("handoff", "proof", "receipt")}
    native_path = base / "native-resume-proof.json"
    state_names = ("tgt", "fr", "x", "freelist", "fmask", "nfree", "y", "G", "P", "Q",
                   "found", "ycert", "flag", "act", "stopped", "stop_t", "ok", "bestP", "bestok",
                   "restarts", "t", "next_restart", "interval", "gen")
    native = {"status": "passed", "seed": 4, "from_host": "foureyes", "to_host": "ripper",
              "static_model_and_signature_exact": True,
              "all_state_fields_except_elapsed_exact": dict.fromkeys(state_names, True)}
    put(native_path, native)
    handoff = {"seed": 4, "k": 1, "config_sha256": "frozen-configuration",
               "input_sha256": inputs, "original_writer_stopped": True,
               "checkpoint_archive_path": str(archive.relative_to(root)),
               "source_host": "foureyes", "native_resume_proof_path": str(native_path.relative_to(root)),
               "native_resume_proof_sha256": digest(native_path),
               "test_data_sha256": digest(data)}
    proof = {"status": "validated", "exact_state": True, "speedup": 2.8,
             "devices": [0, 1, 2, 3], "host": "ripper",
             "comparisons": [{"label": "original_checkpoint/pre_refresh", "equal": True}],
             "sharding_sha256": source["sharded_sudoku_moves.py"],
             "script_sha256": source["benchmark_sharded_sudoku.py"]}
    put(paths["handoff"], handoff)
    put(paths["proof"], proof)
    receipt = {"status": "complete", "exit_code": 0, "seed": 4, "k": 1,
               "handoff_parity": {"equal": True, "label": "actual_resume/pre_refresh"},
               "result_sha256": digest(run / "test10k.json"),
               "original_signature": deepcopy(result["signature"]),
               "final_moves": result["per_puzzle"]["moves"], "summary": deepcopy(result["summary"]),
               "search_seconds": result["per_puzzle"]["seconds"],
               "config_sha256": handoff["config_sha256"],
               "handoff_sha256": digest(paths["handoff"]),
               "proof_sha256": digest(paths["proof"]),
               "input_sha256": inputs, "execution_source_sha256": source,
               "devices": [0, 1, 2, 3], "host": "ripper", "starting_move": 100352}
    put(paths["receipt"], receipt)
    migration = {}
    for kind, path in paths.items():
        migration[f"{kind}_path"] = str(path.relative_to(root))
        migration[f"{kind}_sha256"] = digest(path)
    job = {"k": 1, "seed": 4, "id": "k1-s4"}
    meta = {"config_sha256": handoff["config_sha256"], "search_migration": migration}
    targets = paths | {"result": run / "test10k.json", "checkpoint_archive": archive,
                       "source": base / "execution_source/sharded_sudoku_moves.py",
                       "model": run / "model.pt", "training_archive": run / "train.npz", "test_data": data,
                       "native_proof": native_path}
    return {"root": root, "run": run, "job": job, "meta": meta,
            "paths": paths, "targets": targets, "handoff": handoff, "receipt": receipt, "proof": proof,
            "native": native, "native_path": native_path, "result": result}


def check(evidence, hashes=None):
    return C.migration_evidence(evidence["job"], evidence["meta"], evidence["run"], hashes or C.Hashes())


def repin(evidence):
    """Keep outer receipts consistent while testing inner semantic guards."""
    paths = evidence["paths"]
    if "native_resume_proof_path" in evidence["handoff"]:
        put(evidence["native_path"], evidence["native"])
        evidence["handoff"]["native_resume_proof_sha256"] = digest(evidence["native_path"])
    put(paths["handoff"], evidence["handoff"])
    put(paths["proof"], evidence["proof"])
    evidence["receipt"].update(handoff_sha256=digest(paths["handoff"]), proof_sha256=digest(paths["proof"]))
    put(paths["receipt"], evidence["receipt"])
    for kind, path in paths.items():
        evidence["meta"]["search_migration"][f"{kind}_sha256"] = digest(path)


def test_accepts_matching_immutable_continuation_evidence(evidence):
    result = check(evidence)
    assert result["execution_backend"] == "validated exact-state GPU move sharding"
    assert result["resume_host"] == "ripper" and result["starting_move"] == 100352
    assert result["actual_resume_state_parity"]["equal"] is True


def test_original_execution_needs_no_migration_files(evidence):
    evidence["meta"].pop("search_migration")
    assert check(evidence) == {"execution_backend": "original single-GPU move kernel"}


@pytest.mark.parametrize("target", ["receipt", "handoff", "proof", "result", "source",
                                   "checkpoint_archive", "model", "training_archive", "test_data", "native_proof"])
@pytest.mark.parametrize("cache_kind", ["memory", "persisted"])
def test_corruption_invalidates_warm_hash_cache_even_with_size_and_mtime_preserved(evidence, target, cache_kind):
    hashes = C.Hashes()
    check(evidence, hashes)
    check(evidence, hashes)
    assert hashes.hits > 0
    if cache_kind == "persisted":
        hashes.save()
        hashes = C.Hashes()
    path = evidence["targets"][target]
    before = path.stat()
    content = bytearray(path.read_bytes())
    if target == "result":
        # Keep JSON parseable so this specifically exercises hash invalidation.
        content[content.index(b"102400") + 5] = ord("1")
    else:
        content[len(content) // 2] ^= 1
    path.write_bytes(content)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size and path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(AssertionError):
        check(evidence, hashes)


@pytest.mark.parametrize("field,value", [("seed", 5), ("k", 3)])
def test_rejects_wrong_job_identity(evidence, field, value):
    evidence["job"][field] = value
    with pytest.raises(AssertionError):
        check(evidence)


@pytest.mark.parametrize("fault", ["unretired_writer", "failed_handoff_parity", "failed_proof", "no_speedup", "failed_exit"])
def test_rejects_self_consistently_hashed_but_ineligible_handoff(evidence, fault):
    if fault == "unretired_writer":
        evidence["handoff"]["original_writer_stopped"] = False
    elif fault == "failed_handoff_parity":
        evidence["receipt"]["handoff_parity"]["equal"] = False
    elif fault == "failed_proof":
        evidence["proof"]["comparisons"][0]["equal"] = False
    elif fault == "no_speedup":
        evidence["proof"]["speedup"] = 1.0
    else:
        evidence["receipt"]["exit_code"] = -15
    repin(evidence)
    with pytest.raises(AssertionError):
        check(evidence)


def test_checkpoint_archive_must_remain_inside_migration_artifacts(evidence):
    archive = evidence["root"] / "unprotected-checkpoint.pt"
    archive.write_bytes(evidence["targets"]["checkpoint_archive"].read_bytes())
    evidence["handoff"]["checkpoint_archive_path"] = str(archive.relative_to(evidence["root"]))
    repin(evidence)
    with pytest.raises(AssertionError):
        check(evidence)


@pytest.mark.parametrize("fault", ["wrong_host", "wrong_devices", "missing_source", "extra_source"])
def test_receipt_must_match_exact_benchmark_target_and_source_set(evidence, fault):
    receipt = evidence["receipt"]
    if fault == "wrong_host":
        receipt["host"] = "work"
    elif fault == "wrong_devices":
        receipt["devices"] = [0, 1, 2, 4]
    elif fault == "missing_source":
        receipt["execution_source_sha256"].pop("resume_sharded_data_efficiency.py")
    else:
        receipt["execution_source_sha256"]["unexpected.py"] = "unreviewed"
    repin(evidence)
    with pytest.raises(AssertionError):
        check(evidence)


@pytest.mark.parametrize("fault", ["wrong_seed", "wrong_from", "wrong_to", "failed_status", "different_static",
                                   "different_state", "missing_state", "non_boolean_state"])
def test_cross_host_resume_requires_complete_matching_native_parity(evidence, fault):
    native = evidence["native"]
    if fault == "wrong_seed":
        native["seed"] = 5
    elif fault == "wrong_from":
        native["from_host"] = "loki"
    elif fault == "wrong_to":
        native["to_host"] = "work"
    elif fault == "failed_status":
        native["status"] = "failed"
    elif fault == "different_static":
        native["static_model_and_signature_exact"] = False
    elif fault == "different_state":
        native["all_state_fields_except_elapsed_exact"]["y"] = False
    elif fault == "missing_state":
        native["all_state_fields_except_elapsed_exact"].pop("G")
    else:
        native["all_state_fields_except_elapsed_exact"]["G"] = "true"
    repin(evidence)
    with pytest.raises(AssertionError):
        check(evidence)


def test_same_host_continuation_does_not_require_cross_host_proof(evidence):
    evidence["handoff"]["source_host"] = "ripper"
    evidence["handoff"].pop("native_resume_proof_path")
    evidence["handoff"].pop("native_resume_proof_sha256")
    evidence["native_path"].unlink()
    repin(evidence)
    assert check(evidence)["resume_host"] == "ripper"


@pytest.mark.parametrize("field", ["original_signature", "final_moves", "summary", "search_seconds"])
def test_receipt_final_values_must_match_hashed_result(evidence, field):
    if field == "original_signature":
        evidence["receipt"][field]["seed"] = 5
    elif field == "summary":
        evidence["receipt"][field]["solved"] += 1
    else:
        evidence["receipt"][field] += 1
    repin(evidence)
    with pytest.raises(AssertionError):
        check(evidence)
