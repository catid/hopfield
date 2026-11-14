"""Recovery after interrupted publication; synthetic files and no GPU calls."""
import json
from pathlib import Path
import shutil

import pytest

from test_data_efficiency_migration_evidence import evidence, digest, put, repin
import de_checkpoint_handoff as H


@pytest.fixture
def publication(evidence, monkeypatch):
    root = evidence["root"]
    monkeypatch.setattr(H, "ROOT", root)
    base = evidence["paths"]["receipt"].parent
    evidence["paths"]["receipt"] = base / "sharded-search-s4.json"
    canonical = evidence["run"]
    isolated = base / "k1-s4"
    shutil.copytree(canonical, isolated)
    (canonical / "test10k.json").unlink()
    result = {"signature": {"k": 1, "seed": 4}, "summary": {"n": 10000},
              "per_puzzle": {"puzzle": list(range(10000, 20000)), "moves": 102400, "seconds": 210.5}}
    put(isolated / "test10k.json", result)
    evidence["receipt"].update(host="foureyes", original_signature=result["signature"],
        result_sha256=digest(isolated / "test10k.json"), argv=["fixture-resumer"],
        final_moves=result["per_puzzle"]["moves"], search_seconds=result["per_puzzle"]["seconds"],
        summary=result["summary"],
        completed_utc="2026-11-14T00:00:00+00:00")
    evidence["proof"]["host"] = "foureyes"
    repin(evidence)
    meta = evidence["meta"] | {"status": "checkpoint_migration", "phases": [
        {"phase": "fit", "exit_code": 0},
        {"phase": "search", "exit_code": -15, "interrupted_for_checkpoint_migration": True}]}
    meta.pop("search_migration")
    meta_path = root / "artifacts/data_efficiency/jobs/k1-s4.json"
    put(meta_path, meta)
    journal = base.parent / "journal.json"
    item = {"seed": 4, "job": evidence["job"], "target_host": "foureyes",
            "isolated_run": str(isolated), "source_run": str(canonical),
            "original_meta_path": str(meta_path),
            "parent": {"pid": 91000001, "startticks": 1},
            "child": {"pid": 91000002, "startticks": 2}}
    put(journal, {"status": "retired_captured", "jobs": [item]})
    # Process and numerical-result validation have separate tests. These
    # fixtures exercise real evidence validation and publication ordering.
    monkeypatch.setattr(H, "alive", lambda saved: False)
    monkeypatch.setattr(H.Q, "validate_saved_outputs", lambda *args: None)
    return {"journal": journal, "meta": meta_path, "canonical": canonical,
            "isolated": isolated, "receipt": evidence["paths"]["receipt"], "seed": 4}


def read(path):
    return json.loads(path.read_text())


def assert_published_once(publication):
    p = publication
    assert digest(p["canonical"] / "test10k.json") == digest(p["isolated"] / "test10k.json")
    meta = read(p["meta"])
    assert meta["status"] == "complete"
    assert [phase["exit_code"] for phase in meta["phases"]] == [0, -15, 0]
    assert meta["phases"][1]["interrupted_for_checkpoint_migration"] is True
    assert meta["phases"][-1]["migration_receipt_sha256"] == digest(p["receipt"])
    assert read(p["journal"])["jobs"][0]["imported_result_sha256"] == digest(p["isolated"] / "test10k.json")


def test_completed_import_is_idempotent_and_preserves_interrupted_history(publication):
    H.import_result(publication["journal"], 4)
    H.import_result(publication["journal"], 4)
    assert_published_once(publication)


@pytest.mark.parametrize("crash_boundary", ["after_canonical_copy", "after_metadata_write"])
def test_import_recovers_after_each_publication_boundary(publication, monkeypatch, crash_boundary):
    original_write = H.durable_write
    failed = False
    failure_path = publication["meta"] if crash_boundary == "after_canonical_copy" else publication["journal"]

    def injected_failure(path, record):
        nonlocal failed
        if Path(path) == failure_path and not failed:
            failed = True
            raise RuntimeError("synthetic publication interruption")
        return original_write(path, record)

    monkeypatch.setattr(H, "durable_write", injected_failure)
    with pytest.raises(RuntimeError, match="synthetic publication interruption"):
        H.import_result(publication["journal"], 4)
    assert (publication["canonical"] / "test10k.json").exists()
    H.import_result(publication["journal"], 4)
    assert_published_once(publication)


def test_existing_different_final_result_is_preserved(publication):
    target = publication["canonical"] / "test10k.json"
    target.write_bytes(b"different final result must survive")
    before_meta = publication["meta"].read_bytes()
    before_journal = publication["journal"].read_bytes()
    with pytest.raises(AssertionError, match="Never overwrite a different final result"):
        H.import_result(publication["journal"], 4)
    assert target.read_bytes() == b"different final result must survive"
    assert publication["meta"].read_bytes() == before_meta
    assert publication["journal"].read_bytes() == before_journal
