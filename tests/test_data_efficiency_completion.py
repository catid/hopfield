"""CPU-only checks that a saved search is complete and resumes its archive.

The synthetic model and rows exercise the full completed-result validator;
no released data, GPU inference, or live reproduction outputs are needed.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_data_efficiency as C


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def summarize(result):
    """Keep summaries honest so corrupt trajectories reach their own guard."""
    pp = result["per_puzzle"]
    solved = np.asarray(pp["solved"], dtype=bool)
    stopped = np.asarray(pp["stopped"], dtype=bool)
    flops = np.asarray(pp["flops"], dtype=float)
    result["summary"] = {
        "n": 10000, "solved": int(solved.sum()), "stopped": int(stopped.sum()),
        "wrong_stops": int((stopped & ~solved).sum()),
        "budget_failures": int((~stopped).sum()),
        "unstopped_but_right": int((~stopped & solved).sum()),
        "mean_flop": float(flops.mean()), "median_flop": float(np.median(flops)),
        "p99_flop": float(np.quantile(flops, .99)),
        "p999_flop": float(np.quantile(flops, .999)), "max_flop": float(flops.max()),
    }


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "OUT", tmp_path / "audit")
    torch.set_num_threads(1)
    run = tmp_path / "run"
    run.mkdir()
    W = torch.zeros((3240, 9, 9))
    W[0, 0, 0] = 1.
    torch.save({"W": W, "B": torch.zeros((81, 9))}, run / "model.pt")
    dataset = tmp_path / "dataset/train"
    dataset.mkdir(parents=True)
    np.save(dataset / "all__inputs.npy", np.ones((1001, 81), dtype=np.uint8))
    np.save(dataset / "all__labels.npy", np.full((1001, 81), 2, dtype=np.uint8))
    np.savez(run / "train.npz", solutions=np.zeros((100, 81), dtype=np.int16),
             free=np.ones((100, 81), dtype=bool))
    fit = {
        "k": 1, "seed": 4,
        "config": dict(aug=False, lam=3e-3 / 9, lr=540., momentum=.5,
                       updates=32, split=4, seed=4),
        "variants": 99, "per_update_relabeling": False,
        "examples": 100, "params": 263169, "fit_gpu_warm_s": [.1] * 7,
        "warm_bitwise_equal": True, "fit_gpu_warm_median_s": .1,
        "fit_gpu_cold_s": .2, "model_sha256": digest(run / "model.pt"),
        "positive_W": 1, "negative_W": 0, "nonzero_params": 1,
        "tau": -1., "hbar": 1., "D": 8, "gpu": "synthetic CPU fixture",
        "node": "fixture",
    }
    put(run / "fit.json", fit)
    signature = {
        "k": 1, "seed": 4, "data": "test10k", "n": 10000,
        "model_sha256": fit["model_sha256"], "tau": -1., "hbar": 1., "D": 8,
        "rel": [.4, .31, .45], "C": 4, "cap": 4e11, "check": 4096,
        "tol": 1e-3, "fl_move": 1000000,
    }
    result = {
        "signature": signature, "resumes": 0,
        "per_puzzle": {
            "puzzle": list(range(10000, 20000)),
            "solved": [True] + [False] * 9999,
            "stopped": [True, True] + [False] * 9998,
            "flops": [4096 * 4000000] * 2 + [102400 * 4000000] * 9998,
            "moves": 102400, "T_cap": 100000, "seconds": 10.,
        },
    }
    summarize(result)
    put(run / "test10k.json", result)
    meta = {"phases": [{"phase": phase, "exit_code": 0} for phase in ("fit", "search")],
            "output_sha256": {name: digest(run / name)
                              for name in ("model.pt", "fit.json", "test10k.json")}}
    return {"root": tmp_path, "run": run, "result": result, "meta": meta,
            "job": {"k": 1, "seed": 4, "id": "fixture-k1-s4"},
            "data": {"path": str(dataset.parent)}}


def check(fixture):
    summarize(fixture["result"])
    put(fixture["run"] / "test10k.json", fixture["result"])
    fixture["meta"]["output_sha256"]["test10k.json"] = digest(fixture["run"] / "test10k.json")
    return C.validate_completed(fixture["job"], fixture["meta"], fixture["run"],
                                fixture["data"], C.Hashes())


def set_move(fixture, moves):
    pp = fixture["result"]["per_puzzle"]
    pp["moves"] = moves
    unit = fixture["result"]["signature"]["fl_move"] * 4
    pp["flops"] = [min(4096, moves) * unit if stopped else moves * unit
                   for stopped in pp["stopped"]]


def test_accepts_first_check_boundary_beyond_budget(completed):
    row = check(completed)
    assert row["status"] == "complete"
    assert row["accuracy_pct"] == .01
    assert row["summary"]["budget_failures"] == 9998


def test_rejects_partial_search_before_budget_with_unstopped_puzzles(completed):
    set_move(completed, 98304)
    with pytest.raises(AssertionError, match="Partial search"):
        check(completed)


def test_accepts_early_finish_only_when_every_puzzle_stopped(completed):
    pp = completed["result"]["per_puzzle"]
    pp["stopped"] = [True] * 10000
    set_move(completed, 4096)
    row = check(completed)
    assert row["status"] == "complete" and row["summary"]["budget_failures"] == 0
    assert row["summary"]["wrong_stops"] == 9999


@pytest.mark.parametrize("moves", [0, -4096, 102401, 102400., 106496])
def test_rejects_zero_negative_unaligned_noninteger_or_overrun_moves(completed, moves):
    set_move(completed, moves)
    with pytest.raises(AssertionError):
        check(completed)


def test_rejects_invented_move_budget(completed):
    completed["result"]["per_puzzle"]["T_cap"] = 98304
    with pytest.raises(AssertionError):
        check(completed)


@pytest.mark.parametrize("step", [0, .5, 102401])
def test_rejects_zero_fractional_or_future_per_puzzle_steps(completed, step):
    completed["result"]["per_puzzle"]["flops"][0] = step * 4000000
    with pytest.raises(AssertionError):
        check(completed)


def test_unstopped_puzzle_must_have_consumed_every_executed_move(completed):
    completed["result"]["per_puzzle"]["flops"][2] = 98304 * 4000000
    with pytest.raises(AssertionError):
        check(completed)


@pytest.fixture
def resumed(completed):
    base = completed["root"] / "artifacts/data_efficiency/migrations/fixture"
    base.mkdir(parents=True)
    checkpoint = {
        "signature": deepcopy(completed["result"]["signature"]), "resumes": 2,
        "state": {"t": 81920, "elapsed": 7., "act": torch.arange(9998)},
    }
    handoff = {"checkpoint_archive_path": str((base / "initial.pt").relative_to(completed["root"])),
               "checkpoint_move": 81920}
    receipt = {"starting_move": 81920, "prior_search_seconds": 7.,
               "starting_active_puzzles": 9998}
    completed["result"]["resumes"] = 3
    completed["meta"]["search_migration"] = {
        "receipt_path": str((base / "receipt.json").relative_to(completed["root"])),
        "handoff_path": str((base / "handoff.json").relative_to(completed["root"])),
    }
    completed.update(base=base, checkpoint=checkpoint, handoff=handoff, receipt=receipt)
    return completed


def check_resume(fixture):
    torch.save(fixture["checkpoint"], fixture["base"] / "initial.pt")
    put(fixture["base"] / "handoff.json", fixture["handoff"])
    put(fixture["base"] / "receipt.json", fixture["receipt"])
    return check(fixture)


def test_accepts_resume_bound_to_saved_checkpoint(resumed):
    assert check_resume(resumed)["resumes"] == 3


@pytest.mark.parametrize("fault", ["signature", "receipt_move", "handoff_move", "archive_move",
                                   "receipt_elapsed", "archive_elapsed", "receipt_active", "archive_active",
                                   "result_resumes", "archive_resumes"])
def test_rejects_inconsistent_archived_resume_state(resumed, fault):
    if fault == "signature":
        resumed["checkpoint"]["signature"]["seed"] = 5
    elif fault == "receipt_move":
        resumed["receipt"]["starting_move"] += 4096
    elif fault == "handoff_move":
        resumed["handoff"]["checkpoint_move"] += 4096
    elif fault == "archive_move":
        resumed["checkpoint"]["state"]["t"] += 4096
    elif fault == "receipt_elapsed":
        resumed["receipt"]["prior_search_seconds"] += 1.
    elif fault == "archive_elapsed":
        resumed["checkpoint"]["state"]["elapsed"] += 1.
    elif fault == "receipt_active":
        resumed["receipt"]["starting_active_puzzles"] -= 1
    elif fault == "archive_active":
        resumed["checkpoint"]["state"]["act"] = torch.arange(9997)
    elif fault == "result_resumes":
        resumed["result"]["resumes"] += 1
    elif fault == "archive_resumes":
        resumed["checkpoint"]["resumes"] += 1
    with pytest.raises(AssertionError):
        check_resume(resumed)
