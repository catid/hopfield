"""Replay Figure 6's one Maze-Hard example from the fresh seed-0 fit on CPU.

This is a figure-specific replay, not a second full-test evaluation. All adapters,
exports, and checks live in artifacts; upstream source and results stay untouched.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"
SOURCE = ROOT / "artifacts/fresh_maze/maze_hard/seed0"
OUT = ROOT / "artifacts/figures"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    adapter = OUT / "maze-hard-fresh-source"
    adapter.mkdir(exist_ok=True)
    shutil.copy2(SOURCE / "model.pt", adapter / "final-s0.pt")
    (adapter / "final-s0.json").write_text(json.dumps({"recipe": json.loads((SOURCE / "config.json").read_text())}) + "\n")
    (adapter / "test-scores.json").write_text(json.dumps({"seeds": {"0": json.loads((SOURCE / "test.json").read_text())}}) + "\n")
    os.environ.setdefault("CRAM_DATA", str(UP / "data"))
    sys.path[:0] = [str(UP), str(UP / "experiments")]
    import maze_hard_figure_export as E

    E.FIN = adapter
    # HERE only controls the printed model path inside main; this permits a
    # relative path to artifacts outside the original upstream folder.
    E.HERE = UP
    destination = OUT / "maze-hard-figure-fresh.json"
    E.main(E.Args(maze=0, seed=0, out=str(destination)))
    data = json.loads(destination.read_text())
    saved = np.load(SOURCE / "test_predictions.npz")
    final = np.asarray(data["final"], bool)
    assert np.array_equal(final, saved["predictions"][0].astype(bool))
    assert np.array_equal(final, np.asarray(data["label"], bool))
    energy = np.asarray(data["energy"] + data["energy_after_stop"])
    difference = np.diff(energy)
    before = np.diff(data["energy"])
    check = {
        "figure": 6, "split": "test", "maze": 0, "seed": 0,
        "replay_device": "cpu", "threads": 1,
        "fresh_model_sha256": sha(SOURCE / "model.pt"),
        "adapted_model_sha256": sha(adapter / "final-s0.pt"),
        "source_test_record_sha256": sha(SOURCE / "test.json"),
        "saved_predictions_sha256": sha(SOURCE / "test_predictions.npz"),
        "exporter_sha256": sha(UP / "experiments/maze_hard_figure_export.py"),
        "figure_json_sha256": sha(destination),
        "exactly_matches_original_scored_board": True,
        "exactly_matches_label": True,
        "hidden_steps": data["hidden_steps"],
        "descent_steps": data["descent_steps"],
        "energy_start": float(energy[0]), "energy_at_stop": data["energy"][-1],
        "energy_after_48_more_steps": float(energy[-1]),
        "energy_drop_before_stop": float(energy[0] - data["energy"][-1]),
        "max_energy_increase_before_stop": float(max(0, before.max())),
        "max_energy_increase_including_after_stop": float(max(0, difference.max())),
        "positive_energy_differences_before_stop": int((before > 0).sum()),
        "positive_energy_differences_including_after_stop": int((difference > 0).sum()),
        "energy_nonincreasing_with_absolute_tolerance_1e_3": bool((difference <= 1e-3).all()),
        "max_answer_change_first_extra_step": data["max_answer_change_after_stop"][0],
        "max_answer_change_last_extra_step": data["max_answer_change_after_stop"][-1],
        "rounded_board_unchanged_for_48_extra_steps": True,
        "scope": "Single-maze figure replay only. Upstream exporter asserts fixed rounded board for 48 extra steps. Positive floating-point energy differences are reported explicitly.",
    }
    assert check["fresh_model_sha256"] == check["adapted_model_sha256"] == data["model_sha256"]
    (OUT / "maze-hard-figure-audit.json").write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
