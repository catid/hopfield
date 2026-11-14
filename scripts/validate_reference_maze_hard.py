"""Full-test CPU validation of our independent solver on fresh fitted weights.

The solver imports only hopfield_cram. The saved upstream predictions are an
oracle for comparison, never an input to inference. This validates inference;
the checkpoint was trained by the separately pinned optimized implementation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hopfield_cram import MazeHardEnergy, solve_maze_hard


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/reference_maze_hard")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source = ROOT / "artifacts/fresh_maze/maze_hard" / f"seed{args.seed}"
    checkpoint = source / "model.pt"
    dataset = source / "test_predictions.npz"
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / f"seed{args.seed}.json"
    if result_path.exists():
        raise FileExistsError(f"A final result already exists: {result_path}")
    data = np.load(dataset)
    tokens, labels = data["inputs"], data["labels"]
    assert tokens.shape == labels.shape == (1000, 30, 30)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(state) == {"mu", "beta", "kappa", "b", "T", "k"}
    assert bool((state["k"] == 0).all())  # Unit hidden edge weights.
    model = MazeHardEnergy()
    released_offsets = ((0, 0), (-1, 0), (0, -1), (0, 1), (1, 0))
    permutation = [released_offsets.index(d) for d in model.attention_offsets]
    with torch.no_grad():
        model.maze.tables.copy_(state["T"])
        model.maze.unary_cost.copy_(-state["b"])
        model.log_excess_mass.copy_(state["mu"].squeeze())
        model.attention_bias.copy_(state["beta"][permutation])
        model.link_strength.copy_(state["kappa"][permutation])
    model.eval()
    boards, steps, stopped = [], [], []
    begin = time.perf_counter()
    for lo in range(0, len(tokens), args.batch_size):
        batch = torch.as_tensor(tokens[lo:lo + args.batch_size])
        result = solve_maze_hard(model, batch)
        boards.append(result.boards.numpy().reshape(-1, 30, 30))
        steps.append(result.steps.numpy())
        stopped.append(result.stopped.numpy())
        print(json.dumps({"seed": args.seed, "completed": min(lo + len(batch), len(tokens)),
                          "total": len(tokens), "elapsed_s": time.perf_counter() - begin}), flush=True)
    elapsed = time.perf_counter() - begin
    boards, steps, stopped = map(np.concatenate, (boards, steps, stopped))
    exact = (boards == labels).all((1, 2))
    prediction_agrees = (boards == data["predictions"]).all((1, 2))
    arrays_path = args.out / f"seed{args.seed}-predictions.npz"
    np.savez_compressed(arrays_path, predictions=boards, steps=steps, stopped=stopped)
    record = {
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": args.seed, "n": len(tokens), "solved": int(exact.sum()),
        "wrong_indices": np.flatnonzero(~exact).tolist(),
        "stopped": int(stopped.sum()), "wrong_stops": int((stopped & ~exact).sum()),
        "predictions_agree_with_optimized": int(prediction_agrees.sum()),
        "prediction_disagreement_indices": np.flatnonzero(~prediction_agrees).tolist(),
        "stop_steps_agree_with_optimized": int((steps == data["steps"]).sum()),
        "stopped_flags_agree_with_optimized": int((stopped == data["stopped"]).sum()),
        "elapsed_s": elapsed, "batch_size": args.batch_size, "threads": args.threads,
        "device": "cpu", "host": platform.node(), "torch": torch.__version__,
        "python": sys.version, "argv": sys.argv,
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha(checkpoint),
        "dataset": str(dataset.relative_to(ROOT)), "dataset_sha256": sha(dataset),
        "output_sha256": sha(arrays_path),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src/hopfield_cram").glob("*.py"))},
        "script_sha256": sha(__file__),
        "protocol": "Independent float32 CPU solver; complete 1000-puzzle test set; fresh optimized fit weights; no training or hyperparameter selection in this validation.",
    }
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
