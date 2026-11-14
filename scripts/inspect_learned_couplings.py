"""Inspect fresh Figure 2b weights after training; never modify model parameters."""
from pathlib import Path
import hashlib
import itertools
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"
OUT = ROOT / "artifacts/figures"
sys.path.insert(0, str(UP))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(x):
    x = np.asarray(x, float)
    return {"n": x.size, "nonzero": int(np.count_nonzero(x)), "positive": int((x > 0).sum()),
            "negative": int((x < 0).sum()), "min": float(x.min()), "max": float(x.max()),
            "mean": float(x.mean()), "std": float(x.std())}


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    sudoku_path = UP / "experiments/runs/CRAM-S/full/full-aug-s0/model.pt"
    maze_path = ROOT / "artifacts/fresh_maze/maze_unique/seed0/model.pt"
    original_hashes = {"sudoku": sha(sudoku_path), "maze_unique": sha(maze_path)}
    sd = torch.load(sudoku_path, map_location="cpu", weights_only=False)
    weights, bias = sd["W"].double().numpy(), sd["B"].double().numpy()
    i, j = np.triu_indices(81, 1)
    row, col = np.divmod(np.arange(81), 9)
    box = (row // 3) * 3 + col // 3
    conflict = (row[i] == row[j]) | (col[i] == col[j]) | (box[i] == box[j])
    diag = np.diagonal(weights, axis1=1, axis2=2)
    diagonal_mask = np.eye(9, dtype=bool)
    positive_pairs = (weights > 0).any(axis=(1, 2))
    nonzero_pairs = (weights != 0).any(axis=(1, 2))
    false_pairs = nonzero_pairs & ~conflict
    S = np.zeros((81, 81, 9, 9))
    S[i, j] = weights
    S[j, i] = weights.transpose(0, 2, 1)
    expected = np.zeros((81, 81), bool)
    expected[i, j] = conflict
    expected[j, i] = conflict
    actual = np.any(S != 0, axis=(2, 3))
    mean_diag = np.diagonal(S, axis1=2, axis2=3).mean(axis=-1)
    pair_digit_spread = np.ptp(diag[conflict], axis=-1)
    extra = [{"cells": [int(i[k]), int(j[k])], "row_col": [[int(row[i[k]]), int(col[i[k]])],
               [int(row[j[k]]), int(col[j[k]])]], "same_digit_costs": diag[k].tolist()}
             for k in np.flatnonzero(false_pairs)]
    sudoku = {
        "seed": 0, "model": str(sudoku_path.relative_to(ROOT)), "model_sha256": original_hashes["sudoku"],
        "learned_parameters": int(weights.size + bias.size),
        "nonzero_parameters": int(np.count_nonzero(weights) + np.count_nonzero(bias)),
        "rule_graph_used_only_after_training": True,
        "rule_conflict_pairs": int(conflict.sum()), "nonconflict_pairs": int((~conflict).sum()),
        "positive_pair_support": {"true_positive": int((positive_pairs & conflict).sum()),
             "false_positive": int((positive_pairs & ~conflict).sum()),
             "false_negative": int((~positive_pairs & conflict).sum())},
        "nonzero_pair_support": {"true_positive": int((nonzero_pairs & conflict).sum()),
             "false_positive": int(false_pairs.sum()), "false_negative": int((~nonzero_pairs & conflict).sum())},
        "rule_same_digit_costs": summary(diag[conflict]),
        "rule_different_digit_costs": summary(weights[conflict][:, ~diagonal_mask]),
        "nonrule_same_digit_costs": summary(diag[~conflict]),
        "nonrule_different_digit_costs": summary(weights[~conflict][:, ~diagonal_mask]),
        "unary_costs": summary(bias),
        "rule_digit_spread_within_each_pair": summary(pair_digit_spread),
        "rule_mean_pair_cost_spread": summary(diag[conflict].mean(axis=-1)),
        "coupled_cells_per_cell": summary(actual.sum(axis=1)),
        "rule_neighbors_per_cell": summary(expected.sum(axis=1)),
        "extra_nonrule_pairs": extra,
        "largest_extra_absolute_cost_over_mean_rule_cost": float(np.abs(diag[~conflict]).max() / diag[conflict].mean()),
        "focal_cell": 40, "focal_row_col_zero_based": [4, 4],
        "interpretation": "Positive same-digit pair support exactly recovers row/column/box conflicts, but the raw learned energy also has weak negative same-digit costs on non-conflicting pairs and nonuniform positive magnitudes. It is not an exact uniform hand-set Sudoku energy.",
    }

    from cram.cram_maze import CRAMMazeFreeEnergy
    model = CRAMMazeFreeEnergy(window=1, init=0.)
    model.load_state_dict(torch.load(maze_path, map_location="cpu", weights_only=True))
    kernel = model.kernel().detach().double().numpy()
    b = model.b.detach().double().numpy()
    # delta E = E(on)-E(off) = intercept + sum_d coupling_d * on_neighbor_d.
    base = kernel[1, 0] - kernel[0, 0]
    coupling = kernel[1, 1] - kernel[0, 1] - kernel[1, 0] + kernel[0, 0]
    intercept = float(base.sum() - (b[1] - b[0]))
    side_mask = np.array([[False, True, False], [True, False, True], [False, True, False]])
    unit = float(-coupling[side_mask].mean())
    assert unit > 0
    normalized = coupling / unit
    normalized[1, 1] = intercept / unit
    offsets = [(dr, dc) for dr in range(-1, 2) for dc in range(-1, 2) if dr or dc]
    checks = []
    for ri, ci in itertools.product(range(3), repeat=2):
        available = [(dr, dc) for dr, dc in offsets if 0 <= ri + dr < 3 and 0 <= ci + dc < 3]
        states = np.array(list(itertools.product([0, 1], repeat=len(available))), dtype=np.int64)
        yy = np.zeros((len(states), 3, 3), np.int64)
        for k, (dr, dc) in enumerate(available):
            yy[:, ri + dr, ci + dc] = states[:, k]
        onehot = torch.nn.functional.one_hot(torch.from_numpy(yy), 2).permute(0, 3, 1, 2).float()
        x = torch.ones((len(states), 3, 3), dtype=torch.long)
        with torch.no_grad():
            current = model.current(onehot, x)[:, :, ri, ci].numpy()
        measured = current[:, 1] - current[:, 0]
        affine = np.full(len(states), -(b[1] - b[0]))
        cardinal_count = np.zeros(len(states), int)
        for k, (dr, dc) in enumerate(available):
            affine += base[dr + 1, dc + 1] + coupling[dr + 1, dc + 1] * states[:, k]
            if abs(dr) + abs(dc) == 1:
                cardinal_count += states[:, k]
        keep = affine < 0
        expected_keep = cardinal_count >= 2
        checks.append({"cell_position": [ri, ci], "neighbor_configurations": len(states),
            "max_affine_vs_source_field_difference": float(np.abs(measured - affine).max()),
            "dead_end_pruning_rule_disagreements": int((keep != expected_keep).sum()),
            "smallest_absolute_decision_margin": float(np.abs(affine).min()),
            "edge_note": "Finite grid zero-padding honored: absent neighbors contribute neither state cost nor coupling."})
    maze = {
        "seed": 0, "model": str(maze_path.relative_to(ROOT)), "model_sha256": original_hashes["maze_unique"],
        "raw_pair_tables": model.T.detach().tolist(), "raw_bias": b.tolist(),
        "current_kernel": kernel.tolist(), "binary_coupling": coupling.tolist(),
        "interior_intercept": intercept, "normalization_mean_cardinal_magnitude": unit,
        "normalized_kernel_center_is_intercept": normalized.tolist(), "independent_neighborhood_checks": checks,
        "checked_configurations": sum(r["neighbor_configurations"] for r in checks),
        "dead_end_pruning_rule_disagreements": sum(r["dead_end_pruning_rule_disagreements"] for r in checks),
        "interpretation": "For a free cell, the binary field is E(on)-E(off). Negative prefers on. Center in the display is the interior intercept, not a self-coupling. Exact affine expansion is checked against the source field on all local binary configurations at interior, edge, and corner locations.",
    }
    from cram import maze_unique as M
    boundary = np.zeros((30, 30), bool)
    boundary[0] = boundary[-1] = True
    boundary[:, 0] = boundary[:, -1] = True
    boundary_audit = []
    for split in ("train", "dev100", "test"):
        data = M.load(split)
        token = data["inputs"].reshape(-1, 30, 30)
        unique, counts = np.unique(token[:, boundary], return_counts=True)
        boundary_audit.append({"split": split, "mazes": len(token),
            "boundary_token_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
            "free_boundary_cells": int((token[:, boundary] == M.OPEN).sum()),
            "boundary_all_walls": bool((token[:, boundary] == M.WALL).all())})
    maze["dataset_boundary_audit"] = boundary_audit
    maze["boundary_scope"] = "All benchmark boundary cells are clamped walls, so every free cell has the complete 3x3 window. The 64 synthetic boundary disagreements concern out-of-benchmark free boundary cells; the applicable interior rule matches all 256 binary neighborhoods."
    assert all(r["boundary_all_walls"] for r in boundary_audit)
    assert checks[4]["dead_end_pruning_rule_disagreements"] == 0
    assert original_hashes == {"sudoku": sha(sudoku_path), "maze_unique": sha(maze_path)}
    np.savez_compressed(OUT / "learned-couplings-fresh.npz", sudoku_pair_tables=S, sudoku_pair_mean_same_digit=mean_diag,
                        sudoku_rule_graph=expected, maze_normalized=normalized)
    result = {"figure": "2b", "source_commit": "62bbe830c7d6f87f2ba18f30a28039258b986454",
              "weights_unmodified": True, "sudoku": sudoku, "maze_unique": maze}
    (OUT / "learned-couplings-fresh.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"sudoku_positive_pair_support": sudoku["positive_pair_support"],
        "sudoku_nonzero_pair_support": sudoku["nonzero_pair_support"], "sudoku_rule_costs": sudoku["rule_same_digit_costs"],
        "sudoku_extra_costs": sudoku["nonrule_same_digit_costs"], "maze_normalized": normalized.tolist(),
        "maze_checked_configurations": maze["checked_configurations"],
        "maze_dead_end_rule_disagreements": maze["dead_end_pruning_rule_disagreements"]}, indent=2))


if __name__ == "__main__":
    main()
