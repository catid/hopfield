"""CPU audits for the Maze-Unique initialization claim and Figure 5 hand weights."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"
OUT = ROOT / "artifacts/figures"
sys.path[:0] = [str(UP), str(UP / "experiments")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manual_energy(model, board, free):
    """Independent NumPy sum over unordered offset pairs and free-cell unary costs."""
    states = np.eye(2)[board]
    total = np.zeros(len(board), float)
    height, width = board.shape[1:]
    for table, (dr, dc) in zip(model.T.detach().double().numpy(), model.offsets):
        r0, r1 = max(0, -dr), min(height, height - dr)
        c0, c1 = max(0, -dc), min(width, width - dc)
        s = states[:, r0:r1, c0:c1]
        t = states[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        total += np.einsum("bhwa,ac,bhwc->b", s, table, t)
    total -= (states * model.b.detach().double().numpy() * free[..., None]).sum(axis=(1, 2, 3))
    return total


def main():
    torch.set_num_threads(1)
    from cram import maze_unique as M
    from cram.cram_maze import CRAMMazeFreeEnergy, board_energy
    import hand_set as H
    import graph_coloring_hand_set as GC
    import nqueens_free as NQ
    from cram import cram_nqueens as Q

    OUT.mkdir(exist_ok=True)
    data = M.load("dev100")
    x, target, free = M.to_device(data, np.arange(100), "cpu")
    empty = ((x == M.S_TOK) | (x == M.G_TOK)).long()
    rows = []
    for seed in (0, 1, 2):
        path = ROOT / f"artifacts/fresh_maze/maze_unique/seed{seed}/model.pt"
        model = CRAMMazeFreeEnergy(window=1, init=0.)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        with torch.no_grad():
            exact = board_energy(model, target, x, free).double().numpy()
            off = board_energy(model, empty, x, free).double().numpy()
        independent_exact = manual_energy(model, target.numpy(), free.numpy())
        independent_empty = manual_energy(model, empty.numpy(), free.numpy())
        assert np.allclose(exact, independent_exact, rtol=1e-6, atol=1e-4)
        assert np.allclose(off, independent_empty, rtol=1e-6, atol=1e-4)
        difference = independent_exact - independent_empty
        assert (difference > 0).all()
        rows.append({"seed": seed, "model_sha256": sha(path), "dev_mazes": 100,
            "empty_lower_energy_count": int((difference > 0).sum()),
            "solution_minus_empty_energy": {"minimum": float(difference.min()), "mean": float(difference.mean()),
                "maximum": float(difference.max()), "per_maze": difference.tolist()},
            "max_source_vs_independent_energy_discrepancy": float(max(np.abs(exact - independent_exact).max(),
                np.abs(off - independent_empty).max())),
            "source_solution_energy": exact.tolist(), "source_empty_energy": off.tolist()})
    initial = {"claim": "Appendix C.4: empty board has lower energy than solution on every development maze.",
        "split": "dev100", "seeds": [0, 1, 2], "per_seed": rows,
        "scope": "Empty means all free cells off while walls remain off and S/G remain clamped on. No inference or test scoring is performed; independent NumPy pair/unary sum checks source energies.",
        "source_sha256": sha(UP / "cram/cram_maze.py")}
    (OUT / "maze-empty-board-energy-audit.json").write_text(json.dumps(initial, indent=2) + "\n")

    A, B = H.sudoku_hand()
    sud = (A + A.T).double().numpy()
    r, c = np.divmod(np.arange(81), 9)
    box = 3 * (r // 3) + c // 3
    independent_sud = (r[:, None] == r) | (c[:, None] == c) | (box[:, None] == box)
    np.fill_diagonal(independent_sud, False)
    assert np.array_equal(sud != 0, independent_sud) and np.array_equal(sud, independent_sud.astype(float))
    assert np.all(sud.sum(axis=1) == 20) and not torch.count_nonzero(B)
    handmaze = H.maze_hand()
    kernel = handmaze.kernel().detach().double().numpy()
    b = handmaze.b.detach().double().numpy()
    binary = kernel[1, 1] - kernel[0, 1] - kernel[1, 0] + kernel[0, 0]
    intercept = float((kernel[1, 0] - kernel[0, 0]).sum() - (b[1] - b[0]))
    binary[1, 1] = intercept
    assert np.allclose(binary, [[-.1, -1, -.1], [-1, 1.5, -1], [-.1, -1, -.1]], rtol=0, atol=1e-7)
    n = 8
    i, j = np.triu_indices(n * n, 1)
    ri, ci, rj, cj = i // n, i % n, j // n, j % n
    attack = (ri == rj) | (ci == cj) | (np.abs(ri - rj) == np.abs(ci - cj))
    assert np.array_equal(attack, Q.attack_classes(n) > 0)
    nq = np.zeros((64, 64))
    nq[i, j] = nq[j, i] = attack * NQ.HAND["attack"]
    gc = GC.model()
    assert np.array_equal(gc["D"], np.eye(3)) and not np.any(gc["W0"]) and not np.any(gc["B"])
    source_paths = [UP / "experiments/hand_set.py", UP / "experiments/nqueens_free.py",
                    UP / "experiments/graph_coloring_hand_set.py"]
    meta = {"figure": 5, "learned_parameters": 0, "sudoku_unique_nonzero_energy_values": [1.0],
        "sudoku_conflict_pairs": int(independent_sud.sum() // 2), "sudoku_neighbors_per_cell": 20,
        "maze_unique_nonzero_effective_values": [-1., -.1, 1.5], "maze_effective_kernel": binary.tolist(),
        "nqueens_unique_nonzero_energy_values": [1., -.5], "nqueens_unary_on_cost": -NQ.HAND["queen"],
        "nqueens_attack_pairs": int(attack.sum()), "nqueens_size": 8,
        "graph_coloring_unique_nonzero_energy_values": [1.0], "graph_coloring_edge_cost": gc["D"].tolist(),
        "graph_coloring_nonedge_cost": gc["W0"].tolist(),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        "validation": "Sudoku and N-Queens supports checked against independently assembled row/column/box or row/column/diagonal masks; Maze and graph-coloring values checked directly against the claimed energies.",
        "scope": "Visualization and source consistency only; full control inference results are recorded separately."}
    np.savez_compressed(OUT / "hand-energies-fresh.npz", sudoku=sud, maze=binary, nqueens=nq, gc=gc["D"])
    (OUT / "hand-energies-fresh.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"empty_board_claim": {str(r["seed"]): r["empty_lower_energy_count"] for r in rows},
        "sudoku_pairs": meta["sudoku_conflict_pairs"], "nqueens_pairs": meta["nqueens_attack_pairs"]}))


if __name__ == "__main__":
    main()
