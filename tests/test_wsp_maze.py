"""Comprehensive unit, AST, parity, FullCounter (FC2), and edge-case tests for Maze WSP."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (
    ROOT / "src",
    ROOT / "upstream",
    ROOT / "upstream/baselines/flops",
    ROOT / "reproduction/baselines/code/Weight-Space-Programming",
):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

from fullcount import FullCounter  # noqa: E402
from wsp.free_model import static_problems  # noqa: E402
from hopfield_cram import wsp_maze as M  # noqa: E402


def _make_spiral_maze(h: int, w: int) -> np.ndarray:
    """Create a deterministic winding corridor maze of arbitrary shape (h, w)."""
    grid = np.zeros((h, w), dtype=np.int64)
    path_cells: list[tuple[int, int]] = []
    for r_idx, r in enumerate(range(0, h, 2)):
        cols = range(0, w) if r_idx % 2 == 0 else range(w - 1, -1, -1)
        for c in cols:
            grid[r, c] = 1
            path_cells.append((r, c))
        if r + 2 < h:
            conn_c = w - 1 if r_idx % 2 == 0 else 0
            grid[r + 1, conn_c] = 1
            path_cells.append((r + 1, conn_c))
    grid[path_cells[0]] = 2
    grid[path_cells[-1]] = 3
    return grid


def _make_exact_distance_corridor(dist: int, side: int = 24) -> np.ndarray:
    """Create a snake corridor whose shortest path from S to G has exact edge length `dist`."""
    grid = np.zeros((side, side), dtype=np.int64)
    cells: list[tuple[int, int]] = []
    r, c, dc = 1, 1, 1
    while len(cells) < dist + 1:
        grid[r, c] = 1
        cells.append((r, c))
        if len(cells) == dist + 1:
            break
        nxt_c = c + dc
        if 1 <= nxt_c < side - 1:
            c = nxt_c
        else:
            grid[r + 1, c] = 1
            cells.append((r + 1, c))
            if len(cells) == dist + 1:
                break
            r += 2
            dc = -dc
    grid[cells[0]] = 2
    grid[cells[-1]] = 3
    return grid


def test_static_problems_ast_contract() -> None:
    """All three WSP `arch.py` sources must pass upstream `static_problems` with zero errors."""
    for name, source in (
        ("WSP-DeadEnd", M.ARCH_SOURCE_DEADEND),
        ("WSP-Wavefront", M.ARCH_SOURCE_WAVEFRONT),
        ("WSP-Potential", M.ARCH_SOURCE_POTENTIAL),
    ):
        problems = static_problems(source, signature="forward(w, x)")
        assert problems == [], f"{name} failed static_problems: {problems}"

    # Verify the AST checker is non-vacuous on forbidden constructs
    bad_if = "import torch\ndef forward(w, x):\n    if True:\n        return x\n    return x\n"
    bad_item = "import torch\ndef forward(w, x):\n    y = x.sum().item()\n    return x\n"
    assert len(static_problems(bad_if, signature="forward(w, x)")) > 0
    assert len(static_problems(bad_item, signature="forward(w, x)")) > 0


def test_weight_metadata_and_zero_learned_params() -> None:
    """Verify zero learned params, exact coefficient counts, and deterministic construction."""
    meta_de = M.weight_metadata(M.build_weights_deadend())
    assert meta_de["params_learned"] == 0
    assert meta_de["fit_time"] == "none"
    assert meta_de["distinct_nonzero_coefficients"] == 3
    assert meta_de["distinct_nonzero_values"] == [-4.5, -1.5, 1.0]

    meta_wf = M.weight_metadata(M.build_weights_wavefront())
    assert meta_wf["params_learned"] == 0
    assert meta_wf["fit_time"] == "none"
    assert meta_wf["distinct_nonzero_coefficients"] == 5
    assert meta_wf["distinct_nonzero_values"] == [0.125, 0.25, 0.375, 1.0, 1048576.0]

    meta_pot = M.weight_metadata(M.build_weights_potential())
    assert meta_pot["params_learned"] == 0
    assert meta_pot["fit_time"] == "none"
    assert meta_pot["distinct_nonzero_coefficients"] == 11

    for builder in (M.build_weights_deadend, M.build_weights_wavefront, M.build_weights_potential):
        w1 = builder()
        w2 = builder()
        assert set(w1.keys()) == set(w2.keys())
        for k in w1:
            assert not w1[k].requires_grad
            assert torch.equal(w1[k], w2[k])


def test_zero_label_leakage_and_input_immutability() -> None:
    """Ensure solvers never mutate inputs/weights and strictly respect wall=0 and S/G clamps."""
    tok = _make_spiral_maze(7, 11)[None]
    x = torch.as_tensor(tok, dtype=torch.long)
    x_orig = x.clone()

    for builder, solver, arch_fn in (
        (M.build_weights_deadend, M.solve_wsp_deadend, M.arch_deadend_forward),
        (M.build_weights_wavefront, M.solve_wsp_wavefront, M.arch_wavefront_forward),
        (M.build_weights_potential, M.solve_wsp_potential, M.arch_potential_forward),
    ):
        w = builder()
        w_orig = {k: v.clone() for k, v in w.items()}
        res = solver(x, w)
        arch_pred = arch_fn(w, x)
        assert torch.equal(x, x_orig)
        for k in w:
            assert torch.equal(w[k], w_orig[k])
        assert torch.equal(res.predictions, arch_pred)
        assert not bool((res.predictions & (x == 0)).any())
        assert bool(res.predictions[x == 2].all())
        assert bool(res.predictions[x == 3].all())


@pytest.mark.parametrize("shape", [(2, 2), (7, 11), (13, 17)])
def test_adversarial_shapes_and_prime_grids(shape: tuple[int, int]) -> None:
    """Test minimal (2x2), non-square (7x11), and prime (13x17) grids across all 3 models."""
    h, w_dim = shape
    if shape == (2, 2):
        g1 = np.array([[2, 3], [0, 1]], dtype=np.int64)
        g2 = np.array([[2, 1], [0, 3]], dtype=np.int64)
        toks = np.stack([g1, g2])
    else:
        s1 = _make_spiral_maze(h, w_dim)
        # Add dead-end branches off the spiral (still a tree!)
        s2 = s1.copy()
        for r in range(1, h - 1):
            for c in range(1, w_dim - 1):
                if s2[r, c] == 0:
                    nbr_open = sum(int(s2[r + dr, c + dc] > 0) for dr, dc in M.SIDES)
                    if nbr_open == 1:
                        s2[r, c] = 1
        toks = np.stack([s1, s2])

    x = torch.as_tensor(toks, dtype=torch.long)
    for builder, solver, arch_fn in (
        (M.build_weights_deadend, M.solve_wsp_deadend, M.arch_deadend_forward),
        (M.build_weights_wavefront, M.solve_wsp_wavefront, M.arch_wavefront_forward),
        (M.build_weights_potential, M.solve_wsp_potential, M.arch_potential_forward),
    ):
        wt = builder()
        res = solver(x, wt)
        arch_out = arch_fn(wt, x)
        assert bool(res.halted.all())
        assert bool(res.finite.all())
        assert torch.equal(res.predictions, arch_out)
        audit = M.path_audit_batch(toks, res.predictions.numpy(), halted=res.halted.numpy())["summary"]
        assert audit["exact_solved"] == len(toks)
        assert audit["shortest_paths"] == len(toks)
        assert audit["wrong_halts_shortest"] == 0


def test_exact_check_boundary_distances_and_disconnected_guard() -> None:
    """Test exact D in {15, 16, 17, 31, 32, 33} check boundaries plus disconnected mazes."""
    corridors = np.stack([_make_exact_distance_corridor(d, side=20) for d in (15, 16, 17, 31, 32, 33)])
    x = torch.as_tensor(corridors, dtype=torch.long)
    w = M.build_weights_wavefront()
    res = M.solve_wsp_wavefront(x, w, check_every=16, sync_every=16, backend="pytorch")
    assert bool(res.halted.all())
    audit = M.path_audit_batch(corridors, res.predictions.numpy(), halted=res.halted.numpy())["summary"]
    assert audit["exact_solved"] == 6
    assert audit["wrong_halts_shortest"] == 0

    # Disconnected maze: wall blocks S from G
    disc = np.zeros((1, 7, 7), dtype=np.int64)
    disc[0, 1, 1:3] = 1
    disc[0, 1, 1] = 2
    disc[0, 5, 4:6] = 1
    disc[0, 5, 5] = 3
    xd = torch.as_tensor(disc, dtype=torch.long)
    res_d = M.solve_wsp_wavefront(xd, w, backend="pytorch")
    assert not bool(res_d.halted.any())
    assert bool(res_d.finite.all())
    assert int(res_d.stage_steps["trace_steps"].item()) == 0
    # Verify FullCounter executes 0 trace iterations when trace_steps == 0!
    disc16 = np.zeros((1, 18, 18), dtype=np.int64)
    disc16[0, 1, 1:4] = 1
    disc16[0, 1, 1] = 2
    disc16[0, 15, 13:16] = 1
    disc16[0, 15, 15] = 3
    cal_d = M.calibrate_fc2_affine(
        "wavefront",
        np.stack([_make_exact_distance_corridor(15, side=18), _make_exact_distance_corridor(16, side=18)]),
        FullCounter,
        max_samples=4,
    )
    rec_d = M.count_single_maze_fc2(
        lambda t, ww: M.solve_wsp_wavefront(t, ww, check_every=16, sync_every=16, backend="pytorch"),
        torch.as_tensor(disc16, dtype=torch.long),
        w,
        FullCounter,
    )
    pred_d = M.evaluate_fc2_from_steps(
        cal_d,
        {k: np.array([v]) for k, v in rec_d["stage_steps"].items()},
    )
    assert rec_d["stage_steps"]["trace_steps"] == 0
    assert rec_d["total"] == pred_d["sum_total"]
    assert rec_d["total_with_compare"] == pred_d["sum_total_with_compare"]
    audit_d = M.path_audit_batch(disc, res_d.predictions.numpy(), halted=res_d.halted.numpy())["summary"]
    assert audit_d["halted"] == 0
    assert audit_d["wrong_halts_shortest"] == 0
    assert audit_d["valid_simple_paths"] == 0

    # Empty batch (0, 7, 11)
    empty_x = torch.zeros((0, 7, 11), dtype=torch.long)
    for solver in (M.solve_wsp_deadend, M.solve_wsp_wavefront, M.solve_wsp_potential):
        r0 = solver(empty_x)
        assert r0.predictions.shape == (0, 7, 11)
        assert r0.steps.shape == (0,)

    with pytest.raises(ValueError):
        M.solve_wsp_wavefront(torch.zeros((7, 7), dtype=torch.long))
    with pytest.raises(ValueError):
        M.solve_wsp_wavefront(torch.zeros((1, 7, 7), dtype=torch.float32))


def test_exact_bfs_tie_breaking_on_open_and_random_loopy_grids() -> None:
    """Verify 100% exact `(down, left, right, up)` BFS tie parity on heavily tied loopy grids."""
    # 1. Completely open 6x8 grids with opposite corners (4 orientations)
    open_grids = []
    for (sr, sc), (gr, gc) in (
        ((0, 0), (5, 7)),
        ((5, 7), (0, 0)),
        ((0, 7), (5, 0)),
        ((5, 0), (0, 7)),
    ):
        g = np.ones((6, 8), dtype=np.int64)
        g[sr, sc] = 2
        g[gr, gc] = 3
        open_grids.append(g)
    tok_open = np.stack(open_grids)
    res_open = M.solve_wsp_wavefront(torch.as_tensor(tok_open), M.build_weights_wavefront())
    aud_open = M.path_audit_batch(tok_open, res_open.predictions.numpy(), halted=res_open.halted.numpy())["summary"]
    assert aud_open["exact_solved"] == 4
    assert aud_open["wrong_halts_exact"] == 0

    # 2. 48 random 85%-open prime-shaped 13x17 loopy grids
    rng = np.random.default_rng(20261117)
    loopy = []
    while len(loopy) < 48:
        g = (rng.random((13, 17)) < 0.85).astype(np.int64)
        pts = np.argwhere(g > 0)
        if len(pts) < 4:
            continue
        ix = rng.choice(len(pts), size=2, replace=False)
        g[pts[ix[0], 0], pts[ix[0], 1]] = 2
        g[pts[ix[1], 0], pts[ix[1], 1]] = 3
        _, dist, _ = M.bfs_shortest_path_np(g)
        if dist > 0:
            loopy.append(g)
    tok_loopy = np.stack(loopy)
    res_loopy = M.solve_wsp_wavefront(torch.as_tensor(tok_loopy), M.build_weights_wavefront())
    aud_loopy = M.path_audit_batch(tok_loopy, res_loopy.predictions.numpy(), halted=res_loopy.halted.numpy())["summary"]
    assert aud_loopy["exact_solved"] == 48
    assert aud_loopy["shortest_paths"] == 48
    assert aud_loopy["wrong_halts_exact"] == 0


def test_deadend_tree_vs_cycle_contrast() -> None:
    """Verify WSP-DeadEnd solves tree mazes and halts on the 2-core when a cycle is added."""
    tree = np.zeros((9, 9), dtype=np.int64)
    tree[1, 1:8] = 1
    tree[1:8, 7] = 1
    # Dead-end branches
    tree[2:6, 3] = 1
    tree[5, 3:6] = 1
    tree[1, 1] = 2
    tree[7, 7] = 3

    cyclic = tree.copy()
    # Close a loop off the dead-end branch back to the main path!
    cyclic[5, 6] = 1

    pair = np.stack([tree, cyclic])
    x = torch.as_tensor(pair, dtype=torch.long)

    r_de = M.solve_wsp_deadend(x, M.build_weights_deadend())
    aud_de = M.path_audit_batch(pair, r_de.predictions.numpy(), halted=r_de.halted.numpy())
    assert bool(aud_de["arrays"]["shortest_path"][0]) is True
    assert bool(aud_de["arrays"]["shortest_path"][1]) is False
    assert aud_de["summary"]["halted"] == 2
    assert aud_de["summary"]["wrong_halts_shortest"] == 1

    # Both WSP-Wavefront and WSP-Potential solve BOTH the tree and cyclic maze!
    for builder, solver in (
        (M.build_weights_wavefront, M.solve_wsp_wavefront),
        (M.build_weights_potential, M.solve_wsp_potential),
    ):
        r = solver(x, builder())
        aud = M.path_audit_batch(pair, r.predictions.numpy(), halted=r.halted.numpy())["summary"]
        assert aud["exact_solved"] == 2
        assert aud["shortest_paths"] == 2
        assert aud["wrong_halts_shortest"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton parity check")
def test_cuda_triton_vs_pytorch_vs_arch_exact_parity() -> None:
    """Verify state-for-state bit-exact parity across PyTorch, Triton, arch, and batch sizes."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    rng = np.random.default_rng(777)
    grids = []
    for d in (5, 15, 16, 17, 28, 32, 41):
        g = _make_exact_distance_corridor(d, side=17)
        # Add random loops/dead-ends
        extra = (rng.random(g.shape) < 0.25) & (g == 0)
        g[extra] = 1
        grids.append(g)
    while len(grids) < 17:  # Prime batch size B=17!
        g = _make_spiral_maze(17, 17)
        grids.append(g)
    tok = np.stack(grids)
    x_cpu = torch.as_tensor(tok, dtype=torch.long)
    x_gpu = x_cpu.cuda()

    for builder, solver, arch_fn in (
        (M.build_weights_deadend, M.solve_wsp_deadend, M.arch_deadend_forward),
        (M.build_weights_wavefront, M.solve_wsp_wavefront, M.arch_wavefront_forward),
    ):
        w_cpu = builder("cpu")
        w_gpu = builder("cuda")
        r_cpu = solver(x_cpu, w_cpu, backend="pytorch", sync_every=8)
        r_py_gpu = solver(x_gpu, w_gpu, backend="pytorch", sync_every=64)
        r_tr_gpu = solver(x_gpu, w_gpu, backend="triton", sync_every=64)
        r_arch = arch_fn(w_gpu, x_gpu)
        # Also verify B=1 single-item independence
        r_single = torch.cat(
            [solver(x_gpu[i : i + 1], w_gpu, backend="triton", sync_every=16).predictions for i in range(len(tok))],
            dim=0,
        )

        assert torch.equal(r_cpu.predictions, r_py_gpu.predictions.cpu())
        assert torch.equal(r_py_gpu.predictions, r_tr_gpu.predictions)
        assert torch.equal(r_tr_gpu.predictions, r_arch)
        assert torch.equal(r_tr_gpu.predictions, r_single)
        assert torch.equal(r_cpu.steps, r_py_gpu.steps.cpu())
        assert torch.equal(r_py_gpu.steps, r_tr_gpu.steps)
        assert torch.equal(r_py_gpu.halted, r_tr_gpu.halted)


def test_fullcounter_fc2_exact_affine_zero_residual() -> None:
    """Verify FullCounter (FC2) exact affine calibration (`max_abs_residual == 0`) on all 3 models."""
    # Build 14 synthetic 18x18 tree mazes with varied path lengths AND varied dead-end chain depths
    mazes = []
    for idx, d in enumerate((4, 8, 12, 15, 16, 17, 20, 24, 28, 31, 32, 36, 40, 44)):
        g = np.zeros((18, 18), dtype=np.int64)
        # Main corridor along top/right border of length d
        cells: list[tuple[int, int]] = []
        for c in range(1, min(17, d + 2)):
            g[1, c] = 1
            cells.append((1, c))
        rem = (d + 1) - len(cells)
        r_cur = 1
        while rem > 0 and r_cur + 1 < 17:
            r_cur += 1
            g[r_cur, 16] = 1
            cells.append((r_cur, 16))
            rem -= 1
        c_cur = 16
        while rem > 0 and c_cur - 1 >= 1:
            c_cur -= 1
            g[16, c_cur] = 1
            cells.append((16, c_cur))
            rem -= 1
        g[cells[0]] = 2
        g[cells[-1]] = 3
        # Add a single winding dead-end chain starting at (2, 2) attached to (1, 2) of depth (idx + 1) * 2
        if d >= 2:
            spur_len = min(22, (idx % 6) * 4 + 1)
            g[2, 2] = 1
            sr, sc, sdc = 2, 2, 1
            for _ in range(spur_len - 1):
                if 2 <= sc + sdc <= 13:
                    sc += sdc
                    g[sr, sc] = 1
                elif sr + 2 <= 14:
                    g[sr + 1, sc] = 1
                    sr += 2
                    g[sr, sc] = 1
                    sdc = -sdc
        mazes.append(g)
    train_tok = np.stack(mazes[:10])
    heldout_tok = np.stack(mazes[10:])

    for kind, check_ev, solver_fn, builder in (
        (
            "deadend",
            8,
            lambda t, ww: M.solve_wsp_deadend(t, ww, check_every=8, sync_every=8, backend="pytorch"),
            M.build_weights_deadend,
        ),
        (
            "wavefront",
            16,
            lambda t, ww: M.solve_wsp_wavefront(t, ww, check_every=16, sync_every=16, backend="pytorch"),
            M.build_weights_wavefront,
        ),
        (
            "potential",
            16,
            lambda t, ww: M.solve_wsp_potential(t, ww, hidden_check=16, descend_check=16, sync_every=16),
            M.build_weights_potential,
        ),
    ):
        cal = M.calibrate_fc2_affine(kind, train_tok, FullCounter, max_samples=10, check_every=check_ev)
        assert cal["max_abs_residual_total"] == 0.0
        assert cal["max_abs_residual_total_with_compare"] == 0.0
        assert cal["unknown_ops_count"] == 0

        # Independently count every held-out maze directly with FullCounter and compare to affine formula!
        w = builder("cpu")
        x_hold = torch.as_tensor(heldout_tok, dtype=torch.long)
        res_hold = solver_fn(x_hold, w)
        pred_fc2 = M.evaluate_fc2_from_steps(cal, res_hold.stage_steps)

        direct_tot = []
        direct_cmp = []
        for i in range(len(heldout_tok)):
            rec = M.count_single_maze_fc2(solver_fn, x_hold[i : i + 1], w, FullCounter)
            direct_tot.append(rec["total"])
            direct_cmp.append(rec["total_with_compare"])
            assert rec["total_with_compare"] > rec["total"] > 0

        assert sum(direct_tot) == pred_fc2["sum_total"]
        assert sum(direct_cmp) == pred_fc2["sum_total_with_compare"]


def test_potential_hopfield_energy_and_single_flip_margin() -> None:
    """Verify continuous Hopfield energy differences match exact single-flip margin formula."""
    tok = _make_spiral_maze(9, 9)[None]
    x = torch.as_tensor(tok, dtype=torch.long)
    w64 = M.build_weights_potential(dtype=torch.float64)
    res = M.solve_wsp_potential(x, w64)
    assert bool(res.halted.all())

    board = res.predictions.to(torch.float64)
    e0, min_margin = M.wsp_potential_energy_and_margin(x, board, w64)
    assert float(min_margin.item()) >= -1e-6

    # Check brute-force single-cell flip energy deltas against min_margin!
    deltas = []
    for r in range(9):
        for c in range(9):
            if tok[0, r, c] == 1:
                flipped = board.clone()
                flipped[0, r, c] = 1.0 - flipped[0, r, c]
                ef, _ = M.wsp_potential_energy_and_margin(x, flipped, w64)
                deltas.append(float((ef - e0).item()))
    assert abs(min(deltas) - float(min_margin.item())) < 1e-5


def test_path_audit_detects_longer_detour_and_disconnected_cycle() -> None:
    """Verify path_audit separates valid-but-longer detours and disconnected cycles from shortest paths."""
    tok = np.ones((3, 5, 5), dtype=np.int64)
    tok[:, 0, 0] = 2
    tok[:, 4, 4] = 3

    # Case 0: canonical shortest path (down 4, right 4 -> 9 cells, 8 edges)
    p0 = np.zeros((5, 5), dtype=bool)
    p0[:, 0] = True
    p0[4, :] = True

    # Case 1: valid simple detour path (12 edges vs shortest 8 edges, separated by wall row 1!)
    tok[1] = 0
    short_cells = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
    detour_cells = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 4), (2, 4), (2, 3), (2, 2),
        (3, 2), (4, 2), (4, 3), (4, 4),
    ]
    for r, c in short_cells + detour_cells:
        tok[1, r, c] = 1
    tok[1, 0, 0] = 2
    tok[1, 4, 4] = 3
    p1 = np.zeros((5, 5), dtype=bool)
    for r, c in detour_cells:
        p1[r, c] = True

    # Case 2: canonical shortest path PLUS a disjoint 2x2 cycle (degree=2 everywhere interior, disconnected!)
    tok[2] = 0
    for r, c in short_cells + [(0, 3), (0, 4), (1, 3), (1, 4)]:
        tok[2, r, c] = 1
    tok[2, 0, 0] = 2
    tok[2, 4, 4] = 3
    p2 = p0.copy()
    for r, c in [(0, 3), (0, 4), (1, 3), (1, 4)]:
        p2[r, c] = True

    preds = np.stack([p0, p1, p2])
    halted = np.array([True, True, True])
    res = M.path_audit_batch(tok, preds, halted=halted)
    assert list(res["arrays"]["valid_path"]) == [True, True, False]
    assert list(res["arrays"]["shortest_path"]) == [True, False, False]
    assert list(res["arrays"]["canonical_exact"]) == [True, False, False]
    assert res["summary"]["wrong_halts_shortest"] == 2


def test_runner_atomic_io_and_verifier_gate(tmp_path: Path) -> None:
    """Verify atomic JSON rejects NaN/Inf and `full-test` blocks without `--verifier-approved`."""
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import run_wsp_maze as R

    with pytest.raises(ValueError):
        R.atomic_write_json(tmp_path / "bad.json", {"x": float("nan")})
    with pytest.raises(ValueError):
        R.atomic_write_json(tmp_path / "bad_inf.json", {"x": float("inf")})

    ok_path = tmp_path / "good.json"
    digest = R.atomic_write_json(ok_path, {"ok": 1, "items": [2, 3]})
    assert ok_path.exists() and len(digest) == 64

    with pytest.raises(PermissionError):
        R.cmd_full_test(tmp_path / "art", tmp_path / "res", approved=False)

    sav = R.exact_saving_against_baseline([5.0, 6.0, 5.0, 6.0], [10.0, 12.0, 20.0, 24.0])
    assert abs(sav["median_saving"] - 0.625) < 1e-12
    assert sav["exact_median_saving"] == "5/8"

    # General Clopper-Pearson bounds (k=0, k=n, and interior k=4044/4096 matching H2be REPORT.md)
    lo0, hi0 = R.clopper_pearson_one_sided(0, 4096, 0.05)
    assert lo0 == 0.0 and abs(100.0 * hi0 - 0.0731113) < 1e-5
    lon, hin = R.clopper_pearson_one_sided(4096, 4096, 0.05)
    assert abs(100.0 * lon - 99.92688874435301) < 1e-9 and hin == 1.0
    lo_mid, hi_mid = R.clopper_pearson_one_sided(4044, 4096, 0.05)
    assert abs(100.0 * lo_mid - 98.403139) < 1e-5
    assert abs(100.0 * hi_mid - 99.004269) < 1e-5

    # Test deterministic ProcessPoolExecutor regeneration into isolated own_cache_dir
    fake_orig = [
        {"shard": 0, "path": "nonexistent_dir_do_not_create/shard00.npz", "sha256": "0" * 64, "n": 2},
        {"shard": 1, "path": "nonexistent_dir_do_not_create/shard01.npz", "sha256": "0" * 64, "n": 2},
    ]
    own_cache = tmp_path / "regen_shards"
    paths, info = R.ensure_180_cohort_shards("uniform", 888888, fake_orig, own_cache, max_workers=2)
    assert info["regenerated"] is True
    assert len(paths) == 2 and all(p.exists() for p in paths)
    assert not (ROOT / "nonexistent_dir_do_not_create").exists()
    with np.load(paths[0], allow_pickle=False) as z0:
        assert z0["tokens"].shape == (2, 180, 180)
        assert np.array_equal(z0["tokens"][0], R._gen_180_item(("uniform", 888888, 0))[0])

