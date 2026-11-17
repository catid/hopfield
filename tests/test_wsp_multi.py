"""Comprehensive verification suite for multi-solution Weight-Space Programming (`wsp_multi`)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (
    ROOT / "src",
    ROOT / "upstream",
    ROOT / "reproduction" / "baselines" / "code",
    ROOT / "reproduction" / "baselines" / "code" / "Weight-Space-Programming",
):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

from cram import graph_coloring_data as GC_DATA
from cram import nqueens_data as NQ_DATA
from flops.fullcount import FullCounter
from wsp import free_model as upstream_free_model

from hopfield_cram import wsp_multi as W


ALL_ARCH_SOURCES = [
    ("ARCH_NQUEENS_DET_SOURCE", W.ARCH_NQUEENS_DET_SOURCE),
    ("ARCH_NQUEENS_ARBITRARY_SOURCE", W.ARCH_NQUEENS_ARBITRARY_SOURCE),
    ("ARCH_PERM_SWAP_SOURCE", W.ARCH_PERM_SWAP_SOURCE),
    ("ARCH_ARB_SWAP_SOURCE", W.ARCH_ARB_SWAP_SOURCE),
    ("ARCH_GC_DET_SOURCE", W.ARCH_GC_DET_SOURCE),
    ("ARCH_GC_ASSOC_SOURCE", W.ARCH_GC_ASSOC_SOURCE),
]


def _independent_leaper_valid(boards: np.ndarray, givens: np.ndarray, n: int, leaper: tuple[int, int]) -> np.ndarray:
    """Independent NumPy rule checker for generalized Queen+Leaper completions."""
    R = boards.shape[0]
    la, lb = leaper
    out = np.zeros(R, dtype=bool)
    for idx in range(R):
        b = boards[idx]
        if b.sum() != n or not np.all(b >= givens[idx]):
            continue
        cells = np.flatnonzero(b)
        rows, cols = cells // n, cells % n
        ok = True
        for u in range(n):
            for v in range(u + 1, n):
                dr = abs(int(rows[u] - rows[v]))
                dc = abs(int(cols[u] - cols[v]))
                if dr == 0 or dc == 0 or dr == dc or (dr == la and dc == lb) or (dr == lb and dc == la):
                    ok = False
                    break
            if not ok:
                break
        out[idx] = ok
    return out


# ---------------------------------------------------------------------------
# 1. Upstream WSP `free_model.static_problems` AST Compliance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,source", ALL_ARCH_SOURCES)
def test_upstream_wsp_static_ast_contract(name: str, source: str) -> None:
    assert W.static_problems(source, signature="forward(w, x)") == [], f"{name} failed local AST check"
    assert upstream_free_model.static_problems(source, signature="forward(w, x)") == [], (
        f"{name} failed upstream wsp.free_model.static_problems"
    )
    mod = W.compile_wsp_arch(source, name.lower())
    assert hasattr(mod, "forward")


def test_static_ast_checker_rejects_forbidden_constructs() -> None:
    bad_if = "import torch\ndef forward(w, x):\n    if True:\n        return x\n    return x\n"
    bad_item = "import torch\ndef forward(w, x):\n    _ = x.sum().item()\n    return x\n"
    bad_dict = "import torch\ndef forward(w, x):\n    _ = dict(a=x)\n    return x\n"
    for src in (bad_if, bad_item, bad_dict):
        assert len(W.static_problems(src)) > 0
        assert len(upstream_free_model.static_problems(src)) > 0


def test_direct_arch_forward_calls_execute_cleanly() -> None:
    giv4 = torch.zeros((3, 16), dtype=torch.bool)
    giv4[0, 1] = True
    for builder, src in [
        (lambda: W.build_nqueens_det_wsp(4, passes=4), W.ARCH_NQUEENS_DET_SOURCE),
        (lambda: W.build_nqueens_arbitrary_wsp(4, cap=16, temperature=0.3), W.ARCH_NQUEENS_ARBITRARY_SOURCE),
        (lambda: W.build_nqueens_perm_wsp(4, cap=16), W.ARCH_PERM_SWAP_SOURCE),
        (lambda: W.build_queen_leaper_wsp(5, leaper=(2, 3), cap=16), W.ARCH_ARB_SWAP_SOURCE),
    ]:
        w = builder()
        n = int(round(w["W_attack"].shape[0] ** 0.5))
        x_in = torch.zeros((3, n * n), dtype=torch.bool)
        x_in[0, 0] = True
        out = W.compile_wsp_arch(src).forward(w, x_in)
        assert out.shape == (3, n * n) and out.dtype == torch.bool
        assert bool((out >= x_in).all())

    edges5 = torch.zeros((4, 10), dtype=torch.float32)
    edges5[:, :3] = 1.0
    for builder, src in [
        (lambda: W.build_graph_coloring_det_wsp(5, passes=6), W.ARCH_GC_DET_SOURCE),
        (lambda: W.build_graph_coloring_assoc_wsp(5, cap=16, temperature=0.8), W.ARCH_GC_ASSOC_SOURCE),
    ]:
        w = builder()
        out = W.compile_wsp_arch(src).forward(w, edges5)
        assert out.shape == (4, 5) and out.dtype == torch.int64
        assert bool(((out >= 0) & (out < 3)).all())


# ---------------------------------------------------------------------------
# 2. Parameter Disclosures & Attack Geometry Parity
# ---------------------------------------------------------------------------

def test_parameter_and_coefficient_disclosures() -> None:
    d_nq_det = W.weight_disclosure(
        W.build_nqueens_det_wsp(8), energy_keys=("W_attack", "bias"), prior_disclosure="rule-programmed-det"
    )
    assert d_nq_det["params_learned"] == 0
    assert d_nq_det["distinct_programmed_coefficients"] == 2
    assert d_nq_det["distinct_programmed_values"] == [-0.5, 1.0]
    assert d_nq_det["fit_display"] == "none"

    d_nq_arb = W.weight_disclosure(
        W.build_nqueens_arbitrary_wsp(10, cap=100),
        energy_keys=("W_attack", "bias_scalar"),
        prior_disclosure="rule-programmed-arbitrary",
    )
    assert d_nq_arb["params_learned"] == 0 and d_nq_arb["distinct_programmed_coefficients"] == 2

    d_nq_perm = W.weight_disclosure(
        W.build_nqueens_perm_wsp(10, cap=100),
        energy_keys=("W_attack",),
        prior_disclosure="rule-programmed-permutation",
    )
    assert d_nq_perm["params_learned"] == 0 and d_nq_perm["distinct_programmed_coefficients"] == 1

    d_gc = W.weight_disclosure(
        W.build_graph_coloring_assoc_wsp(8, cap=100),
        energy_keys=("color_eye",),
        prior_disclosure="rule-programmed-potts",
    )
    assert d_gc["params_learned"] == 0 and d_gc["distinct_programmed_coefficients"] == 1

    for leaper in [(2, 3), (2, 5)]:
        w_leap = W.build_queen_leaper_wsp(13, leaper=leaper, cap=64)
        d_leap = W.weight_disclosure(w_leap, energy_keys=("W_attack",), prior_disclosure="rule-programmed-leaper")
        assert d_leap["params_learned"] == 0 and d_leap["distinct_programmed_coefficients"] == 1
        mat = w_leap["W_attack"].numpy()
        assert np.array_equal(mat, mat.T) and not np.any(np.diag(mat))
        # Verify exact match to h3ad geometry
        idx = np.arange(169)
        r, c = idx // 13, idx % 13
        dr, dc = np.abs(r[:, None] - r[None, :]), np.abs(c[:, None] - c[None, :])
        expected = (
            (r[:, None] == r[None, :])
            | (c[:, None] == c[None, :])
            | (dr == dc)
            | ((dr == leaper[0]) & (dc == leaper[1]))
            | ((dr == leaper[1]) & (dc == leaper[0]))
        )
        np.fill_diagonal(expected, False)
        assert np.array_equal(mat > 0.5, expected)


# ---------------------------------------------------------------------------
# 3. `FullCounter` (FC2) Dispatch Audit & Exact Zero-Residual Affine Fit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kind,n,leaper",
    [
        ("nqueens_det", 8, None),
        ("nqueens_arbitrary", 8, None),
        ("nqueens_arbitrary", 10, None),
        ("nqueens_perm", 8, None),
        ("leaper_arbitrary", 13, (2, 3)),
        ("leaper_perm", 13, (2, 5)),
        ("gc_det", 8, None),
        ("gc_assoc", 8, None),
        ("gc_assoc", 10, None),
    ],
)
def test_fullcounter_zero_residual_affine_extrapolation(
    kind: str, n: int, leaper: tuple[int, int] | None
) -> None:
    cal = W.calibrate_affine_chain_flops(kind, n, leaper=leaper)
    assert cal["max_abs_residual_total"] == 0
    assert cal["max_abs_residual_total_with_compare"] == 0
    assert cal["per_step_total"] > 0
    assert cal["per_step_total_with_compare"] >= cal["per_step_total"]
    if kind in ("leaper_arbitrary", "leaper_perm"):
        cal_init = W.calibrate_affine_chain_flops(kind, n, leaper=leaper, use_given_initial=True)
        assert cal_init["max_abs_residual_total"] == 0
        assert cal_init["per_step_total"] == cal["per_step_total"]


def test_rng_seed_isolation_and_reproducibility() -> None:
    torch.manual_seed(12345)
    before = torch.rand(4)
    torch.manual_seed(12345)
    giv = np.zeros((2, 16), dtype=bool)
    w = W.build_nqueens_arbitrary_wsp(4, cap=64, temperature=0.3)
    r0a = W.solve_nqueens_arbitrary(w, giv, chains=6, seed=0, use_compile=False)
    after = torch.rand(4)
    assert torch.equal(before, after), "Solver mutated global torch RNG state"
    r0b = W.solve_nqueens_arbitrary(w, giv, chains=6, seed=0, use_compile=False)
    r1 = W.solve_nqueens_arbitrary(w, giv, chains=6, seed=1, use_compile=False)
    assert np.array_equal(r0a["boards"], r0b["boards"])
    assert not np.array_equal(r0a["moves"], r1["moves"]) or not np.array_equal(r0a["boards"], r1["boards"])


def test_eager_batched_solvers_run_under_fullcounter_without_unknown_ops() -> None:
    giv = np.zeros((2, 16), dtype=bool)
    giv[0, 1] = True
    with FullCounter() as fc_nq:
        res_nq = W.solve_nqueens_arbitrary(
            W.build_nqueens_arbitrary_wsp(4, cap=32, temperature=0.3),
            giv, chains=3, seed=7, device="cpu", chunk_steps=8, use_compile=False,
        )
    assert fc_nq.get_total_flops() > 0 and res_nq["boards"].shape == (2, 3, 16)

    edges = np.zeros((2, 10), dtype=np.float32)
    edges[:, [0, 4, 7]] = 1.0
    with FullCounter() as fc_gc:
        res_gc = W.solve_graph_coloring_assoc(
            W.build_graph_coloring_assoc_wsp(5, cap=32, temperature=0.8),
            edges, samples=3, seed=7, device="cpu", chunk_steps=8, use_compile=False,
        )
    assert fc_gc.get_total_flops() > 0 and res_gc["boards"].shape == (2, 3, 5)


# ---------------------------------------------------------------------------
# 4. Canonical Coloring Exhaustive Parity & Solver Correctness / Edge Cases
# ---------------------------------------------------------------------------

def test_exhaustive_tensor_color_canonicalizer_matches_gram() -> None:
    mod = W.compile_wsp_arch(W.ARCH_GC_ASSOC_SOURCE, "wsp_gc_assoc")
    all_6 = np.array(list(itertools.product(range(3), repeat=6)), dtype=np.int64)
    expected = GC_DATA.canonical(all_6)
    got = mod.canonicalize_colorings(torch.from_numpy(all_6), torch.arange(3, dtype=torch.int64)).numpy()
    assert np.array_equal(got, expected)
    assert np.all(GC_DATA.is_canonical(got))


def test_nqueens_solvers_zero_wrong_stops_and_mode_collapse_contrast() -> None:
    # 4x4 empty board has exactly 2 valid solutions: [1,3,0,2] and [2,0,3,1]
    giv = np.zeros((2, 16), dtype=bool)
    giv[1, 1] = True  # Clamps row 0 col 1 -> unique completion

    res_det = W.solve_nqueens_det(W.build_nqueens_det_wsp(4, passes=16), giv, chains=20)
    ok_det = NQ_DATA.completes(res_det["boards"], giv, 4)
    assert int((res_det["stopped"] & ~ok_det).sum()) == 0
    # Deterministic relaxation produces at most 1 distinct board across all 20 chains
    assert len({b.tobytes() for b in res_det["boards"][0]}) == 1

    res_arb = W.solve_nqueens_arbitrary(
        W.build_nqueens_arbitrary_wsp(4, cap=2000, temperature=0.3),
        giv, chains=20, seed=0, chunk_steps=32, use_compile=False,
    )
    ok_arb = NQ_DATA.completes(res_arb["boards"], giv, 4)
    assert int((res_arb["stopped"] & ~ok_arb).sum()) == 0
    assert np.all(ok_arb)
    assert len({b.tobytes() for b in res_arb["boards"][0, ok_arb[0]]}) == 2

    res_perm = W.solve_permutation_swap(
        W.build_nqueens_perm_wsp(4, cap=200, temp_start=1.0, end_ratio=0.05),
        giv, chains=20, seed=0, chunk_steps=32, use_compile=False,
    )
    ok_perm = NQ_DATA.completes(res_perm["boards"], giv, 4)
    assert int((res_perm["stopped"] & ~ok_perm).sum()) == 0
    assert np.all(ok_perm)
    assert len({b.tobytes() for b in res_perm["boards"][0, ok_perm[0]]}) == 2


def test_nqueens_boundary_and_unsatisfiable_edge_cases() -> None:
    # 1. All-clamped valid 4-queens solution: immediate stop at moves=0, zero clue corruption
    sol4 = np.zeros((1, 16), dtype=bool)
    for r, c in [(0, 1), (1, 3), (2, 0), (3, 2)]:
        sol4[0, r * 4 + c] = True
    for res in [
        W.solve_nqueens_arbitrary(W.build_nqueens_arbitrary_wsp(4, cap=32), sol4, chains=4, use_compile=False),
        W.solve_permutation_swap(W.build_nqueens_perm_wsp(4, cap=32), sol4, chains=4, use_compile=False),
    ]:
        assert np.all(res["stopped"]) and np.all(res["moves"] == 0)
        assert np.all(res["boards"] == sol4[:, None, :])

    # 2. (n-1)=3 clues leaving <2 free rows on an UNSATISFIABLE 4x4 board (3 queens on main diagonal!)
    unsat_perm = np.zeros((1, 16), dtype=bool)
    for r in (0, 1, 2):
        unsat_perm[0, r * 4 + r] = True
    res_up = W.solve_permutation_swap(W.build_nqueens_perm_wsp(4, cap=32), unsat_perm, chains=4, use_compile=False)
    assert not np.any(res_up["stopped"])
    assert np.all(res_up["boards"] >= unsat_perm[:, None, :])
    assert np.all(res_up["moves"] == 32)

    # 3. Unsatisfiable arbitrary clue (two given queens attacking in row 0): zero wrong stops, clues preserved
    unsat_arb = np.zeros((1, 16), dtype=bool)
    unsat_arb[0, [0, 1]] = True
    res_ua = W.solve_nqueens_arbitrary(W.build_nqueens_arbitrary_wsp(4, cap=32), unsat_arb, chains=4, use_compile=False)
    assert not np.any(res_ua["stopped"])
    assert np.all(res_ua["boards"] >= unsat_arb[:, None, :])
    with pytest.raises(ValueError, match="at most one given queen"):
        W.solve_permutation_swap(W.build_nqueens_perm_wsp(4, cap=16), unsat_arb, chains=2, use_compile=False)


def test_graph_coloring_edgeless_k3_and_uncolorable_k4() -> None:
    # n=4: graph 0 = edgeless, graph 1 = K_3 + isolated vertex, graph 2 = K_4 (not 3-colorable!)
    edges = np.zeros((3, 6), dtype=np.float32)
    # pairs for n=4: (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
    edges[1, [0, 1, 3]] = 1.0  # K_3 on {0,1,2}
    edges[2, :] = 1.0          # K_4 on {0,1,2,3}
    A = GC_DATA.adjacency(edges, 4)

    res = W.solve_graph_coloring_assoc(
        W.build_graph_coloring_assoc_wsp(4, cap=128, temperature=0.8, canonicalize_in_arch=True),
        edges, samples=20, seed=3, chunk_steps=32, use_compile=False,
    )
    conf = GC_DATA.conflicts(A, res["boards"])
    proper = conf == 0
    assert int((res["stopped"] & ~proper).sum()) == 0
    # Edgeless graph stops at step 0 on all 20 chains and outputs canonical colorings
    assert np.all(res["stopped"][0]) and np.all(res["moves"][0] == 0)
    assert np.all(proper[0]) and np.all(proper[1])
    assert not np.any(res["stopped"][2]) and not np.any(proper[2])
    assert np.all(GC_DATA.is_canonical(res["boards"]))

    # Deterministic GC collapses to 1 output across all 20 chains
    res_det = W.solve_graph_coloring_det(W.build_graph_coloring_det_wsp(4, passes=12), edges[:2], samples=20)
    conf_det = GC_DATA.conflicts(A[:2], res_det["boards"])
    assert np.all(conf_det == 0)
    assert len({tuple(r) for r in res_det["boards"][1]}) == 1


def test_generalized_queen_leaper_solves_and_preserves_invariants() -> None:
    giv = np.zeros((2, 169), dtype=bool)
    giv[0, 0] = True
    giv[1, 1 * 13 + 4] = True
    w = W.build_queen_leaper_wsp(13, leaper=(2, 3), cap=1500, temp_start=1.0, end_ratio=0.01)
    res_perm = W.solve_permutation_swap(w, giv, chains=8, seed=11, chunk_steps=64, use_compile=False)
    boards = res_perm["boards"]
    assert np.all(boards.reshape(-1, 13, 13).sum(axis=1) == 1)
    assert np.all(boards.reshape(-1, 13, 13).sum(axis=2) == 1)
    valid = _independent_leaper_valid(boards.reshape(-1, 169), np.repeat(giv, 8, axis=0), 13, (2, 3)).reshape(2, 8)
    assert int((res_perm["stopped"] & ~valid).sum()) == 0
    assert np.all(valid.any(axis=1))


# ---------------------------------------------------------------------------
# 5. GPU `torch.compile` vs Eager Bitwise State-for-State Parity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compiled vs eager GPU parity check")
def test_gpu_compiled_vs_eager_bitwise_parity() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    giv_nq = np.zeros((8, 64), dtype=bool)
    for r in range(8):
        giv_nq[r, r] = True
    w_nq = W.build_nqueens_arbitrary_wsp(8, cap=80, temperature=0.35)
    r_eager = W.solve_nqueens_arbitrary(w_nq, giv_nq, chains=16, seed=5, device="cuda", chunk_steps=32, use_compile=False)
    r_comp = W.solve_nqueens_arbitrary(w_nq, giv_nq, chains=16, seed=5, device="cuda", chunk_steps=32, use_compile=True)
    for key in ("boards", "stopped", "moves"):
        assert np.array_equal(r_eager[key], r_comp[key]), f"NQ arbitrary mismatch on {key}"
    assert np.allclose(r_eager["energy"], r_comp["energy"], atol=1e-6, rtol=0.0)

    giv_leap = np.zeros((8, 169), dtype=bool)
    for r in range(8):
        giv_leap[r, r * 13 + ((2 * r) % 13)] = True
    w_leap = W.build_queen_leaper_wsp(13, leaper=(2, 3), cap=80)
    for solver in (W.solve_permutation_swap, W.solve_arbitrary_swap):
        e_out = solver(w_leap, giv_leap, chains=16, seed=9, device="cuda", chunk_steps=32, use_compile=False)
        c_out = solver(w_leap, giv_leap, chains=16, seed=9, device="cuda", chunk_steps=32, use_compile=True)
        for key in ("boards", "stopped", "moves", "conflicts", "accepted"):
            assert np.array_equal(e_out[key], c_out[key]), f"{solver.__name__} mismatch on {key}"

    edges_gc = np.ones((8, 28), dtype=np.float32)  # K_8 keeps all 128 chains active across compiled 32-chunks + eager 16-tail
    w_gc = W.build_graph_coloring_assoc_wsp(8, cap=80, temperature=0.8)
    e_gc = W.solve_graph_coloring_assoc(w_gc, edges_gc, samples=16, seed=4, device="cuda", chunk_steps=32, use_compile=False)
    c_gc = W.solve_graph_coloring_assoc(w_gc, edges_gc, samples=16, seed=4, device="cuda", chunk_steps=32, use_compile=True)
    for key in ("boards", "boards_raw_uncanonicalized", "stopped", "moves"):
        assert np.array_equal(e_gc[key], c_gc[key]), f"GC assoc mismatch on {key}"


def test_runner_atomic_nan_guard_and_test_gate(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_wsp_multi as runner

    with pytest.raises(RuntimeError, match="confirm-verifier-signoff"):
        runner.run_test_stage("cpu", confirm_signoff=False)

    good_path = tmp_path / "good.json"
    runner.save_json_atomic(good_path, {"ok": 1, "val": 2.5})
    assert good_path.exists()

    bad_path = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        runner.save_json_atomic(bad_path, {"nan_val": float("nan")})
    assert not bad_path.exists()

    giv4 = np.zeros((3, 16), dtype=bool)
    giv4[1, 1] = True
    giv4[2, 1] = True  # duplicate of row 1
    inv, S = runner.compute_nqueens_puzzle_counts(giv4, 4)
    assert len(S) == 2
    assert S[inv[0]] == 2 and S[inv[1]] == 1 and inv[1] == inv[2]
