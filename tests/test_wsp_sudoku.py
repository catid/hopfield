"""Comprehensive unit, parity, FullCounter (FC2), AST-contract, and adversarial tests
for Weight-Space Programming (WSP) Sudoku models (`wsp_contract.py` & `wsp_sudoku.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "reproduction" / "baselines" / "code" / "flops") not in sys.path:
    sys.path.insert(0, str(ROOT / "reproduction" / "baselines" / "code" / "flops"))
if str(ROOT / "reproduction" / "baselines" / "code" / "Weight-Space-Programming") not in sys.path:
    sys.path.insert(0, str(ROOT / "reproduction" / "baselines" / "code" / "Weight-Space-Programming"))

from fullcount import FullCounter  # noqa: E402
from wsp import free_model as upstream_free_model  # noqa: E402

from hopfield_cram.wsp_contract import (  # noqa: E402
    atomic_write_json,
    check_wsp_static_tensor_program,
    static_problems,
    verify_fullcounter_affine,
    wsp_weight_metadata,
)
from hopfield_cram.wsp_sudoku import (  # noqa: E402
    N_CELLS,
    N_DIGITS,
    N_PEERS,
    WSP_ASSOC_ARCH_SOURCE,
    WSP_ASSOC_PROP_STEP_SOURCE,
    WSP_ASSOC_STAGE1_SOURCE,
    WSP_DET_ARCH_SOURCE,
    WSP_TRANSFORMER_ARCH_SOURCE,
    WSPAssocSearch,
    _extract_triton_move_rng,
    _wsp_domain_moves,
    build_sudoku_incidence_tensors,
    build_wsp_assoc_weights,
    build_wsp_det_weights,
    build_wsp_transformer_weights,
    compute_wsp_assoc_puzzle_flops,
    count_wsp_sudoku_flops,
    encode_sudoku_clues,
    evaluate_deterministic_wsp,
    load_sudoku_split,
    make_synthetic_assoc_chain_state,
    verify_sudoku_boards,
    wsp_assoc_move_pytorch,
    wsp_assoc_prop_step_forward,
    wsp_assoc_stage1_forward,
    wsp_assoc_static_forward,
    wsp_det_forward,
    wsp_transformer_forward,
)


def test_wsp_ast_checker_matches_upstream_exact():
    """Verify our `static_problems` matches upstream `wsp/free_model.py:static_problems` exactly."""
    upstream_tf_src = (
        ROOT / "reproduction" / "baselines" / "code" / "Weight-Space-Programming" / "wsp" / "given_archs" / "transformer.py"
    ).read_text(encoding="utf-8")
    valid_sources = [
        WSP_TRANSFORMER_ARCH_SOURCE,
        WSP_DET_ARCH_SOURCE,
        WSP_ASSOC_PROP_STEP_SOURCE,
        WSP_ASSOC_STAGE1_SOURCE,
        WSP_ASSOC_ARCH_SOURCE,
        upstream_tf_src,
    ]
    for src in valid_sources:
        ours = check_wsp_static_tensor_program(src, signature="forward(w, x)")
        theirs = upstream_free_model.static_problems(src, signature="forward(w, x)")
        assert ours == [], f"Unexpected AST violation: {ours}"
        assert theirs == [], f"Upstream AST violation: {theirs}"

    invalid_cases = [
        "import torch\ndef forward(w, x):\n    if x.shape[0] > 1:\n        return x\n    return x\n",
        "import torch\ndef forward(w, x):\n    return x if True else x\n",
        "import torch\ndef forward(w, x):\n    while False:\n        pass\n    return x\n",
        "import torch\ndef forward(w, x):\n    return (x > 0) and (x < 2)\n",
        "import torch\ndef forward(w, x):\n    return x.item()\n",
        "import torch\ndef forward(w, x):\n    return x.numpy()\n",
        "import torch\ndef forward(w, x):\n    return float(x.shape[0])\n",
        "import os, torch\ndef forward(w, x):\n    return x\n",
        "import torch\ndef forward(w, x):\n    return [i for i in range(4) if i > 1]\n",
        "import torch\ndef helper_only(w, x):\n    return x\n",
    ]
    for bad in invalid_cases:
        ours = static_problems(bad, signature="forward(w, x)")
        theirs = upstream_free_model.static_problems(bad, signature="forward(w, x)")
        assert len(ours) > 0
        assert ours == theirs


def test_weight_metadata_and_chromatic_sudoku_geometry():
    """Verify zero-training metadata and exact 9-coloring of the 81-cell Sudoku peer graph."""
    inc = build_sudoku_incidence_tensors("cpu")
    peer = inc["peer_adj"]
    assert peer.shape == (81, 81)
    assert int(peer.count_nonzero().item()) == 81 * 20  # 1620 directed edges = 810 undirected pairs
    assert torch.equal(peer, peer.T)
    assert int(torch.diag(peer).count_nonzero().item()) == 0

    colors = inc["color_cells"]
    assert colors.shape == (9, 9)
    assert sorted(colors.reshape(-1).tolist()) == list(range(81))
    for k in range(9):
        c_idx = colors[k]
        assert int(peer[c_idx][:, c_idx].count_nonzero().item()) == 0, f"Color {k} has adjacent peers!"

    meta_tf = wsp_weight_metadata(build_wsp_transformer_weights(steps=16), index_keys={"steps"})
    meta_det = wsp_weight_metadata(build_wsp_det_weights(16, 32), index_keys={"prop_steps", "relax_steps"})
    meta_assoc = wsp_weight_metadata(
        build_wsp_assoc_weights(),
        index_keys={"peer_nbr", "color_cells", "prop_steps", "anneal_period"},
    )
    for meta in (meta_tf, meta_det, meta_assoc):
        assert meta["params_learned"] == 0
        assert meta["training_examples"] == 0
        assert meta["fit_time_display"] == "none"
        assert meta["materialized_entries"] > 0
        assert 0 < meta["nnz"] <= meta["materialized_entries"]
    assert meta_det["distinct_programmed_coefficients"] == 3  # {1.0, 0.5, 4.0}
    # Verify int8 tensors automatically receive routing_or_mask role even when index_keys is omitted
    meta_no_keys = wsp_weight_metadata(build_wsp_det_weights(16, 32))
    assert meta_no_keys["per_tensor"]["prop_steps"]["role"] == "routing_or_mask"
    assert meta_no_keys["distinct_programmed_coefficients"] == 3


def test_stage1_fixed_vs_adaptive_exact_equality_at_partial_prop_steps():
    """Verify `WSP_ASSOC_STAGE1_SOURCE` and `WSP_ASSOC_PROP_STEP_SOURCE` match on non-converged boards at prop_steps=2."""
    sol, free, ratings, _ = load_sudoku_split("dev10k", 0, 200)
    x_tok = encode_sudoku_clues(sol, free)
    w2 = build_wsp_assoc_weights(prop_steps=2, device="cpu")
    with torch.no_grad():
        m_fixed2 = wsp_assoc_stage1_forward(w2, x_tok)
        is_clue = (x_tok > 0).unsqueeze(-1)
        clue_oh = F.one_hot((x_tok - 1).clamp(min=0), N_DIGITS).float() * is_clue.float()
        mb = torch.where(is_clue, clue_oh, torch.ones_like(clue_oh))
        act = torch.arange(len(x_tok), dtype=torch.int64)
        for _ in range(2):
            m_prev = mb[act]
            m_next = wsp_assoc_prop_step_forward(w2, x_tok[act], m_prev)
            mb[act] = m_next
            unchanged = (m_next == m_prev).all(dim=(1, 2))
            all_single = (m_next.sum(dim=-1) == 1.0).all(dim=1)
            act = act[(~unchanged) & (~all_single)]
        assert len(act) > 0, "Expected non-converged boards at prop_steps=2"
        assert torch.equal(m_fixed2, mb)


def test_independent_sudoku_board_verifier():
    """Verify `verify_sudoku_boards` accepts valid solutions and rejects every violation class."""
    sol, free, _, _ = load_sudoku_split("dev10k", 0, 16)
    x_tok = encode_sudoku_clues(sol, free)
    ok = verify_sudoku_boards(sol, x_tok)
    assert ok["n_valid"] == 16
    assert ok["n_clue_violations"] == 0

    # 1. Mutate a free cell to break row/col/box uniqueness
    bad_dup = sol.copy()
    free_idx = int(np.flatnonzero(free[0])[0])
    bad_dup[0, free_idx] = (bad_dup[0, free_idx] + 1) % 9
    r_dup = verify_sudoku_boards(bad_dup, x_tok)
    assert not r_dup["valid_mask"][0]
    assert r_dup["valid_mask"][1:].all()

    # 2. Valid Sudoku grid via global digit relabeling, but violates original clues!
    perm_board = (sol[:1] + 1) % 9
    r_clue = verify_sudoku_boards(perm_board, x_tok[:1])
    assert r_clue["n_valid"] == 0
    assert r_clue["n_clue_violations"] == 1

    # 3. Valid row and column permutations (shifted Latin square) that violate 3x3 boxes!
    latin = np.array([[(r + c) % 9 for c in range(9)] for r in range(9)], dtype=np.int64).reshape(1, 81)
    r_latin = verify_sudoku_boards(latin, torch.zeros((1, 81), dtype=torch.int64))
    assert r_latin["n_valid"] == 0

    # 4. Out-of-range digit
    bad_rng = sol[:2].copy()
    bad_rng[0, free_idx] = 9
    bad_rng[1, int(np.flatnonzero(free[1])[0])] = -1
    r_rng = verify_sudoku_boards(bad_rng, x_tok[:2])
    assert r_rng["n_valid"] == 0


def test_zero_free_cell_label_leakage():
    """Verify corrupting ground-truth digits at free cells changes nothing in any WSP model."""
    sol, free, _, _ = load_sudoku_split("dev10k", 0, 7)
    corrupted_sol = sol.copy()
    corrupted_sol[free] = (corrupted_sol[free] * 3 + 7) % 9

    x_clean = encode_sudoku_clues(sol, free)
    x_corrupt = encode_sudoku_clues(corrupted_sol, free)
    assert torch.equal(x_clean, x_corrupt)

    w_tf = build_wsp_transformer_weights(steps=2, device="cpu")
    w_det = build_wsp_det_weights(4, 4, device="cpu")
    assert torch.equal(wsp_transformer_forward(w_tf, x_clean), wsp_transformer_forward(w_tf, x_corrupt))
    assert torch.equal(wsp_det_forward(w_det, x_clean), wsp_det_forward(w_det, x_corrupt))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w_assoc = build_wsp_assoc_weights(device=dev)
    res_a = WSPAssocSearch(w_assoc, x_clean, C=4, check=64, max_moves=256, seed=3, device=dev).run()
    res_b = WSPAssocSearch(w_assoc, x_corrupt, C=4, check=64, max_moves=256, seed=3, device=dev).run()
    assert torch.equal(res_a["pred"], res_b["pred"])
    assert torch.equal(res_a["stopped"], res_b["stopped"])
    assert torch.equal(res_a["steps_used"], res_b["steps_used"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton parity test requires CUDA")
def test_three_way_exact_parity_triton_vs_pytorch_vs_static_ast():
    """Verify exact state-for-state equivalence across:
      (1) Triton GPU kernel `_wsp_domain_moves`,
      (2) PyTorch `FullCounter` transcription `wsp_assoc_move_pytorch`, and
      (3) Pure static-AST tensor program `wsp_assoc_static_forward`.
    Tested on prime batch size B=7 puzzles * C=4 chains = 28 chains over 48 moves.
    """
    sol, free, ratings, _ = load_sudoku_split("dev10k", 0, 50)
    hard_idx = np.flatnonzero(ratings >= 20)[:7]
    assert len(hard_idx) == 7
    x_tok = encode_sudoku_clues(sol[hard_idx], free[hard_idx]).cuda()

    B, C, nmoves = 7, 4, 48
    H = B * C
    w = build_wsp_assoc_weights(prop_steps=16, temp_hi=0.52, temp_lo=0.52, device="cuda")

    gen = torch.Generator(device="cuda").manual_seed(42)
    u_init = torch.rand((B, C, N_CELLS, N_DIGITS), device="cuda", generator=gen)

    seed_t = torch.tensor([1234567], dtype=torch.int64, device="cuda")
    u0_buf = torch.empty((H, nmoves), dtype=torch.float32, device="cuda")
    uf_buf = torch.empty((H, nmoves, N_CELLS), dtype=torch.float32, device="cuda")
    uu_buf = torch.empty((H, nmoves, N_DIGITS), dtype=torch.float32, device="cuda")
    _extract_triton_move_rng[(H,)](
        seed_t, u0_buf, uf_buf, uu_buf, nmoves, N=N_CELLS, V=N_DIGITS, NB=128, VB=16, num_warps=1
    )

    # Path (3): Static-AST tensor program `wsp_assoc_static_forward`
    with torch.no_grad():
        ycert_ast, found_ast, P_ast = wsp_assoc_static_forward(w, x_tok, u_init, u0_buf, uf_buf, uu_buf)

    # Shared initial state for Path (1) Triton and Path (2) PyTorch transcription
    with torch.no_grad():
        m_puz = wsp_assoc_stage1_forward(w, x_tok)
        m = m_puz.unsqueeze(1).expand(B, C, N_CELLS, N_DIGITS).reshape(H, N_CELLS, N_DIGITS)
        fr = m.sum(dim=-1) > 1.5
        fmask = fr.to(torch.int32).contiguous()
        nfree = fmask.sum(dim=-1).to(torch.int32).contiguous()
        freelist = torch.argsort((~fr).to(torch.int8), dim=1, stable=True).to(torch.int32).contiguous()
        gum0 = -torch.log(-torch.log(torch.maximum(u_init.reshape(H, N_CELLS, N_DIGITS), w["eps"])))
        y0 = torch.where(m > 0.5, gum0, w["neg_inf"]).argmax(dim=-1).to(torch.int32).contiguous()
        Y0_oh = F.one_hot(y0.long(), N_DIGITS).float()
        G0 = torch.matmul(w["peer_adj"], Y0_oh).contiguous()
        P0 = (0.5 * (Y0_oh * G0).sum(dim=(1, 2))).contiguous()
        own0 = G0.gather(2, y0.long().unsqueeze(-1)).squeeze(-1)
        flag0 = ((own0 > w["theta"]) & fr).to(torch.int32).contiguous()
        found0 = torch.where(P0 <= w["tau_tol"], torch.zeros_like(nfree), torch.full_like(nfree, -1)).contiguous()
        cand_i32 = (m > 0.5).to(torch.int32).contiguous()

        # Path (1): Fused Triton kernel
        y_tri, G_tri, flag_tri, P_tri = y0.clone(), G0.clone(), flag0.clone(), P0.clone()
        found_tri, ycert_tri = found0.clone(), y0.clone()
        temp_t = torch.full((H,), float(w["temp"].cpu()), dtype=torch.float32, device="cuda")
        _wsp_domain_moves[(H,)](
            w["peer_nbr"],
            cand_i32,
            y_tri,
            G_tri,
            flag_tri,
            P_tri,
            freelist,
            fmask,
            nfree,
            temp_t,
            found_tri,
            ycert_tri,
            seed_t,
            nmoves,
            0,
            float(w["theta"].cpu()),
            float(w["tau_tol"].cpu()),
            N=N_CELLS,
            V=N_DIGITS,
            D=N_PEERS,
            NB=128,
            VB=16,
            DB=32,
            num_warps=1,
        )

        # Path (2): Per-chain PyTorch transcription `wsp_assoc_move_pytorch`
        y_py, G_py, flag_py, P_py = y0.clone(), G0.clone(), flag0.clone(), P0.clone()
        found_py, ycert_py = found0.clone(), y0.clone()
        for h_idx in range(H):
            st = {
                "G": G_py[h_idx],
                "y": y_py[h_idx],
                "ycert": ycert_py[h_idx],
                "flag": flag_py[h_idx],
                "fmask": fmask[h_idx],
                "free": freelist[h_idx],
                "cand": cand_i32[h_idx],
                "nbr": w["peer_nbr"],
                "P": P_py[h_idx],
                "T": w["temp"],
                "nf": nfree[h_idx],
                "found": found_py[h_idx],
                "thr": w["theta"],
                "tau": w["tau_tol"],
                "move0": torch.tensor(0, dtype=torch.int32, device="cuda"),
                "k": torch.tensor(0, dtype=torch.int32, device="cuda"),
            }
            for k in range(nmoves):
                st["k"] = torch.tensor(k, dtype=torch.int32, device="cuda")
                wsp_assoc_move_pytorch(
                    st,
                    u0_in=u0_buf[h_idx, k],
                    uf_in=uf_buf[h_idx, k],
                    uu_in=uu_buf[h_idx, k],
                )
            P_py[h_idx] = st["P"]
            found_py[h_idx] = st["found"]

    # Exact integer & float equality checks across all 3 paths!
    assert torch.equal(y_tri, y_py)
    assert torch.equal(flag_tri, flag_py)
    assert torch.equal(found_tri, found_py)
    assert torch.equal(ycert_tri, ycert_py)
    assert float((G_tri - G_py).abs().max().item()) == 0.0
    assert float((P_tri - P_py).abs().max().item()) == 0.0

    assert torch.equal(ycert_tri, ycert_ast)
    assert torch.equal(found_tri, found_ast)
    assert float((P_tri - P_ast).abs().max().item()) == 0.0


def test_fullcounter_fc2_and_zero_affine_residuals():
    """Verify `FullCounter` (`FC2`) runs cleanly on all 3 WSP models with `max_abs_residual == 0`."""
    counts = count_wsp_sudoku_flops(FullCounter)
    assert counts["WSP-Transformer"]["per_step_total"] > 0
    assert counts["WSP-Det"]["per_prop_step_total"] > 0
    assert counts["WSP-Det"]["per_relax_step_total"] > 0
    assert counts["WSP-Assoc"]["move_per_chain"]["total"] > 0
    assert counts["WSP-Assoc"]["move_per_chain"]["total_with_compare"] > counts["WSP-Assoc"]["move_per_chain"]["total"]
    assert counts["WSP-Assoc"]["check_per_puzzle"]["total"] == 2  # matches cram-hand-sudoku.json check_per_puzzle == 2

    x1 = torch.zeros((1, N_CELLS), dtype=torch.int64)
    x1[0, 0] = 1
    step_grid = (0, 1, 2, 3, 5, 8, 13, 21)

    # 1. WSP-Transformer affine check
    def tf_at(s: int):
        w = build_wsp_transformer_weights(steps=s, device="cpu")
        with torch.no_grad(), FullCounter() as fc:
            _ = wsp_transformer_forward(w, x1)
        return fc.summary()

    aff_tf = verify_fullcounter_affine(tf_at, step_grid)
    assert aff_tf["max_abs_residual_total"] == 0
    assert aff_tf["max_abs_residual_total_with_compare"] == 0

    # 2. WSP-Det affine check (both prop_steps and relax_steps)
    def det_prop_at(s: int):
        w = build_wsp_det_weights(prop_steps=s, relax_steps=2, device="cpu")
        with torch.no_grad(), FullCounter() as fc:
            _ = wsp_det_forward(w, x1)
        return fc.summary()

    def det_rel_at(s: int):
        w = build_wsp_det_weights(prop_steps=2, relax_steps=s, device="cpu")
        with torch.no_grad(), FullCounter() as fc:
            _ = wsp_det_forward(w, x1)
        return fc.summary()

    aff_dp = verify_fullcounter_affine(det_prop_at, step_grid)
    aff_dr = verify_fullcounter_affine(det_rel_at, step_grid)
    assert aff_dp["max_abs_residual_total"] == 0 and aff_dp["max_abs_residual_total_with_compare"] == 0
    assert aff_dr["max_abs_residual_total"] == 0 and aff_dr["max_abs_residual_total_with_compare"] == 0

    # 3. WSP-Assoc move affine check
    w_assoc = build_wsp_assoc_weights(device="cpu")

    def assoc_moves_at(s: int):
        st = make_synthetic_assoc_chain_state(w_assoc, seed=0, device="cpu")
        st["tau"] = torch.tensor(-1e9)  # ensure constant trajectory
        with torch.no_grad(), FullCounter() as fc:
            for k in range(s):
                st["k"] = torch.tensor(k, dtype=torch.int32)
                wsp_assoc_move_pytorch(st)
        return fc.summary()

    aff_am = verify_fullcounter_affine(assoc_moves_at, step_grid)
    assert aff_am["max_abs_residual_total"] == 0
    assert aff_am["max_abs_residual_total_with_compare"] == 0

    def assoc_s1_at(s: int):
        with torch.no_grad(), FullCounter() as fc:
            is_clue = (x1 > 0).unsqueeze(-1)
            clue_oh = F.one_hot((x1 - 1).clamp(min=0), N_DIGITS).float() * is_clue.float()
            mb = torch.where(is_clue, clue_oh, torch.ones_like(clue_oh))
            for _ in range(s):
                m_prev = mb
                mb = wsp_assoc_prop_step_forward(w_assoc, x1, m_prev)
                _ = (mb == m_prev).all(dim=(1, 2))
                _ = (mb.sum(dim=-1) == 1.0).all(dim=1)
        return fc.summary()

    aff_s1 = verify_fullcounter_affine(assoc_s1_at, step_grid)
    assert aff_s1["max_abs_residual_total"] == 0
    assert aff_s1["max_abs_residual_total_with_compare"] == 0


def test_edge_boards_and_impossible_contradictions():
    """Test 0-free-cell, 1-free-cell, prime batch sizes, and impossible contradictory boards."""
    sol, free, _, _ = load_sudoku_split("dev10k", 0, 13)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w_assoc = build_wsp_assoc_weights(device=dev)
    w_det = build_wsp_det_weights(16, 16, device=dev)
    w_tf = build_wsp_transformer_weights(steps=8, device=dev)
    peer_adj = w_det["peer_adj"]

    # 1. Already-solved board (0 free cells) & 1-free-cell board
    no_free = np.zeros((1, 81), dtype=bool)
    x_solved = encode_sudoku_clues(sol[:1], no_free)
    res0 = WSPAssocSearch(w_assoc, x_solved, C=4, check=64, max_moves=256, seed=0, device=dev).run()
    assert bool(res0["stopped"][0]) and int(res0["steps_used"][0]) == 0
    assert verify_sudoku_boards(res0["pred"], x_solved)["n_valid"] == 1

    one_free = np.zeros((1, 81), dtype=bool)
    one_free[0, 40] = True
    x_one = encode_sudoku_clues(sol[1:2], one_free)
    res1 = WSPAssocSearch(w_assoc, x_one, C=4, check=64, max_moves=256, seed=0, device=dev).run()
    assert bool(res1["stopped"][0]) and int(res1["steps_used"][0]) == 0
    assert verify_sudoku_boards(res1["pred"], x_one)["n_valid"] == 1

    # 2. Adversarial impossible boards MUST NEVER trigger a wrong stop!
    # Board A: Two identical clues in row 0 (cells 0 and 1 both given digit 1)
    imp_a = torch.zeros((1, 81), dtype=torch.int64)
    imp_a[0, 0] = 1
    imp_a[0, 1] = 1
    # Board B: Cell 0 blank, but digits 1..9 all appear as clues among row 0 + col 0 peers!
    imp_b = torch.zeros((1, 81), dtype=torch.int64)
    for d in range(1, 9):
        imp_b[0, d] = d       # row 0 cols 1..8 hold 1..8
    imp_b[0, 9] = 9           # col 0 row 1 holds 9 -> cell 0 has 0 valid candidates!
    x_imp = torch.cat([imp_a, imp_b], dim=0)

    res_imp_assoc = WSPAssocSearch(w_assoc, x_imp, C=4, check=64, max_moves=512, seed=0, device=dev).run()
    assert int(res_imp_assoc["stopped"].sum().item()) == 0, "Impossible board falsely halted in WSP-Assoc!"

    res_imp_det = evaluate_deterministic_wsp(wsp_det_forward, w_det, x_imp, peer_adj)
    assert int(res_imp_det["stopped"].sum().item()) == 0, "Impossible board falsely halted in WSP-Det!"

    res_imp_tf = evaluate_deterministic_wsp(wsp_transformer_forward, w_tf, x_imp, peer_adj)
    assert int(res_imp_tf["stopped"].sum().item()) == 0, "Impossible board falsely halted in WSP-Transformer!"

    # 3. Prime batch size B=13 smoke check with NaN/Inf guards
    x_13 = encode_sudoku_clues(sol, free)
    out_det = wsp_det_forward(w_det, x_13.to(dev))
    out_tf = wsp_transformer_forward(w_tf, x_13.to(dev))
    assert torch.isfinite(out_det).all() and torch.isfinite(out_tf).all()


def test_atomic_write_json_rejects_nan_and_writes_cleanly(tmp_path: Path):
    target = tmp_path / "out.json"
    digest = atomic_write_json({"ok": True, "val": 123}, target)
    assert target.exists() and len(digest) == 64
    with pytest.raises(ValueError):
        atomic_write_json({"bad": float("nan")}, tmp_path / "nan.json")


def test_runner_blocks_test_splits_without_verifier_approval():
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import run_wsp_sudoku

    with pytest.raises(RuntimeError, match="TEST GATE BLOCKED"):
        run_wsp_sudoku.enforce_dev_and_verifier_gate(verifier_approved=False)

    # Also test Figure 3a 8-bin difficulty slicing helper on synthetic (3, 412786) -> (3, 10000) slice
    rat = np.array([0, 1, 2, 5, 10, 20, 40, 80, 0, 100], dtype=np.int64)
    solved_full = np.ones((3, 25), dtype=bool)
    fl_full = np.full((3, 25), 1e7, dtype=np.float64)
    det_full = np.ones(25, dtype=bool)
    tf_full = np.zeros(25, dtype=bool)
    b_cnts, a_bins, d_bins, t_bins = run_wsp_sudoku.build_fig3a_bins(
        rat, solved_full[:, :10], fl_full[:, :10], det_full[:10], 1.5e7, tf_full[:10], 1.7e9
    )
    assert sum(b_cnts) == 10 and len(a_bins) == 8 and len(d_bins) == 8 and len(t_bins) == 8


def test_dev10k_slice_end_to_end_zero_wrong_stops():
    """End-to-end smoke check on 128 dev10k puzzles across all 3 WSP models."""
    sol, free, _, _ = load_sudoku_split("dev10k", 0, 128)
    x_tok = encode_sudoku_clues(sol, free)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    w_det = build_wsp_det_weights(16, 16, device=dev)
    r_det = evaluate_deterministic_wsp(wsp_det_forward, w_det, x_tok, w_det["peer_adj"])
    if bool(r_det["stopped"].any()):
        v_det = verify_sudoku_boards(r_det["pred"][r_det["stopped"]], x_tok[r_det["stopped"]])
        assert v_det["n_valid"] == int(r_det["stopped"].sum().item())

    w_tf = build_wsp_transformer_weights(steps=16, device=dev)
    r_tf = evaluate_deterministic_wsp(wsp_transformer_forward, w_tf, x_tok, w_det["peer_adj"])
    if bool(r_tf["stopped"].any()):
        v_tf = verify_sudoku_boards(r_tf["pred"][r_tf["stopped"]], x_tok[r_tf["stopped"]])
        assert v_tf["n_valid"] == int(r_tf["stopped"].sum().item())

    max_m = 65536 if dev == "cuda" else 256
    w_assoc = build_wsp_assoc_weights(device=dev)
    r_assoc = WSPAssocSearch(w_assoc, x_tok, C=4, check=min(4096, max_m), max_moves=max_m, seed=0, device=dev).run()
    stopped_mask = r_assoc["stopped"].numpy()
    exact_mask = (r_assoc["pred"].numpy() == sol).all(axis=1)
    assert int((stopped_mask & ~exact_mask).sum()) == 0, "Wrong stop detected!"
    if stopped_mask.any():
        v_assoc = verify_sudoku_boards(r_assoc["pred"][r_assoc["stopped"]], x_tok[r_assoc["stopped"]])
        assert v_assoc["n_valid"] == int(stopped_mask.sum())

    counts = count_wsp_sudoku_flops(FullCounter)
    flops = compute_wsp_assoc_puzzle_flops(r_assoc["steps_used"], r_assoc["stage2_entered"], counts["WSP-Assoc"])
    assert np.isfinite(flops).all() and (flops > 0).all()
