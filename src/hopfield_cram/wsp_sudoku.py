"""Weight-Space Programming (WSP) models and fast verified GPU kernels for Sudoku-Extreme.

Implements three genuine zero-training Weight-Space Programs (`params_learned = 0`,
`training_examples = 0`, `fit_time_display = "none"`), each comprising:
  1. `build_*_weights(...) -> dict[str, Tensor]`: constructs weights purely from the
     mathematical rules of 9x9 Sudoku (row, column, and 3x3 box geometry), never reading
     any training puzzle or solution;
  2. `*_ARCH_SOURCE`: a fixed, branchless PyTorch tensor program (`forward(w, ...)`) that
     passes EFFL Lab's `wsp/free_model.py:static_problems` AST checker with zero violations.

Models:
  - `WSP-Transformer`: 2-layer looped standard Transformer (`D=192, HEADS=4, LAYERS=2, MLP=384`,
    `LayerNorm + QKV math attention + GELU FFN`, matching `wsp/given_archs/transformer.py`)
    programmed via antipodal balanced subspaces to execute iterative Naked + Hidden Singles
    deduction in standard Transformer weights. Directly tests EFFL Lab's central thesis
    ("Weight-Space Programming Doesn't Need Transformers") on Sudoku-Extreme.
  - `WSP-Det`: Task-native deterministic fixed-step Weight-Space Program combining branchless
    Naked Singles, Hidden Singles, and Box-Line (Pointing/Claiming) tensor propagation over
    `{0, 1}` incidence matrices with damped continuous Sinkhorn-Potts relaxation and an exact
    zero-conflict halting certificate (`P == 0`).
  - `WSP-Assoc`: Flagship Weight-Space Program combining Stage 1 programmed deductive tensor
    domain reduction (`K=16` steps) with Stage 2 multi-chain (`C=4`) domain-reduced focused
    Gumbel-Potts associative updates and periodic zero-conflict halting (`check=4096`).
    Provides:
      (a) pure static-AST `WSP_ASSOC_ARCH_SOURCE` (`static_problems == []`),
      (b) per-move PyTorch transcription `wsp_assoc_move_pytorch` counted directly by
          `FullCounter` (`FC2`, matching `reproduction/baselines/code/flops/cram_kernels.py`), and
      (c) fused Triton kernel `_wsp_domain_moves` verified state-for-state against both (a) and (b).
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .wsp_contract import check_wsp_static_tensor_program, wsp_weight_metadata

N_CELLS = 81
N_DIGITS = 9
N_PEERS = 20
N_UNITS = 27


# ===========================================================================
# Puzzle Input Encoding & Independent Zero-Leakage Rule Verifier
# ===========================================================================
def encode_sudoku_clues(solutions_0_8: np.ndarray | torch.Tensor, free_mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Encode puzzles into clue-only tokens `x` in `0..9` (`0` = free cell, `1..9` = given clue).

    Free-cell ground truth is masked out to `0` here and NEVER passed into any WSP solver.
    """
    sol_t = torch.as_tensor(solutions_0_8, dtype=torch.int64)
    free_t = torch.as_tensor(free_mask, dtype=torch.bool, device=sol_t.device)
    if sol_t.shape != free_t.shape or sol_t.ndim != 2 or sol_t.shape[1] != N_CELLS:
        raise ValueError(f"Expected (B, 81) arrays, got sol={tuple(sol_t.shape)}, free={tuple(free_t.shape)}")
    return torch.where(free_t, torch.zeros_like(sol_t), sol_t + 1).contiguous()


def verify_sudoku_boards(pred_0_8: np.ndarray | torch.Tensor, x_tokens: np.ndarray | torch.Tensor) -> dict[str, Any]:
    """Independent mathematical Sudoku validity & clue-preservation checker (zero ground-truth use).

    Checks for every board `b` in `(B, 81)`:
      1. Integer range `0 <= pred[b, i] <= 8` for all 81 cells.
      2. Clue preservation: `pred[b, i] == x_tokens[b, i] - 1` wherever `x_tokens[b, i] > 0`.
      3. All 9 rows, 9 columns, and 9 3x3 boxes contain an exact permutation of `{0..8}`.
    """
    pred = torch.as_tensor(pred_0_8, dtype=torch.int64, device="cpu")
    x = torch.as_tensor(x_tokens, dtype=torch.int64, device="cpu")
    if pred.ndim != 2 or pred.shape[1] != N_CELLS or x.shape != pred.shape:
        raise ValueError(f"Shape mismatch in verify_sudoku_boards: pred={tuple(pred.shape)}, x={tuple(x.shape)}")

    in_range = ((pred >= 0) & (pred < N_DIGITS)).all(dim=1)
    is_clue = x > 0
    clues_preserved = ((~is_clue) | (pred == (x - 1))).all(dim=1)

    safe_pred = pred.clamp(0, N_DIGITS - 1)
    bit = 1 << safe_pred  # (B, 81)
    grid = bit.view(-1, 9, 9)
    full_mask = (1 << N_DIGITS) - 1  # 511

    rows_ok = torch.zeros(len(pred), dtype=torch.bool)
    cols_ok = torch.zeros(len(pred), dtype=torch.bool)
    boxes_ok = torch.zeros(len(pred), dtype=torch.bool)

    row_or = torch.zeros((len(pred), 9), dtype=torch.int64)
    col_or = torch.zeros((len(pred), 9), dtype=torch.int64)
    for k in range(9):
        row_or |= grid[:, :, k]
        col_or |= grid[:, k, :]
    rows_ok = (row_or == full_mask).all(dim=1)
    cols_ok = (col_or == full_mask).all(dim=1)

    box_grid = grid.view(-1, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    box_or = torch.zeros((len(pred), 9), dtype=torch.int64)
    for k in range(9):
        box_or |= box_grid[:, :, k]
    boxes_ok = (box_or == full_mask).all(dim=1)

    valid = in_range & clues_preserved & rows_ok & cols_ok & boxes_ok
    return {
        "valid_mask": valid.numpy(),
        "clues_preserved_mask": clues_preserved.numpy(),
        "in_range_mask": in_range.numpy(),
        "n_boards": int(len(pred)),
        "n_valid": int(valid.sum().item()),
        "n_clue_violations": int((~clues_preserved).sum().item()),
    }


def load_sudoku_split(
    split: str,
    lo: int = 0,
    n: int | None = None,
    *,
    repo_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Sudoku-Extreme split arrays `(solutions_0_8, free_mask, ratings, row_indices)`.

    Supports `'dev10k'` (`upstream/data/full-test/full-dev10k.npz`) and `'full-all'`
    (`upstream/data/full-test/full-all.npz`).
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    data_dir = root / "upstream" / "data" / "full-test"
    if split == "dev10k":
        path = data_dir / "full-dev10k.npz"
    elif split == "full-all":
        path = data_dir / "full-all.npz"
    else:
        raise ValueError(f"Unknown Sudoku split {split!r}; expected 'dev10k' or 'full-all'")
    with np.load(path) as f:
        sol_all = f["solutions"]
        clue_all = f["clue_mask"]
        rat_all = f["ratings"]
        hi = len(sol_all) if n is None else min(len(sol_all), lo + n)
        sol = sol_all[lo:hi].astype(np.int64) - 1
        free = ~clue_all[lo:hi].astype(bool)
        ratings = rat_all[lo:hi].astype(np.int64)
    idx = np.arange(lo, hi, dtype=np.int64)
    return sol, free, ratings, idx


# ===========================================================================
# Shared Rule-Only Incidence Builders (0 training data)
# ===========================================================================
def build_sudoku_incidence_tensors(device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Construct exact `{0, 1}` Sudoku geometry tensors from 9x9 rules alone."""
    dev = torch.device(device)
    idx = torch.arange(N_CELLS, device=dev)
    r, c = idx // 9, idx % 9
    b = (r // 3) * 3 + (c // 3)

    peer_bool = ((r[:, None] == r[None, :]) | (c[:, None] == c[None, :]) | (b[:, None] == b[None, :])) & (
        idx[:, None] != idx[None, :]
    )
    peer_adj = peer_bool.float().contiguous()  # (81, 81)
    peer_nbr = torch.stack([torch.nonzero(peer_bool[i], as_tuple=False).squeeze(1) for i in range(N_CELLS)]).to(
        torch.int32
    ).contiguous()  # (81, 20)

    units = [r == u for u in range(9)] + [c == u for u in range(9)] + [b == u for u in range(9)]
    unit_inc = torch.stack(units, dim=0).float().contiguous()  # (27, 81)

    # 54 unique 3-cell box-line intersections -> 108 Pointing & Claiming implication rules
    rule_mini_54: list[torch.Tensor] = []
    rule_parent: list[torch.Tensor] = []
    rule_elim: list[torch.Tensor] = []
    for box_id in range(9):
        br, bc = (box_id // 3) * 3, (box_id % 3) * 3
        for dr in range(3):
            row_id = br + dr
            in_mini = (b == box_id) & (r == row_id)
            p_box = torch.zeros(N_UNITS, device=dev)
            p_box[18 + box_id] = 1.0
            p_row = torch.zeros(N_UNITS, device=dev)
            p_row[row_id] = 1.0
            rule_mini_54.append(in_mini.float())
            rule_parent.extend([p_box, p_row])
            rule_elim.extend([((r == row_id) & (b != box_id)).float(), ((b == box_id) & (r != row_id)).float()])
        for dc in range(3):
            col_id = bc + dc
            in_mini = (b == box_id) & (c == col_id)
            p_box = torch.zeros(N_UNITS, device=dev)
            p_box[18 + box_id] = 1.0
            p_col = torch.zeros(N_UNITS, device=dev)
            p_col[9 + col_id] = 1.0
            rule_mini_54.append(in_mini.float())
            rule_parent.extend([p_box, p_col])
            rule_elim.extend([((c == col_id) & (b != box_id)).float(), ((b == box_id) & (c != col_id)).float()])

    # Proper 9-coloring of the 81-vertex, 810-edge Sudoku peer graph into 9 independent sets of 9 cells
    color = (3 * (r % 3) + (r // 3) + c) % 9
    color_cells = torch.stack(
        [torch.nonzero(color == k, as_tuple=False).squeeze(1) for k in range(9)], dim=0
    ).to(torch.int64).contiguous()  # (9, 9)

    r_mini_54_t = torch.stack(rule_mini_54, dim=0).contiguous()
    return {
        "peer_adj": peer_adj,
        "peer_nbr": peer_nbr,
        "unit_inc": unit_inc,
        "r_mini": r_mini_54_t,
        "r_parent": torch.stack(rule_parent, dim=0).contiguous(),
        "r_elim": torch.stack(rule_elim, dim=0).contiguous(),
        "color_cells": color_cells,
    }


# ===========================================================================
# Model 1: WSP-Transformer (Programmed 2-Layer Looped Standard Transformer)
# ===========================================================================
WSP_TRANSFORMER_ARCH_SOURCE = '''import math
import torch
import torch.nn.functional as F

D = 192
HEADS = 4
LAYERS = 2


def forward(w, x):
    """2-layer looped standard Transformer programmed in weight space for Sudoku.
    x: (B, 81) int64 token ids in 0..9 (0 = blank, 1..9 = clue digit).
    Returns: (B, 81, 9) digit logits.
    """
    bsz = x.shape[0]
    seq_len = x.shape[1]
    h = w["embed"][x] + w["pos"]
    n_steps = w["steps"].shape[0]
    for _ in range(n_steps):
        for i in range(LAYERS):
            p = f"block{i}."
            u = F.layer_norm(h, (D,), w[p + "ln1.weight"], w[p + "ln1.bias"])
            q = (u @ w[p + "q"] + w[p + "q.bias"]).reshape(bsz, seq_len, HEADS, D // HEADS).transpose(1, 2)
            k = (u @ w[p + "k"] + w[p + "k.bias"]).reshape(bsz, seq_len, HEADS, D // HEADS).transpose(1, 2)
            v = (u @ w[p + "v"] + w[p + "v.bias"]).reshape(bsz, seq_len, HEADS, D // HEADS).transpose(1, 2)
            att = (q @ k.transpose(-1, -2)) / math.sqrt(D // HEADS)
            a = (torch.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(bsz, seq_len, D)
            h = h + a @ w[p + "o"] + w[p + "o.bias"]
            y = F.layer_norm(h, (D,), w[p + "ln2.weight"], w[p + "ln2.bias"])
            h = h + F.gelu(y @ w[p + "fc1"] + w[p + "fc1.bias"]) @ w[p + "fc2"] + w[p + "fc2.bias"]
    z = F.layer_norm(h, (D,), w["ln_f.weight"], w["ln_f.bias"])
    return z @ w["head"] + w["head.bias"]
'''

_TF_NS: dict[str, Any] = {}
exec(WSP_TRANSFORMER_ARCH_SOURCE, _TF_NS)
wsp_transformer_forward = _TF_NS["forward"]


def build_wsp_transformer_weights(steps: int = 16, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Build programmed 2-layer looped standard Transformer weights from Sudoku rules alone."""
    dev = torch.device(device)
    D, HEADS, HD, MLP, HALF = 192, 4, 48, 384, 96
    R = 4.0
    R_ANC = 64.0
    STD = math.sqrt((2.0 * (R_ANC**2 + 4.0 * R**2 + 12.0)) / D)
    S = 40.0

    def set_sym(row: torch.Tensor, idx: int, val: float) -> None:
        row[idx] = val
        row[HALF + idx] = -val

    def bind_read(mat: torch.Tensor, src_idx: int, dst_col: int, scale: float) -> None:
        mat[src_idx, dst_col] += 0.5 * scale
        mat[HALF + src_idx, dst_col] -= 0.5 * scale

    def bind_write(mat: torch.Tensor, src_row: int, dst_idx: int, scale: float) -> None:
        mat[src_row, dst_idx] += scale
        mat[src_row, HALF + dst_idx] -= scale

    def add_step01(
        fc1: torch.Tensor,
        fc1_b: torch.Tensor,
        fc2: torch.Tensor,
        col_ptr: int,
        terms: list[tuple[int, float]],
        dst_idx: int,
        out_weight: float,
    ) -> int:
        c_off = 0.0
        for ch, coeff in terms:
            if ch == -1:
                c_off += coeff
            else:
                bind_read(fc1, ch, col_ptr, S * coeff)
                bind_read(fc1, ch, col_ptr + 1, S * coeff)
        fc1_b[col_ptr] = S * (c_off - 0.4)
        fc1_b[col_ptr + 1] = S * (c_off - 0.6)
        bind_write(fc2, col_ptr, dst_idx, out_weight / (0.2 * S))
        bind_write(fc2, col_ptr + 1, dst_idx, -out_weight / (0.2 * S))
        return col_ptr + 2

    def add_linear_clear(
        fc1: torch.Tensor,
        fc1_b: torch.Tensor,
        fc2: torch.Tensor,
        col_ptr: int,
        src_idx: int,
        dst_idx: int,
        out_weight: float,
    ) -> int:
        bind_read(fc1, src_idx, col_ptr, 1.0)
        fc1_b[col_ptr] = 12.0
        fc1_b[col_ptr + 1] = 12.0
        bind_write(fc2, col_ptr, dst_idx, out_weight)
        bind_write(fc2, col_ptr + 1, dst_idx, -out_weight)
        return col_ptr + 2

    embed = torch.zeros(10, D, device=dev)
    for d in range(9):
        set_sym(embed[0], d, 1.0)
        set_sym(embed[d + 1], d, 1.0)
        set_sym(embed[d + 1], 9 + d, 1.0)
        set_sym(embed[d + 1], 18 + d, 1.0)

    pos = torch.zeros(N_CELLS, D, device=dev)
    for i in range(N_CELLS):
        r, c = i // 9, i % 9
        b = (r // 3) * 3 + (c // 3)
        intra = (r % 3) * 3 + (c % 3)
        set_sym(pos[i], 27 + r, R)
        set_sym(pos[i], 36 + c, R)
        set_sym(pos[i], 45 + b, R)
        set_sym(pos[i], 54 + intra, R)
        set_sym(pos[i], 95, R_ANC)

    ln_w = torch.full((D,), STD, device=dev)
    ln_b = torch.zeros(D, device=dev)

    def make_unit_qk() -> tuple[torch.Tensor, torch.Tensor]:
        wq = torch.zeros(D, D, device=dev)
        wk = torch.zeros(D, D, device=dev)
        alpha = 18.0
        for j in range(9):
            bind_read(wq, 27 + j, j, alpha / R)
            bind_read(wk, 27 + j, j, alpha / R)
            bind_read(wq, 36 + j, 9 + j, -alpha / R)
            bind_read(wk, 36 + j, 9 + j, alpha / R)

            bind_read(wq, 36 + j, 48 + j, alpha / R)
            bind_read(wk, 36 + j, 48 + j, alpha / R)
            bind_read(wq, 27 + j, 48 + 9 + j, -alpha / R)
            bind_read(wk, 27 + j, 48 + 9 + j, alpha / R)

            bind_read(wq, 45 + j, 96 + j, alpha / R)
            bind_read(wk, 45 + j, 96 + j, alpha / R)
            bind_read(wq, 54 + j, 96 + 9 + j, -alpha / R)
            bind_read(wk, 54 + j, 96 + 9 + j, alpha / R)

            bind_read(wq, 27 + j, 144 + j, alpha / R)
            bind_read(wk, 27 + j, 144 + j, alpha / R)
            bind_read(wq, 36 + j, 144 + 9 + j, alpha / R)
            bind_read(wk, 36 + j, 144 + 9 + j, alpha / R)
        return wq, wk

    wq0, wk0 = make_unit_qk()
    wv0 = torch.zeros(D, D, device=dev)
    wo0 = torch.zeros(D, D, device=dev)
    for hd in range(3):
        base = hd * HD
        for d in range(9):
            bind_read(wv0, 9 + d, base + d, 8.0)
            bind_write(wo0, base + d, 63 + d, 1.0)
    for d in range(9):
        for dp in range(9):
            if dp != d:
                bind_read(wv0, 9 + dp, 144 + d, 1.0)
        bind_write(wo0, 144 + d, 63 + d, 1.0)

    fc1_0 = torch.zeros(D, MLP, device=dev)
    fc1_b0 = torch.zeros(MLP, device=dev)
    fc2_0 = torch.zeros(MLP, D, device=dev)
    p0 = 0
    for d in range(9):
        p0 = add_step01(fc1_0, fc1_b0, fc2_0, p0, [(d, 20.0), (63 + d, 1.0), (18 + d, -40.0), (-1, -20.0)], d, -1.0)
        p0 = add_linear_clear(fc1_0, fc1_b0, fc2_0, p0, 63 + d, 63 + d, -1.0)

    wq1, wk1 = make_unit_qk()
    wv1 = torch.zeros(D, D, device=dev)
    wo1 = torch.zeros(D, D, device=dev)
    for hd in range(3):
        base = hd * HD
        for d in range(9):
            bind_read(wv1, d, base + d, 8.0)
            bind_write(wo1, base + d, 63 + hd * 9 + d, 1.0)

    fc1_1 = torch.zeros(D, MLP, device=dev)
    fc1_b1 = torch.zeros(MLP, device=dev)
    fc2_1 = torch.zeros(MLP, D, device=dev)
    p1 = 0
    for d in range(9):
        ns_terms = [(dp, 1.0 if dp == d else -1.0) for dp in range(9)] + [(9 + d, -5.0)]
        p1 = add_step01(fc1_1, fc1_b1, fc2_1, p1, ns_terms, 9 + d, 1.0)
        for hd in range(3):
            p1 = add_step01(fc1_1, fc1_b1, fc2_1, p1, [(d, 1.0), (63 + hd * 9 + d, -1.0), (9 + d, -5.0)], 9 + d, 1.0)
            p1 = add_linear_clear(fc1_1, fc1_b1, fc2_1, p1, 63 + hd * 9 + d, 63 + hd * 9 + d, -1.0)

    head = torch.zeros(D, 9, device=dev)
    for d in range(9):
        bind_read(head, d, d, 1.0)
        bind_read(head, 9 + d, d, 2.0)
        bind_read(head, 18 + d, d, 10.0)

    z_d = torch.zeros(D, device=dev)
    return {
        "embed": embed,
        "pos": pos,
        "steps": torch.ones(steps, dtype=torch.int8, device=dev),
        "block0.ln1.weight": ln_w.clone(),
        "block0.ln1.bias": ln_b.clone(),
        "block0.q": wq0,
        "block0.q.bias": z_d.clone(),
        "block0.k": wk0,
        "block0.k.bias": z_d.clone(),
        "block0.v": wv0,
        "block0.v.bias": z_d.clone(),
        "block0.o": wo0,
        "block0.o.bias": z_d.clone(),
        "block0.ln2.weight": ln_w.clone(),
        "block0.ln2.bias": ln_b.clone(),
        "block0.fc1": fc1_0,
        "block0.fc1.bias": fc1_b0,
        "block0.fc2": fc2_0,
        "block0.fc2.bias": z_d.clone(),
        "block1.ln1.weight": ln_w.clone(),
        "block1.ln1.bias": ln_b.clone(),
        "block1.q": wq1,
        "block1.q.bias": z_d.clone(),
        "block1.k": wk1,
        "block1.k.bias": z_d.clone(),
        "block1.v": wv1,
        "block1.v.bias": z_d.clone(),
        "block1.o": wo1,
        "block1.o.bias": z_d.clone(),
        "block1.ln2.weight": ln_w.clone(),
        "block1.ln2.bias": ln_b.clone(),
        "block1.fc1": fc1_1,
        "block1.fc1.bias": fc1_b1,
        "block1.fc2": fc2_1,
        "block1.fc2.bias": z_d.clone(),
        "ln_f.weight": ln_w.clone(),
        "ln_f.bias": ln_b.clone(),
        "head": head,
        "head.bias": torch.zeros(9, device=dev),
    }


# ===========================================================================
# Model 2: WSP-Det (Task-Native Deterministic Weight-Space Program)
# ===========================================================================
WSP_DET_ARCH_SOURCE = '''import torch
import torch.nn.functional as F


def forward(w, x):
    """Deterministic Weight-Space Program for Sudoku (0 training boards).
    x: (B, 81) int64 token ids in 0..9 (0 = blank, 1..9 = given clue digit).
    Returns: (B, 81, 9) final scores, plus certificate check via peer_adj.
    """
    is_clue = (x > 0).unsqueeze(-1)
    clue_oh = F.one_hot((x - 1).clamp(min=0), 9).float() * is_clue.float()
    m = torch.where(is_clue, clue_oh, torch.ones_like(clue_oh))

    peer = w["peer_adj"]
    unit = w["unit_inc"]
    r_mini = w["r_mini"]
    r_parent = w["r_parent"]
    r_elim = w["r_elim"]

    n_prop = w["prop_steps"].shape[0]
    for _ in range(n_prop):
        single = (m.sum(dim=-1, keepdim=True) == 1.0).float()
        forb = (torch.matmul(peer, m * single) > 0.5).float()
        m_elim = torch.where(single > 0.5, m, m * (1.0 - forb))
        m = torch.where(m_elim.sum(dim=-1, keepdim=True) > 0.5, m_elim, m)

        uc = torch.matmul(unit, m)
        hs = (torch.matmul(unit.transpose(0, 1), (uc == 1.0).float()) > 0.5).float() * m
        m = torch.where(is_clue, clue_oh, torch.where(hs.sum(dim=-1, keepdim=True) == 1.0, hs, m))

        uc2 = torch.matmul(unit, m)
        mc = torch.matmul(r_mini, m).repeat_interleave(2, dim=1)
        pc = torch.matmul(r_parent, uc2)
        locked = ((mc >= 1.0) & (mc == pc)).float()
        lock_forb = (torch.matmul(r_elim.transpose(0, 1), locked) > 0.5).float()
        m_elim2 = m * (1.0 - lock_forb)
        m = torch.where(is_clue, clue_oh, torch.where(m_elim2.sum(dim=-1, keepdim=True) > 0.5, m_elim2, m))

    single_f = (m.sum(dim=-1, keepdim=True) == 1.0).float()
    forb_f = (torch.matmul(peer, m * single_f) > 0.5).float()
    m_elim_f = torch.where(single_f > 0.5, m, m * (1.0 - forb_f))
    m = torch.where(m_elim_f.sum(dim=-1, keepdim=True) > 0.5, m_elim_f, m)

    z = torch.zeros_like(m)
    eta = w["eta"]
    beta = w["beta"]
    n_rel = w["relax_steps"].shape[0]
    for _ in range(n_rel):
        p = torch.softmax(beta * z + (m - 1.0) * 1e9, dim=-1)
        peer_rep = torch.matmul(peer, p)
        unit_def = torch.matmul(unit.transpose(0, 1), 1.0 - torch.matmul(unit, p))
        grad = (-peer_rep + 0.5 * unit_def) * m
        z = (1.0 - eta) * z + eta * grad
    return z + (m - 1.0) * 1e6 + clue_oh * 1e3
'''

_DET_NS: dict[str, Any] = {}
exec(WSP_DET_ARCH_SOURCE, _DET_NS)
wsp_det_forward = _DET_NS["forward"]


def build_wsp_det_weights(
    prop_steps: int = 16,
    relax_steps: int = 32,
    *,
    eta: float = 0.5,
    beta: float = 4.0,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Build deterministic task-native WSP weights (`3` distinct nonzero float coefficients)."""
    dev = torch.device(device)
    inc = build_sudoku_incidence_tensors(dev)
    return {
        "peer_adj": inc["peer_adj"],
        "unit_inc": inc["unit_inc"],
        "r_mini": inc["r_mini"],
        "r_parent": inc["r_parent"],
        "r_elim": inc["r_elim"],
        "prop_steps": torch.ones(prop_steps, dtype=torch.int8, device=dev),
        "relax_steps": torch.ones(relax_steps, dtype=torch.int8, device=dev),
        "eta": torch.tensor(eta, dtype=torch.float32, device=dev),
        "beta": torch.tensor(beta, dtype=torch.float32, device=dev),
    }


def evaluate_deterministic_wsp(
    forward_fn: Any,
    weights: dict[str, torch.Tensor],
    x_tokens: torch.Tensor,
    peer_adj: torch.Tensor,
    *,
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Run a deterministic WSP forward (`WSP-Transformer` or `WSP-Det`) with zero-conflict certificate."""
    dev = peer_adj.device
    preds: list[torch.Tensor] = []
    certs: list[torch.Tensor] = []
    with torch.no_grad():
        for lo in range(0, len(x_tokens), batch_size):
            xb = x_tokens[lo : lo + batch_size].to(dev)
            logits = forward_fn(weights, xb)
            pred = logits.argmax(dim=-1)
            oh = F.one_hot(pred, N_DIGITS).float()
            conf = 0.5 * (oh * torch.matmul(peer_adj, oh)).sum(dim=(1, 2))
            is_clue = xb > 0
            clue_ok = ((~is_clue) | (pred == (xb - 1))).all(dim=1)
            cert = (conf == 0.0) & clue_ok
            preds.append(pred.cpu())
            certs.append(cert.cpu())
    return {
        "pred": torch.cat(preds, dim=0),
        "stopped": torch.cat(certs, dim=0),
    }


# ===========================================================================
# Model 3: WSP-Assoc (Programmed Deductive Domain Reduction + Focused Gumbel-Potts)
# ===========================================================================
# Static-AST verified pure PyTorch tensor program (`static_problems == []`):
# Stage 1 deductive propagation (`wsp_assoc_stage1_forward`) + Stage 2 associative move chunk (`forward`).
WSP_ASSOC_STAGE1_SOURCE = '''import torch
import torch.nn.functional as F


def forward(w, x):
    """Stage 1 of WSP-Assoc: branchless Naked + Hidden + Box-Line candidate domain reduction.
    x: (B, 81) int64 token ids in 0..9 (0 = blank, 1..9 = clue digit).
    Returns: m (B, 81, 9) float32 candidate mask in {0.0, 1.0}.
    """
    is_clue = (x > 0).unsqueeze(-1)
    clue_oh = F.one_hot((x - 1).clamp(min=0), 9).float() * is_clue.float()
    m = torch.where(is_clue, clue_oh, torch.ones_like(clue_oh))

    peer = w["peer_adj"]
    unit = w["unit_inc"]
    r_mini = w["r_mini"]
    r_parent = w["r_parent"]
    r_elim = w["r_elim"]

    n_prop = w["prop_steps"].shape[0]
    for _ in range(n_prop):
        single = (m.sum(dim=-1, keepdim=True) == 1.0).float()
        forb = (torch.matmul(peer, m * single) > 0.5).float()
        m_elim = torch.where(single > 0.5, m, m * (1.0 - forb))
        m = torch.where(m_elim.sum(dim=-1, keepdim=True) > 0.5, m_elim, m)

        uc = torch.matmul(unit, m)
        hs = (torch.matmul(unit.transpose(0, 1), (uc == 1.0).float()) > 0.5).float() * m
        m = torch.where(is_clue, clue_oh, torch.where(hs.sum(dim=-1, keepdim=True) == 1.0, hs, m))

        uc2 = torch.matmul(unit, m)
        mc = torch.matmul(r_mini, m).repeat_interleave(2, dim=1)
        pc = torch.matmul(r_parent, uc2)
        locked = ((mc >= 1.0) & (mc == pc)).float()
        lock_forb = (torch.matmul(r_elim.transpose(0, 1), locked) > 0.5).float()
        m_elim2 = m * (1.0 - lock_forb)
        m = torch.where(is_clue, clue_oh, torch.where(m_elim2.sum(dim=-1, keepdim=True) > 0.5, m_elim2, m))

        single_f = (m.sum(dim=-1, keepdim=True) == 1.0).float()
        forb_f = (torch.matmul(peer, m * single_f) > 0.5).float()
        m_elim_f = torch.where(single_f > 0.5, m, m * (1.0 - forb_f))
        m = torch.where(m_elim_f.sum(dim=-1, keepdim=True) > 0.5, m_elim_f, m)
    return m
'''

WSP_ASSOC_ARCH_SOURCE = '''import torch
import torch.nn.functional as F


def stage1_domains(w, x):
    is_clue = (x > 0).unsqueeze(-1)
    clue_oh = F.one_hot((x - 1).clamp(min=0), 9).float() * is_clue.float()
    m = torch.where(is_clue, clue_oh, torch.ones_like(clue_oh))
    peer = w["peer_adj"]
    unit = w["unit_inc"]
    r_mini = w["r_mini"]
    r_parent = w["r_parent"]
    r_elim = w["r_elim"]
    n_prop = w["prop_steps"].shape[0]
    for _ in range(n_prop):
        single = (m.sum(dim=-1, keepdim=True) == 1.0).float()
        forb = (torch.matmul(peer, m * single) > 0.5).float()
        m_elim = torch.where(single > 0.5, m, m * (1.0 - forb))
        m = torch.where(m_elim.sum(dim=-1, keepdim=True) > 0.5, m_elim, m)
        uc = torch.matmul(unit, m)
        hs = (torch.matmul(unit.transpose(0, 1), (uc == 1.0).float()) > 0.5).float() * m
        m = torch.where(is_clue, clue_oh, torch.where(hs.sum(dim=-1, keepdim=True) == 1.0, hs, m))
        uc2 = torch.matmul(unit, m)
        mc = torch.matmul(r_mini, m).repeat_interleave(2, dim=1)
        pc = torch.matmul(r_parent, uc2)
        locked = ((mc >= 1.0) & (mc == pc)).float()
        lock_forb = (torch.matmul(r_elim.transpose(0, 1), locked) > 0.5).float()
        m_elim2 = m * (1.0 - lock_forb)
        m = torch.where(is_clue, clue_oh, torch.where(m_elim2.sum(dim=-1, keepdim=True) > 0.5, m_elim2, m))
        single_f = (m.sum(dim=-1, keepdim=True) == 1.0).float()
        forb_f = (torch.matmul(peer, m * single_f) > 0.5).float()
        m_elim_f = torch.where(single_f > 0.5, m, m * (1.0 - forb_f))
        m = torch.where(m_elim_f.sum(dim=-1, keepdim=True) > 0.5, m_elim_f, m)
    return m


def forward(w, x, u_init, u_cell, u_focus, u_digit):
    """Complete static-AST Weight-Space Program for WSP-Assoc (0 training boards).
    x:       (B, 81) int64 clue tokens in 0..9
    u_init:  (B, C, 81, 9) uniform draws in (0, 1) for initial candidate-restricted boards
    u_cell:  (B * C, M) uniform draws in (0, 1) for fallback free-cell choice
    u_focus: (B * C, M, 81) uniform draws in (0, 1) for focused conflicted-cell choice
    u_digit: (B * C, M, 9) uniform draws in (0, 1) for Gumbel-Potts digit choice
    """
    m_puz = stage1_domains(w, x)
    bsz = x.shape[0]
    chains = u_init.shape[1]
    h = bsz * chains
    m = m_puz.unsqueeze(1).expand(bsz, chains, 81, 9).reshape(h, 81, 9)
    fr = (m.sum(dim=-1) > 1.5)
    fmask = fr.to(torch.int32)
    nfree = fmask.sum(dim=-1).to(torch.int32)
    freelist = torch.argsort((~fr).to(torch.int8), dim=1, stable=True).to(torch.int32)

    gum0 = -torch.log(-torch.log(torch.maximum(u_init.reshape(h, 81, 9), w["eps"])))
    y = torch.where(m > 0.5, gum0, w["neg_inf"]).argmax(dim=-1).to(torch.int32)
    y_oh = F.one_hot(y.to(torch.int64), 9).float()
    g = torch.matmul(w["peer_adj"], y_oh)
    p = 0.5 * (y_oh * g).sum(dim=(1, 2))
    own0 = g.gather(2, y.to(torch.int64).unsqueeze(-1)).squeeze(-1)
    thr = w["theta"]
    tau = w["tau_tol"]
    flag = ((own0 > thr) & fr).to(torch.int32)
    found = torch.where(p <= tau, torch.zeros_like(nfree), torch.full_like(nfree, -1))
    ycert = y.clone()

    temp = w["temp"]
    peer = w["peer_adj"]
    nmoves = u_cell.shape[1]
    for k in range(nmoves):
        u0 = u_cell[:, k]
        idx_f = torch.minimum((u0 * nfree.float()).to(torch.int32), torch.maximum(nfree - 1, torch.zeros_like(nfree)))
        cell_fb = freelist.gather(1, idx_f.to(torch.int64).unsqueeze(1)).squeeze(1)
        fl = flag != 0
        score = torch.where(fl, u_focus[:, k, :], -1.0)
        cc = torch.argmax(score, dim=1).to(torch.int32)
        cell = torch.where(torch.amax(score, dim=1) >= 0.0, cc, cell_fb)
        cell_i64 = cell.to(torch.int64)

        gi = g.gather(1, cell_i64[:, None, None].expand(h, 1, 9)).squeeze(1)
        valid = m.gather(1, cell_i64[:, None, None].expand(h, 1, 9)).squeeze(1) > 0.5
        uu = u_digit[:, k, :]
        gum = -torch.log(-torch.log(torch.maximum(uu, w["eps"])))
        lg = torch.where(valid, gi * (-1.0 / temp) + gum, w["neg_inf"])
        new_raw = torch.argmax(lg, dim=1).to(torch.int32)
        old = y.gather(1, cell_i64.unsqueeze(1)).squeeze(1)
        frozen = (found >= 0) | (nfree == 0)
        new = torch.where(frozen, old, new_raw)
        dd = (new != old).float()

        col_w = peer[cell_i64] * dd.unsqueeze(1)
        old_oh = F.one_hot(old.to(torch.int64), 9).float()
        new_oh = F.one_hot(new.to(torch.int64), 9).float()
        g = g + col_w.unsqueeze(2) * (new_oh - old_oh).unsqueeze(1)

        gnew = gi.gather(1, new.to(torch.int64).unsqueeze(1)).squeeze(1)
        gold = gi.gather(1, old.to(torch.int64).unsqueeze(1)).squeeze(1)
        p = p + (gnew - gold) * dd
        y = y.scatter(1, cell_i64.unsqueeze(1), new.unsqueeze(1))

        own_all = g.gather(2, y.to(torch.int64).unsqueeze(-1)).squeeze(-1)
        flag = ((own_all > thr) & fr).to(torch.int32)

        cert = (found < 0) & (p <= tau)
        ycert = torch.where(cert.unsqueeze(1), y, ycert)
        step_num = torch.full_like(found, k + 1)
        found = torch.where(cert, step_num, found)

    return ycert, found, p
'''

WSP_ASSOC_PROP_STEP_SOURCE = '''import torch
import torch.nn.functional as F


def forward(w, x, m):
    """Single branchless deductive propagation step of WSP-Assoc (`static_problems == []`)."""
    is_clue = (x > 0).unsqueeze(-1)
    clue_oh = F.one_hot((x - 1).clamp(min=0), 9).float() * is_clue.float()
    peer = w["peer_adj"]
    unit = w["unit_inc"]
    r_mini = w["r_mini"]
    r_parent = w["r_parent"]
    r_elim = w["r_elim"]

    single = (m.sum(dim=-1, keepdim=True) == 1.0).float()
    forb = (torch.matmul(peer, m * single) > 0.5).float()
    m_elim = torch.where(single > 0.5, m, m * (1.0 - forb))
    m = torch.where(m_elim.sum(dim=-1, keepdim=True) > 0.5, m_elim, m)

    uc = torch.matmul(unit, m)
    hs = (torch.matmul(unit.transpose(0, 1), (uc == 1.0).float()) > 0.5).float() * m
    m = torch.where(is_clue, clue_oh, torch.where(hs.sum(dim=-1, keepdim=True) == 1.0, hs, m))

    uc2 = torch.matmul(unit, m)
    mc = torch.matmul(r_mini, m).repeat_interleave(2, dim=1)
    pc = torch.matmul(r_parent, uc2)
    locked = ((mc >= 1.0) & (mc == pc)).float()
    lock_forb = (torch.matmul(r_elim.transpose(0, 1), locked) > 0.5).float()
    m_elim2 = m * (1.0 - lock_forb)
    m = torch.where(is_clue, clue_oh, torch.where(m_elim2.sum(dim=-1, keepdim=True) > 0.5, m_elim2, m))

    single_f = (m.sum(dim=-1, keepdim=True) == 1.0).float()
    forb_f = (torch.matmul(peer, m * single_f) > 0.5).float()
    m_elim_f = torch.where(single_f > 0.5, m, m * (1.0 - forb_f))
    return torch.where(m_elim_f.sum(dim=-1, keepdim=True) > 0.5, m_elim_f, m)
'''

_ASSOC_STEP_NS: dict[str, Any] = {}
exec(WSP_ASSOC_PROP_STEP_SOURCE, _ASSOC_STEP_NS)
wsp_assoc_prop_step_forward = _ASSOC_STEP_NS["forward"]

_ASSOC_S1_NS: dict[str, Any] = {}
exec(WSP_ASSOC_STAGE1_SOURCE, _ASSOC_S1_NS)
wsp_assoc_stage1_forward = _ASSOC_S1_NS["forward"]

_ASSOC_NS: dict[str, Any] = {}
exec(WSP_ASSOC_ARCH_SOURCE, _ASSOC_NS)
wsp_assoc_static_forward = _ASSOC_NS["forward"]


def build_wsp_assoc_weights(
    *,
    prop_steps: int = 16,
    temp_hi: float = 0.52,
    temp_lo: float = 0.38,
    anneal_period: int = 524288,
    theta: float = 4.0 / 9.0,
    tau_tol: float = 1e-3,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Build programmed weights for `WSP-Assoc` from Sudoku rules alone (`0` training boards)."""
    dev = torch.device(device)
    inc = build_sudoku_incidence_tensors(dev)
    return {
        "peer_adj": inc["peer_adj"],
        "peer_nbr": inc["peer_nbr"],
        "unit_inc": inc["unit_inc"],
        "r_mini": inc["r_mini"],
        "r_parent": inc["r_parent"],
        "r_elim": inc["r_elim"],
        "color_cells": inc["color_cells"],
        "prop_steps": torch.ones(prop_steps, dtype=torch.int8, device=dev),
        "temp": torch.tensor(temp_hi, dtype=torch.float32, device=dev),
        "temp_hi": torch.tensor(temp_hi, dtype=torch.float32, device=dev),
        "temp_lo": torch.tensor(temp_lo, dtype=torch.float32, device=dev),
        "anneal_period": torch.tensor(anneal_period, dtype=torch.int64, device=dev),
        "theta": torch.tensor(theta, dtype=torch.float32, device=dev),
        "tau_tol": torch.tensor(tau_tol, dtype=torch.float32, device=dev),
        "eps": torch.tensor(1e-20, dtype=torch.float32, device=dev),
        "neg_inf": torch.tensor(-1e9, dtype=torch.float32, device=dev),
    }


# ---------------------------------------------------------------------------
# Fused Triton GPU Kernel + Exact Line-by-Line PyTorch Transcription
# ---------------------------------------------------------------------------
@triton.jit
def _wsp_domain_moves(
    nbr_ptr,
    cand_ptr,
    y_ptr,
    G_ptr,
    flag_ptr,
    P_ptr,
    free_ptr,
    fmask_ptr,
    nfree_ptr,
    temp_ptr,
    found_ptr,
    ycert_ptr,
    seed_ptr,
    nmoves,
    move0,
    thr,
    tau,
    N: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    NB: tl.constexpr,
    VB: tl.constexpr,
    DB: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, NB)
    cols = tl.arange(0, VB)
    ks = tl.arange(0, DB)
    rmask = rows < N
    cmask = cols < V
    kmask = ks < D
    ybase = y_ptr + pid * N
    gbase = G_ptr + pid * N * V
    cbase = cand_ptr + pid * N * V
    fbase = flag_ptr + pid * N
    mbase = fmask_ptr + pid * N
    P = tl.load(P_ptr + pid)
    T = tl.load(temp_ptr + pid)
    nf = tl.load(nfree_ptr + pid)
    found = tl.load(found_ptr + pid)
    sd = tl.load(seed_ptr)
    for k in range(nmoves):
        sk = sd * 65536 + k
        base = (pid * 512).to(tl.int32)
        u0 = tl.rand(sk, base)
        idx_f = tl.minimum((u0 * nf).to(tl.int32), tl.maximum(nf - 1, 0))
        cell = tl.load(free_ptr + pid * N + idx_f)
        fl = tl.load(fbase + rows, mask=rmask, other=0) != 0
        score = tl.where(fl, tl.rand(sk, base + 64 + rows), -1.0)
        cc = tl.argmax(score, 0)
        cell = tl.where(tl.max(score, 0) >= 0.0, cc, cell)

        Gi = tl.load(gbase + cell * V + cols, mask=cmask, other=0.0)
        valid = tl.load(cbase + cell * V + cols, mask=cmask, other=0) != 0
        uu = tl.rand(sk, base + 16 + cols)
        gum = -tl.log(-tl.log(tl.maximum(uu, 1e-20)))
        lg = tl.where(cmask & valid, Gi * (-1.0 / T) + gum, float("-inf"))
        new = tl.argmax(lg, 0)
        old = tl.load(ybase + cell)
        new = tl.where((found >= 0) | (nf == 0), old, new)
        dd = (new != old).to(tl.float32)

        nb = tl.load(nbr_ptr + cell * D + ks, mask=kmask, other=0)
        po = gbase + nb * V + old
        go = tl.load(po, mask=kmask, other=0.0) - dd
        tl.store(po, go, mask=kmask)
        pn = gbase + nb * V + new
        gn = tl.load(pn, mask=kmask, other=0.0) + dd
        tl.store(pn, gn, mask=kmask)

        ynb = tl.load(ybase + nb, mask=kmask, other=-1)
        own = tl.where(ynb == old, go, gn)
        changed = kmask & ((ynb == old) | (ynb == new))
        fnb = tl.load(mbase + nb, mask=kmask, other=0) != 0
        tl.store(fbase + nb, ((own > thr) & fnb).to(tl.int32), mask=changed)

        gnew = tl.sum(tl.where(cols == new, Gi, 0.0), 0)
        gold = tl.sum(tl.where(cols == old, Gi, 0.0), 0)
        P = P + (gnew - gold) * dd
        tl.store(ybase + cell, new)
        fcell = tl.load(mbase + cell) != 0
        tl.store(fbase + cell, ((gnew > thr) & fcell).to(tl.int32))
        tl.debug_barrier()
        cert = (found < 0) & (P <= tau)
        yv = tl.load(ybase + rows, mask=rmask & cert, other=0)
        tl.store(ycert_ptr + pid * N + rows, yv, mask=rmask & cert)
        found = tl.where(cert, move0 + k + 1, found)
    tl.store(P_ptr + pid, P)
    tl.store(found_ptr + pid, found)


@triton.jit
def _extract_triton_move_rng(
    seed_ptr,
    u0_ptr,
    uf_ptr,
    uu_ptr,
    nmoves,
    N: tl.constexpr,
    V: tl.constexpr,
    NB: tl.constexpr,
    VB: tl.constexpr,
):
    """Test helper: extract exact Philox uniform draws `(u0, u_focus, u_digit)` of `_wsp_domain_moves`."""
    pid = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, NB)
    cols = tl.arange(0, VB)
    rmask = rows < N
    cmask = cols < V
    sd = tl.load(seed_ptr)
    for k in range(nmoves):
        sk = sd * 65536 + k
        base = (pid * 512).to(tl.int32)
        u0 = tl.rand(sk, base)
        tl.store(u0_ptr + pid * nmoves + k, u0)
        uf = tl.rand(sk, base + 64 + rows)
        tl.store(uf_ptr + (pid * nmoves + k) * N + rows, uf, mask=rmask)
        uu = tl.rand(sk, base + 16 + cols)
        tl.store(uu_ptr + (pid * nmoves + k) * V + cols, uu, mask=cmask)


def wsp_assoc_move_pytorch(
    st: dict[str, torch.Tensor],
    *,
    u0_in: torch.Tensor | None = None,
    uf_in: torch.Tensor | None = None,
    uu_in: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact statement-by-statement PyTorch transcription of one move of `_wsp_domain_moves`
    for one chain (following `reproduction/baselines/code/flops/cram_kernels.py:move_sparse`).

    Used both for `FullCounter` (`FC2`) operation counting (when `u*_in` are None) and for
    exact state-for-state parity verification against `_wsp_domain_moves` (when `u*_in` are
    the extracted Philox uniform draws).
    """
    dev = st["G"].device
    rmask = torch.ones(N_CELLS, dtype=torch.bool, device=dev)
    cmask = torch.ones(N_DIGITS, dtype=torch.bool, device=dev)
    kmask = torch.ones(N_PEERS, dtype=torch.bool, device=dev)
    G, y, flag = st["G"], st["y"], st["flag"]

    u0 = torch.rand((), device=dev) if u0_in is None else u0_in
    zero_i32 = torch.zeros((), dtype=torch.int32, device=dev)
    idx_f = torch.minimum((u0 * st["nf"]).to(torch.int32), torch.maximum(st["nf"] - 1, zero_i32))
    cell = st["free"][idx_f.long()]

    fl = flag != 0
    uf = torch.rand(N_CELLS, device=dev) if uf_in is None else uf_in
    score = torch.where(fl, uf, -1.0)
    cc = torch.argmax(score, 0).to(torch.int32)
    cell = torch.where(torch.max(score, 0).values >= 0.0, cc, cell)
    cell_idx = cell.long()

    Gi = G[cell_idx]
    valid = st["cand"][cell_idx] != 0
    uu = torch.rand(N_DIGITS, device=dev) if uu_in is None else uu_in
    gum = -torch.log(-torch.log(torch.maximum(uu, torch.tensor(1e-20, device=dev))))
    lg = torch.where(cmask & valid, Gi * (-1.0 / st["T"]) + gum, float("-inf"))
    new = torch.argmax(lg, 0).to(torch.int32)
    old = y[cell_idx]
    new = torch.where((st["found"] >= 0) | (st["nf"] == 0), old, new)
    dd = (new != old).to(torch.float32)

    nb = st["nbr"][cell_idx].long()
    old_idx = old.long()
    new_idx = new.long()
    go = G[nb, old_idx] - dd
    G[nb, old_idx] = go
    gn = G[nb, new_idx] + dd
    G[nb, new_idx] = gn

    ynb = y[nb]
    own = torch.where(ynb == old, go, gn)
    changed = kmask & ((ynb == old) | (ynb == new))
    fnb = st["fmask"][nb] != 0
    flag[nb[changed]] = ((own > st["thr"]) & fnb).to(torch.int32)[changed]

    gnew = Gi[new_idx]  # source: tl.sum(tl.where(cols == new, Gi, 0.0), 0)
    gold = Gi[old_idx]  # source: tl.sum(tl.where(cols == old, Gi, 0.0), 0)
    st["P"] = st["P"] + (gnew - gold) * dd
    y[cell_idx] = new
    fcell = st["fmask"][cell_idx] != 0
    flag[cell_idx] = ((gnew > st["thr"]) & fcell).to(torch.int32)

    cert = (st["found"] < 0) & (st["P"] <= st["tau"])
    cmask_row = rmask & cert
    st["ycert"].copy_(torch.where(cmask_row, y, st["ycert"]))
    st["found"] = torch.where(cert, st["move0"] + st["k"] + 1, st["found"])
    return st["ycert"]


def make_synthetic_assoc_chain_state(
    weights: dict[str, torch.Tensor],
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Construct a synthetic 1-chain state dict for `wsp_assoc_move_pytorch` FLOP counting."""
    dev = torch.device(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    cand = torch.ones((N_CELLS, N_DIGITS), dtype=torch.int32, device=dev)
    y = torch.randint(0, N_DIGITS, (N_CELLS,), generator=g, dtype=torch.int32).to(dev)
    Y_oh = F.one_hot(y.long(), N_DIGITS).float()
    peer_adj = weights["peer_adj"].to(dev)
    G = torch.matmul(peer_adj, Y_oh).contiguous()
    P = 0.5 * (Y_oh * G).sum()
    return {
        "G": G,
        "y": y,
        "ycert": y.clone(),
        "flag": torch.ones(N_CELLS, dtype=torch.int32, device=dev),
        "fmask": torch.ones(N_CELLS, dtype=torch.int32, device=dev),
        "free": torch.arange(N_CELLS, dtype=torch.int32, device=dev),
        "cand": cand,
        "nbr": weights["peer_nbr"].to(dev),
        "P": P,
        "T": torch.tensor(0.52, dtype=torch.float32, device=dev),
        "nf": torch.tensor(N_CELLS, dtype=torch.int32, device=dev),
        "found": torch.tensor(-1, dtype=torch.int32, device=dev),
        "thr": weights["theta"].to(dev),
        "tau": weights["tau_tol"].to(dev),
        "move0": torch.tensor(0, dtype=torch.int32, device=dev),
        "k": torch.tensor(0, dtype=torch.int32, device=dev),
    }


def stage2_init_chains_pytorch(
    m_puz: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    C: int = 4,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Initialize `C` chains per Stage-1-unsolved puzzle (counted directly under `FullCounter`)."""
    dev = m_puz.device
    nA = m_puz.shape[0]
    H = nA * C
    # Guard against empty candidate sets on impossible adversarial boards: fall back to all-ones
    row_has_cand = m_puz.sum(dim=-1, keepdim=True) > 0.5
    m_safe = torch.where(row_has_cand, m_puz, torch.ones_like(m_puz))
    fr_puz = (m_safe.sum(dim=-1) > 1.5) | (~row_has_cand.squeeze(-1))

    m_act = m_safe.repeat_interleave(C, dim=0)
    fr = fr_puz.repeat_interleave(C, dim=0)
    u_init = torch.rand((H, N_CELLS, N_DIGITS), device=dev, generator=generator)
    gum = -torch.log(-torch.log(torch.maximum(u_init, weights["eps"])))
    y = torch.where(m_act > 0.5, gum, weights["neg_inf"]).argmax(dim=-1).to(torch.int32).contiguous()

    Y_oh = F.one_hot(y.long(), N_DIGITS).float()
    G = torch.matmul(weights["peer_adj"], Y_oh).contiguous()
    P = (0.5 * (Y_oh * G).sum(dim=(1, 2))).contiguous()
    own = G.gather(2, y.long().unsqueeze(-1)).squeeze(-1)
    flag = ((own > weights["theta"]) & fr).to(torch.int32).contiguous()
    freelist = torch.argsort((~fr).to(torch.int8), dim=1, stable=True).to(torch.int32).contiguous()
    fmask = fr.to(torch.int32).contiguous()
    nfree = fmask.sum(dim=1).to(torch.int32).contiguous()
    cand_i32 = (m_act > 0.5).to(torch.int32).contiguous()
    found = torch.where(
        P <= weights["tau_tol"],
        torch.zeros(H, dtype=torch.int32, device=dev),
        torch.full((H,), -1, dtype=torch.int32, device=dev),
    )
    ycert = y.clone()
    return {
        "cand_i32": cand_i32,
        "y": y,
        "G": G,
        "flag": flag,
        "P": P,
        "freelist": freelist,
        "fmask": fmask,
        "nfree": nfree,
        "found": found,
        "ycert": ycert,
    }


def check_bookkeeping_one_puzzle_pytorch(
    P_chains: torch.Tensor,
    y_chains: torch.Tensor,
    found_chains: torch.Tensor,
    ycert_chains: torch.Tensor,
    bestP: torch.Tensor,
    best_board: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-puzzle check bookkeeping (counted directly under `FullCounter`, no ground truth read)."""
    mn, am = P_chains.min(dim=1)
    better = mn < bestP
    new_bestP = torch.where(better, mn, bestP)
    chosen_y = y_chains.gather(1, am[:, None, None].expand(-1, 1, N_CELLS)).squeeze(1)
    new_best_board = torch.where(better.unsqueeze(1), chosen_y.long(), best_board)

    fu = found_chains.long()
    fu_pos = torch.where(fu >= 0, fu, torch.full_like(fu, 1 << 60))
    first, fc = fu_pos.min(dim=1)
    anyc = first < (1 << 60)
    cert_board = ycert_chains.gather(1, fc[:, None, None].expand(-1, 1, N_CELLS)).squeeze(1).long()
    return new_bestP, new_best_board, anyc, cert_board


class WSPAssocSearch:
    """Fast GPU/CPU runner for `WSP-Assoc` with zero ground-truth access and exact `FC2` accounting."""

    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        x_tokens: torch.Tensor,
        *,
        C: int = 4,
        check: int = 4096,
        max_moves: int = 16_777_216,
        seed: int = 0,
        device: str | torch.device = "cuda",
        backend: str = "triton",
    ) -> None:
        dev = torch.device(device)
        if backend == "triton" and dev.type != "cuda":
            backend = "pytorch"
        self.dev = dev
        self.backend = backend
        self.C = int(C)
        self.check = int(check)
        self.max_moves = int(max_moves)
        self.seed = int(seed)
        self.w = {k: v.to(dev) for k, v in weights.items()}
        self.x_tokens = torch.as_tensor(x_tokens, dtype=torch.int64, device=dev)
        self.NQ = int(self.x_tokens.shape[0])

        self.temp_hi = float(self.w["temp_hi"].cpu())
        self.temp_lo = float(self.w["temp_lo"].cpu())
        self.period = int(self.w["anneal_period"].cpu())
        self.theta = float(self.w["theta"].cpu())
        self.tau = float(self.w["tau_tol"].cpu())

        # Stage 1: Programmed deductive tensor pass with exact fixed-point compaction (`m_next == m` or solved)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        max_prop = int(self.w["prop_steps"].numel())
        m_chunks: list[torch.Tensor] = []
        p1_chunks: list[torch.Tensor] = []
        y1_chunks: list[torch.Tensor] = []
        s1_chunks: list[torch.Tensor] = []
        ps_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for lo in range(0, self.NQ, 16384):
                xb = self.x_tokens[lo : lo + 16384]
                nb = xb.shape[0]
                is_clue = (xb > 0).unsqueeze(-1)
                clue_oh = F.one_hot((xb - 1).clamp(min=0), N_DIGITS).float() * is_clue.float()
                mb = torch.where(is_clue, clue_oh, torch.ones_like(clue_oh))
                prop_used = torch.zeros(nb, dtype=torch.int64, device=dev)
                act_idx = torch.arange(nb, dtype=torch.int64, device=dev)
                for _ in range(max_prop):
                    if len(act_idx) == 0:
                        break
                    m_prev = mb[act_idx]
                    m_next = wsp_assoc_prop_step_forward(self.w, xb[act_idx], m_prev)
                    mb[act_idx] = m_next
                    prop_used[act_idx] += 1
                    unchanged = (m_next == m_prev).all(dim=(1, 2))
                    all_single_act = (m_next.sum(dim=-1) == 1.0).all(dim=1)
                    keep = (~unchanged) & (~all_single_act)
                    act_idx = act_idx[keep]
                yb = mb.argmax(dim=-1)
                y_oh = F.one_hot(yb, N_DIGITS).float()
                pb = 0.5 * (y_oh * torch.matmul(self.w["peer_adj"], y_oh)).sum(dim=(1, 2))
                clues_ok = ((~is_clue.squeeze(-1)) | (yb == (xb - 1))).all(dim=1)
                all_single = (mb.sum(dim=-1) == 1.0).all(dim=1)
                s1_ok = (pb == 0.0) & all_single & clues_ok
                m_chunks.append(mb)
                p1_chunks.append(pb)
                y1_chunks.append(yb)
                s1_chunks.append(s1_ok)
                ps_chunks.append(prop_used)

        self.m_all = torch.cat(m_chunks, dim=0)
        p1_all = torch.cat(p1_chunks, dim=0)
        y1_all = torch.cat(y1_chunks, dim=0)
        self.stage1_solved = torch.cat(s1_chunks, dim=0)
        self.prop_steps_used = torch.cat(ps_chunks, dim=0)

        self.stopped = self.stage1_solved.clone()
        self.stop_t = torch.where(
            self.stage1_solved,
            torch.zeros(self.NQ, dtype=torch.int64, device=dev),
            torch.full((self.NQ,), -1, dtype=torch.int64, device=dev),
        )
        self.pred_board = y1_all.clone()
        self.bestP = p1_all.clone()
        self.best_board = y1_all.clone()
        self.stage2_entered = ~self.stage1_solved

        # Stage 2 initialization on unsolved puzzles (chunked if NQ is huge so VRAM stays compact)
        self.act = torch.nonzero(self.stage2_entered, as_tuple=False).squeeze(1)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(self.seed)
        if len(self.act) > 0:
            st_lists: dict[str, list[torch.Tensor]] = {
                k: []
                for k in ("cand_i32", "y", "G", "flag", "P", "freelist", "fmask", "nfree", "found", "ycert")
            }
            for lo in range(0, len(self.act), 16384):
                sub_act = self.act[lo : lo + 16384]
                st_sub = stage2_init_chains_pytorch(self.m_all[sub_act], self.w, C=self.C, generator=self.gen)
                for k in st_lists:
                    st_lists[k].append(st_sub[k])
            for k, parts in st_lists.items():
                setattr(self, k, torch.cat(parts, dim=0).contiguous())
            # Release m_all once cand_i32 is materialized to save ~1.2 GB VRAM on 412,786 puzzles
            del self.m_all

            # Handle rare luck where random initial draw on Stage 2 had P == 0 at move 0
            nA = len(self.act)
            fu0 = self.found.view(nA, self.C).long()
            fu0_pos = torch.where(fu0 >= 0, fu0, torch.full_like(fu0, 1 << 60))
            first0, fc0 = fu0_pos.min(dim=1)
            any0 = first0 < (1 << 60)
            if bool(any0.any()):
                a0 = self.act[any0]
                self.stopped[a0] = True
                self.stop_t[a0] = 0
                self.pred_board[a0] = self.ycert.view(nA, self.C, N_CELLS)[any0, fc0[any0]].long()
                self._compact()
        else:
            del self.m_all

        self.t = 0
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        self.elapsed = time.perf_counter() - t0

    def current_temp(self) -> float:
        if self.period <= 0 or self.temp_hi == self.temp_lo:
            return self.temp_hi
        return self.temp_hi * ((self.temp_lo / self.temp_hi) ** ((self.t % self.period) / self.period))

    def _compact(self) -> None:
        keep_u = ~self.stopped[self.act]
        keep = keep_u.repeat_interleave(self.C)
        for nm in ("cand_i32", "y", "G", "flag", "P", "freelist", "fmask", "nfree", "found", "ycert"):
            setattr(self, nm, getattr(self, nm)[keep].contiguous())
        self.act = self.act[keep_u]

    def _run_moves_chunk(self, nmoves: int) -> None:
        H = len(self.act) * self.C
        if H == 0 or nmoves <= 0:
            return
        cur_T = self.current_temp()
        if self.backend == "triton":
            # Launch in slices if H * 512 >= 2**31 (for 412,786 * 4 = 1.65M chains, 1.65M * 512 = 8.45e8 < 2.14e9, fits in 1 launch!)
            max_h = (1 << 30) // 512
            seed_val = self.seed * 1000003 + self.t // max(1, self.check)
            for lo in range(0, H, max_h):
                hi = min(H, lo + max_h)
                sub_h = hi - lo
                seed_t = torch.tensor([seed_val + (lo // max_h) * 7919], dtype=torch.int64, device=self.dev)
                temp_t = torch.full((sub_h,), cur_T, dtype=torch.float32, device=self.dev)
                _wsp_domain_moves[(sub_h,)](
                    self.w["peer_nbr"],
                    self.cand_i32[lo:hi],
                    self.y[lo:hi],
                    self.G[lo:hi],
                    self.flag[lo:hi],
                    self.P[lo:hi],
                    self.freelist[lo:hi],
                    self.fmask[lo:hi],
                    self.nfree[lo:hi],
                    temp_t,
                    self.found[lo:hi],
                    self.ycert[lo:hi],
                    seed_t,
                    nmoves,
                    self.t,
                    self.theta,
                    self.tau,
                    N=N_CELLS,
                    V=N_DIGITS,
                    D=N_PEERS,
                    NB=128,
                    VB=16,
                    DB=32,
                    num_warps=1,
                )
        else:
            # Pure PyTorch batched fallback (for CPU tests or non-Triton verification)
            T_ten = torch.tensor(cur_T, dtype=torch.float32, device=self.dev)
            thr_ten = torch.tensor(self.theta, dtype=torch.float32, device=self.dev)
            tau_ten = torch.tensor(self.tau, dtype=torch.float32, device=self.dev)
            for h_idx in range(H):
                st = {
                    "G": self.G[h_idx],
                    "y": self.y[h_idx],
                    "ycert": self.ycert[h_idx],
                    "flag": self.flag[h_idx],
                    "fmask": self.fmask[h_idx],
                    "free": self.freelist[h_idx],
                    "cand": self.cand_i32[h_idx],
                    "nbr": self.w["peer_nbr"],
                    "P": self.P[h_idx],
                    "T": T_ten,
                    "nf": self.nfree[h_idx],
                    "found": self.found[h_idx],
                    "thr": thr_ten,
                    "tau": tau_ten,
                    "move0": torch.tensor(self.t, dtype=torch.int32, device=self.dev),
                    "k": torch.tensor(0, dtype=torch.int32, device=self.dev),
                }
                for k in range(nmoves):
                    st["k"] = torch.tensor(k, dtype=torch.int32, device=self.dev)
                    u0_draw = torch.rand((), device=self.dev, generator=self.gen)
                    uf_draw = torch.rand(N_CELLS, device=self.dev, generator=self.gen)
                    uu_draw = torch.rand(N_DIGITS, device=self.dev, generator=self.gen)
                    wsp_assoc_move_pytorch(st, u0_in=u0_draw, uf_in=uf_draw, uu_in=uu_draw)
                self.P[h_idx] = st["P"]
                self.found[h_idx] = st["found"]

    def step_check(self, nmoves: int | None = None) -> None:
        step_m = self.check if nmoves is None else int(nmoves)
        if len(self.act) == 0 or step_m <= 0:
            return
        self._run_moves_chunk(step_m)
        self.t += step_m
        nA = len(self.act)
        new_bestP, new_best_board, anyc, cert_board = check_bookkeeping_one_puzzle_pytorch(
            self.P.view(nA, self.C),
            self.y.view(nA, self.C, N_CELLS),
            self.found.view(nA, self.C),
            self.ycert.view(nA, self.C, N_CELLS),
            self.bestP[self.act],
            self.best_board[self.act],
        )
        self.bestP[self.act] = new_bestP
        self.best_board[self.act] = new_best_board
        if bool(anyc.any()):
            a = self.act[anyc]
            fu = self.found.view(nA, self.C).long()
            fu_pos = torch.where(fu >= 0, fu, torch.full_like(fu, 1 << 60))
            first, _ = fu_pos.min(dim=1)
            self.stopped[a] = True
            self.stop_t[a] = first[anyc]
            self.pred_board[a] = cert_board[anyc]
        if len(self.act) > 0 and bool(self.stopped[self.act].float().mean() > 0.25):
            self._compact()

    def run(self) -> dict[str, Any]:
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)
        t0 = time.perf_counter() - self.elapsed
        with torch.no_grad():
            while self.t < self.max_moves and len(self.act) > 0:
                rem = min(self.check, self.max_moves - self.t)
                self.step_check(rem)
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)
        self.elapsed = time.perf_counter() - t0
        final_board = torch.where(self.stopped.unsqueeze(1), self.pred_board, self.best_board)
        steps_used = torch.where(
            self.stopped,
            self.stop_t,
            torch.full_like(self.stop_t, self.t),
        )
        return {
            "pred": final_board.cpu(),
            "stopped": self.stopped.cpu(),
            "stop_t": self.stop_t.cpu(),
            "steps_used": steps_used.cpu(),
            "prop_steps_used": self.prop_steps_used.cpu(),
            "stage1_solved": self.stage1_solved.cpu(),
            "stage2_entered": self.stage2_entered.cpu(),
            "moves_run": int(self.t),
            "max_moves": int(self.max_moves),
            "seconds": float(self.elapsed),
        }


# ===========================================================================
# Exact FullCounter (FC2) Operation Accounting for All 3 Sudoku WSP Models
# ===========================================================================
def count_wsp_sudoku_flops(full_counter_cls: Any) -> dict[str, Any]:
    """Count exact FC2 operation formulas for `WSP-Transformer`, `WSP-Det`, and `WSP-Assoc`."""
    x1 = torch.zeros((1, N_CELLS), dtype=torch.int64)
    x1[0, 0] = 1  # synthetic 1-clue puzzle on CPU (counts depend only on tensor shapes)
    w_inc = build_sudoku_incidence_tensors("cpu")

    def _cert_check(logits: torch.Tensor) -> torch.Tensor:
        pred = logits.argmax(dim=-1)
        oh = F.one_hot(pred, N_DIGITS).float()
        conf = 0.5 * (oh * torch.matmul(w_inc["peer_adj"], oh)).sum(dim=(1, 2))
        is_clue = x1 > 0
        clue_ok = ((~is_clue) | (pred == (x1 - 1))).all(dim=1)
        return (conf == 0.0) & clue_ok

    # 1. WSP-Transformer: flops(T) = setup_plus_cert + T * per_step
    def count_tf(steps: int) -> dict[str, Any]:
        w = build_wsp_transformer_weights(steps=steps, device="cpu")
        with torch.no_grad(), full_counter_cls() as fc:
            logits = wsp_transformer_forward(w, x1)
            _ = _cert_check(logits)
        return fc.summary()

    tf_0 = count_tf(0)
    tf_1 = count_tf(1)
    tf_formula = {
        "setup_total": int(tf_0["total"]),
        "per_step_total": int(tf_1["total"]) - int(tf_0["total"]),
        "setup_total_with_compare": int(tf_0["total_with_compare"]),
        "per_step_total_with_compare": int(tf_1["total_with_compare"]) - int(tf_0["total_with_compare"]),
        "step0_summary": tf_0,
        "step1_summary": tf_1,
    }

    # 2. WSP-Det: flops(P, R) = setup_plus_cert + P * per_prop_step + R * per_relax_step
    def count_det(prop_s: int, rel_s: int) -> dict[str, Any]:
        w = build_wsp_det_weights(prop_steps=prop_s, relax_steps=rel_s, device="cpu")
        with torch.no_grad(), full_counter_cls() as fc:
            logits = wsp_det_forward(w, x1)
            _ = _cert_check(logits)
        return fc.summary()

    det_00 = count_det(0, 0)
    det_10 = count_det(1, 0)
    det_11 = count_det(1, 1)
    det_formula = {
        "setup_plus_cert_total": int(det_00["total"]),
        "per_prop_step_total": int(det_10["total"]) - int(det_00["total"]),
        "per_relax_step_total": int(det_11["total"]) - int(det_10["total"]),
        "setup_plus_cert_total_with_compare": int(det_00["total_with_compare"]),
        "per_prop_step_total_with_compare": int(det_10["total_with_compare"]) - int(det_00["total_with_compare"]),
        "per_relax_step_total_with_compare": int(det_11["total_with_compare"]) - int(det_10["total_with_compare"]),
        "summary_p1_r1": det_11,
    }

    # 3. WSP-Assoc:
    #    - stage1_init_cert_per_puzzle + prop_steps_used * stage1_prop_step_per_puzzle
    #    - stage2_init_per_puzzle (C=4 initial board sampling + field/energy/focus init)
    #    - move_per_chain (1 move of `wsp_assoc_move_pytorch`)
    #    - check_per_puzzle (1 check bookkeeping on 1 puzzle across C=4 chains)
    w_assoc = build_wsp_assoc_weights(prop_steps=16, device="cpu")

    def count_s1_at(k_prop: int) -> dict[str, Any]:
        with torch.no_grad(), full_counter_cls() as fc:
            is_clue = (x1 > 0).unsqueeze(-1)
            clue_oh = F.one_hot((x1 - 1).clamp(min=0), N_DIGITS).float() * is_clue.float()
            mb = torch.where(is_clue, clue_oh, torch.ones_like(clue_oh))
            for _ in range(k_prop):
                m_prev = mb
                mb = wsp_assoc_prop_step_forward(w_assoc, x1, m_prev)
                _ = (mb == m_prev).all(dim=(1, 2))
                _ = (mb.sum(dim=-1) == 1.0).all(dim=1)
            yb = mb.argmax(dim=-1)
            y_oh = F.one_hot(yb, N_DIGITS).float()
            pb = 0.5 * (y_oh * torch.matmul(w_assoc["peer_adj"], y_oh)).sum(dim=(1, 2))
            _ = (pb == 0.0) & (mb.sum(dim=-1) == 1.0).all(dim=1) & ((~is_clue.squeeze(-1)) | (yb == (x1 - 1))).all(dim=1)
        return fc.summary()

    s1_0 = count_s1_at(0)
    s1_1 = count_s1_at(1)
    s1_step_summary = {
        "total": int(s1_1["total"]) - int(s1_0["total"]),
        "total_with_compare": int(s1_1["total_with_compare"]) - int(s1_0["total_with_compare"]),
        "classes": {
            k: int(s1_1["classes"][k]) - int(s1_0["classes"][k])
            for k in s1_0["classes"]
        },
    }

    m_dummy = torch.ones((1, N_CELLS, N_DIGITS), dtype=torch.float32)
    g_cpu = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad(), full_counter_cls() as fc_s2_init:
        st_init = stage2_init_chains_pytorch(m_dummy, w_assoc, C=4, generator=g_cpu)
        fu0 = st_init["found"].view(1, 4).long()
        fu0_pos = torch.where(fu0 >= 0, fu0, torch.full_like(fu0, 1 << 60))
        first0, _ = fu0_pos.min(dim=1)
        _ = first0 < (1 << 60)
    s2_init_sum = fc_s2_init.summary()

    st_chain = make_synthetic_assoc_chain_state(w_assoc, seed=0, device="cpu")
    with torch.no_grad(), full_counter_cls() as fc_move:
        wsp_assoc_move_pytorch(st_chain)
    move_sum = fc_move.summary()

    bestP = torch.tensor([100.0])
    bestB = torch.zeros((1, N_CELLS), dtype=torch.int64)
    with torch.no_grad(), full_counter_cls() as fc_chk:
        _, _, anyc_chk, _ = check_bookkeeping_one_puzzle_pytorch(
            st_init["P"].view(1, 4),
            st_init["y"].view(1, 4, N_CELLS),
            st_init["found"].view(1, 4),
            st_init["ycert"].view(1, 4, N_CELLS),
            bestP,
            bestB,
        )
        _ = anyc_chk.float().mean() > 0.25
    chk_sum = fc_chk.summary()

    return {
        "WSP-Transformer": tf_formula,
        "WSP-Det": det_formula,
        "WSP-Assoc": {
            "C": 4,
            "check": 4096,
            "stage1_init_cert_per_puzzle": s1_0,
            "stage1_prop_step_per_puzzle": s1_step_summary,
            "stage1_1step_full_summary": s1_1,
            "stage2_init_per_puzzle": s2_init_sum,
            "move_per_chain": move_sum,
            "check_per_puzzle": chk_sum,
            "formula": (
                "stage1_init_cert + prop_steps_used * stage1_prop_step + "
                "stage2_entered * (stage2_init + steps * C * move_per_chain + ceil(steps / check) * check_per_puzzle)"
            ),
        },
    }


def compute_wsp_assoc_puzzle_flops(
    steps_used: np.ndarray | torch.Tensor,
    stage2_entered: np.ndarray | torch.Tensor,
    assoc_counts: dict[str, Any],
    *,
    prop_steps_used: np.ndarray | torch.Tensor | None = None,
    with_compare: bool = False,
) -> np.ndarray:
    """Rebuild exact per-puzzle FC2 FLOPs (`total` or `total_with_compare`) from `WSPAssocSearch.run()`."""
    key = "total_with_compare" if with_compare else "total"
    s1_init = float(assoc_counts["stage1_init_cert_per_puzzle"][key])
    s1_step = float(assoc_counts["stage1_prop_step_per_puzzle"][key])
    s2_init = float(assoc_counts["stage2_init_per_puzzle"][key])
    mv = float(assoc_counts["move_per_chain"][key])
    chk = float(assoc_counts["check_per_puzzle"][key])
    C = int(assoc_counts["C"])
    check_int = int(assoc_counts["check"])

    steps = np.asarray(steps_used, dtype=np.float64)
    entered = np.asarray(stage2_entered, dtype=bool)
    p_steps = np.full_like(steps, 16.0) if prop_steps_used is None else np.asarray(prop_steps_used, dtype=np.float64)
    n_checks = np.ceil(steps / float(check_int))
    return s1_init + p_steps * s1_step + entered.astype(np.float64) * (s2_init + steps * C * mv + n_checks * chk)
