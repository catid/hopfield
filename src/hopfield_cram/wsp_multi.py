"""Weight-Space Programming (WSP) baselines and ablations for multi-solution CSPs.

Implements genuine Weight-Space Programs following EFFL Lab's upstream WSP contract
(`reproduction/baselines/code/Weight-Space-Programming/wsp/free_model.py`):
each model consists of a pure rule-programmed weight builder ``build_*(...) -> dict[str, Tensor]``
(zero learned parameters, ``Fit time = "none"``) paired with a static branchless PyTorch
tensor program ``ARCH_*_SOURCE`` (`forward(w, x)`) passing ``free_model.static_problems``
with zero AST violations.

Covered multi-solution benchmark families:
1. N-Queens (``n = 8, 10``):
   - ``WSP-NQ-Det``: Deterministic coordinate continuous-Hopfield tensor relaxation
     (`2` distinct programmed energy coefficients `{1.0, -0.5}`), demonstrating single-mode
     deterministic limits and mode collapse on ``Cov@20``.
   - ``WSP-NQ-Arbitrary``: Batched ``C = 20`` stochastic focused associative heat-bath
     tensor updates over unconstrained binary boards `{0, 1}^{n^2}` (`2` distinct programmed
     energy coefficients `{1.0, -0.5}`, exact `E <= -0.5 * n` certificate).
   - ``WSP-NQ-Permutation``: Batched ``C = 20`` row-column permutation-preserving 2-opt swap
     associative tensor updates (`1` distinct programmed energy coefficient `{1.0}`, explicit
     rook-permutation proposal prior disclosure).
2. Graph Coloring (``n = 8, 10``, ``C = 3`` colors):
   - ``WSP-GC-Det``: Deterministic Gauss-Seidel Potts associative relaxation (`1` distinct
     programmed coefficient `{1.0}`), demonstrating mode collapse on ``Cov@20``.
   - ``WSP-GC-Associative``: Batched ``C = 20`` stochastic focused Gumbel-Potts associative
     updates (`1` distinct programmed coefficient `{1.0}`), with optional pure-tensor
     first-appearance color canonicalizer inside ``arch.forward(w, edges)``.
3. Generalized Queen+Leaper Families (``N = 13`` ``queen-plus-zebra`` ``[2, 3]`` &
   ``queen-plus-2-5-leaper`` ``[2, 5]``, ``4,096`` fresh inputs x 3 seeds):
   - ``WSP-Leaper-Det``: Deterministic continuous-Hopfield tensor relaxation (`0` training boards).
   - ``WSP-Leaper-Arbitrary`` vs ``WSP-Leaper-Permutation``: Fixed-cardinality arbitrary-cell
     swap vs row-column permutation-preserving 2-opt swap WSP tensor programs (`1` distinct
     programmed coefficient `{1.0}`, `0` training boards) evaluated from shared initial boards
     at source-calibrated matched GPU wall-clock solve time.
"""
from __future__ import annotations

import ast
import functools
import math
import sys
import time
import types
from typing import Any

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Upstream WSP Free-Model Static AST Contract & Weight Disclosure Helpers
# ---------------------------------------------------------------------------

ARCH_IMPORTS = {"torch", "math"}
ARCH_FORBIDDEN_NODES = (
    ast.If,
    ast.IfExp,
    ast.While,
    ast.Assert,
    ast.BoolOp,
    ast.Try,
    ast.With,
    ast.Match,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
    ast.ClassDef,
    ast.AsyncFunctionDef,
)
ARCH_FORBIDDEN_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "input", "globals", "locals",
    "getattr", "setattr", "vars", "dict", "set", "str", "repr", "int", "float",
    "bool", "hash", "print", "sorted", "min", "max", "any", "all", "map", "filter",
    "iter", "next",
}
ARCH_FORBIDDEN_ATTRS = {
    "tolist", "item", "numpy", "load", "save", "data_ptr", "storage", "apply_", "map_",
}


def static_problems(source: str, signature: str = "forward(w, x)") -> list[str]:
    """Exact mirror of upstream `wsp.free_model.static_problems` AST checker."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"arch.py syntax error: {exc}"]
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            problems += [
                f"arch.py: import {a.name} not allowed"
                for a in node.names
                if a.name not in ARCH_IMPORTS and a.name != "torch.nn.functional"
            ]
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ("torch", "torch.nn", "torch.nn.functional", "math"):
                problems.append(f"arch.py: from {node.module} import not allowed (line {node.lineno})")
        elif isinstance(node, ARCH_FORBIDDEN_NODES):
            problems.append(f"arch.py: {type(node).__name__} not allowed (line {node.lineno})")
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            if any(g.ifs for g in node.generators):
                problems.append(f"arch.py: comprehension condition not allowed (line {node.lineno})")
        elif isinstance(node, ast.Name) and (node.id in ARCH_FORBIDDEN_NAMES or node.id.startswith("__")):
            problems.append(f"arch.py: name {node.id} not allowed (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and (node.attr in ARCH_FORBIDDEN_ATTRS or node.attr.startswith("__")):
            problems.append(f"arch.py: attribute .{node.attr} not allowed (line {node.lineno})")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "forward" for n in tree.body):
        problems.append(f"arch.py: no top-level {signature}")
    return problems[:20]


@functools.lru_cache(maxsize=16)
def compile_wsp_arch(source: str, module_name: str = "wsp_arch") -> types.ModuleType:
    """Verify `source` has zero static AST problems and load it as a Python module."""
    problems = static_problems(source)
    if problems:
        raise ValueError(f"WSP architecture {module_name} failed static AST check: {problems}")
    full_name = f"hopfield_cram._dyn_{module_name}"
    mod = types.ModuleType(full_name)
    sys.modules[full_name] = mod
    exec(compile(source, f"<{full_name}>", "exec"), mod.__dict__)
    return mod


def weight_disclosure(
    w: dict[str, Tensor],
    *,
    energy_keys: tuple[str, ...],
    prior_disclosure: str,
) -> dict[str, Any]:
    """Compute exact parameter / programmed-coefficient disclosures per repo convention."""
    energy_vals: set[float] = set()
    all_vals: set[float] = set()
    materialized_entries = 0
    materialized_nonzero = 0
    energy_entries = 0
    energy_nonzero = 0
    for key, tensor in w.items():
        t_cpu = tensor.detach().cpu()
        numel = int(t_cpu.numel())
        nz_mask = t_cpu != 0
        nz = int(nz_mask.sum().item())
        materialized_entries += numel
        materialized_nonzero += nz
        if nz > 0:
            uniq = torch.unique(t_cpu[nz_mask].to(torch.float64)).tolist()
            all_vals.update(round(float(v), 12) for v in uniq)
            if key in energy_keys:
                energy_vals.update(round(float(v), 12) for v in uniq)
        if key in energy_keys:
            energy_entries += numel
            energy_nonzero += nz
    return {
        "params_learned": 0,
        "distinct_programmed_coefficients": len(energy_vals),
        "distinct_programmed_values": sorted(energy_vals),
        "distinct_nonzero_all_w_tensors": len(all_vals),
        "energy_entries": energy_entries,
        "energy_nonzero": energy_nonzero,
        "materialized_entries": materialized_entries,
        "materialized_nonzero": materialized_nonzero,
        "fit_seconds": None,
        "fit_display": "none",
        "prior_disclosure": prior_disclosure,
    }


def _sync(device: torch.device | str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Rule-Programmed Attack Matrices (N-Queens & Generalized Queen+Leaper)
# ---------------------------------------------------------------------------

def build_queen_attack_matrix(
    n: int,
    leaper: tuple[int, int] | None = None,
    *,
    diagonal_only: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build symmetric binary `(n^2, n^2)` attack adjacency from rule offsets."""
    if n < 1:
        raise ValueError("Board size n must be >= 1")
    dev = torch.device(device)
    idx = torch.arange(n * n, device=dev)
    r, col = idx // n, idx % n
    dr = (r[:, None] - r[None, :]).abs()
    dc = (col[:, None] - col[None, :]).abs()
    diag = (dr == dc) & (dr > 0)
    if diagonal_only:
        attack = diag
    else:
        line = ((dr == 0) ^ (dc == 0))
        attack = line | diag
    if leaper is not None:
        la, lb = abs(int(leaper[0])), abs(int(leaper[1]))
        if la == 0 and lb == 0:
            raise ValueError("Leaper offset must be nonzero")
        leap = ((dr == la) & (dc == lb)) | ((dr == lb) & (dc == la))
        attack = attack | leap
    attack.fill_diagonal_(False)
    return attack.to(dtype)


# ---------------------------------------------------------------------------
# 1A. WSP-NQ-Det: Deterministic Coordinate Continuous-Hopfield Relaxation
# ---------------------------------------------------------------------------

ARCH_NQUEENS_DET_SOURCE = """import torch


def forward_with_certificates(w, givens):
    W = w["W_attack"]
    bias = w["bias"]
    betas = w["betas"]
    alphas = w["alphas"]
    tau_thresh = w["tau_thresh"]
    giv = givens > 0
    x_f = giv.to(W.dtype)
    R = x_f.shape[0]
    c = W.shape[0]
    p = x_f.clone()
    field = p @ W + bias[None, :]
    ones = torch.ones(R, device=W.device, dtype=W.dtype)
    for s in range(betas.shape[0]):
        beta = betas[s]
        alpha = alphas[s]
        for i in range(c):
            target = torch.where(giv[:, i], ones, torch.sigmoid(-beta * field[:, i]))
            dp = alpha * (target - p[:, i])
            p[:, i] = p[:, i] + dp
            field = field + dp[:, None] * W[i][None, :]
    board = (p >= 0.5) | giv
    bf = board.to(W.dtype)
    attack = bf @ W
    energy = 0.5 * (bf * attack).sum(dim=1) + (bf * bias[None, :]).sum(dim=1)
    stopped = energy <= tau_thresh
    moves = torch.full((R,), betas.shape[0] * c, device=W.device, dtype=torch.int64)
    return board, stopped, energy, moves


def forward(w, x):
    board, stopped, energy, moves = forward_with_certificates(w, x)
    return board
"""


def build_nqueens_det_wsp(
    n: int,
    *,
    passes: int = 12,
    alpha: float = 0.5,
    beta_start: float = 2.0,
    beta_end: float = 24.0,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Programmed deterministic Hopfield N-Queens weights (`2` distinct energy values)."""
    dev = torch.device(device)
    c = n * n
    W_attack = build_queen_attack_matrix(n, device=dev, dtype=torch.float32)
    bias = torch.full((c,), -0.5, device=dev, dtype=torch.float32)
    if passes < 1:
        raise ValueError("passes must be >= 1")
    if passes == 1:
        betas = torch.tensor([beta_end], device=dev, dtype=torch.float32)
    else:
        ratio = beta_end / beta_start
        ks = torch.arange(passes, device=dev, dtype=torch.float32) / float(passes - 1)
        betas = beta_start * (ratio ** ks)
    alphas = torch.full((passes,), alpha, device=dev, dtype=torch.float32)
    tau_thresh = torch.tensor(-0.5 * n + 1e-3, device=dev, dtype=torch.float32)
    return {
        "W_attack": W_attack,
        "bias": bias,
        "betas": betas,
        "alphas": alphas,
        "tau_thresh": tau_thresh,
    }


# ---------------------------------------------------------------------------
# 1B. WSP-NQ-Arbitrary: Batched 20-Chain Stochastic Focused Heat-Bath WSP
# ---------------------------------------------------------------------------

ARCH_NQUEENS_ARBITRARY_SOURCE = """import torch


def step_chunk(W_attack, bias_val, tau_thresh, free, x, attack, energy, best_x, best_e, stopped, moves, u_cell, u_flip, inv_temps):
    R = x.shape[0]
    a = torch.arange(R, device=x.device)
    has_free = free.any(dim=1)
    for k in range(inv_temps.shape[0]):
        active = ~stopped
        can_move = active & has_free
        unhappy = free & torch.where(x > 0.5, attack > 0.5, attack < 0.5)
        has_unhappy = unhappy.any(dim=1, keepdim=True)
        pool = torch.where(has_unhappy, unhappy, free)
        cell = torch.where(pool, u_cell[k], -1.0).argmax(dim=1)
        dfield = attack[a, cell] + bias_val
        prob1 = torch.sigmoid(-dfield * inv_temps[k])
        old_val = x[a, cell]
        proposed = (u_flip[k] < prob1).to(x.dtype)
        new_val = torch.where(can_move, proposed, old_val)
        delta = new_val - old_val
        energy = energy + delta * dfield
        attack = attack + delta[:, None] * W_attack[cell]
        x[a, cell] = new_val
        moves = moves + active.to(torch.int64)
        better = active & (energy < best_e)
        best_x = torch.where(better[:, None], x, best_x)
        best_e = torch.where(better, energy, best_e)
        stopped = stopped | (energy <= tau_thresh)
    return x, attack, energy, best_x, best_e, stopped, moves


def forward_with_certificates(w, givens):
    W_attack = w["W_attack"]
    bias_val = w["bias_scalar"]
    tau_thresh = w["tau_thresh"]
    inv_temps = w["inv_temps"]
    giv = givens > 0
    free = ~giv
    x = giv.to(W_attack.dtype)
    attack = x @ W_attack
    energy = 0.5 * (x * attack).sum(dim=1) + bias_val * x.sum(dim=1)
    best_x = x.clone()
    best_e = energy.clone()
    stopped = energy <= tau_thresh
    R, c = x.shape
    moves = torch.zeros(R, device=x.device, dtype=torch.int64)
    steps = inv_temps.shape[0]
    u_cell = torch.rand((steps, R, c), device=x.device, dtype=W_attack.dtype)
    u_flip = torch.rand((steps, R), device=x.device, dtype=W_attack.dtype)
    x, attack, energy, best_x, best_e, stopped, moves = step_chunk(
        W_attack, bias_val, tau_thresh, free, x, attack, energy, best_x, best_e, stopped, moves, u_cell, u_flip, inv_temps
    )
    out_x = torch.where(stopped[:, None], x, best_x) > 0.5
    out_e = torch.where(stopped, energy, best_e)
    return out_x, stopped, out_e, moves


def forward(w, x):
    out_x, stopped, out_e, moves = forward_with_certificates(w, x)
    return out_x
"""


def build_nqueens_arbitrary_wsp(
    n: int,
    *,
    cap: int = 20000,
    temperature: float = 0.3,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Programmed N-Queens unconstrained binary WSP weights (`2` distinct energy values `{1.0, -0.5}`)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if cap < 0:
        raise ValueError("cap must be >= 0")
    dev = torch.device(device)
    W_attack = build_queen_attack_matrix(n, device=dev, dtype=torch.float32)
    bias_scalar = torch.tensor(-0.5, device=dev, dtype=torch.float32)
    tau_thresh = torch.tensor(-0.5 * n + 1e-3, device=dev, dtype=torch.float32)
    inv_temps = torch.full((cap,), 1.0 / float(temperature), device=dev, dtype=torch.float32)
    return {
        "W_attack": W_attack,
        "bias_scalar": bias_scalar,
        "tau_thresh": tau_thresh,
        "inv_temps": inv_temps,
    }


# ---------------------------------------------------------------------------
# 1C & 3B. WSP Permutation 2-Opt Swap Architecture (N-Queens & Queen+Leaper)
# ---------------------------------------------------------------------------

ARCH_PERM_SWAP_SOURCE = """import torch


def init_permutation_boards(givens, n_board, u_init):
    R = givens.shape[0]
    c = n_board * n_board
    g = givens.view(R, n_board, n_board)
    fixed = g.any(dim=2)
    used = g.any(dim=1)
    given_col = g.to(torch.int64).argmax(dim=2)
    row_rank = (~fixed).to(torch.int64).cumsum(dim=1) - 1
    scores = torch.where(used, -1.0, u_init)
    perm = scores.argsort(dim=1, descending=True)
    cols = torch.where(fixed, given_col, perm.gather(1, row_rank.clamp_min(0)))
    cells = torch.arange(n_board, device=givens.device)[None, :] * n_board + cols
    x = torch.zeros((R, c), device=givens.device, dtype=givens.dtype).scatter_(1, cells, torch.ones_like(cells, dtype=givens.dtype))
    return x, cols, fixed


def step_chunk(W_int, fixed, x, cols, counts, conflict, best_x, best_conf, stopped, moves, accepted, u_row, u_acc, inv_temps):
    R, n = cols.shape
    a = torch.arange(R, device=x.device)
    can_swap_rows = ((~fixed).sum(dim=1) >= 2)
    false_vec = torch.zeros(R, device=x.device, dtype=x.dtype)
    true_vec = torch.ones(R, device=x.device, dtype=x.dtype)
    for k in range(inv_temps.shape[0]):
        active = ~stopped
        can_swap = active & can_swap_rows
        score = torch.where(fixed, -1.0, u_row[k])
        rows = score.topk(2, dim=1).indices
        r = rows[:, 0]
        s = rows[:, 1]
        cr = cols[a, r]
        cs = cols[a, s]
        i = r * n + cr
        j = s * n + cs
        kk = r * n + cs
        ll = s * n + cr
        dc = (
            counts[a, kk] + counts[a, ll] - counts[a, i] - counts[a, j]
            + W_int[i, j] + W_int[kk, ll]
            - W_int[i, kk] - W_int[i, ll] - W_int[j, kk] - W_int[j, ll]
        )
        prob = torch.exp(-dc.clamp_min(0).to(torch.float32) * inv_temps[k])
        take = can_swap & ((dc <= 0) | (u_acc[k] < prob))
        cols[a, r] = torch.where(take, cs, cr)
        cols[a, s] = torch.where(take, cr, cs)
        x[a, i] = torch.where(take, false_vec, x[a, i])
        x[a, j] = torch.where(take, false_vec, x[a, j])
        x[a, kk] = torch.where(take, true_vec, x[a, kk])
        x[a, ll] = torch.where(take, true_vec, x[a, ll])
        counts = counts + take[:, None].to(torch.int32) * (W_int[kk] + W_int[ll] - W_int[i] - W_int[j])
        conflict = conflict + take.to(torch.int32) * dc
        accepted = accepted + take.to(torch.int64)
        moves = moves + active.to(torch.int64)
        better = active & (conflict < best_conf)
        best_x = torch.where(better[:, None], x, best_x)
        best_conf = torch.where(better, conflict, best_conf)
        stopped = stopped | (conflict == 0)
    return x, cols, counts, conflict, best_x, best_conf, stopped, moves, accepted


def forward_from_initial(w, givens, x_init):
    W_int = w["W_attack_i32"]
    W_f = w["W_attack"]
    inv_temps = w["inv_temps"]
    n_board = w["row_arange"].shape[0]
    giv = givens > 0
    R = giv.shape[0]
    x = x_init > 0
    fixed = giv.view(R, n_board, n_board).any(dim=2)
    cols = x.view(R, n_board, n_board).to(torch.int64).argmax(dim=2)
    counts = (x.to(torch.float32) @ W_f).to(torch.int32)
    conflict = (x.to(torch.int32) * counts).sum(dim=1, dtype=torch.int32) // 2
    best_x = x.clone()
    best_conf = conflict.clone()
    stopped = conflict == 0
    moves = torch.zeros(R, device=giv.device, dtype=torch.int64)
    accepted = torch.zeros(R, device=giv.device, dtype=torch.int64)
    steps = inv_temps.shape[0]
    u_row = torch.rand((steps, R, n_board), device=giv.device, dtype=torch.float32)
    u_acc = torch.rand((steps, R), device=giv.device, dtype=torch.float32)
    x, cols, counts, conflict, best_x, best_conf, stopped, moves, accepted = step_chunk(
        W_int, fixed, x, cols, counts, conflict, best_x, best_conf, stopped, moves, accepted, u_row, u_acc, inv_temps
    )
    out_x = torch.where(stopped[:, None], x, best_x)
    out_conf = torch.where(stopped, conflict, best_conf)
    return out_x, stopped, out_conf.to(torch.float32), moves, out_conf, accepted


def forward_with_certificates(w, givens):
    n_board = w["row_arange"].shape[0]
    giv = givens > 0
    R = giv.shape[0]
    u_init = torch.rand((R, n_board), device=giv.device, dtype=torch.float32)
    x, cols, fixed = init_permutation_boards(giv, n_board, u_init)
    return forward_from_initial(w, giv, x)


def forward(w, x):
    out_x, stopped, energy, moves, out_conf, accepted = forward_with_certificates(w, x)
    return out_x
"""


# ---------------------------------------------------------------------------
# 3A. WSP Arbitrary Fixed-Cardinality Swap Architecture (Generalized Queen+Leaper)
# ---------------------------------------------------------------------------

ARCH_ARB_SWAP_SOURCE = """import torch


def init_arbitrary_boards(givens, q_count, u_init):
    R, c = givens.shape
    scores = torch.where(givens, -1.0, u_init)
    order = scores.argsort(dim=1, descending=True)
    rank = torch.zeros_like(order).scatter_(1, order, torch.arange(c, device=givens.device)[None, :].expand(R, c))
    needed = q_count - givens.to(torch.int64).sum(dim=1)
    x = givens | (rank < needed[:, None])
    return x


def step_chunk(W_int, free, x, counts, conflict, best_x, best_conf, stopped, moves, accepted, u_cell, u_acc, inv_temps):
    R = x.shape[0]
    a = torch.arange(R, device=x.device)
    false_vec = torch.zeros(R, device=x.device, dtype=x.dtype)
    true_vec = torch.ones(R, device=x.device, dtype=x.dtype)
    for k in range(inv_temps.shape[0]):
        active = ~stopped
        occupied = x & free
        empty = ~x
        can_swap = active & occupied.any(dim=1) & empty.any(dim=1)
        i = torch.where(occupied, u_cell[k], -1.0).argmax(dim=1)
        j = torch.where(empty, u_cell[k], -1.0).argmax(dim=1)
        dc = counts[a, j] - counts[a, i] - W_int[i, j]
        prob = torch.exp(-dc.clamp_min(0).to(torch.float32) * inv_temps[k])
        take = can_swap & ((dc <= 0) | (u_acc[k] < prob))
        x[a, i] = torch.where(take, false_vec, x[a, i])
        x[a, j] = torch.where(take, true_vec, x[a, j])
        counts = counts + take[:, None].to(torch.int32) * (W_int[j] - W_int[i])
        conflict = conflict + take.to(torch.int32) * dc
        accepted = accepted + take.to(torch.int64)
        moves = moves + active.to(torch.int64)
        better = active & (conflict < best_conf)
        best_x = torch.where(better[:, None], x, best_x)
        best_conf = torch.where(better, conflict, best_conf)
        stopped = stopped | (conflict == 0)
    return x, counts, conflict, best_x, best_conf, stopped, moves, accepted


def forward_from_initial(w, givens, x_init):
    W_int = w["W_attack_i32"]
    W_f = w["W_attack"]
    inv_temps = w["inv_temps"]
    giv = givens > 0
    free = ~giv
    R, c = giv.shape
    x = x_init > 0
    counts = (x.to(torch.float32) @ W_f).to(torch.int32)
    conflict = (x.to(torch.int32) * counts).sum(dim=1, dtype=torch.int32) // 2
    best_x = x.clone()
    best_conf = conflict.clone()
    stopped = conflict == 0
    moves = torch.zeros(R, device=giv.device, dtype=torch.int64)
    accepted = torch.zeros(R, device=giv.device, dtype=torch.int64)
    steps = inv_temps.shape[0]
    u_cell = torch.rand((steps, R, c), device=giv.device, dtype=torch.float32)
    u_acc = torch.rand((steps, R), device=giv.device, dtype=torch.float32)
    x, counts, conflict, best_x, best_conf, stopped, moves, accepted = step_chunk(
        W_int, free, x, counts, conflict, best_x, best_conf, stopped, moves, accepted, u_cell, u_acc, inv_temps
    )
    out_x = torch.where(stopped[:, None], x, best_x)
    out_conf = torch.where(stopped, conflict, best_conf)
    return out_x, stopped, out_conf.to(torch.float32), moves, out_conf, accepted


def forward_with_certificates(w, givens):
    q_count = w["row_arange"].shape[0]
    giv = givens > 0
    R, c = giv.shape
    u_init = torch.rand((R, c), device=giv.device, dtype=torch.float32)
    x = init_arbitrary_boards(giv, q_count, u_init)
    return forward_from_initial(w, giv, x)


def forward(w, x):
    out_x, stopped, energy, moves, out_conf, accepted = forward_with_certificates(w, x)
    return out_x
"""


def build_queen_leaper_wsp(
    n: int = 13,
    leaper: tuple[int, int] | None = None,
    *,
    cap: int = 20000,
    temp_start: float = 1.0,
    end_ratio: float = 0.01,
    diagonal_only: bool = False,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Programmed generalized N-Queens / Queen+Leaper WSP weights (`1` distinct energy coefficient `{1.0}`)."""
    if temp_start <= 0 or end_ratio <= 0:
        raise ValueError("temp_start and end_ratio must be positive")
    if cap < 0:
        raise ValueError("cap must be >= 0")
    dev = torch.device(device)
    W_attack = build_queen_attack_matrix(n, leaper=leaper, diagonal_only=diagonal_only, device=dev, dtype=torch.float32)
    W_attack_i32 = W_attack.to(torch.int32)
    if cap == 0:
        inv_temps = torch.empty(0, device=dev, dtype=torch.float32)
    elif cap == 1:
        inv_temps = torch.tensor([1.0 / temp_start], device=dev, dtype=torch.float32)
    else:
        steps = torch.arange(cap, device=dev, dtype=torch.float32) / float(cap - 1)
        temps = float(temp_start) * (float(end_ratio) ** steps)
        inv_temps = 1.0 / temps
    row_arange = torch.arange(n, device=dev, dtype=torch.int64)
    return {
        "W_attack": W_attack,
        "W_attack_i32": W_attack_i32,
        "inv_temps": inv_temps,
        "row_arange": row_arange,
    }


def build_nqueens_perm_wsp(
    n: int,
    *,
    cap: int = 20000,
    temp_start: float = 1.0,
    end_ratio: float = 0.05,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Programmed permutation-swap N-Queens WSP weights (`1` distinct energy coefficient `{1.0}`)."""
    return build_queen_leaper_wsp(
        n=n,
        leaper=None,
        cap=cap,
        temp_start=temp_start,
        end_ratio=end_ratio,
        diagonal_only=False,
        device=device,
    )


def build_queen_leaper_det_wsp(
    n: int = 13,
    leaper: tuple[int, int] | None = (2, 3),
    *,
    passes: int = 12,
    alpha: float = 0.5,
    beta_start: float = 2.0,
    beta_end: float = 24.0,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Programmed deterministic Hopfield relaxation weights for Generalized Queen+Leaper."""
    dev = torch.device(device)
    c = n * n
    W_attack = build_queen_attack_matrix(n, leaper=leaper, device=dev, dtype=torch.float32)
    bias = torch.full((c,), -0.5, device=dev, dtype=torch.float32)
    ks = torch.arange(passes, device=dev, dtype=torch.float32) / float(max(passes - 1, 1))
    betas = beta_start * ((beta_end / beta_start) ** ks)
    alphas = torch.full((passes,), alpha, device=dev, dtype=torch.float32)
    tau_thresh = torch.tensor(-0.5 * n + 1e-3, device=dev, dtype=torch.float32)
    return {
        "W_attack": W_attack,
        "bias": bias,
        "betas": betas,
        "alphas": alphas,
        "tau_thresh": tau_thresh,
    }


# ---------------------------------------------------------------------------
# 2A & 2B. Graph Coloring WSP: Deterministic Potts & Batched 20-Chain Associative Potts
# ---------------------------------------------------------------------------

ARCH_GC_DET_SOURCE = """import torch


def canonicalize_colorings(y, color_ids):
    R, n = y.shape
    v_pos = torch.arange(n, device=y.device, dtype=torch.int64)
    match = y[:, :, None] == color_ids[None, None, :]
    first = torch.where(match, v_pos[None, :, None], n).amin(dim=1)
    order = (first * 3 + color_ids[None, :]).argsort(dim=1)
    lut = torch.zeros_like(order).scatter_(1, order, color_ids[None, :].expand_as(order))
    return lut.gather(1, y)


def build_adj(edges, pair_u, pair_v, n_verts):
    B = edges.shape[0]
    A = torch.zeros((B, n_verts, n_verts), device=edges.device, dtype=edges.dtype)
    A[:, pair_u, pair_v] = edges
    A[:, pair_v, pair_u] = edges
    return A


def forward_with_certificates(w, edges):
    pair_u = w["pair_u"]
    pair_v = w["pair_v"]
    color_eye = w["color_eye"]
    color_ids = w["color_ids"]
    alphas = w["alphas"]
    n_verts = w["vert_arange"].shape[0]
    edges_f = edges.to(color_eye.dtype)
    A = build_adj(edges_f, pair_u, pair_v, n_verts)
    B = A.shape[0]
    P = torch.zeros((B, n_verts, 3), device=A.device, dtype=A.dtype)
    for s in range(alphas.shape[0]):
        alpha = alphas[s]
        for i in range(n_verts):
            field_i = torch.einsum("bj,bjc->bc", A[:, i, :], P) @ color_eye
            best_c = field_i.argmin(dim=1)
            target = color_eye[best_c]
            P[:, i, :] = (1.0 - alpha) * P[:, i, :] + alpha * target
    y = P.argmax(dim=2)
    y_canon = canonicalize_colorings(y, color_ids)
    oh = color_eye[y_canon]
    g = (A @ oh) @ color_eye
    own = g.gather(2, y_canon[:, :, None])[:, :, 0]
    energy = 0.5 * own.sum(dim=1)
    stopped = energy <= 1e-3
    moves = torch.full((B,), alphas.shape[0] * n_verts, device=A.device, dtype=torch.int64)
    return y_canon, stopped, energy, moves


def forward(w, x):
    y_canon, stopped, energy, moves = forward_with_certificates(w, x)
    return y_canon
"""


ARCH_GC_ASSOC_SOURCE = """import torch


def canonicalize_colorings(y, color_ids):
    R, n = y.shape
    v_pos = torch.arange(n, device=y.device, dtype=torch.int64)
    match = y[:, :, None] == color_ids[None, None, :]
    first = torch.where(match, v_pos[None, :, None], n).amin(dim=1)
    order = (first * 3 + color_ids[None, :]).argsort(dim=1)
    lut = torch.zeros_like(order).scatter_(1, order, color_ids[None, :].expand_as(order))
    return lut.gather(1, y)


def build_adj(edges, pair_u, pair_v, n_verts):
    B = edges.shape[0]
    A = torch.zeros((B, n_verts, n_verts), device=edges.device, dtype=edges.dtype)
    A[:, pair_u, pair_v] = edges
    A[:, pair_v, pair_u] = edges
    return A


def step_chunk(A_act, color_eye, focus_thr, y, g, energy, best_y, best_e, stopped, moves, u_vert, u_gumbel, inv_temps):
    R, n = y.shape
    a = torch.arange(R, device=y.device)
    for k in range(inv_temps.shape[0]):
        active = ~stopped
        own = g.gather(2, y[:, :, None])[:, :, 0]
        hot = own > focus_thr
        has_hot = hot.any(dim=1, keepdim=True)
        scores = torch.where(has_hot, torch.where(hot, u_vert[k], -1.0), u_vert[k])
        v = scores.argmax(dim=1)
        local = g[a, v]
        u_clamped = u_gumbel[k].clamp(1e-30, 1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(u_clamped))
        proposed = (-local * inv_temps[k] + gumbel).argmax(dim=1)
        old_c = y[a, v]
        new_c = torch.where(active, proposed, old_c)
        energy = energy + local[a, new_c] - local[a, old_c]
        adj_col = A_act[a, :, v]
        delta_c = color_eye[new_c] - color_eye[old_c]
        g = g + adj_col[:, :, None] * delta_c[:, None, :]
        y[a, v] = new_c
        moves = moves + active.to(torch.int64)
        better = active & (energy < best_e)
        best_y = torch.where(better[:, None], y, best_y)
        best_e = torch.where(better, energy, best_e)
        stopped = stopped | (energy <= 1e-3)
    return y, g, energy, best_y, best_e, stopped, moves


def forward_with_certificates(w, edges):
    pair_u = w["pair_u"]
    pair_v = w["pair_v"]
    color_eye = w["color_eye"]
    color_ids = w["color_ids"]
    focus_thr = w["focus_thr"]
    inv_temps = w["inv_temps"]
    apply_canon = w["apply_canon"]
    n_verts = w["vert_arange"].shape[0]
    edges_f = edges.to(color_eye.dtype)
    A = build_adj(edges_f, pair_u, pair_v, n_verts)
    R = A.shape[0]
    y = torch.randint(0, 3, (R, n_verts), device=A.device, dtype=torch.int64)
    oh = color_eye[y]
    g = (A @ oh) @ color_eye
    own = g.gather(2, y[:, :, None])[:, :, 0]
    energy = 0.5 * own.sum(dim=1)
    best_y = y.clone()
    best_e = energy.clone()
    stopped = energy <= 1e-3
    moves = torch.zeros(R, device=A.device, dtype=torch.int64)
    steps = inv_temps.shape[0]
    u_vert = torch.rand((steps, R, n_verts), device=A.device, dtype=color_eye.dtype)
    u_gumbel = torch.rand((steps, R, 3), device=A.device, dtype=color_eye.dtype)
    y, g, energy, best_y, best_e, stopped, moves = step_chunk(
        A, color_eye, focus_thr, y, g, energy, best_y, best_e, stopped, moves, u_vert, u_gumbel, inv_temps
    )
    out_raw = torch.where(stopped[:, None], y, best_y)
    out_canon = canonicalize_colorings(out_raw, color_ids)
    out_y = torch.where(apply_canon, out_canon, out_raw)
    out_e = torch.where(stopped, energy, best_e)
    return out_y, stopped, out_e, moves, out_raw


def forward(w, x):
    out_y, stopped, out_e, moves, out_raw = forward_with_certificates(w, x)
    return out_y
"""


def build_graph_coloring_det_wsp(
    n: int,
    *,
    passes: int = 16,
    alpha: float = 0.5,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Programmed deterministic Potts Graph-Coloring WSP weights (`1` distinct energy coefficient `{1.0}`)."""
    if n < 2:
        raise ValueError("n must be >= 2")
    dev = torch.device(device)
    iu = torch.triu_indices(n, n, 1, device=dev)
    return {
        "pair_u": iu[0],
        "pair_v": iu[1],
        "color_eye": torch.eye(3, device=dev, dtype=torch.float32),
        "color_ids": torch.arange(3, device=dev, dtype=torch.int64),
        "vert_arange": torch.arange(n, device=dev, dtype=torch.int64),
        "alphas": torch.full((passes,), alpha, device=dev, dtype=torch.float32),
    }


def build_graph_coloring_assoc_wsp(
    n: int,
    *,
    cap: int = 10000,
    temperature: float = 1.0,
    theta: float = 0.45,
    canonicalize_in_arch: bool = True,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Programmed stochastic associative Graph-Coloring WSP weights (`1` distinct energy coefficient `{1.0}`)."""
    if n < 2:
        raise ValueError("n must be >= 2")
    if temperature <= 0 or cap < 0:
        raise ValueError("temperature > 0 and cap >= 0 required")
    dev = torch.device(device)
    iu = torch.triu_indices(n, n, 1, device=dev)
    return {
        "pair_u": iu[0],
        "pair_v": iu[1],
        "color_eye": torch.eye(3, device=dev, dtype=torch.float32),
        "color_ids": torch.arange(3, device=dev, dtype=torch.int64),
        "vert_arange": torch.arange(n, device=dev, dtype=torch.int64),
        "focus_thr": torch.tensor(float(theta), device=dev, dtype=torch.float32),
        "inv_temps": torch.full((cap,), 1.0 / float(temperature), device=dev, dtype=torch.float32),
        "apply_canon": torch.tensor(bool(canonicalize_in_arch), device=dev),
    }


# ---------------------------------------------------------------------------
# Compiled Chunked GPU / CPU Solvers (Bitwise-Matched to Eager Arch Chunks)
# ---------------------------------------------------------------------------

_COMPILED_CACHE: dict[tuple[str, str], Any] = {}


def _get_step_fn(
    arch_source: str,
    mod_name: str,
    use_compile: bool,
    device: torch.device,
    target_chunk: int = 32,
) -> Any:
    mod = compile_wsp_arch(arch_source, mod_name)
    eager_fn = mod.step_chunk
    if not use_compile or device.type != "cuda":
        return eager_fn
    key = (mod_name, device.type, int(target_chunk))
    if key not in _COMPILED_CACHE:
        _COMPILED_CACHE[key] = torch.compile(eager_fn, dynamic=True)
    comp_fn = _COMPILED_CACHE[key]

    def _dispatch(*args: Any) -> Any:
        # Avoid TorchDynamo loop-unroll recompiles on tail k_len != target_chunk and 0/1 R_act specialization
        inv_t = args[-1]
        u_t = args[-2]
        if inv_t.shape[0] != target_chunk or u_t.shape[1] < 64:
            return eager_fn(*args)
        return comp_fn(*args)

    return _dispatch


@torch.no_grad()
def solve_nqueens_det(
    w: dict[str, Tensor],
    givens: np.ndarray | Tensor,
    *,
    chains: int = 20,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Run deterministic WSP Hopfield N-Queens relaxation (`chains` copies demonstrate mode collapse)."""
    dev = torch.device(device)
    w_dev = {k: v.to(dev) for k, v in w.items()}
    mod = compile_wsp_arch(ARCH_NQUEENS_DET_SOURCE, "wsp_nq_det")
    giv_t = torch.as_tensor(givens, device=dev, dtype=torch.bool)
    B, c = giv_t.shape
    _sync(dev)
    t0 = time.perf_counter()
    boards_1, stopped_1, energy_1, moves_1 = mod.forward_with_certificates(w_dev, giv_t)
    boards = boards_1[:, None, :].expand(B, chains, c).contiguous()
    stopped = stopped_1[:, None].expand(B, chains).contiguous()
    energy = energy_1[:, None].expand(B, chains).contiguous()
    moves = moves_1[:, None].expand(B, chains).contiguous()
    _sync(dev)
    gpu_seconds = time.perf_counter() - t0
    out = {
        "boards": boards.cpu().numpy(),
        "stopped": stopped.cpu().numpy(),
        "energy": energy.cpu().numpy(),
        "moves": moves.cpu().numpy(),
    }
    _sync(dev)
    out["gpu_kernel_seconds"] = gpu_seconds
    out["worker_solve_seconds"] = time.perf_counter() - t0
    return out


@torch.no_grad()
def solve_nqueens_arbitrary(
    w: dict[str, Tensor],
    givens: np.ndarray | Tensor,
    *,
    chains: int = 20,
    seed: int = 0,
    device: torch.device | str = "cpu",
    chunk_steps: int = 32,
    block_chains: int = 65536,
    use_compile: bool = True,
) -> dict[str, Any]:
    """Batched 20-chain stochastic focused heat-bath WSP solver for N-Queens."""
    dev = torch.device(device)
    w_dev = {k: v.to(dev) for k, v in w.items()}
    W_attack = w_dev["W_attack"]
    bias_val = w_dev["bias_scalar"]
    tau_thresh = w_dev["tau_thresh"]
    inv_temps = w_dev["inv_temps"]
    cap = int(inv_temps.shape[0])
    step_fn = _get_step_fn(ARCH_NQUEENS_ARBITRARY_SOURCE, "wsp_nq_arb", use_compile, dev, target_chunk=chunk_steps)

    _sync(dev)
    t_total0 = time.perf_counter()
    giv_all = torch.as_tensor(givens, device=dev, dtype=torch.bool).repeat_interleave(chains, dim=0)
    total, c = giv_all.shape
    gen = torch.Generator(device=dev).manual_seed(int(seed))

    out_x = torch.zeros((total, c), device=dev, dtype=torch.bool)
    out_stop = torch.zeros(total, device=dev, dtype=torch.bool)
    out_energy = torch.zeros(total, device=dev, dtype=torch.float32)
    out_moves = torch.zeros(total, device=dev, dtype=torch.int64)

    for lo in range(0, total, block_chains):
        hi = min(lo + block_chains, total)
        ids = torch.arange(lo, hi, device=dev)
        giv = giv_all[ids]
        free = ~giv
        x = giv.to(torch.float32)
        attack = x @ W_attack
        energy = 0.5 * (x * attack).sum(dim=1) + bias_val * x.sum(dim=1)
        best_x = x.clone()
        best_e = energy.clone()
        stopped = energy <= tau_thresh
        moves = torch.zeros(len(ids), device=dev, dtype=torch.int64)

        if bool(stopped.any()):
            out_x[ids[stopped]] = x[stopped] > 0.5
            out_stop[ids[stopped]] = True
            out_energy[ids[stopped]] = energy[stopped]
            keep = ~stopped
            ids, free, x, attack, energy, best_x, best_e, stopped, moves = (
                t[keep] for t in (ids, free, x, attack, energy, best_x, best_e, stopped, moves)
            )

        for step_lo in range(0, cap, chunk_steps):
            if len(ids) == 0:
                break
            step_hi = min(step_lo + chunk_steps, cap)
            k_len = step_hi - step_lo
            R_act = len(ids)
            u_cell = torch.rand((k_len, R_act, c), device=dev, dtype=torch.float32, generator=gen)
            u_flip = torch.rand((k_len, R_act), device=dev, dtype=torch.float32, generator=gen)
            x, attack, energy, best_x, best_e, stopped, moves = step_fn(
                W_attack, bias_val, tau_thresh, free, x, attack, energy, best_x, best_e, stopped, moves,
                u_cell, u_flip, inv_temps[step_lo:step_hi]
            )
            is_last = step_hi == cap
            if is_last or bool(stopped.any()):
                fin = torch.ones_like(stopped) if is_last else stopped
                res_x = torch.where(stopped[:, None], x, best_x) > 0.5
                res_e = torch.where(stopped, energy, best_e)
                out_x[ids[fin]] = res_x[fin]
                out_stop[ids[fin]] = stopped[fin]
                out_energy[ids[fin]] = res_e[fin]
                out_moves[ids[fin]] = moves[fin]
                keep = ~fin
                ids, free, x, attack, energy, best_x, best_e, stopped, moves = (
                    t[keep] for t in (ids, free, x, attack, energy, best_x, best_e, stopped, moves)
                )

    _sync(dev)
    gpu_kernel_seconds = time.perf_counter() - t_total0
    B = len(givens)
    result = {
        "boards": out_x.view(B, chains, c).cpu().numpy(),
        "stopped": out_stop.view(B, chains).cpu().numpy(),
        "energy": out_energy.view(B, chains).cpu().numpy(),
        "moves": out_moves.view(B, chains).cpu().numpy(),
    }
    _sync(dev)
    worker_solve_seconds = time.perf_counter() - t_total0
    result["gpu_kernel_seconds"] = gpu_kernel_seconds
    result["worker_solve_seconds"] = worker_solve_seconds
    return result


@torch.no_grad()
def solve_permutation_swap(
    w: dict[str, Tensor],
    givens: np.ndarray | Tensor,
    *,
    chains: int = 20,
    seed: int = 0,
    device: torch.device | str = "cpu",
    initial: np.ndarray | Tensor | None = None,
    chunk_steps: int = 32,
    block_chains: int = 81920,
    use_compile: bool = True,
) -> dict[str, Any]:
    """Batched 20-chain row-column permutation 2-opt swap WSP solver (N-Queens & Queen+Leaper)."""
    dev = torch.device(device)
    w_dev = {k: v.to(dev) for k, v in w.items()}
    W_f = w_dev["W_attack"]
    W_int = w_dev["W_attack_i32"]
    inv_temps = w_dev["inv_temps"]
    n = int(w_dev["row_arange"].shape[0])
    c = n * n
    cap = int(inv_temps.shape[0])
    mod = compile_wsp_arch(ARCH_PERM_SWAP_SOURCE, "wsp_perm_swap")
    step_fn = _get_step_fn(ARCH_PERM_SWAP_SOURCE, "wsp_perm_swap", use_compile, dev, target_chunk=chunk_steps)

    _sync(dev)
    t_total0 = time.perf_counter()
    giv_raw = torch.as_tensor(givens, device=dev, dtype=torch.bool)
    B = giv_raw.shape[0]
    G = giv_raw.view(B, n, n)
    if not bool((G.sum(dim=1) <= 1).all() and (G.sum(dim=2) <= 1).all()):
        raise ValueError("Permutation solver requires at most one given queen per row and per column")

    giv_all = giv_raw.repeat_interleave(chains, dim=0)
    total = giv_all.shape[0]
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    init_all = None
    if initial is not None:
        init_all = torch.as_tensor(initial, device=dev, dtype=torch.bool).reshape(total, c)

    out_x = torch.zeros((total, c), device=dev, dtype=torch.bool)
    out_stop = torch.zeros(total, device=dev, dtype=torch.bool)
    out_energy = torch.zeros(total, device=dev, dtype=torch.float32)
    out_moves = torch.zeros(total, device=dev, dtype=torch.int64)
    out_conflict = torch.zeros(total, device=dev, dtype=torch.int32)
    out_accept = torch.zeros(total, device=dev, dtype=torch.int64)

    for lo in range(0, total, block_chains):
        hi = min(lo + block_chains, total)
        ids = torch.arange(lo, hi, device=dev)
        giv = giv_all[ids]
        if init_all is None:
            u_init = torch.rand((len(ids), n), device=dev, dtype=torch.float32, generator=gen)
            x, cols, fixed = mod.init_permutation_boards(giv, n, u_init)
        else:
            x = init_all[ids].clone()
            fixed = giv.view(-1, n, n).any(dim=2)
            cols = x.view(-1, n, n).to(torch.int64).argmax(dim=2)
            if not bool(
                (x.view(-1, n, n).sum(dim=1) == 1).all()
                and (x.view(-1, n, n).sum(dim=2) == 1).all()
                and (x >= giv).all()
            ):
                raise ValueError("Supplied initial boards violate row/column permutation or givens")

        counts = (x.to(torch.float32) @ W_f).to(torch.int32)
        conflict = (x.to(torch.int32) * counts).sum(dim=1, dtype=torch.int32) // 2
        best_x = x.clone()
        best_conf = conflict.clone()
        stopped = conflict == 0
        moves = torch.zeros(len(ids), device=dev, dtype=torch.int64)
        accepted = torch.zeros(len(ids), device=dev, dtype=torch.int64)

        if bool(stopped.any()):
            out_x[ids[stopped]] = x[stopped]
            out_stop[ids[stopped]] = True
            out_energy[ids[stopped]] = conflict[stopped].to(torch.float32)
            out_conflict[ids[stopped]] = conflict[stopped]
            keep = ~stopped
            ids, fixed, x, cols, counts, conflict, best_x, best_conf, stopped, moves, accepted = (
                t[keep] for t in (ids, fixed, x, cols, counts, conflict, best_x, best_conf, stopped, moves, accepted)
            )

        for step_lo in range(0, cap, chunk_steps):
            if len(ids) == 0:
                break
            step_hi = min(step_lo + chunk_steps, cap)
            k_len = step_hi - step_lo
            R_act = len(ids)
            u_row = torch.rand((k_len, R_act, n), device=dev, dtype=torch.float32, generator=gen)
            u_acc = torch.rand((k_len, R_act), device=dev, dtype=torch.float32, generator=gen)
            x, cols, counts, conflict, best_x, best_conf, stopped, moves, accepted = step_fn(
                W_int, fixed, x, cols, counts, conflict, best_x, best_conf, stopped, moves, accepted,
                u_row, u_acc, inv_temps[step_lo:step_hi]
            )
            is_last = step_hi == cap
            if is_last or bool(stopped.any()):
                fin = torch.ones_like(stopped) if is_last else stopped
                res_x = torch.where(stopped[:, None], x, best_x)
                res_c = torch.where(stopped, conflict, best_conf)
                out_x[ids[fin]] = res_x[fin]
                out_stop[ids[fin]] = stopped[fin]
                out_energy[ids[fin]] = res_c[fin].to(torch.float32)
                out_conflict[ids[fin]] = res_c[fin]
                out_moves[ids[fin]] = moves[fin]
                out_accept[ids[fin]] = accepted[fin]
                keep = ~fin
                ids, fixed, x, cols, counts, conflict, best_x, best_conf, stopped, moves, accepted = (
                    t[keep] for t in (ids, fixed, x, cols, counts, conflict, best_x, best_conf, stopped, moves, accepted)
                )

    _sync(dev)
    gpu_kernel_seconds = time.perf_counter() - t_total0
    result = {
        "boards": out_x.view(B, chains, c).cpu().numpy(),
        "stopped": out_stop.view(B, chains).cpu().numpy(),
        "energy": out_energy.view(B, chains).cpu().numpy(),
        "moves": out_moves.view(B, chains).cpu().numpy(),
        "conflicts": out_conflict.view(B, chains).cpu().numpy(),
        "accepted": out_accept.view(B, chains).cpu().numpy(),
    }
    _sync(dev)
    worker_solve_seconds = time.perf_counter() - t_total0
    result["gpu_kernel_seconds"] = gpu_kernel_seconds
    result["worker_solve_seconds"] = worker_solve_seconds
    return result


@torch.no_grad()
def solve_arbitrary_swap(
    w: dict[str, Tensor],
    givens: np.ndarray | Tensor,
    *,
    chains: int = 20,
    seed: int = 0,
    device: torch.device | str = "cpu",
    initial: np.ndarray | Tensor | None = None,
    chunk_steps: int = 32,
    block_chains: int = 81920,
    use_compile: bool = True,
) -> dict[str, Any]:
    """Batched 20-chain fixed-cardinality arbitrary-cell swap WSP solver (Generalized Queen+Leaper)."""
    dev = torch.device(device)
    w_dev = {k: v.to(dev) for k, v in w.items()}
    W_f = w_dev["W_attack"]
    W_int = w_dev["W_attack_i32"]
    inv_temps = w_dev["inv_temps"]
    q = int(w_dev["row_arange"].shape[0])
    c = int(W_f.shape[0])
    cap = int(inv_temps.shape[0])
    mod = compile_wsp_arch(ARCH_ARB_SWAP_SOURCE, "wsp_arb_swap")
    step_fn = _get_step_fn(ARCH_ARB_SWAP_SOURCE, "wsp_arb_swap", use_compile, dev, target_chunk=chunk_steps)

    _sync(dev)
    t_total0 = time.perf_counter()
    giv_raw = torch.as_tensor(givens, device=dev, dtype=torch.bool)
    B = giv_raw.shape[0]
    if not bool((giv_raw.sum(dim=1) <= q).all()):
        raise ValueError("givens cannot contain more than q queens")

    giv_all = giv_raw.repeat_interleave(chains, dim=0)
    total = giv_all.shape[0]
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    init_all = None
    if initial is not None:
        init_all = torch.as_tensor(initial, device=dev, dtype=torch.bool).reshape(total, c)

    out_x = torch.zeros((total, c), device=dev, dtype=torch.bool)
    out_stop = torch.zeros(total, device=dev, dtype=torch.bool)
    out_energy = torch.zeros(total, device=dev, dtype=torch.float32)
    out_moves = torch.zeros(total, device=dev, dtype=torch.int64)
    out_conflict = torch.zeros(total, device=dev, dtype=torch.int32)
    out_accept = torch.zeros(total, device=dev, dtype=torch.int64)

    for lo in range(0, total, block_chains):
        hi = min(lo + block_chains, total)
        ids = torch.arange(lo, hi, device=dev)
        giv = giv_all[ids]
        free = ~giv
        if init_all is None:
            u_init = torch.rand((len(ids), c), device=dev, dtype=torch.float32, generator=gen)
            x = mod.init_arbitrary_boards(giv, q, u_init)
        else:
            x = init_all[ids].clone()
            if not bool((x.sum(dim=1) == q).all() and (x >= giv).all()):
                raise ValueError("Supplied initial boards violate cardinality q or givens")

        counts = (x.to(torch.float32) @ W_f).to(torch.int32)
        conflict = (x.to(torch.int32) * counts).sum(dim=1, dtype=torch.int32) // 2
        best_x = x.clone()
        best_conf = conflict.clone()
        stopped = conflict == 0
        moves = torch.zeros(len(ids), device=dev, dtype=torch.int64)
        accepted = torch.zeros(len(ids), device=dev, dtype=torch.int64)

        if bool(stopped.any()):
            out_x[ids[stopped]] = x[stopped]
            out_stop[ids[stopped]] = True
            out_energy[ids[stopped]] = conflict[stopped].to(torch.float32)
            out_conflict[ids[stopped]] = conflict[stopped]
            keep = ~stopped
            ids, free, x, counts, conflict, best_x, best_conf, stopped, moves, accepted = (
                t[keep] for t in (ids, free, x, counts, conflict, best_x, best_conf, stopped, moves, accepted)
            )

        for step_lo in range(0, cap, chunk_steps):
            if len(ids) == 0:
                break
            step_hi = min(step_lo + chunk_steps, cap)
            k_len = step_hi - step_lo
            R_act = len(ids)
            u_cell = torch.rand((k_len, R_act, c), device=dev, dtype=torch.float32, generator=gen)
            u_acc = torch.rand((k_len, R_act), device=dev, dtype=torch.float32, generator=gen)
            x, counts, conflict, best_x, best_conf, stopped, moves, accepted = step_fn(
                W_int, free, x, counts, conflict, best_x, best_conf, stopped, moves, accepted,
                u_cell, u_acc, inv_temps[step_lo:step_hi]
            )
            is_last = step_hi == cap
            if is_last or bool(stopped.any()):
                fin = torch.ones_like(stopped) if is_last else stopped
                res_x = torch.where(stopped[:, None], x, best_x)
                res_c = torch.where(stopped, conflict, best_conf)
                out_x[ids[fin]] = res_x[fin]
                out_stop[ids[fin]] = stopped[fin]
                out_energy[ids[fin]] = res_c[fin].to(torch.float32)
                out_conflict[ids[fin]] = res_c[fin]
                out_moves[ids[fin]] = moves[fin]
                out_accept[ids[fin]] = accepted[fin]
                keep = ~fin
                ids, free, x, counts, conflict, best_x, best_conf, stopped, moves, accepted = (
                    t[keep] for t in (ids, free, x, counts, conflict, best_x, best_conf, stopped, moves, accepted)
                )

    _sync(dev)
    gpu_kernel_seconds = time.perf_counter() - t_total0
    result = {
        "boards": out_x.view(B, chains, c).cpu().numpy(),
        "stopped": out_stop.view(B, chains).cpu().numpy(),
        "energy": out_energy.view(B, chains).cpu().numpy(),
        "moves": out_moves.view(B, chains).cpu().numpy(),
        "conflicts": out_conflict.view(B, chains).cpu().numpy(),
        "accepted": out_accept.view(B, chains).cpu().numpy(),
    }
    _sync(dev)
    worker_solve_seconds = time.perf_counter() - t_total0
    result["gpu_kernel_seconds"] = gpu_kernel_seconds
    result["worker_solve_seconds"] = worker_solve_seconds
    return result


@torch.no_grad()
def solve_graph_coloring_det(
    w: dict[str, Tensor],
    edges: np.ndarray | Tensor,
    *,
    samples: int = 20,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Run deterministic WSP Potts Graph-Coloring relaxation (`samples` copies demonstrate mode collapse)."""
    dev = torch.device(device)
    w_dev = {k: v.to(dev) for k, v in w.items()}
    mod = compile_wsp_arch(ARCH_GC_DET_SOURCE, "wsp_gc_det")
    edges_t = torch.as_tensor(edges, device=dev, dtype=torch.float32)
    B = edges_t.shape[0]
    n = int(w_dev["vert_arange"].shape[0])
    _sync(dev)
    t0 = time.perf_counter()
    y_1, stopped_1, energy_1, moves_1 = mod.forward_with_certificates(w_dev, edges_t)
    y_canon = y_1[:, None, :].expand(B, samples, n).contiguous()
    stopped = stopped_1[:, None].expand(B, samples).contiguous()
    energy = energy_1[:, None].expand(B, samples).contiguous()
    moves = moves_1[:, None].expand(B, samples).contiguous()
    _sync(dev)
    gpu_seconds = time.perf_counter() - t0
    out = {
        "boards": y_canon.cpu().numpy(),
        "stopped": stopped.cpu().numpy(),
        "energy": energy.cpu().numpy(),
        "moves": moves.cpu().numpy(),
    }
    _sync(dev)
    out["gpu_kernel_seconds"] = gpu_seconds
    out["worker_solve_seconds"] = time.perf_counter() - t0
    return out


@torch.no_grad()
def solve_graph_coloring_assoc(
    w: dict[str, Tensor],
    edges: np.ndarray | Tensor,
    *,
    samples: int = 20,
    seed: int = 0,
    device: torch.device | str = "cpu",
    chunk_steps: int = 32,
    use_compile: bool = True,
) -> dict[str, Any]:
    """Batched 20-chain stochastic focused Gumbel-Potts WSP solver for Graph Coloring."""
    dev = torch.device(device)
    w_dev = {k: v.to(dev) for k, v in w.items()}
    pair_u = w_dev["pair_u"]
    pair_v = w_dev["pair_v"]
    color_eye = w_dev["color_eye"]
    color_ids = w_dev["color_ids"]
    focus_thr = w_dev["focus_thr"]
    inv_temps = w_dev["inv_temps"]
    apply_canon = w_dev["apply_canon"]
    n = int(w_dev["vert_arange"].shape[0])
    cap = int(inv_temps.shape[0])
    mod = compile_wsp_arch(ARCH_GC_ASSOC_SOURCE, "wsp_gc_assoc")
    step_fn = _get_step_fn(ARCH_GC_ASSOC_SOURCE, "wsp_gc_assoc", use_compile, dev, target_chunk=chunk_steps)

    _sync(dev)
    t0 = time.perf_counter()
    edges_t = torch.as_tensor(edges, device=dev, dtype=torch.float32)
    B = edges_t.shape[0]
    A_graph = mod.build_adj(edges_t, pair_u, pair_v, n)
    A_all = A_graph.repeat_interleave(samples, dim=0)
    total = A_all.shape[0]
    gen = torch.Generator(device=dev).manual_seed(int(seed))

    y = torch.randint(0, 3, (total, n), device=dev, dtype=torch.int64, generator=gen)
    oh = color_eye[y]
    g = (A_all @ oh) @ color_eye
    own = g.gather(2, y[:, :, None])[:, :, 0]
    energy = 0.5 * own.sum(dim=1)
    best_y = y.clone()
    best_e = energy.clone()
    stopped = energy <= 1e-3
    moves = torch.zeros(total, device=dev, dtype=torch.int64)

    out_raw = torch.zeros((total, n), device=dev, dtype=torch.int64)
    out_stop = torch.zeros(total, device=dev, dtype=torch.bool)
    out_energy = torch.zeros(total, device=dev, dtype=torch.float32)
    out_moves = torch.zeros(total, device=dev, dtype=torch.int64)

    ids = torch.arange(total, device=dev)
    A_act = A_all
    if bool(stopped.any()):
        out_raw[ids[stopped]] = y[stopped]
        out_stop[ids[stopped]] = True
        out_energy[ids[stopped]] = energy[stopped]
        keep = ~stopped
        ids, A_act, y, g, energy, best_y, best_e, stopped, moves = (
            t[keep] for t in (ids, A_act, y, g, energy, best_y, best_e, stopped, moves)
        )

    for step_lo in range(0, cap, chunk_steps):
        if len(ids) == 0:
            break
        step_hi = min(step_lo + chunk_steps, cap)
        k_len = step_hi - step_lo
        R_act = len(ids)
        u_vert = torch.rand((k_len, R_act, n), device=dev, dtype=torch.float32, generator=gen)
        u_gumbel = torch.rand((k_len, R_act, 3), device=dev, dtype=torch.float32, generator=gen)
        y, g, energy, best_y, best_e, stopped, moves = step_fn(
            A_act, color_eye, focus_thr, y, g, energy, best_y, best_e, stopped, moves,
            u_vert, u_gumbel, inv_temps[step_lo:step_hi]
        )
        is_last = step_hi == cap
        if is_last or bool(stopped.any()):
            fin = torch.ones_like(stopped) if is_last else stopped
            res_y = torch.where(stopped[:, None], y, best_y)
            res_e = torch.where(stopped, energy, best_e)
            out_raw[ids[fin]] = res_y[fin]
            out_stop[ids[fin]] = stopped[fin]
            out_energy[ids[fin]] = res_e[fin]
            out_moves[ids[fin]] = moves[fin]
            keep = ~fin
            ids, A_act, y, g, energy, best_y, best_e, stopped, moves = (
                t[keep] for t in (ids, A_act, y, g, energy, best_y, best_e, stopped, moves)
            )

    out_canon = mod.canonicalize_colorings(out_raw, color_ids)
    out_y = torch.where(apply_canon, out_canon, out_raw)
    _sync(dev)
    gpu_kernel_seconds = time.perf_counter() - t0
    result = {
        "boards": out_y.view(B, samples, n).cpu().numpy(),
        "boards_raw_uncanonicalized": out_raw.view(B, samples, n).cpu().numpy(),
        "stopped": out_stop.view(B, samples).cpu().numpy(),
        "energy": out_energy.view(B, samples).cpu().numpy(),
        "moves": out_moves.view(B, samples).cpu().numpy(),
    }
    _sync(dev)
    result["gpu_kernel_seconds"] = gpu_kernel_seconds
    result["worker_solve_seconds"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Exact `FullCounter` (FC2) PyTorch Transcriptions & Affine FLOP Accounting
# ---------------------------------------------------------------------------

@torch.no_grad()
def transcribe_single_chain_flops(
    kind: str,
    n: int,
    steps: int,
    *,
    leaper: tuple[int, int] | None = None,
    canonicalize_in_arch: bool = True,
    use_given_initial: bool = False,
) -> dict[str, int]:
    """Run exact 1-chain WSP setup + `steps` moves under `FullCounter` (`FC2`) on CPU."""
    from flops.fullcount import FullCounter

    if kind == "nqueens_det":
        w = build_nqueens_det_wsp(n, passes=steps, device="cpu")
        giv = torch.zeros((1, n * n), dtype=torch.bool)
        mod = compile_wsp_arch(ARCH_NQUEENS_DET_SOURCE, "wsp_nq_det")
        with FullCounter() as fc:
            mod.forward_with_certificates(w, giv)
    elif kind == "nqueens_arbitrary":
        w = build_nqueens_arbitrary_wsp(n, cap=steps, temperature=0.3, device="cpu")
        giv = torch.zeros((1, n * n), dtype=torch.bool)
        giv[0, 0] = True
        mod = compile_wsp_arch(ARCH_NQUEENS_ARBITRARY_SOURCE, "wsp_nq_arb")
        with FullCounter() as fc:
            mod.forward_with_certificates(w, giv)
    elif kind in ("nqueens_perm", "leaper_perm"):
        w = build_queen_leaper_wsp(n=n, leaper=leaper, cap=steps, device="cpu")
        giv = torch.zeros((1, n * n), dtype=torch.bool)
        giv[0, 0] = True
        mod = compile_wsp_arch(ARCH_PERM_SWAP_SOURCE, "wsp_perm_swap")
        if use_given_initial:
            init = torch.zeros((1, n * n), dtype=torch.bool)
            for r in range(n):
                init[0, r * n + r] = True
            with FullCounter() as fc:
                mod.forward_from_initial(w, giv, init)
        else:
            with FullCounter() as fc:
                mod.forward_with_certificates(w, giv)
    elif kind == "leaper_arbitrary":
        w = build_queen_leaper_wsp(n=n, leaper=leaper, cap=steps, device="cpu")
        giv = torch.zeros((1, n * n), dtype=torch.bool)
        giv[0, 0] = True
        mod = compile_wsp_arch(ARCH_ARB_SWAP_SOURCE, "wsp_arb_swap")
        if use_given_initial:
            init = torch.zeros((1, n * n), dtype=torch.bool)
            for r in range(n):
                init[0, r * n + r] = True
            with FullCounter() as fc:
                mod.forward_from_initial(w, giv, init)
        else:
            with FullCounter() as fc:
                mod.forward_with_certificates(w, giv)
    elif kind == "gc_det":
        p = n * (n - 1) // 2
        w = build_graph_coloring_det_wsp(n, passes=steps, device="cpu")
        edges = torch.zeros((1, p), dtype=torch.float32)
        edges[0, 0] = 1.0
        mod = compile_wsp_arch(ARCH_GC_DET_SOURCE, "wsp_gc_det")
        with FullCounter() as fc:
            mod.forward_with_certificates(w, edges)
    elif kind == "gc_assoc":
        p = n * (n - 1) // 2
        w = build_graph_coloring_assoc_wsp(
            n, cap=steps, temperature=1.0, canonicalize_in_arch=canonicalize_in_arch, device="cpu"
        )
        edges = torch.ones((1, p), dtype=torch.float32)  # K_n (>3) ensures 0 early halt during step count calibration
        mod = compile_wsp_arch(ARCH_GC_ASSOC_SOURCE, "wsp_gc_assoc")
        with FullCounter() as fc:
            mod.forward_with_certificates(w, edges)
    else:
        raise ValueError(f"Unknown WSP model kind: {kind}")

    sm = fc.summary()
    cls = sm["classes"]
    return {
        "total": int(sm["total"]),
        "total_with_compare": int(sm["total_with_compare"]),
        "matmul": int(cls["matmul"]),
        "arith": int(cls["arith"]),
        "transc": int(cls["transc"]),
        "compare": int(cls["compare"]),
        "rng": int(cls["rng"]),
    }


def calibrate_affine_chain_flops(
    kind: str,
    n: int,
    *,
    leaper: tuple[int, int] | None = None,
    canonicalize_in_arch: bool = True,
    use_given_initial: bool = False,
    step_grid: tuple[int, ...] = (0, 1, 2, 3, 5, 8, 13, 21),
) -> dict[str, Any]:
    """Calibrate and assert exact zero-residual affine `FullCounter` FLOP cost `setup + steps * per_step`."""
    if kind in ("nqueens_det", "gc_det"):
        step_grid = tuple(s for s in step_grid if s >= 1)
    samples = [
        (
            s,
            transcribe_single_chain_flops(
                kind, n, s,
                leaper=leaper,
                canonicalize_in_arch=canonicalize_in_arch,
                use_given_initial=use_given_initial,
            ),
        )
        for s in step_grid
    ]
    s0, c0 = samples[0]
    s1, c1 = samples[1]
    ds = s1 - s0
    per_step_total = (c1["total"] - c0["total"]) // ds
    setup_total = c0["total"] - s0 * per_step_total
    per_step_cmp = (c1["total_with_compare"] - c0["total_with_compare"]) // ds
    setup_cmp = c0["total_with_compare"] - s0 * per_step_cmp

    residuals_total = [abs(c["total"] - (setup_total + s * per_step_total)) for s, c in samples]
    residuals_cmp = [abs(c["total_with_compare"] - (setup_cmp + s * per_step_cmp)) for s, c in samples]
    max_res_total = max(residuals_total)
    max_res_cmp = max(residuals_cmp)
    if max_res_total != 0 or max_res_cmp != 0:
        raise AssertionError(
            f"Nonzero FullCounter affine residual for {kind} (n={n}): "
            f"total_res={residuals_total}, cmp_res={residuals_cmp}"
        )
    return {
        "kind": kind,
        "n": n,
        "setup_total": int(setup_total),
        "per_step_total": int(per_step_total),
        "setup_total_with_compare": int(setup_cmp),
        "per_step_total_with_compare": int(per_step_cmp),
        "step_grid": list(step_grid),
        "max_abs_residual_total": int(max_res_total),
        "max_abs_residual_total_with_compare": int(max_res_cmp),
    }


def compute_trajectory_flops(
    moves: np.ndarray,
    affine: dict[str, Any],
    *,
    chains_used: int | None = None,
    chunk_steps: int = 32,
) -> dict[str, Any]:
    """Compute exact per-instance FC2 FLOPs across chains up to first-pass `moves` (`stop_t` or `cap`)."""
    m = np.asarray(moves, dtype=np.int64)
    kind_is_det = affine["kind"] in ("nqueens_det", "gc_det")
    if kind_is_det:
        # Deterministic relaxation executes 1 deterministic trajectory per instance (broadcast for Cov@20 scoring)
        m = m[:, :1]
        coords = affine["n"] * affine["n"] if affine["kind"] == "nqueens_det" else affine["n"]
        steps = m // coords
        chunk_checks = np.zeros_like(steps)
    else:
        if chains_used is not None:
            m = m[:, :chains_used]
        steps = m
        # Periodic outer `stopped.any()` check every `chunk_steps` adds 1 compare per active chain per chunk
        chunk_checks = (steps + chunk_steps - 1) // chunk_steps
    chain_total = affine["setup_total"] + steps * affine["per_step_total"]
    chain_cmp = affine["setup_total_with_compare"] + steps * affine["per_step_total_with_compare"] + chunk_checks
    per_instance_total = chain_total.sum(axis=1, dtype=np.int64)
    per_instance_cmp = chain_cmp.sum(axis=1, dtype=np.int64)
    return {
        "chains_counted": int(m.shape[1]),
        "mean_flops_per_instance": float(per_instance_total.mean()),
        "median_flops_per_instance": float(np.median(per_instance_total)),
        "mean_flops_with_compare_per_instance": float(per_instance_cmp.mean()),
        "median_flops_with_compare_per_instance": float(np.median(per_instance_cmp)),
        "total_flops_all_instances": int(per_instance_total.sum()),
        "total_flops_with_compare_all_instances": int(per_instance_cmp.sum()),
        "deterministic_passes_mode": bool(kind_is_det),
    }
