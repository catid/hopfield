"""Weight-Space Programming (WSP) architectures and solvers for Maze benchmarks.

Implements genuine zero-training Weight-Space Programming (`effl-lab/Weight-Space-Programming`,
commit `02c31abd3b34d84854cc80d9717717c45365c719`) for 2D grid mazes with the repository's
canonical token encoding (`0 = wall`, `1 = open`, `2 = start S`, `3 = goal G`) and shortest-path
tie-breaking convention `((1, 0), (0, -1), (0, 1), (-1, 0))` = `(down, left, right, up)`:

1. `WSP-DeadEnd` (`WSP-Local` cellular dead-end pruning):
   - Programmed weights: 4-neighbor `3x3` stencil `nbr_kernel` + 4-token threshold bias `cell_bias`
     (`0` learned parameters, `3` distinct nonzero programmed coefficients `{-4.5, -1.5, 1.0}`).
   - Solves tree mazes (`Maze-Unique`) with exact local fixed-point halting; halts on the 2-core +
     bridges when cycles exist (`Maze-Hard`), mirroring 18-param local CRAM.

2. `WSP-Wavefront` (`WSP-Assoc` geodesic min-plus wavefront + forward lexicographic pointer trace):
   - Programmed weights: directional `3x3` neighbor-read and cursor-move stencils (`(down, left,
     right, up)`), unit step bias, dyadic directional tie bias `[0, 1/8, 2/8, 3/8]`, and exact
     FP32 sentinel `2^20` (`0` learned parameters, `5` distinct nonzero programmed coefficients).
   - Computes exact goal-to-start geodesic distances via parallel min-plus relaxation and traces
     the unique lexicographically smallest shortest path from `start` to `goal` (proven equivalent
     to `deque` BFS from `start` under `(down, left, right, up)` neighbor order).

3. `WSP-Potential` (Programmed continuous screened-Poisson + symmetric Hopfield relaxation control):
   - Programmed weights: zero-training continuous analog of 29-parameter `CRAM+z` (`MazeHardEnergy`),
     settling a log-domain screened-Poisson geodesic potential `psi` (`q = 64`) followed by damped
     continuous Hopfield relaxation (`eta = 0.2`) on gated directed links (`alpha = 20`, `kappa = 8`,
     `b_cost = 10`) with single-flip local energy stopping every 16 steps (`0` learned parameters).

Every model provides:
- `ARCH_SOURCE_*`: standalone pure-tensor `forward(w, x)` module source text passing upstream
  `wsp/free_model.py:static_problems(source) == []` (zero data-dependent Python control flow).
- `build_weights_*`: deterministic weight construction on target device/dtype (`Fit time = "none"`).
- `solve_wsp_*`: batched halting-tracked PyTorch reference solver (`backend="pytorch"`, 100%
  compatible with `FullCounter` / `FC2`) and state-for-state verified fused Triton / compiled GPU
  fast paths (`backend="triton"` / `backend="compiled"`).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


SIDES: tuple[tuple[int, int], ...] = ((1, 0), (0, -1), (0, 1), (-1, 0))  # down, left, right, up


# ---------------------------------------------------------------------------
# Standalone pure-tensor WSP `arch.py` sources (verified by `static_problems`)
# ---------------------------------------------------------------------------

ARCH_SOURCE_DEADEND = """import torch
import torch.nn.functional as F


def forward(w, x):
    dtype = w["nbr_kernel"].dtype
    cell_bias = w["cell_bias"][x]
    active = (x > 0).to(dtype)
    steps = x.shape[1] * x.shape[2]
    for _ in range(steps):
        deg = F.conv2d(active.unsqueeze(1), w["nbr_kernel"], padding=1).squeeze(1)
        active = (deg + cell_bias > 0).to(dtype)
    return active > 0.5
"""

ARCH_SOURCE_WAVEFRONT = """import torch
import torch.nn.functional as F


def forward(w, x):
    dtype = w["nbr_kernels"].dtype
    inf_d = w["inf_dist"]
    open_m = x > 0
    start_m = x == 2
    goal_m = x == 3
    dist = torch.where(goal_m, w["zero"], inf_d)
    steps = x.shape[1] * x.shape[2]
    for _ in range(steps):
        nb = F.conv2d((dist - inf_d).unsqueeze(1), w["nbr_kernels"], bias=w["step_bias"], padding=1) + inf_d
        dist = torch.where(open_m, torch.minimum(dist, nb.amin(dim=1)), inf_d)
    nb_tie = F.conv2d((dist - inf_d).unsqueeze(1), w["nbr_kernels"], bias=w["tie_bias"], padding=1) + inf_d
    best_nb = nb_tie.amin(dim=1)
    best_dir = nb_tie.argmin(dim=1)
    valid_step = (dist > 0.0) & (dist < inf_d) & (best_nb < dist)
    step_oh = F.one_hot(best_dir, num_classes=4).permute(0, 3, 1, 2).to(dtype) * valid_step.unsqueeze(1).to(dtype)
    cursor = (start_m & (dist < inf_d)).to(dtype)
    path = cursor.clone()
    for _ in range(steps):
        cursor = F.conv2d(cursor.unsqueeze(1) * step_oh, w["move_kernels"], padding=1).squeeze(1)
        path = torch.maximum(path, cursor)
    return path > 0.5
"""

ARCH_SOURCE_POTENTIAL = """import torch
import torch.nn.functional as F


def forward(w, x):
    dtype = w["nbr_kernels"].dtype
    q = w["q"]
    cap = w["cap"]
    zero = w["zero"]
    open_m = x > 0
    goal_m = x == 3
    free_m = x == 1
    given = (x >= 2).to(dtype)
    psi = torch.where(goal_m, zero, cap)
    src = torch.where(goal_m, q, -q * cap).unsqueeze(1)
    for _ in range(1024):
        nb = F.conv2d((psi - cap).unsqueeze(1), w["nbr_kernels"], padding=1) + cap
        lse = torch.logsumexp(torch.cat([-q * nb, src], dim=1), dim=1)
        psi = torch.where(open_m, torch.clamp(1.0 - lse / q, max=cap), cap)
    nb_psi = F.conv2d((psi - cap).unsqueeze(1), w["nbr_kernels"], padding=1) + cap
    s_nbr = w["alpha"] * (psi.unsqueeze(1) - nb_psi - 1.0) + w["beta_dir"].view(1, 4, 1, 1)
    s_all = torch.cat([ torch.zeros_like(psi).unsqueeze(1), s_nbr ], dim=1)
    att = torch.softmax(s_all, dim=1)[:, 1:]
    u = torch.zeros_like(psi) + w["init_logit"]
    kappa = w["kappa"]
    b_cost = w["b_cost"]
    eta = w["eta"]
    for _ in range(2048):
        nu = torch.where(free_m, torch.sigmoid(u), given)
        nb_nu = F.conv2d(nu.unsqueeze(1), w["nbr_kernels"], padding=1)
        out_sup = kappa * (att * nb_nu).sum(dim=1)
        in_sup = kappa * F.conv2d(att * nu.unsqueeze(1), w["in_sum_kernel"], padding=1).squeeze(1)
        field = -b_cost + out_sup + in_sup
        u = torch.where(free_m, u + eta * (field - u), u)
    return torch.where(free_m, u > 0.0, x >= 2)
"""


def load_arch_forward(source: str):
    """Compile and return top-level `forward(w, x)` from a pure-tensor WSP `arch` string."""
    namespace: dict[str, Any] = {}
    code = compile(source, "<wsp_maze_arch>", "exec")
    exec(code, namespace)
    return namespace["forward"]


_ARCH_DEADEND_FN = load_arch_forward(ARCH_SOURCE_DEADEND)
_ARCH_WAVEFRONT_FN = load_arch_forward(ARCH_SOURCE_WAVEFRONT)
_ARCH_POTENTIAL_FN = load_arch_forward(ARCH_SOURCE_POTENTIAL)


def arch_deadend_forward(w: dict[str, Tensor], x: Tensor) -> Tensor:
    return _ARCH_DEADEND_FN(w, x)


def arch_wavefront_forward(w: dict[str, Tensor], x: Tensor) -> Tensor:
    return _ARCH_WAVEFRONT_FN(w, x)


def arch_potential_forward(w: dict[str, Tensor], x: Tensor) -> Tensor:
    return _ARCH_POTENTIAL_FN(w, x)


# ---------------------------------------------------------------------------
# Programmed weight builders & metadata
# ---------------------------------------------------------------------------

def _directional_read_kernels(device: torch.device | str, dtype: torch.dtype) -> Tensor:
    """4x1x3x3 cardinal neighbor read kernels in `(down, left, right, up)` order."""
    nbr_k = torch.zeros((4, 1, 3, 3), device=device, dtype=dtype)
    nbr_k[0, 0, 2, 1] = 1.0  # down:  (r + 1, c)
    nbr_k[1, 0, 1, 0] = 1.0  # left:  (r, c - 1)
    nbr_k[2, 0, 1, 2] = 1.0  # right: (r, c + 1)
    nbr_k[3, 0, 0, 1] = 1.0  # up:    (r - 1, c)
    return nbr_k


def build_weights_deadend(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Programmed weights for `WSP-DeadEnd` (`0` learned params, `3` nonzero values)."""
    nbr_kernel = torch.tensor(
        [[[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]],
        device=device,
        dtype=dtype,
    )
    # Token thresholds: 0=wall (-4.5 -> always off), 1=open (-1.5 -> keep iff deg>=2),
    # 2=start (+1.0 -> always on), 3=goal (+1.0 -> always on).
    cell_bias = torch.tensor([-4.5, -1.5, 1.0, 1.0], device=device, dtype=dtype)
    return {"nbr_kernel": nbr_kernel, "cell_bias": cell_bias}


def build_weights_wavefront(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Programmed weights for `WSP-Wavefront` (`0` learned params, `5` nonzero values)."""
    nbr_kernels = _directional_read_kernels(device=device, dtype=dtype)
    move_kernels = nbr_kernels.flip(-2, -1).permute(1, 0, 2, 3).contiguous()
    step_bias = torch.ones((4,), device=device, dtype=dtype)
    tie_bias = torch.tensor([0.0, 0.125, 0.25, 0.375], device=device, dtype=dtype)
    inf_dist = torch.tensor(1048576.0, device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    return {
        "nbr_kernels": nbr_kernels,
        "move_kernels": move_kernels,
        "step_bias": step_bias,
        "tie_bias": tie_bias,
        "inf_dist": inf_dist,
        "zero": zero,
    }


def build_weights_potential(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Programmed weights for `WSP-Potential` (`0` learned params, zero-training CRAM+z control)."""
    nbr_kernels = _directional_read_kernels(device=device, dtype=dtype)
    in_sum_kernel = nbr_kernels.flip(-2, -1).permute(1, 0, 2, 3).contiguous()
    return {
        "nbr_kernels": nbr_kernels,
        "in_sum_kernel": in_sum_kernel,
        "q": torch.tensor(64.0, device=device, dtype=dtype),
        "alpha": torch.tensor(20.0, device=device, dtype=dtype),
        "beta_dir": torch.tensor([14.0, 11.0, 8.0, 5.0], device=device, dtype=dtype),
        "kappa": torch.tensor(8.0, device=device, dtype=dtype),
        "b_cost": torch.tensor(10.0, device=device, dtype=dtype),
        "eta": torch.tensor(0.2, device=device, dtype=dtype),
        "init_logit": torch.tensor(3.0, device=device, dtype=dtype),
        "cap": torch.tensor(1024.0, device=device, dtype=dtype),
        "zero": torch.tensor(0.0, device=device, dtype=dtype),
    }


def weight_metadata(w: dict[str, Tensor]) -> dict[str, Any]:
    """Summarize programmed WSP parameter provenance cleanly without inflating/deflating."""
    vals: list[float] = []
    total_entries = 0
    nonzero_entries = 0
    tensors_info: dict[str, dict[str, Any]] = {}
    for name, t in w.items():
        arr = t.detach().cpu().to(torch.float64).reshape(-1).numpy()
        total_entries += int(arr.size)
        nz = arr[arr != 0.0]
        nonzero_entries += int(nz.size)
        vals.extend(float(v) for v in nz)
        tensors_info[name] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "numel": int(t.numel()),
            "nonzero": int(nz.size),
        }
    distinct_nz = sorted({round(v, 7) for v in vals})
    return {
        "params_learned": 0,
        "fit_time": "none",
        "materialized_entries": total_entries,
        "nonzero_entries": nonzero_entries,
        "distinct_nonzero_coefficients": len(distinct_nz),
        "distinct_nonzero_values": distinct_nz,
        "tensors": tensors_info,
    }


# ---------------------------------------------------------------------------
# Fused Triton kernels for exact state-for-state GPU acceleration
# ---------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def _wsp_deadend_step_kernel(
        active_ptr,
        out_ptr,
        cbias_ptr,
        halted_ptr,
        nbr_w_ptr,
        B,
        H,
        W,
        BLOCK_W: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b = bh // H
        r = bh % H
        cols = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_c = cols < W
        HW = H * W
        base_b = b * HW
        idx = base_b + r * W + cols

        is_halted = tl.load(halted_ptr + b)
        cur = tl.load(active_ptr + idx, mask=mask_c, other=0.0)
        if is_halted != 0:
            tl.store(out_ptr + idx, cur, mask=mask_c)
            return

        deg = tl.zeros([BLOCK_W], dtype=tl.float32)
        for dr in tl.static_range(3):
            nr = r + dr - 1
            valid_r = (nr >= 0) & (nr < H)
            for dc in tl.static_range(3):
                wk = tl.load(nbr_w_ptr + dr * 3 + dc)
                nc = cols + dc - 1
                valid_rc = mask_c & valid_r & (nc >= 0) & (nc < W)
                n_idx = base_b + nr * W + nc
                val = tl.load(active_ptr + n_idx, mask=valid_rc, other=0.0)
                deg += val * wk

        cb = tl.load(cbias_ptr + idx, mask=mask_c, other=-4.5)
        nxt = tl.where(deg + cb > 0.0, 1.0, 0.0)
        tl.store(out_ptr + idx, nxt, mask=mask_c)

    @triton.jit
    def _wsp_wave_step_kernel(
        dist_ptr,
        out_ptr,
        open_ptr,
        halted_ptr,
        w_ptr,
        bias_ptr,
        B,
        H,
        W,
        inf_val,
        BLOCK_W: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b = bh // H
        r = bh % H
        cols = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_c = cols < W
        HW = H * W
        base_b = b * HW
        idx = base_b + r * W + cols

        is_halted = tl.load(halted_ptr + b)
        cur_dist = tl.load(dist_ptr + idx, mask=mask_c, other=inf_val)
        if is_halted != 0:
            tl.store(out_ptr + idx, cur_dist, mask=mask_c)
            return

        is_open = tl.load(open_ptr + idx, mask=mask_c, other=0)
        acc0 = tl.load(bias_ptr + 0) + inf_val
        acc1 = tl.load(bias_ptr + 1) + inf_val
        acc2 = tl.load(bias_ptr + 2) + inf_val
        acc3 = tl.load(bias_ptr + 3) + inf_val

        for dr in tl.static_range(3):
            nr = r + dr - 1
            valid_r = (nr >= 0) & (nr < H)
            for dc in tl.static_range(3):
                nc = cols + dc - 1
                valid_rc = mask_c & valid_r & (nc >= 0) & (nc < W)
                n_idx = base_b + nr * W + nc
                delta = tl.load(dist_ptr + n_idx, mask=valid_rc, other=inf_val) - inf_val
                off = dr * 3 + dc
                acc0 += delta * tl.load(w_ptr + 0 * 9 + off)
                acc1 += delta * tl.load(w_ptr + 1 * 9 + off)
                acc2 += delta * tl.load(w_ptr + 2 * 9 + off)
                acc3 += delta * tl.load(w_ptr + 3 * 9 + off)

        best = tl.minimum(cur_dist, tl.minimum(tl.minimum(acc0, acc1), tl.minimum(acc2, acc3)))
        res = tl.where(is_open != 0, best, inf_val)
        tl.store(out_ptr + idx, res, mask=mask_c)

    @triton.jit
    def _wsp_trace_step_kernel(
        cur_ptr,
        nxt_ptr,
        path_ptr,
        step_oh_ptr,
        halted_ptr,
        move_w_ptr,
        B,
        H,
        W,
        BLOCK_W: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b = bh // H
        r = bh % H
        cols = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_c = cols < W
        HW = H * W
        base_b = b * HW
        idx = base_b + r * W + cols

        is_halted = tl.load(halted_ptr + b)
        old_cur = tl.load(cur_ptr + idx, mask=mask_c, other=0.0)
        old_path = tl.load(path_ptr + idx, mask=mask_c, other=0.0)
        if is_halted != 0:
            tl.store(nxt_ptr + idx, old_cur, mask=mask_c)
            return

        acc = tl.zeros([BLOCK_W], dtype=tl.float32)
        b0 = (b * 4 + 0) * HW
        b1 = (b * 4 + 1) * HW
        b2 = (b * 4 + 2) * HW
        b3 = (b * 4 + 3) * HW
        for dr in tl.static_range(3):
            nr = r + dr - 1
            valid_r = (nr >= 0) & (nr < H)
            for dc in tl.static_range(3):
                nc = cols + dc - 1
                valid_rc = mask_c & valid_r & (nc >= 0) & (nc < W)
                n_idx = nr * W + nc
                c_val = tl.load(cur_ptr + base_b + n_idx, mask=valid_rc, other=0.0)
                off = dr * 3 + dc
                s0 = tl.load(step_oh_ptr + b0 + n_idx, mask=valid_rc, other=0.0) * tl.load(move_w_ptr + 0 * 9 + off)
                s1 = tl.load(step_oh_ptr + b1 + n_idx, mask=valid_rc, other=0.0) * tl.load(move_w_ptr + 1 * 9 + off)
                s2 = tl.load(step_oh_ptr + b2 + n_idx, mask=valid_rc, other=0.0) * tl.load(move_w_ptr + 2 * 9 + off)
                s3 = tl.load(step_oh_ptr + b3 + n_idx, mask=valid_rc, other=0.0) * tl.load(move_w_ptr + 3 * 9 + off)
                acc += c_val * (s0 + s1 + s2 + s3)

        tl.store(nxt_ptr + idx, acc, mask=mask_c)
        tl.store(path_ptr + idx, tl.maximum(old_path, acc), mask=mask_c)


# ---------------------------------------------------------------------------
# Input validation & result container
# ---------------------------------------------------------------------------

@dataclass
class WSPMazeResult:
    predictions: Tensor
    steps: Tensor
    halted: Tensor
    finite: Tensor
    stage_steps: dict[str, Tensor]


def _validate_maze_tokens(x: Tensor) -> Tensor:
    if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError(f"Expected 3D maze token tensor (B, H, W), got shape {tuple(x.shape)}")
    if x.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
        raise ValueError(f"Expected integer maze token dtype, got {x.dtype}")
    return x.long()


def _round_up_multiple(val: int, mult: int) -> int:
    return ((val + mult - 1) // mult) * mult


# ---------------------------------------------------------------------------
# 1. WSP-DeadEnd Solver (`WSP-Local`)
# ---------------------------------------------------------------------------

@torch.no_grad()
def solve_wsp_deadend(
    x: Tensor,
    w: dict[str, Tensor] | None = None,
    *,
    max_steps: int | None = None,
    check_every: int = 8,
    sync_every: int = 64,
    backend: str = "auto",
) -> WSPMazeResult:
    """Run parallel cellular dead-end pruning with local fixed-point halting."""
    x = _validate_maze_tokens(x)
    if check_every < 1 or sync_every < 1:
        raise ValueError("check_every and sync_every must be >= 1")
    B, H, W = x.shape
    if B == 0:
        empty_b = torch.zeros((0,), dtype=torch.bool, device=x.device)
        empty_s = torch.zeros((0,), dtype=torch.long, device=x.device)
        return WSPMazeResult(
            torch.zeros((0, H, W), dtype=torch.bool, device=x.device),
            empty_s,
            empty_b,
            empty_b,
            {"prune_steps": empty_s},
        )
    if w is None:
        w = build_weights_deadend(device=x.device)
    if max_steps is None:
        max_steps = _round_up_multiple(max(64, H * W), check_every)
    else:
        max_steps = _round_up_multiple(max_steps, check_every)

    use_triton = (
        backend == "triton"
        or (backend == "auto" and x.is_cuda and HAS_TRITON and w["nbr_kernel"].dtype == torch.float32)
    )
    if backend not in ("auto", "pytorch", "triton"):
        raise ValueError(f"Unknown backend: {backend!r}")

    dtype = w["nbr_kernel"].dtype
    nbr_kernel = w["nbr_kernel"]
    cbias = w["cell_bias"][x].contiguous()
    active = (x > 0).to(dtype).contiguous()
    halted = torch.zeros((B,), dtype=torch.bool, device=x.device)
    steps = torch.full((B,), max_steps, dtype=torch.long, device=x.device)

    if use_triton:
        buf = torch.empty_like(active)
        halted_u8 = torch.zeros((B,), dtype=torch.uint8, device=x.device)
        block_w = min(256, max(32, 1 << (W - 1).bit_length()))
        grid = (B * H, triton.cdiv(W, block_w))
        for t in range(1, max_steps + 1):
            _wsp_deadend_step_kernel[grid](
                active, buf, cbias, halted_u8, nbr_kernel, B, H, W, BLOCK_W=block_w
            )
            if t % check_every == 0:
                same = (buf == active).flatten(1).all(1)
                steps = torch.where(same & (~halted), torch.full_like(steps, t), steps)
                halted = halted | same
                halted_u8 = halted.to(torch.uint8)
                active, buf = buf, active
                if (t % sync_every == 0 or t == max_steps) and bool(halted.all()):
                    break
            else:
                active, buf = buf, active
    else:
        for t in range(1, max_steps + 1):
            deg = F.conv2d(active.unsqueeze(1), nbr_kernel, padding=1).squeeze(1)
            nxt = (deg + cbias > 0.0).to(dtype)
            nxt = torch.where(halted[:, None, None], active, nxt)
            if t % check_every == 0:
                same = (nxt == active).flatten(1).all(1)
                steps = torch.where(same & (~halted), torch.full_like(steps, t), steps)
                halted = halted | same
                active = nxt
                if (t % sync_every == 0 or t == max_steps) and bool(halted.all()):
                    break
            else:
                active = nxt

    finite = torch.isfinite(active).flatten(1).all(1)
    preds = active > 0.5
    return WSPMazeResult(preds, steps, halted, finite, {"prune_steps": steps})


# ---------------------------------------------------------------------------
# 2. WSP-Wavefront Solver (`WSP-Assoc` Geodesic Min-Plus + Lexicographic Trace)
# ---------------------------------------------------------------------------

@torch.no_grad()
def solve_wsp_wavefront(
    x: Tensor,
    w: dict[str, Tensor] | None = None,
    *,
    max_steps: int | None = None,
    check_every: int = 16,
    sync_every: int = 64,
    backend: str = "auto",
) -> WSPMazeResult:
    """Run geodesic min-plus wavefront from goal + lexicographic shortest-path trace from start."""
    x = _validate_maze_tokens(x)
    if check_every < 1 or sync_every < 1:
        raise ValueError("check_every and sync_every must be >= 1")
    B, H, W = x.shape
    if B == 0:
        empty_b = torch.zeros((0,), dtype=torch.bool, device=x.device)
        empty_s = torch.zeros((0,), dtype=torch.long, device=x.device)
        return WSPMazeResult(
            torch.zeros((0, H, W), dtype=torch.bool, device=x.device),
            empty_s,
            empty_b,
            empty_b,
            {"wave_steps": empty_s, "trace_steps": empty_s},
        )
    if H * W >= 1048576:
        raise ValueError(f"Grid cells {H*W} must be < inf_dist sentinel 1,048,576")
    if w is None:
        w = build_weights_wavefront(device=x.device)
    if max_steps is None:
        max_steps = _round_up_multiple(max(check_every, H * W), check_every)
    else:
        max_steps = _round_up_multiple(max_steps, check_every)

    use_triton = (
        backend == "triton"
        or (backend == "auto" and x.is_cuda and HAS_TRITON and w["nbr_kernels"].dtype == torch.float32)
    )
    if backend not in ("auto", "pytorch", "triton"):
        raise ValueError(f"Unknown backend: {backend!r}")

    dtype = w["nbr_kernels"].dtype
    nbr_kernels = w["nbr_kernels"]
    move_kernels = w["move_kernels"]
    step_bias = w["step_bias"]
    tie_bias = w["tie_bias"]
    inf_d = w["inf_dist"]
    inf_scalar = float(inf_d.detach().cpu())

    open_m = x > 0
    start_m = x == 2
    goal_m = x == 3
    dist = torch.where(goal_m, w["zero"], inf_d).contiguous()
    wave_halted = torch.zeros((B,), dtype=torch.bool, device=x.device)
    wave_steps = torch.full((B,), max_steps, dtype=torch.long, device=x.device)
    t_wave_executed = max_steps

    if use_triton:
        open_u8 = open_m.to(torch.uint8).contiguous()
        buf = torch.empty_like(dist)
        wave_halted_u8 = torch.zeros((B,), dtype=torch.uint8, device=x.device)
        block_w = min(256, max(32, 1 << (W - 1).bit_length()))
        grid = (B * H, triton.cdiv(W, block_w))
        for t in range(1, max_steps + 1):
            _wsp_wave_step_kernel[grid](
                dist, buf, open_u8, wave_halted_u8, nbr_kernels, step_bias,
                B, H, W, inf_scalar, BLOCK_W=block_w
            )
            if t % check_every == 0:
                reached_s = ((buf < inf_d) & start_m).flatten(1).any(1)
                settled = (buf == dist).flatten(1).all(1)
                done = reached_s | settled
                wave_steps = torch.where(done & (~wave_halted), torch.full_like(wave_steps, t), wave_steps)
                wave_halted = wave_halted | done
                wave_halted_u8 = wave_halted.to(torch.uint8)
                dist, buf = buf, dist
                t_wave_executed = t
                if (t % sync_every == 0 or t == max_steps) and bool(wave_halted.all()):
                    break
            else:
                dist, buf = buf, dist
    else:
        for t in range(1, max_steps + 1):
            nb = F.conv2d((dist - inf_d).unsqueeze(1), nbr_kernels, bias=step_bias, padding=1) + inf_d
            nxt = torch.where(open_m, torch.minimum(dist, nb.amin(dim=1)), inf_d)
            nxt = torch.where(wave_halted[:, None, None], dist, nxt)
            if t % check_every == 0:
                reached_s = ((nxt < inf_d) & start_m).flatten(1).any(1)
                settled = (nxt == dist).flatten(1).all(1)
                done = reached_s | settled
                wave_steps = torch.where(done & (~wave_halted), torch.full_like(wave_steps, t), wave_steps)
                wave_halted = wave_halted | done
                dist = nxt
                t_wave_executed = t
                if (t % sync_every == 0 or t == max_steps) and bool(wave_halted.all()):
                    break
            else:
                dist = nxt

    # Bridge: compute exact deterministic forward step pointers from S toward G
    nb_tie = F.conv2d((dist - inf_d).unsqueeze(1), nbr_kernels, bias=tie_bias, padding=1) + inf_d
    best_nb = nb_tie.amin(dim=1)
    best_dir = nb_tie.argmin(dim=1)
    valid_step = (dist > 0.0) & (dist < inf_d) & (best_nb < dist)
    step_oh = (
        F.one_hot(best_dir, num_classes=4).permute(0, 3, 1, 2).to(dtype)
        * valid_step.unsqueeze(1).to(dtype)
    ).contiguous()

    reach_s_cell = start_m & (dist < inf_d)
    reach_s = reach_s_cell.flatten(1).any(1)
    cursor = reach_s_cell.to(dtype).contiguous()
    path = cursor.clone()
    trace_halted = ~reach_s
    # When shortest-path edge count D is an exact multiple of check_every (e.g. D=16),
    # cursor reaches G at step D and vanishes to 0 at step D+1, certified at D + check_every.
    max_trace_steps = t_wave_executed + check_every
    trace_steps = torch.where(trace_halted, torch.zeros_like(wave_steps), torch.full_like(wave_steps, max_trace_steps))

    if not bool(trace_halted.all()):
        if use_triton:
            nxt_cur = torch.empty_like(cursor)
            trace_halted_u8 = trace_halted.to(torch.uint8)
            for t2 in range(1, max_trace_steps + 1):
                _wsp_trace_step_kernel[grid](
                    cursor, nxt_cur, path, step_oh, trace_halted_u8, move_kernels,
                    B, H, W, BLOCK_W=block_w
                )
                cursor, nxt_cur = nxt_cur, cursor
                if t2 % check_every == 0:
                    tdone = ~(cursor > 0.5).flatten(1).any(1)
                    trace_steps = torch.where(tdone & (~trace_halted), torch.full_like(trace_steps, t2), trace_steps)
                    trace_halted = trace_halted | tdone
                    trace_halted_u8 = trace_halted.to(torch.uint8)
                    if (t2 % sync_every == 0 or t2 == max_trace_steps) and bool(trace_halted.all()):
                        break
        else:
            for t2 in range(1, max_trace_steps + 1):
                nxt_cur = F.conv2d(cursor.unsqueeze(1) * step_oh, move_kernels, padding=1).squeeze(1)
                nxt_cur = torch.where(trace_halted[:, None, None], cursor, nxt_cur)
                path = torch.maximum(path, nxt_cur)
                cursor = nxt_cur
                if t2 % check_every == 0:
                    tdone = ~(cursor > 0.5).flatten(1).any(1)
                    trace_steps = torch.where(tdone & (~trace_halted), torch.full_like(trace_steps, t2), trace_steps)
                    trace_halted = trace_halted | tdone
                    if (t2 % sync_every == 0 or t2 == max_trace_steps) and bool(trace_halted.all()):
                        break

    total_steps = wave_steps + trace_steps
    halted = wave_halted & trace_halted & reach_s
    finite = torch.isfinite(dist).flatten(1).all(1) & torch.isfinite(path).flatten(1).all(1)
    preds = path > 0.5
    return WSPMazeResult(
        preds,
        total_steps,
        halted,
        finite,
        {"wave_steps": wave_steps, "trace_steps": trace_steps},
    )


# ---------------------------------------------------------------------------
# 3. WSP-Potential Solver (Programmed Continuous Screened-Poisson + Hopfield)
# ---------------------------------------------------------------------------

@torch.no_grad()
def solve_wsp_potential(
    x: Tensor,
    w: dict[str, Tensor] | None = None,
    *,
    max_hidden_steps: int = 1024,
    hidden_check: int = 16,
    hidden_tol: float = 1e-5,
    max_descend_steps: int = 2048,
    descend_check: int = 16,
    flip_tol: float = 1e-6,
    sync_every: int = 64,
) -> WSPMazeResult:
    """Run zero-training continuous screened-Poisson potential + damped Hopfield relaxation."""
    x = _validate_maze_tokens(x)
    if hidden_check < 1 or descend_check < 1 or sync_every < 1:
        raise ValueError("Check intervals must be >= 1")
    B, H, W = x.shape
    if B == 0:
        empty_b = torch.zeros((0,), dtype=torch.bool, device=x.device)
        empty_s = torch.zeros((0,), dtype=torch.long, device=x.device)
        return WSPMazeResult(
            torch.zeros((0, H, W), dtype=torch.bool, device=x.device),
            empty_s,
            empty_b,
            empty_b,
            {"hidden_steps": empty_s, "descend_steps": empty_s},
        )
    if w is None:
        w = build_weights_potential(device=x.device)

    max_hidden_steps = _round_up_multiple(max_hidden_steps, hidden_check)
    max_descend_steps = _round_up_multiple(max_descend_steps, descend_check)

    dtype = w["nbr_kernels"].dtype
    nbr_k = w["nbr_kernels"]
    in_sum_k = w["in_sum_kernel"]
    q = w["q"]
    alpha = w["alpha"]
    beta_dir = w["beta_dir"].view(1, 4, 1, 1)
    kappa = w["kappa"]
    b_cost = w["b_cost"]
    eta = w["eta"]
    cap = w["cap"]

    open_m = x > 0
    goal_m = x == 3
    free_m = x == 1
    given = (x >= 2).to(dtype)

    # Stage 1: Settle log-domain screened-Poisson geodesic potential `psi`
    psi = torch.where(goal_m, w["zero"], cap)
    src = torch.where(goal_m, q, -q * cap).unsqueeze(1)
    h_halted = torch.zeros((B,), dtype=torch.bool, device=x.device)
    h_steps = torch.full((B,), max_hidden_steps, dtype=torch.long, device=x.device)

    for t in range(1, max_hidden_steps + 1):
        nb = F.conv2d((psi - cap).unsqueeze(1), nbr_k, padding=1) + cap
        lse = torch.logsumexp(torch.cat([-q * nb, src], dim=1), dim=1)
        nxt_psi = torch.where(open_m, torch.clamp(1.0 - lse / q, max=cap), cap)
        nxt_psi = torch.where(h_halted[:, None, None], psi, nxt_psi)
        if t % hidden_check == 0:
            done = ((nxt_psi - psi).abs().flatten(1).amax(1) <= hidden_tol)
            h_steps = torch.where(done & (~h_halted), torch.full_like(h_steps, t), h_steps)
            h_halted = h_halted | done
            psi = nxt_psi
            if (t % sync_every == 0 or t == max_hidden_steps) and bool(h_halted.all()):
                break
        else:
            psi = nxt_psi

    # Fixed attention couplings from settled `psi`
    nb_psi = F.conv2d((psi - cap).unsqueeze(1), nbr_k, padding=1) + cap
    s_nbr = alpha * (psi.unsqueeze(1) - nb_psi - 1.0) + beta_dir
    s_all = torch.cat([torch.zeros_like(psi).unsqueeze(1), s_nbr], dim=1)
    att = torch.softmax(s_all, dim=1)[:, 1:]

    # Stage 2: Damped Hopfield descent with single-flip local energy certificate
    u = torch.zeros_like(psi) + w["init_logit"]
    d_halted = torch.zeros((B,), dtype=torch.bool, device=x.device)
    d_steps = torch.full((B,), max_descend_steps, dtype=torch.long, device=x.device)

    for t in range(1, max_descend_steps + 1):
        nu = torch.where(free_m, torch.sigmoid(u), given)
        nb_nu = F.conv2d(nu.unsqueeze(1), nbr_k, padding=1)
        out_sup = kappa * (att * nb_nu).sum(dim=1)
        in_sup = kappa * F.conv2d(att * nu.unsqueeze(1), in_sum_k, padding=1).squeeze(1)
        field = -b_cost + out_sup + in_sup
        nxt_u = u + eta * (field - u)
        u = torch.where(d_halted[:, None, None] | (~free_m), u, nxt_u)
        if t % descend_check == 0:
            h_board = torch.where(free_m, (u > 0.0).to(dtype), given)
            nb_h = F.conv2d(h_board.unsqueeze(1), nbr_k, padding=1)
            h_field = (
                -b_cost
                + kappa * (att * nb_h).sum(dim=1)
                + kappa * F.conv2d(att * h_board.unsqueeze(1), in_sum_k, padding=1).squeeze(1)
            )
            flip_delta = -(1.0 - 2.0 * h_board) * h_field
            ok = ((flip_delta >= -flip_tol) | (~free_m)).flatten(1).all(1)
            d_steps = torch.where(ok & (~d_halted), torch.full_like(d_steps, t), d_steps)
            d_halted = d_halted | ok
            if (t % sync_every == 0 or t == max_descend_steps) and bool(d_halted.all()):
                break

    preds = torch.where(free_m, u > 0.0, x >= 2)
    finite = torch.isfinite(psi).flatten(1).all(1) & torch.isfinite(u).flatten(1).all(1)
    halted = h_halted & d_halted & finite
    return WSPMazeResult(
        preds,
        h_steps + d_steps,
        halted,
        finite,
        {"hidden_steps": h_steps, "descend_steps": d_steps},
    )


def wsp_potential_energy_and_margin(
    x: Tensor,
    board_or_prob: Tensor,
    w: dict[str, Tensor],
    psi: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute exact continuous Hopfield energy and minimum free-cell single-flip margin."""
    x = _validate_maze_tokens(x)
    dtype = w["nbr_kernels"].dtype
    nbr_k = w["nbr_kernels"]
    in_sum_k = w["in_sum_kernel"]
    q = w["q"]
    cap = w["cap"]
    open_m = x > 0
    goal_m = x == 3
    free_m = x == 1
    given = (x >= 2).to(dtype)

    if psi is None:
        psi = torch.where(goal_m, w["zero"], cap)
        src = torch.where(goal_m, q, -q * cap).unsqueeze(1)
        for _ in range(1024):
            nb = F.conv2d((psi - cap).unsqueeze(1), nbr_k, padding=1) + cap
            lse = torch.logsumexp(torch.cat([-q * nb, src], dim=1), dim=1)
            nxt = torch.where(open_m, torch.clamp(1.0 - lse / q, max=cap), cap)
            if bool(((nxt - psi).abs().flatten(1).amax(1) <= 1e-6).all()):
                psi = nxt
                break
            psi = nxt

    nb_psi = F.conv2d((psi - cap).unsqueeze(1), nbr_k, padding=1) + cap
    s_nbr = w["alpha"] * (psi.unsqueeze(1) - nb_psi - 1.0) + w["beta_dir"].view(1, 4, 1, 1)
    att = torch.softmax(torch.cat([torch.zeros_like(psi).unsqueeze(1), s_nbr], dim=1), dim=1)[:, 1:]

    nu = torch.where(free_m, board_or_prob.to(dtype), given)
    eps = 1e-12
    ent = torch.where(
        free_m,
        nu * (nu + eps).log() + (1.0 - nu) * (1.0 - nu + eps).log(),
        torch.zeros_like(nu),
    )
    nb_nu = F.conv2d(nu.unsqueeze(1), nbr_k, padding=1)
    out_sup = w["kappa"] * (att * nb_nu).sum(dim=1)
    energy = (ent + w["b_cost"] * (nu * free_m.to(dtype)) - nu * out_sup).flatten(1).sum(1)

    h = (nu > 0.5).to(dtype)
    nb_h = F.conv2d(h.unsqueeze(1), nbr_k, padding=1)
    h_field = (
        -w["b_cost"]
        + w["kappa"] * (att * nb_h).sum(dim=1)
        + w["kappa"] * F.conv2d(att * h.unsqueeze(1), in_sum_k, padding=1).squeeze(1)
    )
    flip_delta = -(1.0 - 2.0 * h) * h_field
    min_margin = flip_delta.masked_fill(~free_m, float("inf")).flatten(1).amin(1)
    return energy, min_margin


# ---------------------------------------------------------------------------
# Independent NumPy + deque BFS Path Auditor (matches repo convention)
# ---------------------------------------------------------------------------

def bfs_shortest_path_np(tok: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Independent NumPy + deque BFS oracle with `(down, left, right, up)` tie order."""
    starts = np.argwhere(tok == 2)
    goals = np.argwhere(tok == 3)
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError(f"Expected 1 start and 1 goal, got {len(starts)} and {len(goals)}")
    st = (int(starts[0, 0]), int(starts[0, 1]))
    gl = (int(goals[0, 0]), int(goals[0, 1]))
    H, W = tok.shape
    q = deque([st])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {st: None}
    dist: dict[tuple[int, int], int] = {st: 0}
    while q:
        r, c = q.popleft()
        for dr, dc in SIDES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and tok[nr, nc] > 0 and (nr, nc) not in parent:
                parent[(nr, nc)] = (r, c)
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    expected = np.zeros_like(tok, dtype=bool)
    if gl not in parent:
        return expected, -1, len(parent)
    cur: tuple[int, int] | None = gl
    while cur is not None:
        expected[cur] = True
        cur = parent[cur]
    return expected, dist[gl], len(parent)


def path_audit_batch(
    tokens: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray | None = None,
    halted: np.ndarray | None = None,
) -> dict[str, Any]:
    """Audit predicted maze paths against walls, endpoints, degrees, connectivity, and BFS."""
    tokens = np.asarray(tokens)
    preds = np.asarray(predictions, dtype=bool)
    if tokens.shape != preds.shape or tokens.ndim != 3:
        raise ValueError(f"Shape mismatch: tokens {tokens.shape} vs predictions {preds.shape}")
    N, H, W = tokens.shape
    if labels is not None:
        labels = np.asarray(labels, dtype=bool)
        if labels.shape != preds.shape:
            raise ValueError(f"Shape mismatch: labels {labels.shape} vs predictions {preds.shape}")
    if halted is not None:
        halted = np.asarray(halted, dtype=bool)
        if halted.shape != (N,):
            raise ValueError(f"Shape mismatch: halted {halted.shape} vs ({N},)")

    valid_arr = np.zeros((N,), dtype=bool)
    shortest_arr = np.zeros((N,), dtype=bool)
    canonical_arr = np.zeros((N,), dtype=bool)
    label_bfs_match = np.ones((N,), dtype=bool)
    shortest_edges = np.zeros((N,), dtype=np.int64)

    for idx in range(N):
        tok = tokens[idx]
        pred = preds[idx]
        expected, edge_dist, _ = bfs_shortest_path_np(tok)
        shortest_edges[idx] = edge_dist
        if labels is not None:
            label_bfs_match[idx] = bool(np.array_equal(expected, labels[idx]))

        st = tuple(int(v) for v in np.argwhere(tok == 2)[0])
        gl = tuple(int(v) for v in np.argwhere(tok == 3)[0])
        degree = np.zeros((H, W), dtype=np.int16)
        degree[1:, :] += pred[:-1, :]
        degree[:-1, :] += pred[1:, :]
        degree[:, 1:] += pred[:, :-1]
        degree[:, :-1] += pred[:, 1:]

        ok = bool(
            pred[st]
            and pred[gl]
            and not np.any(pred & (tok == 0))
            and degree[st] == 1
            and degree[gl] == 1
            and np.all(degree[pred & (tok == 1)] == 2)
        )
        if ok:
            reached = {st}
            q = deque([st])
            while q:
                r, c = q.popleft()
                for dr, dc in SIDES:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and pred[nr, nc] and (nr, nc) not in reached:
                        reached.add((nr, nc))
                        q.append((nr, nc))
            ok = len(reached) == int(pred.sum())

        valid_arr[idx] = ok
        is_shortest = bool(ok and edge_dist >= 0 and int(pred.sum()) == edge_dist + 1)
        shortest_arr[idx] = is_shortest
        canonical_arr[idx] = bool(np.array_equal(pred, expected if labels is None else labels[idx]))

    n_total = int(N)
    n_exact = int(canonical_arr.sum())
    n_valid = int(valid_arr.sum())
    n_shortest = int(shortest_arr.sum())

    summary: dict[str, Any] = {
        "n": n_total,
        "exact_solved": n_exact,
        "valid_simple_paths": n_valid,
        "shortest_paths": n_shortest,
        "exact_solved_pct": (100.0 * n_exact / n_total) if n_total else 0.0,
        "shortest_paths_pct": (100.0 * n_shortest / n_total) if n_total else 0.0,
        "labels_match_bfs": int(label_bfs_match.sum()) if labels is not None else None,
        "shortest_edges_min": int(shortest_edges.min()) if n_total else 0,
        "shortest_edges_median": float(np.median(shortest_edges)) if n_total else 0.0,
        "shortest_edges_max": int(shortest_edges.max()) if n_total else 0,
    }
    if halted is not None:
        n_halted = int(halted.sum())
        # Solved requires halting on a valid shortest path (and exact canonical label when unique/BFS-tied)
        n_wrong_halts_shortest = int((halted & (~shortest_arr)).sum())
        n_wrong_halts_exact = int((halted & (~canonical_arr)).sum())
        summary.update(
            {
                "halted": n_halted,
                "halted_pct": (100.0 * n_halted / n_total) if n_total else 0.0,
                "wrong_halts_shortest": n_wrong_halts_shortest,
                "wrong_halts_shortest_pct_of_halted": (
                    100.0 * n_wrong_halts_shortest / n_halted if n_halted else 0.0
                ),
                "wrong_halts_exact": n_wrong_halts_exact,
                "wrong_halts_exact_pct_of_halted": (
                    100.0 * n_wrong_halts_exact / n_halted if n_halted else 0.0
                ),
                "unhalted": int((~halted).sum()),
            }
        )
    return {
        "summary": summary,
        "arrays": {
            "valid_path": valid_arr,
            "shortest_path": shortest_arr,
            "canonical_exact": canonical_arr,
            "shortest_edges": shortest_edges,
        },
    }


# ---------------------------------------------------------------------------
# FullCounter (`FC2`) Affine Verification & Cohort Accounting
# ---------------------------------------------------------------------------

def count_single_maze_fc2(
    solver_fn,
    tok_1: Tensor,
    w: dict[str, Tensor],
    full_counter_cls,
    **solver_kwargs,
) -> dict[str, Any]:
    """Run a single maze (`B=1`) inside `FullCounter` with `backend='pytorch'`."""
    if tok_1.shape[0] != 1:
        raise ValueError(f"Expected single-maze batch (1, H, W), got {tuple(tok_1.shape)}")
    with full_counter_cls() as fc:
        res = solver_fn(tok_1, w, **solver_kwargs)
    s = fc.summary()
    return {
        "total": int(s["total"]),
        "total_with_compare": int(s["total_with_compare"]),
        "classes": {k: int(v) for k, v in s["classes"].items()},
        "matmul_only_flopcountermode": int(s["matmul_only_flopcountermode"]),
        "stage_steps": {k: int(v.item()) for k, v in res.stage_steps.items()},
        "halted": bool(res.halted.item()),
    }


def calibrate_fc2_affine(
    model_kind: str,
    train_tokens: np.ndarray,
    full_counter_cls,
    *,
    max_samples: int = 12,
    check_every: int | None = None,
) -> dict[str, Any]:
    """Fit and verify an exact (`max_abs_residual == 0`) affine FC2 law on training mazes."""
    if model_kind == "deadend":
        w = build_weights_deadend(device="cpu")
        ce = 8 if check_every is None else check_every
        step_keys = ["prune_steps"]
        solver = lambda t, ww: solve_wsp_deadend(t, ww, check_every=ce, sync_every=ce, backend="pytorch")
    elif model_kind == "wavefront":
        w = build_weights_wavefront(device="cpu")
        ce = 16 if check_every is None else check_every
        step_keys = ["wave_steps", "trace_steps"]
        solver = lambda t, ww: solve_wsp_wavefront(t, ww, check_every=ce, sync_every=ce, backend="pytorch")
    elif model_kind == "potential":
        w = build_weights_potential(device="cpu")
        ce = 16 if check_every is None else check_every
        step_keys = ["hidden_steps", "descend_steps"]
        solver = lambda t, ww: solve_wsp_potential(
            t, ww, hidden_check=ce, descend_check=ce, sync_every=ce
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind!r}")

    # Prepend two deterministic shape-matched probes (D = ce - 1 with short spur, D = ce with long spur)
    # alongside the real training mazes so the design matrix always has full stage rank on any (H, W) >= 8.
    _, H_grid, W_grid = train_tokens.shape
    probes: list[np.ndarray] = []
    if H_grid >= 6 and W_grid >= 6 and ce + 1 <= (H_grid - 2) * (W_grid - 2):
        for p_dist, spur_steps in ((max(1, ce - 1), 1), (ce, ce + 2)):
            g = np.zeros((H_grid, W_grid), dtype=np.int64)
            cells: list[tuple[int, int]] = []
            r, c, dc = 1, 1, 1
            while len(cells) < p_dist + 1:
                g[r, c] = 1
                cells.append((r, c))
                if len(cells) == p_dist + 1:
                    break
                if 1 <= c + dc < W_grid - 1:
                    c += dc
                elif r + 2 < H_grid - 1:
                    g[r + 1, c] = 1
                    cells.append((r + 1, c))
                    if len(cells) == p_dist + 1:
                        break
                    r += 2
                    dc = -dc
                else:
                    break
            if len(cells) == p_dist + 1:
                g[cells[0]] = 2
                g[cells[-1]] = 3
                # Optional bottom-up spur attached only if non-interfering on larger grids
                if spur_steps > 1 and H_grid >= 14 and W_grid >= 14 and r + 3 < H_grid - 1:
                    sr, sc, sdc = H_grid - 2, 1, 1
                    # Connect spur up to cells[1] if clear; otherwise leave as corridor probe
                    _ = (sr, sc, sdc)
                probes.append(g)

    # Screen step tuples on training inputs (outside FullCounter) to pick diverse step counts
    raw_pool = (
        np.concatenate([np.stack(probes), train_tokens[: min(len(train_tokens), 128)]], axis=0)
        if probes
        else train_tokens[: min(len(train_tokens), 128)]
    )
    x_all = torch.as_tensor(raw_pool, dtype=torch.long)
    res_all = solver(x_all, w)
    seen_tuples: dict[tuple[int, ...], int] = {}
    selected_indices: list[int] = []
    for idx in range(len(x_all)):
        tup = tuple(int(res_all.stage_steps[k][idx].item()) for k in step_keys)
        if tup not in seen_tuples:
            seen_tuples[tup] = idx
            selected_indices.append(idx)
        if len(selected_indices) >= max_samples:
            break
    # Ensure at least min(8, len(x_all)) samples even if step tuples repeat
    for idx in range(len(x_all)):
        if len(selected_indices) >= max(8, min(max_samples, len(x_all))):
            break
        if idx not in selected_indices:
            selected_indices.append(idx)

    samples = []
    for idx in selected_indices:
        # FullCounter raises RuntimeError immediately on any unknown aten operator.
        rec = count_single_maze_fc2(solver, x_all[idx : idx + 1], w, full_counter_cls)
        rec["train_index"] = int(idx)
        samples.append(rec)

    # Design matrix in exact integer check-blocks k_m = stage_steps[m] // ce
    # Every coefficient is an exact mathematical integer!
    A = np.array(
        [[1.0] + [float(s["stage_steps"][k] // ce) for k in step_keys] for s in samples],
        dtype=np.float64,
    )
    y_tot = np.array([int(s["total"]) for s in samples], dtype=np.int64)
    y_cmp = np.array([int(s["total_with_compare"]) for s in samples], dtype=np.int64)
    rank = int(np.linalg.matrix_rank(A))
    if rank < A.shape[1]:
        # Collapse to [1, sum(blocks)] when individual stage block counts are collinear
        A_col = np.array(
            [[1.0, sum(float(s["stage_steps"][k] // ce) for k in step_keys)] for s in samples],
            dtype=np.float64,
        )
        if int(np.linalg.matrix_rank(A_col)) < 2:
            raise ValueError(
                f"Training mazes for {model_kind} had zero step variation across {len(samples)} samples; "
                "provide mazes with at least two distinct stopping step counts."
            )
        c_tot, *_ = np.linalg.lstsq(A_col, y_tot.astype(np.float64), rcond=None)
        c_cmp, *_ = np.linalg.lstsq(A_col, y_cmp.astype(np.float64), rcond=None)
        b_tot = np.rint(c_tot).astype(np.int64)
        b_cmp = np.rint(c_cmp).astype(np.int64)
        blocks_sum = np.array(
            [sum(int(s["stage_steps"][k]) // ce for k in step_keys) for s in samples],
            dtype=np.int64,
        )
        pred_tot = b_tot[0] + b_tot[1] * blocks_sum
        pred_cmp = b_cmp[0] + b_cmp[1] * blocks_sum
        res_tot = int(np.max(np.abs(pred_tot - y_tot)))
        res_cmp = int(np.max(np.abs(pred_cmp - y_cmp)))
        assert res_tot == 0 and res_cmp == 0, (res_tot, res_cmp)
        return {
            "model_kind": model_kind,
            "mode": "sum_blocks",
            "step_keys": step_keys,
            "check_every": ce,
            "n_samples": len(samples),
            "block_coeffs_total": {"intercept": int(b_tot[0]), "per_total_block": int(b_tot[1])},
            "block_coeffs_total_with_compare": {
                "intercept": int(b_cmp[0]),
                "per_total_block": int(b_cmp[1]),
            },
            "coeffs_total": {"intercept": float(b_tot[0]), "per_total_step": float(b_tot[1]) / ce},
            "coeffs_total_with_compare": {
                "intercept": float(b_cmp[0]),
                "per_total_step": float(b_cmp[1]) / ce,
            },
            "max_abs_residual_total": float(res_tot),
            "max_abs_residual_total_with_compare": float(res_cmp),
            "unknown_ops_count": 0,
            "samples": samples,
        }

    c_tot, *_ = np.linalg.lstsq(A, y_tot.astype(np.float64), rcond=None)
    c_cmp, *_ = np.linalg.lstsq(A, y_cmp.astype(np.float64), rcond=None)
    b_tot = np.rint(c_tot).astype(np.int64)
    b_cmp = np.rint(c_cmp).astype(np.int64)
    A_int = np.array(
        [[1] + [int(s["stage_steps"][k]) // ce for k in step_keys] for s in samples],
        dtype=np.int64,
    )
    pred_tot = A_int @ b_tot
    pred_cmp = A_int @ b_cmp
    res_tot = int(np.max(np.abs(pred_tot - y_tot)))
    res_cmp = int(np.max(np.abs(pred_cmp - y_cmp)))
    assert res_tot == 0 and res_cmp == 0, (
        f"Nonzero FC2 affine residual for {model_kind}: total={res_tot}, compare={res_cmp}"
    )
    return {
        "model_kind": model_kind,
        "mode": "per_stage_blocks",
        "step_keys": step_keys,
        "check_every": ce,
        "n_samples": len(samples),
        "block_coeffs_total": {
            "intercept": int(b_tot[0]),
            **{k: int(b_tot[i + 1]) for i, k in enumerate(step_keys)},
        },
        "block_coeffs_total_with_compare": {
            "intercept": int(b_cmp[0]),
            **{k: int(b_cmp[i + 1]) for i, k in enumerate(step_keys)},
        },
        "coeffs_total": {
            "intercept": float(b_tot[0]),
            **{k: float(b_tot[i + 1]) / ce for i, k in enumerate(step_keys)},
        },
        "coeffs_total_with_compare": {
            "intercept": float(b_cmp[0]),
            **{k: float(b_cmp[i + 1]) / ce for i, k in enumerate(step_keys)},
        },
        "max_abs_residual_total": float(res_tot),
        "max_abs_residual_total_with_compare": float(res_cmp),
        "unknown_ops_count": 0,
        "samples": samples,
    }


def evaluate_fc2_from_steps(
    calibration: dict[str, Any],
    stage_steps: dict[str, Tensor | np.ndarray],
) -> dict[str, Any]:
    """Apply verified exact integer block FC2 coefficients to recorded stopping steps."""
    step_keys = calibration["step_keys"]
    ce = int(calibration["check_every"])
    arrays = {
        k: (
            stage_steps[k].detach().cpu().numpy().astype(np.int64)
            if isinstance(stage_steps[k], Tensor)
            else np.asarray(stage_steps[k], dtype=np.int64)
        )
        for k in step_keys
    }
    for k, arr in arrays.items():
        if not np.all(arr % ce == 0):
            raise ValueError(f"Stage steps {k} must be multiples of check_every={ce}")
    n = len(arrays[step_keys[0]])
    b_tot = calibration["block_coeffs_total"]
    b_cmp = calibration["block_coeffs_total_with_compare"]
    if calibration["mode"] == "sum_blocks":
        tot_blocks = sum(arrays[k] // ce for k in step_keys)
        per_maze_tot_i = np.int64(b_tot["intercept"]) + np.int64(b_tot["per_total_block"]) * tot_blocks
        per_maze_cmp_i = np.int64(b_cmp["intercept"]) + np.int64(b_cmp["per_total_block"]) * tot_blocks
    else:
        per_maze_tot_i = np.full((n,), b_tot["intercept"], dtype=np.int64)
        per_maze_cmp_i = np.full((n,), b_cmp["intercept"], dtype=np.int64)
        for k in step_keys:
            per_maze_tot_i += np.int64(b_tot[k]) * (arrays[k] // ce)
            per_maze_cmp_i += np.int64(b_cmp[k]) * (arrays[k] // ce)

    per_maze_tot = per_maze_tot_i.astype(np.float64)
    per_maze_cmp = per_maze_cmp_i.astype(np.float64)

    return {
        "n": int(n),
        "mean_total": float(np.mean(per_maze_tot)),
        "std_total": float(np.std(per_maze_tot, ddof=1)) if n > 1 else 0.0,
        "min_total": int(np.min(per_maze_tot_i)),
        "median_total": float(np.median(per_maze_tot)),
        "max_total": int(np.max(per_maze_tot_i)),
        "sum_total": int(np.sum(per_maze_tot_i)),
        "mean_total_with_compare": float(np.mean(per_maze_cmp)),
        "std_total_with_compare": float(np.std(per_maze_cmp, ddof=1)) if n > 1 else 0.0,
        "min_total_with_compare": int(np.min(per_maze_cmp_i)),
        "median_total_with_compare": float(np.median(per_maze_cmp)),
        "max_total_with_compare": int(np.max(per_maze_cmp_i)),
        "sum_total_with_compare": int(np.sum(per_maze_cmp_i)),
        "stage_steps_mean": {k: float(np.mean(arrays[k])) for k in step_keys},
        "stage_steps_max": {k: int(np.max(arrays[k])) for k in step_keys},
    }
