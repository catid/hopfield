"""Core Weight-Space Programming (WSP) contract, static AST verifier, weight-size metadata,
FullCounter affine-extrapolation audit, and atomic JSON output utilities.

Implements the fixed tensor-program rules of EFFL Lab's Weight-Space Programming repository
(`effl-lab/Weight-Space-Programming`, commit `02c31abd3b34d84854cc80d9717717c45365c719`, MIT,
`wsp/free_model.py:46-52, 554-580`) together with the unified parameter/operation accounting
used across `/home/catid/hopfield`.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import statistics
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

WSP_UPSTREAM_REPO = "https://github.com/effl-lab/Weight-Space-Programming"
WSP_UPSTREAM_COMMIT = "02c31abd3b34d84854cc80d9717717c45365c719"
WSP_UPSTREAM_LICENSE = "MIT"

# Exact constants from Weight-Space-Programming/wsp/free_model.py:46-52
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
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "vars",
    "dict",
    "set",
    "str",
    "repr",
    "int",
    "float",
    "bool",
    "hash",
    "print",
    "sorted",
    "min",
    "max",
    "any",
    "all",
    "map",
    "filter",
    "iter",
    "next",
}
ARCH_FORBIDDEN_ATTRS = {
    "tolist",
    "item",
    "numpy",
    "load",
    "save",
    "data_ptr",
    "storage",
    "apply_",
    "map_",
}


def static_problems(source: str, signature: str = "forward(w, x)") -> list[str]:
    """Exact replica of `wsp/free_model.py:static_problems`.

    Returns a list of violations explaining why `source` is not a fixed branchless
    tensor program (empty list `[]` when valid).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"arch.py syntax error: {e}"]
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
        elif isinstance(node, ast.Attribute) and (
            node.attr in ARCH_FORBIDDEN_ATTRS or node.attr.startswith("__")
        ):
            problems.append(f"arch.py: attribute .{node.attr} not allowed (line {node.lineno})")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "forward" for n in tree.body):
        problems.append(f"arch.py: no top-level {signature}")
    return problems[:20]


def check_wsp_static_tensor_program(
    source_or_fn: str | Callable[..., Any],
    signature: str = "forward(w, x)",
) -> list[str]:
    """Verify a WSP `arch.py` module source string or Python function against `static_problems`."""
    if callable(source_or_fn):
        source = textwrap.dedent(inspect.getsource(source_or_fn))
    else:
        source = source_or_fn
    return static_problems(source, signature=signature)


def _distinct_nonzero_values(tensors: list[torch.Tensor], *, round_sig: int = 12) -> int:
    """Count distinct nonzero scalar values across a list of tensors."""
    uniq: set[float] = set()
    for t in tensors:
        det = t.detach().cpu()
        if det.is_sparse:
            vals = det.coalesce().values().reshape(-1)
        else:
            vals = det.reshape(-1)
        if vals.numel() == 0:
            continue
        nz = vals[vals != 0]
        if nz.numel() == 0:
            continue
        arr = np.unique(nz.to(torch.float64).numpy())
        for v in arr.tolist():
            uniq.add(round(float(v), round_sig))
    return len(uniq)


def wsp_weight_metadata(
    weights: Mapping[str, torch.Tensor],
    *,
    index_keys: set[str] | frozenset[str] = frozenset(),
    params_learned: int = 0,
    training_examples: int = 0,
) -> dict[str, Any]:
    """Compute unified parameter & weight-space metadata for a programmed WSP weight dict `w`.

    Follows both EFFL Lab's `wsp/free_model.py` (`weights` = sum numel, `nonzero` = sum nnz)
    and `/home/catid/hopfield`'s Table 2 convention (`params_learned = 0`,
    `distinct_programmed_coefficients`, `fit_time_display = "none"`).
    """
    if params_learned != 0 or training_examples != 0:
        raise ValueError("Genuine zero-training WSP models must have params_learned=0 and training_examples=0.")

    all_tensors: list[torch.Tensor] = []
    coeff_tensors: list[torch.Tensor] = []
    per_tensor: dict[str, dict[str, Any]] = {}

    materialized_entries = 0
    nnz_total = 0
    storage_bytes = 0

    for name, tensor in weights.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"Weight {name!r} is {type(tensor).__name__}, expected torch.Tensor")
        all_tensors.append(tensor)
        is_index = (name in index_keys) or (not tensor.is_floating_point())
        if name not in index_keys and tensor.is_floating_point():
            coeff_tensors.append(tensor)

        det = tensor.detach()
        if det.is_sparse:
            coal = det.coalesce()
            numel = int(det.numel())
            nnz = int(coal.values().count_nonzero().item())
            nbytes = int(
                coal.indices().numel() * coal.indices().element_size()
                + coal.values().numel() * coal.values().element_size()
            )
        else:
            numel = int(det.numel())
            nnz = int(det.count_nonzero().item())
            nbytes = int(det.numel() * det.element_size())

        materialized_entries += numel
        nnz_total += nnz
        storage_bytes += nbytes
        per_tensor[name] = {
            "shape": list(det.shape),
            "dtype": str(det.dtype).replace("torch.", ""),
            "numel": numel,
            "nnz": nnz,
            "bytes": nbytes,
            "role": "routing_or_mask" if is_index else "programmed_weight",
        }

    distinct_float_coeffs = _distinct_nonzero_values(coeff_tensors) if coeff_tensors else _distinct_nonzero_values(all_tensors)
    distinct_all_nonzero = _distinct_nonzero_values(all_tensors)

    return {
        "params_learned": 0,
        "training_examples": 0,
        "fit_time_display": "none",
        "distinct_programmed_coefficients": distinct_float_coeffs,
        "distinct_nonzero_scalars_all_tensors": distinct_all_nonzero,
        "materialized_entries": materialized_entries,
        "nnz": nnz_total,
        "wsp_size": {"weights": materialized_entries, "nonzero": nnz_total},
        "storage_bytes": storage_bytes,
        "per_tensor": per_tensor,
        "upstream_wsp": {
            "repo": WSP_UPSTREAM_REPO,
            "commit": WSP_UPSTREAM_COMMIT,
            "license": WSP_UPSTREAM_LICENSE,
        },
    }


def measure_build_times(
    build_fn: Callable[[str | torch.device], Mapping[str, torch.Tensor]],
    *,
    warm_repeats: int = 7,
    cuda_device: str | None = "cuda",
) -> dict[str, Any]:
    """Measure weight-construction (`build()`) cold + warm wall-clock time on 1-thread CPU and GPU."""
    out: dict[str, Any] = {}

    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        t0 = time.perf_counter()
        _ = build_fn("cpu")
        cpu_cold = time.perf_counter() - t0
        cpu_warm: list[float] = []
        for _ in range(warm_repeats):
            t0 = time.perf_counter()
            _ = build_fn("cpu")
            cpu_warm.append(time.perf_counter() - t0)
        out["cpu_1thread"] = {
            "cold_s": cpu_cold,
            "warm_median_s": float(statistics.median(cpu_warm)),
            "warm_mean_s": float(statistics.mean(cpu_warm)),
            "warm_samples_s": cpu_warm,
        }
    finally:
        torch.set_num_threads(prev_threads)

    if cuda_device is not None and torch.cuda.is_available():
        dev = torch.device(cuda_device)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        _ = build_fn(dev)
        torch.cuda.synchronize(dev)
        gpu_cold = time.perf_counter() - t0

        gpu_warm: list[float] = []
        for _ in range(warm_repeats):
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            _ = build_fn(dev)
            torch.cuda.synchronize(dev)
            gpu_warm.append(time.perf_counter() - t0)
        out["gpu"] = {
            "device_name": torch.cuda.get_device_name(dev),
            "cold_s": gpu_cold,
            "warm_median_s": float(statistics.median(gpu_warm)),
            "warm_mean_s": float(statistics.mean(gpu_warm)),
            "warm_samples_s": gpu_warm,
        }
    return out


def verify_fullcounter_affine(
    count_fn: Callable[[int], dict[str, Any]],
    step_grid: tuple[int, ...] = (0, 1, 2, 3, 5, 8, 13, 21),
) -> dict[str, Any]:
    """Verify exact affine step-count scaling `flops(S) = setup + S * per_step` under FullCounter.

    Requires >= 8 distinct step counts and checks `max_abs_residual == 0` for integer counts.
    """
    if len(step_grid) < 8 or len(set(step_grid)) != len(step_grid):
        raise ValueError(f"step_grid must have >= 8 distinct step counts, got {step_grid}")

    s0, s1 = step_grid[0], step_grid[1]
    if s1 == s0:
        raise ValueError("First two step_grid points must differ.")

    rec0 = count_fn(s0)
    rec1 = count_fn(s1)

    slope_total_num = int(rec1["total"]) - int(rec0["total"])
    slope_cmp_num = int(rec1["total_with_compare"]) - int(rec0["total_with_compare"])
    ds = s1 - s0
    if slope_total_num % ds != 0 or slope_cmp_num % ds != 0:
        raise AssertionError(
            f"Non-integer per-step FLOP slope between S={s0} and S={s1}: "
            f"d_total={slope_total_num}, d_cmp={slope_cmp_num}, ds={ds}"
        )
    per_step_total = slope_total_num // ds
    per_step_cmp = slope_cmp_num // ds
    setup_total = int(rec0["total"]) - s0 * per_step_total
    setup_cmp = int(rec0["total_with_compare"]) - s0 * per_step_cmp

    points: list[dict[str, int]] = []
    max_res_total = 0
    max_res_cmp = 0
    for s in step_grid:
        r = count_fn(s) if s not in (s0, s1) else (rec0 if s == s0 else rec1)
        tot = int(r["total"])
        cmp_tot = int(r["total_with_compare"])
        pred_tot = setup_total + s * per_step_total
        pred_cmp = setup_cmp + s * per_step_cmp
        res_tot = abs(tot - pred_tot)
        res_cmp = abs(cmp_tot - pred_cmp)
        max_res_total = max(max_res_total, res_tot)
        max_res_cmp = max(max_res_cmp, res_cmp)
        points.append(
            {
                "steps": int(s),
                "total": tot,
                "predicted_total": pred_tot,
                "residual_total": res_tot,
                "total_with_compare": cmp_tot,
                "predicted_total_with_compare": pred_cmp,
                "residual_total_with_compare": res_cmp,
            }
        )

    return {
        "setup_total": setup_total,
        "per_step_total": per_step_total,
        "setup_total_with_compare": setup_cmp,
        "per_step_total_with_compare": per_step_cmp,
        "max_abs_residual_total": max_res_total,
        "max_abs_residual_total_with_compare": max_res_cmp,
        "step_grid_points": points,
        "sample_summary_at_s1": rec1,
    }


def sha256_file(path: str | Path) -> str:
    """Compute hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(obj: Any, path: str | Path, *, indent: int = 2) -> str:
    """Atomically write strict JSON (`allow_nan=False`) via PID-tagged `.tmp` + `os.replace`.

    Returns the SHA-256 hex digest of the written file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=indent, allow_nan=False) + "\n"
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    return sha256_file(target)
