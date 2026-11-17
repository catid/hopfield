#!/usr/bin/env python3
"""Benchmark, verify, and audit Weight-Space Programming (WSP) on all Maze tasks.

Subcommands:
- `smoke-dev`: Pre-experiment verification & latency profiling strictly on `train`/`dev` splits
  (`Maze-Unique` `dev100`, `Maze-Hard` 900/100 `rng(0)` dev split from `train.csv`, and synthetic
  `180x180` loop80 mazes with `base_seed=999999`). Hard-blocks all `test` splits and sealed fresh
  `180x180` cohorts.
- `full-test`: Full 3-seed (`0, 1, 2`) evaluations on `Maze-Unique` test (`1,000`), `Maze-Hard`
  test (`1,000`, including explicit maze `106` audit), and the 3 README `180x180` 80%-loop cohorts
  (`H2bo` Corner, `H2bp` Uniform, `H2bq` Opposite-region, `4,096` mazes each, 4 balanced paired
  complete-cohort timing replays + labels-last independent graph audit). Requires `--verifier-approved`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

# Enforce /home temporary directory hygiene before importing torch/triton
for _env_key, _fallback in (
    ("TMPDIR", "/home/catid/.cache/hopfield_tmp"),
    ("TRITON_CACHE_DIR", "/home/catid/.cache/hopfield_tmp/triton_maze"),
    ("TORCHINDUCTOR_CACHE_DIR", "/home/catid/.cache/hopfield_tmp/inductor_maze"),
):
    _val = os.environ.get(_env_key, "")
    if not _val or _val.startswith("/tmp") or _val.startswith("/var/tmp"):
        os.environ[_env_key] = _fallback
    Path(os.environ[_env_key]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[1]
for _p in (
    ROOT / "src",
    ROOT / "upstream",
    ROOT / "upstream/baselines/flops",
    ROOT / "reproduction/baselines/code/Weight-Space-Programming",
    ROOT / "research/20261114/work",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

os.environ.setdefault("CRAM_DATA", str(ROOT / "upstream/data"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cram import maze_h3, maze_unique  # noqa: E402
import endpoint_shift_generator as G_UNIFORM  # noqa: E402
import far_placement_generator as G_FAR  # noqa: E402
from fullcount import FullCounter  # noqa: E402
import study as WORK_STUDY  # noqa: E402
from wsp.free_model import static_problems  # noqa: E402
from hopfield_cram import wsp_maze as M  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ---------------------------------------------------------------------------
# Atomic I/O, Hashing & Provenance Helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    c = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(f"{c.dtype}:{c.shape}:".encode("utf-8"))
    h.update(c.view(np.uint8).data)
    return h.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)


def atomic_savez_compressed(path: Path, **arrays: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Note: np.savez_compressed appends .npz if missing, so end tmp with .npz
    tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return sha256_file(path)


def gpu_inventory() -> dict[str, Any]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    info: dict[str, Any] = {
        "CUDA_VISIBLE_DEVICES": vis,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "TMPDIR": os.environ.get("TMPDIR"),
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
        "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    }
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        info["nvidia_smi_all_gpus"] = smi.splitlines()
    except Exception as exc:  # pragma: no cover
        info["nvidia_smi_error"] = repr(exc)
    return info


def runtime_provenance() -> dict[str, Any]:
    wsp_dir = ROOT / "reproduction/baselines/code/Weight-Space-Programming"
    wsp_commit = None
    if (wsp_dir / ".git").exists():
        wsp_commit = subprocess.check_output(
            ["git", "-C", str(wsp_dir), "rev-parse", "HEAD"], text=True
        ).strip()
    return {
        "utc": utc_now(),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": getattr(M.triton, "__version__", None) if M.HAS_TRITON else None,
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "wsp_upstream_repo": "https://github.com/effl-lab/Weight-Space-Programming",
        "wsp_upstream_commit": wsp_commit,
        "wsp_upstream_license": "MIT",
        "source_sha256": {
            "src/hopfield_cram/wsp_maze.py": sha256_file(ROOT / "src/hopfield_cram/wsp_maze.py"),
            "scripts/run_wsp_maze.py": sha256_file(Path(__file__).resolve()),
            "upstream/baselines/flops/fullcount.py": sha256_file(
                ROOT / "upstream/baselines/flops/fullcount.py"
            ),
        },
        "gpu_environment": gpu_inventory(),
    }


def verify_static_ast_all() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, src in (
        ("WSP-DeadEnd", M.ARCH_SOURCE_DEADEND),
        ("WSP-Wavefront", M.ARCH_SOURCE_WAVEFRONT),
        ("WSP-Potential", M.ARCH_SOURCE_POTENTIAL),
    ):
        probs = static_problems(src, signature="forward(w, x)")
        if probs:
            raise AssertionError(f"static_problems failed for {key}: {probs}")
        out[key] = {
            "static_problems": probs,
            "passed": True,
            "arch_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
        }
    return out


# ---------------------------------------------------------------------------
# Timing Helpers (Synchronized, Warmup-Isolated, Zero D2H Inside Pure Timer)
# ---------------------------------------------------------------------------

def measure_weight_build_times(builder_fn) -> dict[str, Any]:
    """Measure cold and warm (median of 7) weight construction latency on GPU and 1-thread CPU."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        t0 = time.perf_counter()
        w_cpu = builder_fn(device="cpu")
        cpu_cold = time.perf_counter() - t0
        cpu_warm = []
        for _ in range(7):
            t0 = time.perf_counter()
            w_cpu = builder_fn(device="cpu")
            cpu_warm.append(time.perf_counter() - t0)
        del w_cpu
    finally:
        torch.set_num_threads(prev_threads)

    res: dict[str, Any] = {
        "fit_time": "none",
        "cpu_1thread_cold_s": float(cpu_cold),
        "cpu_1thread_warm_median7_s": float(np.median(cpu_warm)),
        "cpu_1thread_warm_all_s": [float(v) for v in cpu_warm],
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        w_gpu = builder_fn(device="cuda")
        torch.cuda.synchronize()
        gpu_cold = time.perf_counter() - t0
        gpu_warm = []
        for _ in range(7):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            w_gpu = builder_fn(device="cuda")
            torch.cuda.synchronize()
            gpu_warm.append(time.perf_counter() - t0)
        del w_gpu
        res.update(
            {
                "gpu_cold_s": float(gpu_cold),
                "gpu_warm_median7_s": float(np.median(gpu_warm)),
                "gpu_warm_all_s": [float(v) for v in gpu_warm],
            }
        )
    return res


def measure_cohort_inference_times(
    tokens_np: np.ndarray,
    builder_fn,
    solver_fn_gpu,
    solver_fn_cpu,
    *,
    gpu_repeats: int = 7,
    cpu_repeats: int = 1,
) -> tuple[M.WSPMazeResult, dict[str, Any]]:
    """Measure warm GPU and 1-thread CPU inference latency (excluding D2H capture)."""
    timing: dict[str, Any] = {"n_mazes": int(len(tokens_np))}

    # 1-thread CPU inference timing
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x_cpu = torch.as_tensor(tokens_np, dtype=torch.long, device="cpu")
        w_cpu = builder_fn(device="cpu")
        # 1-maze warmup on CPU so PyTorch dispatch caches settle without wasting wall time
        _ = solver_fn_cpu(x_cpu[: min(4, len(x_cpu))], w_cpu)
        cpu_times = []
        res_cpu = None
        for _ in range(cpu_repeats):
            t0 = time.perf_counter()
            res_cpu = solver_fn_cpu(x_cpu, w_cpu)
            cpu_times.append(time.perf_counter() - t0)
        timing["cpu_1thread_warm_s"] = float(np.median(cpu_times))
        timing["cpu_1thread_ms_per_maze"] = 1000.0 * timing["cpu_1thread_warm_s"] / len(tokens_np)
    finally:
        torch.set_num_threads(prev_threads)

    if torch.cuda.is_available():
        x_gpu = torch.as_tensor(tokens_np, dtype=torch.long, device="cuda")
        w_gpu = builder_fn(device="cuda")
        # Full warmup outside timed region (includes Triton JIT compilation if cold)
        torch.cuda.synchronize()
        t_warm0 = time.perf_counter()
        res_warm = solver_fn_gpu(x_gpu, w_gpu)
        torch.cuda.synchronize()
        timing["gpu_warmup_including_compile_s"] = float(time.perf_counter() - t_warm0)

        # Verify exact parity between GPU fast path and 1-thread CPU PyTorch path!
        assert res_cpu is not None
        assert torch.equal(res_warm.predictions.cpu(), res_cpu.predictions), "GPU/CPU prediction mismatch"
        assert torch.equal(res_warm.steps.cpu(), res_cpu.steps), "GPU/CPU steps mismatch"
        assert torch.equal(res_warm.halted.cpu(), res_cpu.halted), "GPU/CPU halted mismatch"
        timing["gpu_cpu_exact_parity_verified"] = True

        gpu_times = []
        res_gpu = None
        for _ in range(gpu_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res_gpu = solver_fn_gpu(x_gpu, w_gpu)
            torch.cuda.synchronize()
            gpu_times.append(time.perf_counter() - t0)
        assert res_gpu is not None
        timing["gpu_warm_median_s"] = float(np.median(gpu_times))
        timing["gpu_warm_all_s"] = [float(v) for v in gpu_times]
        timing["gpu_warm_ms_per_maze"] = 1000.0 * timing["gpu_warm_median_s"] / len(tokens_np)
        return res_gpu, timing

    assert res_cpu is not None
    return res_cpu, timing


# ---------------------------------------------------------------------------
# 30x30 Task Evaluation Helpers (`Maze-Unique` & `Maze-Hard`)
# ---------------------------------------------------------------------------

def _model_specs_30x30():
    return [
        (
            "WSP-DeadEnd",
            "deadend",
            M.build_weights_deadend,
            lambda x, w: M.solve_wsp_deadend(x, w, check_every=8, sync_every=64, backend="auto"),
            lambda x, w: M.solve_wsp_deadend(x, w, check_every=8, sync_every=64, backend="pytorch"),
        ),
        (
            "WSP-Wavefront",
            "wavefront",
            M.build_weights_wavefront,
            lambda x, w: M.solve_wsp_wavefront(x, w, check_every=16, sync_every=64, backend="auto"),
            lambda x, w: M.solve_wsp_wavefront(x, w, check_every=16, sync_every=64, backend="pytorch"),
        ),
        (
            "WSP-Potential",
            "potential",
            M.build_weights_potential,
            lambda x, w: M.solve_wsp_potential(x, w, hidden_check=16, descend_check=16, sync_every=64),
            lambda x, w: M.solve_wsp_potential(x, w, hidden_check=16, descend_check=16, sync_every=64),
        ),
    ]


def audit_maze_106(tok_106: np.ndarray, lab_106: np.ndarray) -> dict[str, Any]:
    """Explicit diagnostic audit of Maze-Hard test index 106 across all 3 WSP models."""
    x = torch.as_tensor(tok_106[None], dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
    out: dict[str, Any] = {
        "maze_index": 106,
        "true_path_cells": int(lab_106.sum()),
        "true_shortest_edges": int(lab_106.sum()) - 1,
        "cram_z_reference_note": (
            "In fresh CRAM+z reproduction (seeds 0-2), test maze 106 is the sole failure: "
            "halts at 160 descent steps with all 117 target cells plus 6 extra cells at "
            "(29, 0)..(29, 5) creating invalid interior degrees."
        ),
        "models": {},
    }
    for label, kind, builder, solver_gpu, _ in _model_specs_30x30():
        w = builder(device=x.device)
        res = solver_gpu(x, w)
        pred = res.predictions[0].detach().cpu().numpy().astype(bool)
        halted = bool(res.halted[0].item())
        aud = M.path_audit_batch(tok_106[None], pred[None], lab_106[None], np.array([halted]))["summary"]
        extra_coords = [[int(r), int(c)] for r, c in np.argwhere(pred & (~lab_106))]
        missing_coords = [[int(r), int(c)] for r, c in np.argwhere((~pred) & lab_106)]
        rec: dict[str, Any] = {
            "halted": halted,
            "finite": bool(res.finite[0].item()),
            "total_steps": int(res.steps[0].item()),
            "stage_steps": {k: int(v[0].item()) for k, v in res.stage_steps.items()},
            "predicted_cells": int(pred.sum()),
            "exact_solved": bool(aud["exact_solved"] == 1),
            "valid_simple_path": bool(aud["valid_simple_paths"] == 1),
            "shortest_path": bool(aud["shortest_paths"] == 1),
            "wrong_halt": bool(aud["wrong_halts_shortest"] == 1),
            "extra_cells_count": len(extra_coords),
            "extra_cells_coords_head": extra_coords[:20],
            "missing_cells_count": len(missing_coords),
            "missing_cells_coords_head": missing_coords[:20],
        }
        if kind == "potential":
            w64 = M.build_weights_potential(device=x.device, dtype=torch.float64)
            e_pred, m_pred = M.wsp_potential_energy_and_margin(
                x, torch.as_tensor(pred[None], device=x.device, dtype=torch.float64), w64
            )
            e_true, m_true = M.wsp_potential_energy_and_margin(
                x, torch.as_tensor(lab_106[None], device=x.device, dtype=torch.float64), w64
            )
            rec["hopfield_energy64_predicted"] = float(e_pred[0].item())
            rec["hopfield_min_flip_margin64_predicted"] = float(m_pred[0].item())
            rec["hopfield_energy64_true_label"] = float(e_true[0].item())
            rec["hopfield_min_flip_margin64_true_label"] = float(m_true[0].item())
        out["models"][label] = rec
    return out


# ---------------------------------------------------------------------------
# 180x180 Paired Timing & Independent Graph Audit Helpers
# ---------------------------------------------------------------------------

def _run_180_arm(
    arm_mode: str,
    shards_tokens: list[np.ndarray],
) -> dict[str, Any]:
    """Execute one complete label-blind 180x180 timing arm on GPU (`CUDA_VISIBLE_DEVICES=2`)."""
    if arm_mode not in ("matched_3seed_B32", "unique_1seed_B256"):
        raise ValueError(f"Unknown arm_mode: {arm_mode!r}")

    torch.cuda.synchronize()
    readiness_t0 = time.perf_counter()
    n_models = 3 if arm_mode == "matched_3seed_B32" else 1
    models = {}
    readiness_receipts = []
    for s in range(n_models):
        torch.cuda.synchronize()
        m0 = time.perf_counter()
        models[s] = M.build_weights_wavefront(device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        readiness_receipts.append({"seed": s, "seconds": float(time.perf_counter() - m0)})
    torch.cuda.synchronize()
    readiness_seconds = float(time.perf_counter() - readiness_t0)

    batch_answer_seconds = 0.0
    comp_totals = {
        "input_transfer": 0.0,
        "wsp_wavefront_gpu_solve": 0.0,
        "answer_d2h_return": 0.0,
        "final_clamps_and_acceptance": 0.0,
    }

    n_shards = len(shards_tokens)
    shard_size = len(shards_tokens[0])
    total_unique = n_shards * shard_size

    # Retain primary seed-0 arrays for replay-identity & labels-last audit; verify seed 1 & 2 match seed 0!
    seed0_preds = np.empty((total_unique, 180, 180), dtype=bool)
    seed0_steps = np.empty((total_unique,), dtype=np.int64)
    seed0_wave_steps = np.empty((total_unique,), dtype=np.int64)
    seed0_trace_steps = np.empty((total_unique,), dtype=np.int64)
    seed0_halted = np.empty((total_unique,), dtype=bool)
    seed0_accepted = np.empty((total_unique,), dtype=bool)
    model_inputs = 0
    cross_seed_exact = True

    with torch.no_grad():
        if arm_mode == "matched_3seed_B32":
            for s in range(3):
                w = models[s]
                for sh_idx, sh_tok in enumerate(shards_tokens):
                    for bi in range(0, len(sh_tok), 32):
                        host_tok = sh_tok[bi : bi + 32]
                        global_lo = sh_idx * shard_size + bi
                        global_hi = global_lo + len(host_tok)

                        torch.cuda.synchronize()
                        seg_t0 = time.perf_counter()

                        c0 = time.perf_counter()
                        t_gpu = torch.as_tensor(host_tok, dtype=torch.long, device="cuda")
                        torch.cuda.synchronize()
                        comp_totals["input_transfer"] += time.perf_counter() - c0

                        c0 = time.perf_counter()
                        res = M.solve_wsp_wavefront(t_gpu, w, check_every=16, sync_every=64, backend="triton")
                        torch.cuda.synchronize()
                        comp_totals["wsp_wavefront_gpu_solve"] += time.perf_counter() - c0

                        c0 = time.perf_counter()
                        pred_np = res.predictions.cpu().numpy()
                        steps_np = res.steps.cpu().numpy()
                        wsteps_np = res.stage_steps["wave_steps"].cpu().numpy()
                        tsteps_np = res.stage_steps["trace_steps"].cpu().numpy()
                        halted_np = res.halted.cpu().numpy()
                        finite_np = res.finite.cpu().numpy()
                        torch.cuda.synchronize()
                        comp_totals["answer_d2h_return"] += time.perf_counter() - c0

                        c0 = time.perf_counter()
                        clamps = ((host_tok == 1) | (pred_np == (host_tok >= 2))).all(axis=(1, 2))
                        accepted = halted_np & finite_np & clamps
                        comp_totals["final_clamps_and_acceptance"] += time.perf_counter() - c0

                        torch.cuda.synchronize()
                        batch_answer_seconds += time.perf_counter() - seg_t0
                        model_inputs += len(host_tok)

                        if s == 0:
                            seed0_preds[global_lo:global_hi] = pred_np
                            seed0_steps[global_lo:global_hi] = steps_np
                            seed0_wave_steps[global_lo:global_hi] = wsteps_np
                            seed0_trace_steps[global_lo:global_hi] = tsteps_np
                            seed0_halted[global_lo:global_hi] = halted_np
                            seed0_accepted[global_lo:global_hi] = accepted
                        else:
                            if not (
                                np.array_equal(seed0_preds[global_lo:global_hi], pred_np)
                                and np.array_equal(seed0_steps[global_lo:global_hi], steps_np)
                                and np.array_equal(seed0_accepted[global_lo:global_hi], accepted)
                            ):
                                cross_seed_exact = False
        else:
            w = models[0]
            for sh_idx, sh_tok in enumerate(shards_tokens):
                global_lo = sh_idx * shard_size
                global_hi = global_lo + len(sh_tok)

                torch.cuda.synchronize()
                seg_t0 = time.perf_counter()

                c0 = time.perf_counter()
                t_gpu = torch.as_tensor(sh_tok, dtype=torch.long, device="cuda")
                torch.cuda.synchronize()
                comp_totals["input_transfer"] += time.perf_counter() - c0

                c0 = time.perf_counter()
                res = M.solve_wsp_wavefront(t_gpu, w, check_every=16, sync_every=64, backend="triton")
                torch.cuda.synchronize()
                comp_totals["wsp_wavefront_gpu_solve"] += time.perf_counter() - c0

                c0 = time.perf_counter()
                pred_np = res.predictions.cpu().numpy()
                steps_np = res.steps.cpu().numpy()
                wsteps_np = res.stage_steps["wave_steps"].cpu().numpy()
                tsteps_np = res.stage_steps["trace_steps"].cpu().numpy()
                halted_np = res.halted.cpu().numpy()
                finite_np = res.finite.cpu().numpy()
                torch.cuda.synchronize()
                comp_totals["answer_d2h_return"] += time.perf_counter() - c0

                c0 = time.perf_counter()
                clamps = ((sh_tok == 1) | (pred_np == (sh_tok >= 2))).all(axis=(1, 2))
                accepted = halted_np & finite_np & clamps
                comp_totals["final_clamps_and_acceptance"] += time.perf_counter() - c0

                torch.cuda.synchronize()
                batch_answer_seconds += time.perf_counter() - seg_t0
                model_inputs += len(sh_tok)

                seed0_preds[global_lo:global_hi] = pred_np
                seed0_steps[global_lo:global_hi] = steps_np
                seed0_wave_steps[global_lo:global_hi] = wsteps_np
                seed0_trace_steps[global_lo:global_hi] = tsteps_np
                seed0_halted[global_lo:global_hi] = halted_np
                seed0_accepted[global_lo:global_hi] = accepted

    del models
    return {
        "arm_mode": arm_mode,
        "model_inputs": int(model_inputs),
        "unique_mazes": int(total_unique),
        "model_readiness": readiness_receipts,
        "model_readiness_seconds": float(readiness_seconds),
        "batch_answer_seconds": float(batch_answer_seconds),
        "complete_answer_seconds": float(readiness_seconds + batch_answer_seconds),
        "component_seconds": {k: float(v) for k, v in comp_totals.items()},
        "cross_seed_exact": bool(cross_seed_exact),
        "arrays": {
            "predictions": seed0_preds,
            "steps": seed0_steps,
            "wave_steps": seed0_wave_steps,
            "trace_steps": seed0_trace_steps,
            "halted": seed0_halted,
            "accepted": seed0_accepted,
        },
        "hashes": {
            "predictions_sha256": sha256_array(seed0_preds),
            "steps_sha256": sha256_array(seed0_steps),
            "accepted_sha256": sha256_array(seed0_accepted),
        },
    }


def distance_strata_summary(
    shortest_edges: np.ndarray,
    solved_mask: np.ndarray,
    accepted_mask: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute descriptive distance strata matching H2bp/H2bq (`1..64`, `65..128`, `129..256`, `>=257`)."""
    bins = [
        ("1..64", (shortest_edges >= 1) & (shortest_edges <= 64)),
        ("65..128", (shortest_edges >= 65) & (shortest_edges <= 128)),
        ("129..256", (shortest_edges >= 129) & (shortest_edges <= 256)),
        (">=257", shortest_edges >= 257),
    ]
    rows = []
    for label, m in bins:
        cnt = int(m.sum())
        sol = int((solved_mask & m).sum())
        acc = int((accepted_mask & m).sum())
        rows.append(
            {
                "bin": label,
                "mazes": cnt,
                "joint_solved": sol,
                "model_inputs_denominator": cnt * 3,
                "model_inputs_solved": sol * 3,
                "accepted": acc,
            }
        )
    return rows


def exact_saving_against_baseline(wsp_seconds_list: list[float], base_seconds_list: list[float]) -> dict[str, Any]:
    """Compute exact rational per-pair savings `1 - wsp / baseline` and 4-pair ordinary median."""
    pairs_out = []
    fracs: list[Fraction] = []
    for idx, (w_s, b_s) in enumerate(zip(wsp_seconds_list, base_seconds_list)):
        fw = Fraction(w_s)
        fb = Fraction(b_s)
        sav = Fraction(1, 1) - (fw / fb)
        fracs.append(sav)
        pairs_out.append(
            {
                "pair": idx,
                "wsp_seconds": float(w_s),
                "baseline_seconds": float(b_s),
                "speedup_ratio": float(b_s / w_s),
                "saving": float(sav),
                "exact_saving": f"{sav.numerator}/{sav.denominator}",
            }
        )
    s_sorted = sorted(fracs)
    n = len(s_sorted)
    med = (s_sorted[n // 2 - 1] + s_sorted[n // 2]) / 2 if n % 2 == 0 else s_sorted[n // 2]
    return {
        "pairs": pairs_out,
        "median_saving": float(med),
        "exact_median_saving": f"{med.numerator}/{med.denominator}",
        "median_speedup_ratio": float(np.median([p["speedup_ratio"] for p in pairs_out])),
    }


def clopper_pearson_one_sided(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact one-sided (1 - alpha) Clopper-Pearson bounds via lgamma binomial-tail bisection."""
    if not (0 <= k <= n and n >= 1 and 0.0 < alpha < 1.0):
        raise ValueError(f"Invalid Clopper-Pearson arguments: k={k}, n={n}, alpha={alpha}")
    coefficients = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(n + 1)
    ]

    def tail(p: float, indices: range) -> float:
        if p == 0.0:
            return float(0 in indices)
        if p == 1.0:
            return float(n in indices)
        a, b = math.log(p), math.log1p(-p)
        return math.fsum(math.exp(coefficients[i] + i * a + (n - i) * b) for i in indices)

    def invert(indices: range, increasing: bool) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(70):
            mid = (lo + hi) / 2.0
            value = tail(mid, indices)
            if (value < alpha) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    lower = 0.0 if k == 0 else invert(range(k, n + 1), True)
    upper = 1.0 if k == n else invert(range(k + 1), False)
    return lower, upper


def _gen_180_item(task: tuple[str, int, int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Top-level picklable worker for deterministic 180x180 shard regeneration."""
    law, base_seed, idx = task
    if law == "corner":
        tok, lab, dist, _, _ = WORK_STUDY.maze(
            180, 0.8, np.random.SeedSequence([base_seed, 180, 80, idx])
        )
        return tok.astype(np.uint8), lab.astype(bool), int(dist)
    if law == "uniform":
        tok, lab, dist, *_ = G_UNIFORM.one(idx, base_seed)
        return tok.astype(np.uint8), lab.astype(bool), int(dist)
    if law == "far":
        tok, lab, dist, *_ = G_FAR.one(idx, base_seed)
        return tok.astype(np.uint8), lab.astype(bool), int(dist)
    raise ValueError(f"Unknown 180x180 generator law: {law!r}")


def ensure_180_cohort_shards(
    law: str,
    base_seed: int,
    orig_datasets: list[dict[str, Any]],
    own_cache_dir: Path,
    *,
    max_workers: int = 8,
) -> tuple[list[Path], dict[str, Any]]:
    """Return shard NPZ paths (using existing sealed shards if present, else deterministically
    regenerating into `own_cache_dir` under `artifacts/wsp/maze/` without touching `artifacts/research/`).
    """
    all_orig_exist = bool(orig_datasets) and all(
        (ROOT / rec["path"]).exists() for rec in orig_datasets
    )
    if all_orig_exist:
        paths = []
        for rec in orig_datasets:
            p = ROOT / rec["path"]
            actual_sha = sha256_file(p)
            assert actual_sha == rec["sha256"], f"Original shard SHA-256 mismatch: {p}"
            paths.append(p)
        return paths, {"source": "existing_research_shards", "regenerated": False, "shards": len(paths)}

    own_cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    paths: list[Path] = []
    regen_records: list[dict[str, Any]] = []
    need_any = any(
        not (own_cache_dir / f"shard{int(rec['shard']):02d}.npz").exists() for rec in orig_datasets
    )
    pool = (
        ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"))
        if need_any
        else None
    )
    try:
        for rec in orig_datasets:
            sh = int(rec["shard"])
            n_sh = int(rec["n"])
            sh_path = own_cache_dir / f"shard{sh:02d}.npz"
            if not sh_path.exists():
                assert pool is not None
                indices = list(range(sh * n_sh, (sh + 1) * n_sh))
                items = list(pool.map(_gen_180_item, [(law, base_seed, i) for i in indices], chunksize=8))
                tok_arr = np.stack([it[0] for it in items])
                lab_arr = np.stack([it[1] for it in items])
                edg_arr = np.asarray([it[2] for it in items], dtype=np.int64)
                idx_arr = np.asarray(indices, dtype=np.int64)
                file_sha = atomic_savez_compressed(
                    sh_path,
                    tokens=tok_arr,
                    labels=lab_arr,
                    shortest_edges=edg_arr,
                    indices=idx_arr,
                )
            else:
                file_sha = sha256_file(sh_path)
                with np.load(sh_path, allow_pickle=False) as zf:
                    tok_arr = zf["tokens"]
                    lab_arr = zf["labels"]
                    edg_arr = zf["shortest_edges"]
            paths.append(sh_path)
            regen_records.append(
                {
                    "shard": sh,
                    "n": n_sh,
                    "path": str(sh_path.relative_to(ROOT)) if sh_path.is_relative_to(ROOT) else str(sh_path),
                    "npz_sha256": file_sha,
                    "tokens_array_sha256": sha256_array(tok_arr),
                    "labels_array_sha256": sha256_array(lab_arr),
                    "shortest_edges_array_sha256": sha256_array(edg_arr),
                }
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    manifest_info = {
        "source": "deterministic_seeded_regeneration",
        "regenerated": True,
        "law": law,
        "base_seed": base_seed,
        "regeneration_wall_seconds": float(time.perf_counter() - t0),
        "shards": regen_records,
    }
    atomic_write_json(own_cache_dir / "regenerated_manifest.json", manifest_info)
    return paths, manifest_info


# ---------------------------------------------------------------------------
# Subcommand 1: `smoke-dev` (Pre-Experiment Verification — ZERO Test Access!)
# ---------------------------------------------------------------------------

def cmd_smoke_dev(out_dir: Path) -> None:
    print("[smoke-dev] Starting strict train/dev-only verification and profiling...")
    ast_check = verify_static_ast_all()

    # Load strictly train900 and dev100 splits ONLY!
    mu_tr900 = maze_unique.load("train900")["inputs"].reshape(-1, 30, 30)
    mu_dev = maze_unique.load("dev100")
    mu_dev_tok = mu_dev["inputs"].reshape(-1, 30, 30)
    mu_dev_lab = mu_dev["targets"].reshape(-1, 30, 30).astype(bool)

    mh_tr_all_tok, mh_tr_all_lab = maze_h3.load("train")
    mh_fit_idx, mh_dev_idx = maze_h3.dev_split()
    mh_tr900 = mh_tr_all_tok[mh_fit_idx]
    mh_dev_tok = mh_tr_all_tok[mh_dev_idx]
    mh_dev_lab = mh_tr_all_lab[mh_dev_idx]

    dev_30x30: dict[str, Any] = {"Maze-Unique_dev100": {}, "Maze-Hard_dev100": {}}
    build_benchmarks: dict[str, Any] = {}

    for label, kind, builder, solver_gpu, solver_cpu in _model_specs_30x30():
        w_meta = M.weight_metadata(builder(device="cpu"))
        build_benchmarks[label] = {
            "weight_metadata": w_meta,
            "build_timing": measure_weight_build_times(builder),
        }

        cal_mu = M.calibrate_fc2_affine(kind, mu_tr900, FullCounter, max_samples=12)
        res_mu, tim_mu = measure_cohort_inference_times(
            mu_dev_tok, builder, solver_gpu, solver_cpu, gpu_repeats=7, cpu_repeats=1
        )
        aud_mu = M.path_audit_batch(
            mu_dev_tok,
            res_mu.predictions.detach().cpu().numpy(),
            mu_dev_lab,
            res_mu.halted.detach().cpu().numpy(),
        )["summary"]
        fc2_mu = M.evaluate_fc2_from_steps(cal_mu, res_mu.stage_steps)
        dev_30x30["Maze-Unique_dev100"][label] = {
            "audit": aud_mu,
            "fc2_flops": fc2_mu,
            "fc2_calibration_summary": {
                k: cal_mu[k]
                for k in (
                    "mode",
                    "check_every",
                    "n_samples",
                    "block_coeffs_total",
                    "block_coeffs_total_with_compare",
                    "coeffs_total",
                    "coeffs_total_with_compare",
                    "max_abs_residual_total",
                    "max_abs_residual_total_with_compare",
                )
            },
            "inference_timing_dev100": tim_mu,
        }

        cal_mh = M.calibrate_fc2_affine(kind, mh_tr900, FullCounter, max_samples=12)
        res_mh, tim_mh = measure_cohort_inference_times(
            mh_dev_tok, builder, solver_gpu, solver_cpu, gpu_repeats=7, cpu_repeats=1
        )
        aud_mh = M.path_audit_batch(
            mh_dev_tok,
            res_mh.predictions.detach().cpu().numpy(),
            mh_dev_lab,
            res_mh.halted.detach().cpu().numpy(),
        )["summary"]
        fc2_mh = M.evaluate_fc2_from_steps(cal_mh, res_mh.stage_steps)
        dev_30x30["Maze-Hard_dev100"][label] = {
            "audit": aud_mh,
            "fc2_flops": fc2_mh,
            "fc2_calibration_summary": {
                k: cal_mh[k]
                for k in (
                    "mode",
                    "check_every",
                    "n_samples",
                    "block_coeffs_total",
                    "block_coeffs_total_with_compare",
                    "coeffs_total",
                    "coeffs_total_with_compare",
                    "max_abs_residual_total",
                    "max_abs_residual_total_with_compare",
                )
            },
            "inference_timing_dev100": tim_mh,
        }
        print(
            f"[smoke-dev] {label}: MU_dev100={aud_mu['shortest_paths']}/100 "
            f"(FC2={fc2_mu['mean_total']:.1f}), MH_dev100={aud_mh['shortest_paths']}/100 "
            f"(FC2={fc2_mh['mean_total']:.1f})"
        )

    # Assert expected dev gates!
    assert dev_30x30["Maze-Unique_dev100"]["WSP-DeadEnd"]["audit"]["exact_solved"] == 100
    assert dev_30x30["Maze-Unique_dev100"]["WSP-DeadEnd"]["audit"]["wrong_halts_shortest"] == 0
    assert dev_30x30["Maze-Unique_dev100"]["WSP-Wavefront"]["audit"]["exact_solved"] == 100
    assert dev_30x30["Maze-Hard_dev100"]["WSP-Wavefront"]["audit"]["exact_solved"] == 100
    assert dev_30x30["Maze-Hard_dev100"]["WSP-Potential"]["audit"]["shortest_paths"] == 100

    # Synthetic 180x180 loop80 smoke check (strictly non-test seed 999999, n=64 per law!)
    syn_base = 999999
    cal_180_inputs = np.stack([G_UNIFORM.one(i, syn_base + 10)[0] for i in range(32)]).astype(np.uint8)
    cal_180 = M.calibrate_fc2_affine("wavefront", cal_180_inputs, FullCounter, max_samples=12)
    syn_180: dict[str, Any] = {}
    for law_name, gen_fn in (
        (
            "synthetic_corner",
            lambda idx: WORK_STUDY.maze(180, 0.8, np.random.SeedSequence([syn_base, 180, 80, idx]))[:3],
        ),
        (
            "synthetic_uniform",
            lambda idx: G_UNIFORM.one(idx, syn_base + 1)[:3],
        ),
        (
            "synthetic_opposite_region",
            lambda idx: G_FAR.one(idx, syn_base + 2)[:3],
        ),
    ):
        items = [gen_fn(i) for i in range(64)]
        toks = np.stack([it[0] for it in items]).astype(np.uint8)
        labs = np.stack([it[1] for it in items]).astype(bool)

        # Verify Triton vs PyTorch parity on first 32 180x180 mazes!
        x32 = torch.as_tensor(toks[:32], dtype=torch.long, device="cuda")
        w_gpu = M.build_weights_wavefront("cuda")
        r_tr = M.solve_wsp_wavefront(x32, w_gpu, backend="triton")
        r_py = M.solve_wsp_wavefront(x32, w_gpu, backend="pytorch")
        assert torch.equal(r_tr.predictions, r_py.predictions)
        assert torch.equal(r_tr.steps, r_py.steps)
        assert torch.equal(r_tr.halted, r_py.halted)

        # Mini 2-pair balanced harness on 2 shards of 32
        shards = [toks[:32], toks[32:64]]
        p0_a = _run_180_arm("matched_3seed_B32", shards)
        p0_b = _run_180_arm("unique_1seed_B256", [toks])
        p1_b = _run_180_arm("unique_1seed_B256", [toks])
        p1_a = _run_180_arm("matched_3seed_B32", shards)
        assert p0_a["hashes"] == p1_a["hashes"] == p0_b["hashes"] == p1_b["hashes"]

        aud = M.path_audit_batch(
            toks,
            p0_a["arrays"]["predictions"],
            labs,
            p0_a["arrays"]["halted"],
        )["summary"]
        assert aud["exact_solved"] == 64 and aud["wrong_halts_shortest"] == 0
        fc2_res = M.evaluate_fc2_from_steps(
            cal_180,
            {
                "wave_steps": p0_a["arrays"]["wave_steps"],
                "trace_steps": p0_a["arrays"]["trace_steps"],
            },
        )
        # Project full 4,096-maze (384 B=32 batches = 64x of 6 B=32 batches) runtime!
        b32_sec_per_batch = p0_a["batch_answer_seconds"] / 6.0
        b256_sec_per_64 = p0_b["batch_answer_seconds"]
        syn_180[law_name] = {
            "n": 64,
            "triton_pytorch_bit_exact_parity": True,
            "replay_identity_matched": True,
            "audit": aud,
            "fc2_flops": fc2_res,
            "smoke_matched_3seed_B32_seconds": p0_a["complete_answer_seconds"],
            "projected_full_cohort_12288_B32_seconds_per_pair": float(384.0 * b32_sec_per_batch),
            "projected_full_cohort_4096_B256_seconds_per_pair": float(64.0 * b256_sec_per_64),
        }
        print(
            f"[smoke-dev] 180x180 {law_name}: {aud['exact_solved']}/64 exact, "
            f"projected 12,288-eval B=32 pair = {384.0 * b32_sec_per_batch:.2f}s"
        )

    assert cal_180 is not None
    proj_total_180 = sum(
        4.0
        * (
            syn_180[k]["projected_full_cohort_12288_B32_seconds_per_pair"]
            + syn_180[k]["projected_full_cohort_4096_B256_seconds_per_pair"]
        )
        for k in syn_180
    )
    summary = {
        "status": "passed_pre_experiment_gate",
        "test_splits_opened": False,
        "fresh_4096_cohorts_opened": False,
        "provenance": runtime_provenance(),
        "static_ast_verification": ast_check,
        "models_build_and_metadata": build_benchmarks,
        "dev_30x30_results": dev_30x30,
        "synthetic_180x180_smoke": {
            "fc2_calibration_180x180": {
                k: cal_180[k]
                for k in (
                    "mode",
                    "check_every",
                    "n_samples",
                    "block_coeffs_total",
                    "block_coeffs_total_with_compare",
                    "coeffs_total",
                    "coeffs_total_with_compare",
                    "max_abs_residual_total",
                    "max_abs_residual_total_with_compare",
                )
            },
            "cohorts": syn_180,
            "projected_full_3cohorts_4pairs_gpu_wall_seconds": float(proj_total_180),
        },
    }
    out_file = out_dir / "summary.json"
    digest = atomic_write_json(out_file, summary)
    print(f"[smoke-dev] Wrote {out_file} (sha256={digest})")


# ---------------------------------------------------------------------------
# Subcommand 2: `full-test` (Post-Sign-Off Full Evaluation Across All Benchmarks)
# ---------------------------------------------------------------------------

def _run_30x30_test_benchmark(
    task_name: str,
    train900_tok: np.ndarray,
    test_tok: np.ndarray,
    test_lab: np.ndarray,
    artifacts_dir: Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    """Run 3-seed test evaluation on a 30x30 benchmark (`Maze-Unique` or `Maze-Hard`)."""
    models_out: dict[str, Any] = {}
    for label, kind, builder, solver_gpu, solver_cpu in _model_specs_30x30():
        print(f"[full-test] {task_name} :: {label} calibrating FC2 on train900...")
        cal = M.calibrate_fc2_affine(kind, train900_tok, FullCounter, max_samples=16)
        cal_sha = atomic_write_json(artifacts_dir / f"{kind}_fc2_calibration.json", cal)
        w_meta = M.weight_metadata(builder("cpu"))

        seed_records = []
        ref_preds = None
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            build_time = measure_weight_build_times(builder)
            res, inf_time = measure_cohort_inference_times(
                test_tok,
                builder,
                solver_gpu,
                solver_cpu,
                gpu_repeats=7,
                cpu_repeats=1,
            )
            preds_np = res.predictions.detach().cpu().numpy().astype(bool)
            halted_np = res.halted.detach().cpu().numpy().astype(bool)
            finite_np = res.finite.detach().cpu().numpy().astype(bool)
            steps_np = res.steps.detach().cpu().numpy().astype(np.int64)
            stage_np = {k: v.detach().cpu().numpy().astype(np.int64) for k, v in res.stage_steps.items()}

            if ref_preds is None:
                ref_preds = preds_np
            else:
                assert np.array_equal(ref_preds, preds_np), f"Deterministic WSP differed on seed {seed}"

            aud_full = M.path_audit_batch(test_tok, preds_np, test_lab, halted_np)
            aud = aud_full["summary"]
            fc2 = M.evaluate_fc2_from_steps(cal, stage_np)

            npz_path = artifacts_dir / f"{kind}_seed{seed}_test_predictions.npz"
            npz_sha = atomic_savez_compressed(
                npz_path,
                predictions=preds_np,
                halted=halted_np,
                finite=finite_np,
                steps=steps_np,
                valid_path=aud_full["arrays"]["valid_path"],
                shortest_path=aud_full["arrays"]["shortest_path"],
                canonical_exact=aud_full["arrays"]["canonical_exact"],
                **{f"stage_{k}": v for k, v in stage_np.items()},
            )
            #Solved requires halting on a valid shortest path
            solved_mask = halted_np & aud_full["arrays"]["shortest_path"] & finite_np
            n_solved = int(solved_mask.sum())
            solved_pct = 100.0 * n_solved / len(test_tok)
            seed_records.append(
                {
                    "seed": int(seed),
                    "n": int(len(test_tok)),
                    "solved": n_solved,
                    "solved_pct": solved_pct,
                    "canonical_exact": int((halted_np & aud_full["arrays"]["canonical_exact"] & finite_np).sum()),
                    "halted": int(aud["halted"]),
                    "halted_pct": float(aud["halted_pct"]),
                    "wrong_halts": int(aud["wrong_halts_shortest"]),
                    "wrong_halts_pct_of_halted": float(aud["wrong_halts_shortest_pct_of_halted"]),
                    "wrong_halts_canonical": int(aud["wrong_halts_exact"]),
                    "unhalted": int(aud["unhalted"]),
                    "fc2_flops": fc2,
                    "build_timing": build_time,
                    "inference_timing": inf_time,
                    "path_audit": aud,
                    "predictions_npz": str(npz_path.relative_to(ROOT)),
                    "predictions_sha256": npz_sha,
                }
            )
            print(
                f"[full-test] {task_name} :: {label} seed={seed}: solved={n_solved}/1000 "
                f"({solved_pct:.2f}%), halted={aud['halted']}/1000, wrong_halts={aud['wrong_halts_shortest']}, "
                f"FC2_total={fc2['mean_total']:.2f}, FC2_cmp={fc2['mean_total_with_compare']:.2f}, "
                f"GPU_warm={inf_time.get('gpu_warm_median_s', 0.0)*1000:.3f}ms"
            )

        solved_pcts = [r["solved_pct"] for r in seed_records]
        fc2_tots = [r["fc2_flops"]["mean_total"] for r in seed_records]
        fc2_cmps = [r["fc2_flops"]["mean_total_with_compare"] for r in seed_records]
        gpu_infs = [r["inference_timing"]["gpu_warm_median_s"] for r in seed_records]
        cpu_infs = [r["inference_timing"]["cpu_1thread_warm_s"] for r in seed_records]
        gpu_builds = [r["build_timing"]["gpu_warm_median7_s"] for r in seed_records]
        cpu_builds = [r["build_timing"]["cpu_1thread_warm_median7_s"] for r in seed_records]

        pooled_halted = sum(r["halted"] for r in seed_records)
        pooled_wrong = sum(r["wrong_halts"] for r in seed_records)
        pooled_solved = sum(r["solved"] for r in seed_records)
        pooled_total = sum(r["n"] for r in seed_records)

        models_out[label] = {
            "model": label,
            "kind": kind,
            "weight_metadata": w_meta,
            "fc2_calibration_path": str((artifacts_dir / f"{kind}_fc2_calibration.json").relative_to(ROOT)),
            "fc2_calibration_sha256": cal_sha,
            "fc2_calibration_max_abs_residual_total": cal["max_abs_residual_total"],
            "fc2_calibration_max_abs_residual_total_with_compare": cal[
                "max_abs_residual_total_with_compare"
            ],
            "aggregate_3seeds": {
                "solved_pct_mean": float(np.mean(solved_pcts)),
                "solved_pct_sd": float(np.std(solved_pcts, ddof=1)),
                "solved_per_seed": [r["solved"] for r in seed_records],
                "halted_pct_mean": 100.0 * pooled_halted / pooled_total,
                "wrong_halts_pct_of_halted": (
                    100.0 * pooled_wrong / pooled_halted if pooled_halted else 0.0
                ),
                "pooled_3seed_counts": {
                    "evaluations": pooled_total,
                    "halted": pooled_halted,
                    "wrong_halts": pooled_wrong,
                    "solved": pooled_solved,
                },
                "fc2_mean_total": float(np.mean(fc2_tots)),
                "fc2_mean_total_sd": float(np.std(fc2_tots, ddof=1)),
                "fc2_mean_total_with_compare": float(np.mean(fc2_cmps)),
                "fc2_mean_total_with_compare_sd": float(np.std(fc2_cmps, ddof=1)),
                "fit_time": "none",
                "weight_build_gpu_warm_median_s": float(np.median(gpu_builds)),
                "weight_build_cpu_1thread_warm_median_s": float(np.median(cpu_builds)),
                "inference_gpu_warm_median_s": float(np.median(gpu_infs)),
                "inference_cpu_1thread_warm_median_s": float(np.median(cpu_infs)),
            },
            "seeds": seed_records,
        }
    return models_out


def _run_180x180_cohorts(artifacts_dir: Path) -> dict[str, Any]:
    """Run all 3 README 180x180 loop80 cohorts (H2bo, H2bp, H2bq) with 4 balanced paired replays."""
    # Load published 4-pair LP96 & Cascade costs from results/research/work/endpoint-cost-three-cohorts/provenance.json
    prov_3c_path = ROOT / "results/research/work/endpoint-cost-three-cohorts/provenance.json"
    prov_3c = json.loads(prov_3c_path.read_text(encoding="utf-8"))
    prov_3c_by_slug = {c["cohort"]: c for c in prov_3c["cohorts"]}

    # Calibrate 180x180 FC2 affine law strictly on synthetic uniform-endpoint mazes (seed=999997, zero test access!)
    syn_items = [G_UNIFORM.one(i, 999997)[0] for i in range(32)]
    cal_180 = M.calibrate_fc2_affine("wavefront", np.stack(syn_items), FullCounter, max_samples=12)
    cal_180_path = artifacts_dir / "wavefront_180x180_fc2_calibration.json"
    cal_180_sha = atomic_write_json(cal_180_path, cal_180)

    # Fixed warmup source matching H2bo/H2bp/H2bq: first 32 mazes of H2bh (`loop80_confirmation` shard00)
    warm_shard_path = ROOT / "artifacts/research/20261114/work/loop80_confirmation/shard00.npz"
    if warm_shard_path.exists():
        with np.load(warm_shard_path, allow_pickle=False) as wz:
            warm_tok32 = wz["tokens"][:32].copy()
            warm_tok256 = wz["tokens"].copy()
        warm_source = str(warm_shard_path.relative_to(ROOT))
        warm_sha = sha256_file(warm_shard_path)
    else:
        warm_tok256 = np.stack(
            [WORK_STUDY.maze(180, 0.8, np.random.SeedSequence([999998, 180, 80, i]))[0] for i in range(256)]
        )
        warm_tok32 = warm_tok256[:32].copy()
        warm_source = "synthetic_999998"
        warm_sha = sha256_array(warm_tok256)

    # Source parity gate on warm_tok256: Triton == PyTorch bit-for-bit across all 256 180x180 mazes!
    t_gate = time.perf_counter()
    w_gate = M.build_weights_wavefront("cuda")
    xg = torch.as_tensor(warm_tok256, dtype=torch.long, device="cuda")
    rg_tr = M.solve_wsp_wavefront(xg, w_gate, check_every=16, sync_every=64, backend="triton")
    rg_py = M.solve_wsp_wavefront(xg, w_gate, check_every=16, sync_every=64, backend="pytorch")
    assert torch.equal(rg_tr.predictions, rg_py.predictions)
    assert torch.equal(rg_tr.steps, rg_py.steps)
    assert torch.equal(rg_tr.halted, rg_py.halted)
    del xg, rg_tr, rg_py, w_gate
    torch.cuda.synchronize()
    gate_seconds = float(time.perf_counter() - t_gate)

    # Warmup both paired arms before cohort timing
    _ = _run_180_arm("matched_3seed_B32", [warm_tok32])
    _ = _run_180_arm("unique_1seed_B256", [warm_tok256])

    cohort_specs = [
        (
            "H2bo",
            "bp-first-fresh-confirmation",
            "Corner endpoints (H2bo)",
            "corner",
            2026111529,
            ROOT / "artifacts/research/20261114/work/bp_first_fresh_confirmation",
            ROOT / "results/research/work/bp-first-fresh-confirmation/protocol.json",
            ROOT / "results/research/work/bp-first-fresh-confirmation/independent-input-audit.json",
            lambda idx: WORK_STUDY.maze(180, 0.8, np.random.SeedSequence([2026111529, 180, 80, idx]))[0],
        ),
        (
            "H2bp",
            "bp-first-endpoint-shift",
            "Uniform open-cell endpoints (H2bp)",
            "uniform",
            2026111530,
            ROOT / "artifacts/research/20261114/work/bp_first_endpoint_shift",
            ROOT / "results/research/work/bp-first-endpoint-shift/protocol.json",
            ROOT / "results/research/work/bp-first-endpoint-shift/independent-input-audit.json",
            lambda idx: G_UNIFORM.one(idx, 2026111530)[0],
        ),
        (
            "H2bq",
            "bp-first-far-placement",
            "Opposite-region endpoints (H2bq)",
            "far",
            2026111531,
            ROOT / "artifacts/research/20261114/work/bp_first_far_placement",
            ROOT / "results/research/work/bp-first-far-placement/protocol.json",
            ROOT / "results/research/work/bp-first-far-placement/independent-input-audit.json",
            lambda idx: G_FAR.one(idx, 2026111531)[0],
        ),
    ]

    balanced_schedule = [
        (0, ("matched_3seed_B32", "unique_1seed_B256")),
        (1, ("unique_1seed_B256", "matched_3seed_B32")),
        (2, ("unique_1seed_B256", "matched_3seed_B32")),
        (3, ("matched_3seed_B32", "unique_1seed_B256")),
    ]

    cohorts_out: dict[str, Any] = {}
    for (
        hyp_id,
        slug,
        display_label,
        law_name,
        base_seed,
        data_dir,
        proto_path,
        input_audit_path,
        replay_tok_fn,
    ) in cohort_specs:
        print(f"[full-test] 180x180 {hyp_id} ({slug}): preparing shards & preloading tokens only...")
        ds_manifest_path = data_dir / "datasets.json"
        ds_manifest = json.loads(ds_manifest_path.read_text(encoding="utf-8"))
        assert int(ds_manifest["base_seed"]) == base_seed
        assert len(ds_manifest["datasets"]) == 16

        shard_paths, shard_source_info = ensure_180_cohort_shards(
            law_name,
            base_seed,
            ds_manifest["datasets"],
            artifacts_dir / f"{slug}_shards",
            max_workers=8,
        )

        # Preload ONLY `tokens` (labels remain unloaded on disk until all 4 timing pairs seal!)
        shards_tokens: list[np.ndarray] = []
        for sh_path in shard_paths:
            with np.load(sh_path, allow_pickle=False) as zf:
                tok_arr = zf["tokens"].copy()
            assert tok_arr.shape == (256, 180, 180)
            shards_tokens.append(tok_arr)

        # Spot-check seeded generator identity on shard 0 index 0 and shard 15 index 4095
        assert np.array_equal(shards_tokens[0][0], replay_tok_fn(0))
        assert np.array_equal(shards_tokens[15][255], replay_tok_fn(4095))

        # Execute 4 balanced paired complete-cohort replays (labels untouched!)
        pair_records = []
        primary_arrays = None
        primary_hashes = None
        for pair_idx, order in balanced_schedule:
            runs_in_pair = {}
            for arm_name in order:
                run_res = _run_180_arm(arm_name, shards_tokens)
                assert run_res["cross_seed_exact"] is True
                if primary_arrays is None:
                    primary_arrays = run_res["arrays"]
                    primary_hashes = run_res["hashes"]
                else:
                    assert run_res["hashes"] == primary_hashes, (
                        f"Replay-integrity mismatch in {hyp_id} pair {pair_idx} arm {arm_name}"
                    )
                runs_in_pair[arm_name] = {
                    k: v for k, v in run_res.items() if k != "arrays"
                }
            pair_records.append(
                {
                    "pair": pair_idx,
                    "order": list(order),
                    "runs": runs_in_pair,
                }
            )
            print(
                f"[full-test] 180x180 {hyp_id} pair {pair_idx} ({order[0]} -> {order[1]}): "
                f"matched_3seed_B32={runs_in_pair['matched_3seed_B32']['complete_answer_seconds']:.4f}s, "
                f"unique_1seed_B256={runs_in_pair['unique_1seed_B256']['complete_answer_seconds']:.4f}s"
            )

        assert primary_arrays is not None and primary_hashes is not None
        # Seal timing receipt BEFORE loading labels!
        cohort_art_dir = artifacts_dir / slug
        sealed_receipt_sha = atomic_write_json(
            cohort_art_dir / "sealed_timing_pairs.json",
            {
                "hypothesis_id": hyp_id,
                "cohort": slug,
                "sealed_utc": utc_now(),
                "labels_loaded_during_timing": False,
                "all_8_runs_bit_identical": True,
                "primary_hashes": primary_hashes,
                "pairs": pair_records,
            },
        )

        # NOW load labels & shortest_edges strictly read-only for the independent CPU graph audit!
        all_tokens = np.concatenate(shards_tokens, axis=0)
        all_labels_list = []
        all_edges_list = []
        for sh_path in shard_paths:
            with np.load(sh_path, allow_pickle=False) as zf:
                all_labels_list.append(zf["labels"].astype(bool))
                all_edges_list.append(zf["shortest_edges"].astype(np.int64))
        all_labels = np.concatenate(all_labels_list, axis=0)
        all_edges = np.concatenate(all_edges_list, axis=0)

        # Verify all 4,096 distances match sealed independent-input-audit.json!
        sealed_in_audit = json.loads(input_audit_path.read_text(encoding="utf-8"))
        expected_edges = np.asarray(sealed_in_audit["distances"], dtype=np.int64).ravel()
        assert expected_edges.shape == (4096,)
        assert np.array_equal(all_edges, expected_edges), (
            f"Regenerated shortest_edges mismatch vs {input_audit_path}"
        )

        t_aud0 = time.perf_counter()
        audit_full = M.path_audit_batch(
            all_tokens,
            primary_arrays["predictions"],
            all_labels,
            primary_arrays["halted"],
        )
        audit_seconds = float(time.perf_counter() - t_aud0)
        aud_sum = audit_full["summary"]
        assert np.array_equal(audit_full["arrays"]["shortest_edges"], expected_edges)

        joint_solved_mask = (
            primary_arrays["accepted"]
            & audit_full["arrays"]["shortest_path"]
        )
        n_joint_solved = int(joint_solved_mask.sum())
        n_total = len(all_tokens)
        assert n_total == 4096

        # Exact general one-sided 95% Clopper-Pearson bounds for any 0 <= k <= n
        cp_lo, cp_hi = clopper_pearson_one_sided(n_joint_solved, n_total, alpha=0.05)
        cp_lower_pp = 100.0 * cp_lo
        cp_upper_pp = 100.0 * cp_hi

        fc2_cohort = M.evaluate_fc2_from_steps(
            cal_180,
            {
                "wave_steps": primary_arrays["wave_steps"],
                "trace_steps": primary_arrays["trace_steps"],
            },
        )
        strata = distance_strata_summary(all_edges, joint_solved_mask, primary_arrays["accepted"])

        pred_npz_path = cohort_art_dir / "pair0_predictions_and_audit.npz"
        pred_npz_sha = atomic_savez_compressed(
            pred_npz_path,
            predictions=primary_arrays["predictions"],
            steps=primary_arrays["steps"],
            wave_steps=primary_arrays["wave_steps"],
            trace_steps=primary_arrays["trace_steps"],
            halted=primary_arrays["halted"],
            accepted=primary_arrays["accepted"],
            valid_path=audit_full["arrays"]["valid_path"],
            shortest_path=audit_full["arrays"]["shortest_path"],
            canonical_exact=audit_full["arrays"]["canonical_exact"],
            shortest_edges=all_edges,
        )

        # Exact rational comparisons against published LP96 and BP_geometry_LP96 4-pair timings
        b32_secs = [p["runs"]["matched_3seed_B32"]["complete_answer_seconds"] for p in pair_records]
        b256_secs = [p["runs"]["unique_1seed_B256"]["complete_answer_seconds"] for p in pair_records]
        pub_c = prov_3c_by_slug[slug]
        lp96_secs = [float(p["LP96_seconds"]) for p in pub_c["pairs"]]
        casc_secs = [float(p["cascade_seconds"]) for p in pub_c["pairs"]]

        vs_lp96 = exact_saving_against_baseline(b32_secs, lp96_secs)
        vs_casc = exact_saving_against_baseline(b32_secs, casc_secs)

        cohorts_out[hyp_id] = {
            "hypothesis_id": hyp_id,
            "cohort": slug,
            "label": display_label,
            "base_seed": base_seed,
            "protocol_path": str(proto_path.relative_to(ROOT)),
            "protocol_sha256": sha256_file(proto_path),
            "datasets_manifest_path": str(ds_manifest_path.relative_to(ROOT)),
            "datasets_manifest_sha256": sha256_file(ds_manifest_path),
            "independent_input_audit_path": str(input_audit_path.relative_to(ROOT)),
            "independent_input_audit_sha256": sha256_file(input_audit_path),
            "all_4096_distances_matched_input_audit": True,
            "shard_source_info": shard_source_info,
            "shards_verified": len(shard_paths),
            "seeded_generator_spot_replay_passed": True,
            "sealed_timing_receipt_sha256": sealed_receipt_sha,
            "predictions_and_audit_npz": str(pred_npz_path.relative_to(ROOT)),
            "predictions_and_audit_sha256": pred_npz_sha,
            "independent_audit_wall_seconds": audit_seconds,
            "accuracy": {
                "n_mazes": n_total,
                "joint_solved": n_joint_solved,
                "joint_solved_pct": 100.0 * n_joint_solved / n_total,
                "model_inputs_denominator": n_total * 3,
                "model_inputs_solved": n_joint_solved * 3,
                "canonical_exact": int(aud_sum["exact_solved"]),
                "valid_simple_paths": int(aud_sum["valid_simple_paths"]),
                "shortest_paths": int(aud_sum["shortest_paths"]),
                "halted": int(aud_sum["halted"]),
                "wrong_halts": int(aud_sum["wrong_halts_shortest"]),
                "accepted": int(primary_arrays["accepted"].sum()),
                "one_sided95_clopper_pearson_lower_pp": cp_lower_pp,
                "one_sided95_clopper_pearson_upper_pp": cp_upper_pp,
                "absolute_999_target_decision": "supported" if cp_lower_pp >= 99.9 else "not_supported",
            },
            "fc2_flops_per_unique_maze": fc2_cohort,
            "distance_strata": strata,
            "timing_4_balanced_pairs": {
                "pairs": pair_records,
                "matched_3seed_B32_complete_seconds": b32_secs,
                "matched_3seed_B32_median_seconds": float(np.median(b32_secs)),
                "unique_1seed_B256_complete_seconds": b256_secs,
                "unique_1seed_B256_median_seconds": float(np.median(b256_secs)),
                "comparison_vs_published_LP96_12288": vs_lp96,
                "comparison_vs_published_BP_geometry_LP96_12288": vs_casc,
                "published_reference_cohorts_provenance_sha256": sha256_file(prov_3c_path),
                "hardware_note": (
                    "WSP-Wavefront measured on local RTX PRO 6000 Blackwell Max-Q (GPU 2) under "
                    "identical 12,288-eval B=32 readiness+H2D+GPU+D2H+acceptance cost boundary; "
                    "published H2bo/H2bp/H2bq LP96 and cascade timings ran on Work RTX PRO 6000 Blackwell."
                ),
            },
        }
        print(
            f"[full-test] 180x180 {hyp_id} AUDIT COMPLETE: solved={n_joint_solved}/4096 "
            f"(canonical_exact={aud_sum['exact_solved']}/4096, wrong_halts={aud_sum['wrong_halts_shortest']}), "
            f"median matched_3seed_B32={np.median(b32_secs):.3f}s "
            f"(vs LP96 {np.median(lp96_secs):.2f}s -> {vs_lp96['median_speedup_ratio']:.2f}x, "
            f"vs Cascade {np.median(casc_secs):.2f}s -> {vs_casc['median_speedup_ratio']:.2f}x)"
        )

    return {
        "source_parity_gate": {
            "warmup_source": warm_source,
            "warmup_sha256": warm_sha,
            "triton_vs_pytorch_256_mazes_bit_exact": True,
            "gate_wall_seconds": gate_seconds,
        },
        "fc2_calibration_180x180_path": str(cal_180_path.relative_to(ROOT)),
        "fc2_calibration_180x180_sha256": cal_180_sha,
        "cohorts": cohorts_out,
    }


def cmd_full_test(artifacts_root: Path, results_root: Path, approved: bool) -> None:
    if not approved:
        raise PermissionError(
            "Refusing to open test splits without explicit `--verifier-approved` sign-off!"
        )
    dev_gate_file = artifacts_root / "dev_smoke/summary.json"
    if not dev_gate_file.exists():
        raise PermissionError(
            f"Missing pre-experiment dev smoke receipt at {dev_gate_file}; run `smoke-dev` first!"
        )
    dev_gate = json.loads(dev_gate_file.read_text(encoding="utf-8"))
    if dev_gate.get("status") != "passed_pre_experiment_gate":
        raise PermissionError(f"Dev smoke gate did not pass: {dev_gate.get('status')!r}")
    t_start = time.perf_counter()
    prov = runtime_provenance()
    ast_check = verify_static_ast_all()

    # 1. Maze-Unique (1,000 test mazes, seeds 0, 1, 2)
    mu_tr900 = maze_unique.load("train900")["inputs"].reshape(-1, 30, 30)
    mu_test = maze_unique.load("test")
    mu_test_tok = mu_test["inputs"].reshape(-1, 30, 30)
    mu_test_lab = mu_test["targets"].reshape(-1, 30, 30).astype(bool)
    mu_models = _run_30x30_test_benchmark(
        "Maze-Unique",
        mu_tr900,
        mu_test_tok,
        mu_test_lab,
        artifacts_root / "maze_unique",
    )
    mu_summary = {
        "status": "complete",
        "benchmark": "Maze-Unique",
        "shape": [30, 30],
        "n_test_mazes": 1000,
        "seeds": [0, 1, 2],
        "provenance": prov,
        "static_ast_verification": ast_check,
        "models": mu_models,
    }
    mu_sha = atomic_write_json(results_root / "maze_unique_summary.json", mu_summary)

    # 2. Maze-Hard (1,000 test mazes, seeds 0, 1, 2 + explicit maze 106 audit)
    mh_tr_all_tok, _ = maze_h3.load("train")
    mh_fit_idx, _ = maze_h3.dev_split()
    mh_tr900 = mh_tr_all_tok[mh_fit_idx]
    mh_test_tok, mh_test_lab = maze_h3.load("test")
    mh_models = _run_30x30_test_benchmark(
        "Maze-Hard",
        mh_tr900,
        mh_test_tok,
        mh_test_lab,
        artifacts_root / "maze_hard",
    )
    m106 = audit_maze_106(mh_test_tok[106], mh_test_lab[106])
    mh_summary = {
        "status": "complete",
        "benchmark": "Maze-Hard",
        "shape": [30, 30],
        "n_test_mazes": 1000,
        "seeds": [0, 1, 2],
        "provenance": prov,
        "static_ast_verification": ast_check,
        "maze_106_audit": m106,
        "models": mh_models,
    }
    mh_sha = atomic_write_json(results_root / "maze_hard_summary.json", mh_summary)

    # 3. Dense 180x180 80%-loop cohorts (H2bo, H2bp, H2bq: 4,096 mazes each, 4 balanced paired replays)
    res_180 = _run_180x180_cohorts(artifacts_root / "cohorts_180")
    c180_summary = {
        "status": "complete",
        "benchmark": "Maze-180x180-Loop80-Three-Cohorts",
        "shape": [180, 180],
        "n_mazes_per_cohort": 4096,
        "cohorts_count": 3,
        "provenance": prov,
        "static_ast_verification": ast_check,
        "model": "WSP-Wavefront",
        "weight_metadata": M.weight_metadata(M.build_weights_wavefront("cpu")),
        **res_180,
    }
    c180_sha = atomic_write_json(results_root / "maze_180x180_cohorts_summary.json", c180_summary)

    complete_summary = {
        "status": "complete",
        "total_wall_seconds": float(time.perf_counter() - t_start),
        "provenance": prov,
        "static_ast_verification": ast_check,
        "files": {
            "results/wsp/maze/maze_unique_summary.json": mu_sha,
            "results/wsp/maze/maze_hard_summary.json": mh_sha,
            "results/wsp/maze/maze_180x180_cohorts_summary.json": c180_sha,
        },
        "headline": {
            "Maze-Unique": {
                k: v["aggregate_3seeds"] for k, v in mu_models.items()
            },
            "Maze-Hard": {
                k: v["aggregate_3seeds"] for k, v in mh_models.items()
            },
            "Maze-Hard_maze_106": m106["models"],
            "Maze-180x180": {
                h: {
                    "joint_solved": c["accuracy"]["joint_solved"],
                    "canonical_exact": c["accuracy"]["canonical_exact"],
                    "wrong_halts": c["accuracy"]["wrong_halts"],
                    "fc2_mean_total": c["fc2_flops_per_unique_maze"]["mean_total"],
                    "fc2_mean_total_with_compare": c["fc2_flops_per_unique_maze"][
                        "mean_total_with_compare"
                    ],
                    "matched_3seed_B32_median_s": c["timing_4_balanced_pairs"][
                        "matched_3seed_B32_median_seconds"
                    ],
                    "unique_1seed_B256_median_s": c["timing_4_balanced_pairs"][
                        "unique_1seed_B256_median_seconds"
                    ],
                    "speedup_vs_LP96": c["timing_4_balanced_pairs"][
                        "comparison_vs_published_LP96_12288"
                    ]["median_speedup_ratio"],
                    "saving_vs_LP96": c["timing_4_balanced_pairs"][
                        "comparison_vs_published_LP96_12288"
                    ]["median_saving"],
                    "speedup_vs_cascade": c["timing_4_balanced_pairs"][
                        "comparison_vs_published_BP_geometry_LP96_12288"
                    ]["median_speedup_ratio"],
                    "saving_vs_cascade": c["timing_4_balanced_pairs"][
                        "comparison_vs_published_BP_geometry_LP96_12288"
                    ]["median_saving"],
                }
                for h, c in res_180["cohorts"].items()
            },
        },
    }
    comp_path = results_root / "wsp_maze_complete_summary.json"
    comp_sha = atomic_write_json(comp_path, complete_summary)
    print(f"[full-test] ALL MAZE WSP BENCHMARKS COMPLETE -> {comp_path} (sha256={comp_sha})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dev = sub.add_parser("smoke-dev", help="Run strict train/dev pre-experiment gate & profiling")
    p_dev.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/wsp/maze/dev_smoke",
        help="Output directory for dev smoke summary",
    )

    p_full = sub.add_parser("full-test", help="Run full 3-seed test & 180x180 evaluations")
    p_full.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "artifacts/wsp/maze",
        help="Artifacts directory for per-seed NPZ predictions and calibrations",
    )
    p_full.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results/wsp/maze",
        help="Results directory for machine-readable JSON summaries",
    )
    p_full.add_argument(
        "--verifier-approved",
        action="store_true",
        help="Mandatory gate flag confirming @verifier pre-experiment sign-off",
    )

    args = parser.parse_args()
    if args.command == "smoke-dev":
        cmd_smoke_dev(args.out)
    else:
        cmd_full_test(args.artifacts_dir, args.results_dir, args.verifier_approved)


if __name__ == "__main__":
    main()
