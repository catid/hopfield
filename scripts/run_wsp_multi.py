"""Validation, source-timing calibration, and post-signoff evaluation runner for multi-solution WSP.

Strict zero-leakage two-stage protocol:
1. `--stage dev`:
   - Evaluates ONLY 10% validation splits carved from `train` (`nqueens-8/10`, `graph-coloring-8/10`)
     and `N=12` source-derived timing calibration inputs (`h3ad/b{0,1}/calibration-data.npz`).
   - Freezes hyperparameters and matched-GPU-time step caps in `artifacts/wsp/multi/dev/dev_selections.json`.
   - NEVER opens or touches any `test` split or `h3ad/b{0,1}/data.npz`.
2. `--stage test --confirm-verifier-signoff`:
   - Refuses to run without `--confirm-verifier-signoff` and frozen `dev_selections.json`.
   - Evaluates `seeds = (0, 1, 2)` once per benchmark on `GPU 3`, runs independent rule/certificate/canonical
     audits on every saved output, and writes atomic JSON summaries (`allow_nan=False`) to `results/wsp/multi/`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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

from cram import cram_graph_coloring as GC_CRAM
from cram import graph_coloring_data as GC_DATA
from cram import nqueens_data as NQ_DATA

from hopfield_cram import wsp_multi as W

ARTIFACTS_DIR = ROOT / "artifacts" / "wsp" / "multi"
RESULTS_DIR = ROOT / "results" / "wsp" / "multi"
H3AD_ARTIFACTS = ROOT / "artifacts" / "research" / "20261114" / "loki" / "h3ad"
UPSTREAM_COMMIT = "02c31abd3b34d84854cc80d9717717c45365c719"

FAMILIES = {
    0: {"name": "queen-plus-zebra", "leaper": (2, 3)},
    1: {"name": "queen-plus-2-5-leaper", "leaper": (2, 5)},
}


# ---------------------------------------------------------------------------
# Atomic File I/O, Provenance & Independent Audit Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def hardware_info(device: str) -> dict[str, Any]:
    import subprocess

    info: dict[str, Any] = {
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "tmpdir": os.environ.get("TMPDIR", ""),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", ""),
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
        "wsp_upstream_commit": UPSTREAM_COMMIT,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_id = vis.split(",")[0].strip() if vis else "0"
        try:
            snap = subprocess.check_output(
                ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=index,uuid,utilization.gpu,power.draw,memory.used,memory.free", "--format=csv,noheader"],
                text=True,
                timeout=10,
            ).strip()
            apps = subprocess.check_output(
                ["nvidia-smi", f"--id={gpu_id}", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                text=True,
                timeout=10,
            ).strip()
            info["nvidia_smi_snapshot"] = snap
            info["cotenant_compute_apps"] = [line.strip() for line in apps.splitlines() if line.strip()]
        except Exception as exc:
            info["nvidia_smi_error"] = repr(exc)
    return info


def enumerate_all_nqueens_solutions(n: int) -> np.ndarray:
    """Enumerate all complete `n x n` N-Queens solutions (`92` for n=8, `724` for n=10)."""
    empty = np.zeros(n * n, dtype=bool)
    return np.asarray(NQ_DATA.completions(empty, n), dtype=bool)


def compute_nqueens_puzzle_counts(inputs: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute `(inverse, S)` on validation inputs using complete solution filtering."""
    uniq, inv = np.unique(inputs, axis=0, return_inverse=True)
    all_sols = enumerate_all_nqueens_solutions(n)  # (M, c)
    # A solution completes `u` iff `(all_sols >= u).all(-1)` i.e. `all_sols[:, u].all(-1)`
    S = np.empty(len(uniq), dtype=np.int64)
    for i, u in enumerate(uniq):
        given_idx = np.flatnonzero(u)
        S[i] = len(all_sols) if len(given_idx) == 0 else int(all_sols[:, given_idx].all(axis=1).sum())
    return inv.ravel(), S


def score_nqueens_chains(
    inputs: np.ndarray,
    res: dict[str, Any],
    n: int,
    inverse: np.ndarray,
    S: np.ndarray,
    affine_flops: dict[str, Any],
) -> dict[str, Any]:
    """Exact Table 3 + Halting + FC2 FLOP scoring for N-Queens."""
    boards = res["boards"]
    stopped = res["stopped"]
    energy = res["energy"]
    moves = res["moves"]
    R, C, _ = boards.shape

    # Independent validity & given-preservation checks
    if not bool((boards >= inputs[:, None, :]).all()):
        raise AssertionError("Clue corruption detected: predicted board dropped a given queen!")
    ok = NQ_DATA.valid(boards, n) & (boards >= inputs[:, None, :]).all(axis=-1)
    wrong_stops = int((stopped & ~ok).sum())
    if wrong_stops != 0:
        raise AssertionError(f"Zero-wrong-stop invariant violated on N-Queens (n={n}): {wrong_stops}")

    selected_min_e = np.argmin(energy, axis=1)
    # Majority-vote diagnostic (ties broken by lower mean energy, then first index)
    maj_ok = np.zeros(R, dtype=bool)
    for i in range(R):
        stats: dict[bytes, tuple[int, float, int]] = {}
        for s_idx in range(C):
            key = boards[i, s_idx].tobytes()
            cnt, tot_e, first = stats.get(key, (0, 0.0, s_idx))
            stats[key] = (cnt + 1, tot_e + float(energy[i, s_idx]), first)
        best_s = max(stats.values(), key=lambda v: (v[0], -v[1] / v[0], -v[2]))[2]
        maj_ok[i] = bool(ok[i, best_s])

    cap_s = np.minimum(S, 20)
    found = np.array([len({b.tobytes() for b in boards[i, ok[i]]}) for i in range(R)], dtype=np.float64)
    per_row_found = np.minimum(found, cap_s[inverse])
    found_per_puzzle = (
        np.bincount(inverse, weights=per_row_found, minlength=len(S))
        / np.bincount(inverse, minlength=len(S))
    )

    total_chains = int(stopped.size)
    halted_count = int(stopped.sum())
    fl_n1 = W.compute_trajectory_flops(moves, affine_flops, chains_used=1)
    fl_n20 = W.compute_trajectory_flops(moves, affine_flops, chains_used=C)

    return {
        "rows": int(R),
        "distinct_inputs": int(len(S)),
        "chains": int(C),
        "acc_n1": 100.0 * float(ok[:, 0].mean()),
        "acc_n20": 100.0 * float(ok[np.arange(R), selected_min_e].mean()),
        "acc_n20_majority_vote": 100.0 * float(maj_ok.mean()),
        "cov20_per_puzzle": 100.0 * float(found_per_puzzle.sum() / cap_s.sum()),
        "cov20_per_row": 100.0 * float(per_row_found.sum() / cap_s[inverse].sum()),
        "halted_chains": halted_count,
        "total_chains": total_chains,
        "halted_pct": 100.0 * float(stopped.mean()),
        "wrong_stops": wrong_stops,
        "wrong_halts_pct_of_halted": 100.0 * float(wrong_stops / max(1, halted_count)),
        "budget_failures": int((~stopped).sum()),
        "unstopped_but_right": int((~stopped & ok).sum()),
        "halted_n1_pct": 100.0 * float(stopped[:, 0].mean()),
        "solved_halted_n1_pct": 100.0 * float((stopped[:, 0] & ok[:, 0]).mean()),
        "moves_mean": float(moves.mean()),
        "moves_max": int(moves.max()),
        "fc2_flops_n1": fl_n1,
        "fc2_flops_n20": fl_n20,
        "gpu_kernel_seconds": float(res.get("gpu_kernel_seconds", 0.0)),
        "worker_solve_seconds": float(res.get("worker_solve_seconds", res.get("gpu_kernel_seconds", 0.0))),
    }


def score_graph_coloring_chains(
    A: np.ndarray,
    res: dict[str, Any],
    affine_flops: dict[str, Any],
    *,
    proper_sets: list[set[tuple[int, ...]]] | None = None,
) -> dict[str, Any]:
    """Exact Table 3 + Halting + FC2 FLOP scoring for Graph Coloring."""
    boards = res["boards"]
    stopped = res["stopped"]
    energy = res["energy"]
    moves = res["moves"]
    B, C, _ = boards.shape

    conf = GC_DATA.conflicts(A, boards)
    proper = conf == 0
    canon = GC_DATA.is_canonical(boards)
    wrong_stops = int((stopped & ~proper).sum())
    if wrong_stops != 0:
        raise AssertionError(f"Zero-wrong-stop invariant violated on Graph Coloring: {wrong_stops}")

    sols = proper_sets if proper_sets is not None else GC_DATA.proper_canonical_sets(A)
    total_sols = sum(len(s) for s in sols)

    # N20 lowest-energy selection (ties broken to first chain, matching np.argmin)
    sel_min_e = np.argmin(energy, axis=1)
    conflicts_n20_min_e = int(conf[np.arange(B), sel_min_e].sum())

    # Use upstream GC_CRAM.score compatibility logic for coverage_20, coverage_canonicalized_20, majority vote
    found_strict = 0
    found_canon_posthoc = 0
    maj_conf = 0
    for g in range(B):
        bs = boards[g]
        es = energy[g]
        found_strict += len({tuple(int(x) for x in r) for r, ok in zip(bs, proper[g] & canon[g]) if ok})
        found_canon_posthoc += len({tuple(int(x) for x in r) for r in GC_DATA.canonical(bs[proper[g]])})
        stats: dict[bytes, tuple[int, float, int]] = {}
        for s_idx, (row, en) in enumerate(zip(bs, es)):
            key = row.tobytes()
            cnt, tot_e, first = stats.get(key, (0, 0.0, s_idx))
            stats[key] = (cnt + 1, tot_e + float(en), first)
        maj_idx = max(stats.values(), key=lambda v: (v[0], -v[1] / v[0]))[2]
        maj_conf += int(conf[g, maj_idx])

    fl_n1 = W.compute_trajectory_flops(moves, affine_flops, chains_used=1)
    fl_n20 = W.compute_trajectory_flops(moves, affine_flops, chains_used=C)
    halted_count = int(stopped.sum())

    return {
        "graphs": int(B),
        "samples": int(C),
        "solutions_total": int(total_sols),
        "conflicts_1": int(conf[:, 0].sum()),
        "conflicts_n20_min_energy": conflicts_n20_min_e,
        "conflicts_majority_20": int(maj_conf),
        "proper_1_pct": 100.0 * float(proper[:, 0].mean()),
        "proper_canonical_1_pct": 100.0 * float((proper & canon)[:, 0].mean()),
        "proper_any_20_pct": 100.0 * float(proper.any(axis=1).mean()),
        "all_outputs_canonical": bool(canon.all()),
        "coverage_20_pct": 100.0 * float(found_strict / total_sols),
        "coverage_canonicalized_20_pct": 100.0 * float(found_canon_posthoc / total_sols),
        "wrong_stops_1": int((stopped[:, 0] & ~proper[:, 0]).sum()),
        "wrong_stops_all": wrong_stops,
        "halted_pct": 100.0 * float(stopped.mean()),
        "wrong_halts_pct_of_halted": 100.0 * float(wrong_stops / max(1, halted_count)),
        "budget_failures_1": int((~stopped[:, 0]).sum()),
        "budget_failures_all": int((~stopped).sum()),
        "unstopped_but_proper": int((~stopped & proper).sum()),
        "moves_mean": float(moves.mean()),
        "moves_max": int(moves.max()),
        "fc2_flops_n1": fl_n1,
        "fc2_flops_n20": fl_n20,
        "gpu_kernel_seconds": float(res.get("gpu_kernel_seconds", 0.0)),
        "worker_solve_seconds": float(res.get("worker_solve_seconds", res.get("gpu_kernel_seconds", 0.0))),
    }


def summarize_across_seeds(per_seed: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        vals = [float(r[k]) for r in per_seed]
        out[k] = {
            "mean": float(statistics.mean(vals)),
            "sd": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "per_seed": vals,
        }
    return out


# ---------------------------------------------------------------------------
# Stage 1: Dev Validation & Source-Only Timing Calibration (Zero Test Access)
# ---------------------------------------------------------------------------

def warmup_compiled_solvers(device: str) -> float:
    """Compile all 4 `step_chunk` kernels once on unsatisfiable `R>=128, k_len=32` inputs outside timed regions."""
    if not device.startswith("cuda"):
        return 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # 1. NQ Arbitrary: 2 given queens in row 0 -> unsatisfiable, R=128 stays active for all 32 steps
    g_arb = np.zeros((16, 64), dtype=bool)
    g_arb[:, [0, 1]] = True
    W.solve_nqueens_arbitrary(W.build_nqueens_arbitrary_wsp(8, cap=32), g_arb, chains=8, device=device)

    # 2. Permutation Swap: 7 queens on main diagonal of 8x8 -> unsatisfiable, R=128 stays active
    g_perm = np.zeros((16, 64), dtype=bool)
    for r in range(7):
        g_perm[:, r * 8 + r] = True
    W.solve_permutation_swap(W.build_nqueens_perm_wsp(8, cap=32), g_perm, chains=8, device=device)

    # 3. Arbitrary Swap: 12 queens on main diagonal of 13x13 -> unsatisfiable, R=128 stays active
    g_leap = np.zeros((16, 169), dtype=bool)
    for r in range(12):
        g_leap[:, r * 13 + r] = True
    W.solve_arbitrary_swap(W.build_queen_leaper_wsp(13, (2, 3), cap=32), g_leap, chains=8, device=device)

    # 4. Graph Coloring Assoc: K_8 complete graph -> not 3-colorable, R=128 stays active
    ed_k8 = np.ones((16, 28), dtype=np.float32)
    W.solve_graph_coloring_assoc(W.build_graph_coloring_assoc_wsp(8, cap=32), ed_k8, samples=8, device=device)
    torch.cuda.synchronize()
    return float(time.perf_counter() - t0)


def measure_build_timings(device: str) -> dict[str, Any]:
    """Measure cold + median of 7 warm `build_*()` readiness calls on GPU and 1-thread CPU, verifying bitwise equality."""
    builders = {
        "nqueens_8_arbitrary": lambda d: W.build_nqueens_arbitrary_wsp(8, cap=20000, temperature=0.5, device=d),
        "nqueens_10_arbitrary": lambda d: W.build_nqueens_arbitrary_wsp(10, cap=20000, temperature=0.5, device=d),
        "nqueens_10_permutation": lambda d: W.build_nqueens_perm_wsp(10, cap=20000, device=d),
        "graph_coloring_8_assoc": lambda d: W.build_graph_coloring_assoc_wsp(8, cap=10000, temperature=1.0, device=d),
        "graph_coloring_10_assoc": lambda d: W.build_graph_coloring_assoc_wsp(10, cap=10000, temperature=1.0, device=d),
        "queen_leaper_13_zebra": lambda d: W.build_queen_leaper_wsp(13, (2, 3), cap=20000, device=d),
    }
    out: dict[str, Any] = {}
    targets = ["cpu"] + ([device] if device.startswith("cuda") else [])
    for name, fn in builders.items():
        per_dev: dict[str, Any] = {}
        for tgt in targets:
            is_cuda = tgt.startswith("cuda")
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            w_cold = fn(tgt)
            if is_cuda:
                torch.cuda.synchronize()
            cold_s = float(time.perf_counter() - t0)
            warm_times = []
            bitwise_ok = True
            for _ in range(7):
                if is_cuda:
                    torch.cuda.synchronize()
                tw0 = time.perf_counter()
                w_warm = fn(tgt)
                if is_cuda:
                    torch.cuda.synchronize()
                warm_times.append(float(time.perf_counter() - tw0))
                for k in w_cold:
                    if not bool(torch.equal(w_cold[k], w_warm[k])):
                        bitwise_ok = False
            per_dev[tgt] = {
                "cold_seconds": cold_s,
                "warm_median_seconds": float(statistics.median(warm_times)),
                "warm_repeats_seconds": warm_times,
                "bitwise_equal_across_repeats": bitwise_ok,
            }
        out[name] = per_dev
    return out


def run_dev_stage(device: str) -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.startswith("cuda"):
        t_ctx0 = time.perf_counter()
        torch.zeros(1, device=device)
        torch.cuda.synchronize()
        cuda_context_s = time.perf_counter() - t_ctx0
    else:
        cuda_context_s = 0.0

    build_timings = measure_build_timings(device)
    compile_warmup_s = warmup_compiled_solvers(device)
    print(f"[INIT] cuda_context={cuda_context_s:.3f}s compile_warmup={compile_warmup_s:.3f}s", flush=True)

    selections: dict[str, Any] = {
        "stage": "dev_only_zero_test_access",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cuda_context_seconds": cuda_context_s,
        "compile_warmup_seconds": compile_warmup_s,
        "build_timings": build_timings,
        "hardware": hardware_info(device),
        "nqueens": {},
        "graph_coloring": {},
        "queen_leaper_calibration": {},
    }

    # --- 1. N-Queens 8 & 10 Validation (10% distinct-input split of TRAIN only) ---
    for n in (8, 10):
        tr_in, _, _ = NQ_DATA.load_split(n, "train")
        _, val_rows = NQ_DATA.carve_validation(tr_in, fraction=0.1, seed=0)
        val_in = tr_in[val_rows]
        inv, S = compute_nqueens_puzzle_counts(val_in, n)

        aff_det = W.calibrate_affine_chain_flops("nqueens_det", n)
        aff_arb = W.calibrate_affine_chain_flops("nqueens_arbitrary", n)
        aff_perm = W.calibrate_affine_chain_flops("nqueens_perm", n)

        det_grid = []
        for passes in (8, 12, 16, 24):
            w_det = W.build_nqueens_det_wsp(n, passes=passes, device=device)
            res = W.solve_nqueens_det(w_det, val_in, chains=20, device=device)
            sc = score_nqueens_chains(val_in, res, n, inv, S, aff_det)
            det_grid.append({"passes": passes, **sc})
            print(f"[DEV NQ-{n} Det] passes={passes} N1={sc['acc_n1']:.2f}% Cov20={sc['cov20_per_puzzle']:.2f}%", flush=True)
        best_det = max(det_grid, key=lambda r: (r["acc_n1"], -r["fc2_flops_n1"]["mean_flops_per_instance"]))

        arb_grid = []
        for temp in (0.1, 0.2, 0.3, 0.5, 1.0):
            w_arb = W.build_nqueens_arbitrary_wsp(n, cap=20000, temperature=temp, device=device)
            res = W.solve_nqueens_arbitrary(w_arb, val_in, chains=20, seed=0, device=device)
            sc = score_nqueens_chains(val_in, res, n, inv, S, aff_arb)
            arb_grid.append({"temperature": temp, "cap": 20000, **sc})
            print(
                f"[DEV NQ-{n} Arb] T={temp} N1={sc['acc_n1']:.2f}% Cov20={sc['cov20_per_puzzle']:.2f}% "
                f"s={sc['worker_solve_seconds']:.3f}s",
                flush=True,
            )
        top_arb_n1 = max(r["acc_n1"] for r in arb_grid)
        near_arb = [r for r in arb_grid if r["acc_n1"] >= top_arb_n1 - 0.5]
        best_arb = min(near_arb, key=lambda r: (-r["cov20_per_puzzle"], r["fc2_flops_n20"]["mean_flops_per_instance"]))

        perm_grid = []
        for t_start, end_r in [(0.5, 0.05), (1.0, 0.05), (2.0, 0.05), (1.0, 0.01)]:
            w_perm = W.build_nqueens_perm_wsp(n, cap=20000, temp_start=t_start, end_ratio=end_r, device=device)
            res = W.solve_permutation_swap(w_perm, val_in, chains=20, seed=0, device=device)
            sc = score_nqueens_chains(val_in, res, n, inv, S, aff_perm)
            perm_grid.append({"temp_start": t_start, "end_ratio": end_r, "cap": 20000, **sc})
            print(
                f"[DEV NQ-{n} Perm] T0={t_start} end={end_r} N1={sc['acc_n1']:.2f}% Cov20={sc['cov20_per_puzzle']:.2f}% "
                f"s={sc['worker_solve_seconds']:.3f}s",
                flush=True,
            )
        top_perm_n1 = max(r["acc_n1"] for r in perm_grid)
        near_perm = [r for r in perm_grid if r["acc_n1"] >= top_perm_n1 - 0.5]
        best_perm = min(near_perm, key=lambda r: (-r["cov20_per_puzzle"], r["fc2_flops_n20"]["mean_flops_per_instance"]))

        selections["nqueens"][str(n)] = {
            "val_rows": int(len(val_in)),
            "val_distinct_inputs": int(len(S)),
            "selected_det": {"passes": best_det["passes"]},
            "selected_arbitrary": {"temperature": best_arb["temperature"], "cap": 20000},
            "selected_permutation": {
                "temp_start": best_perm["temp_start"],
                "end_ratio": best_perm["end_ratio"],
                "cap": 20000,
            },
            "affine_flops": {"det": aff_det, "arbitrary": aff_arb, "permutation": aff_perm},
            "dev_grid_det": det_grid,
            "dev_grid_arbitrary": arb_grid,
            "dev_grid_permutation": perm_grid,
        }

    # --- 2. Graph Coloring 8 & 10 Validation (10% distinct-graph split of TRAIN only) ---
    for n in (8, 10):
        x_tr, y_tr = GC_DATA.load_split(n, "train")
        edges_tr, _ = GC_DATA.convert(x_tr, y_tr, n)
        _, val_idx = GC_DATA.validation_graphs(edges_tr, frac=0.1, seed=20261106)
        edges_val = edges_tr[val_idx]
        A_val = GC_DATA.adjacency(edges_val, n)
        proper_val = GC_DATA.proper_canonical_sets(A_val)

        aff_det = W.calibrate_affine_chain_flops("gc_det", n)
        aff_assoc = W.calibrate_affine_chain_flops("gc_assoc", n, canonicalize_in_arch=True)

        det_grid = []
        for passes in (8, 16, 24):
            w_det = W.build_graph_coloring_det_wsp(n, passes=passes, device=device)
            res = W.solve_graph_coloring_det(w_det, edges_val, samples=20, device=device)
            sc = score_graph_coloring_chains(A_val, res, aff_det, proper_sets=proper_val)
            det_grid.append({"passes": passes, **sc})
            print(
                f"[DEV GC-{n} Det] passes={passes} conf1={sc['conflicts_1']} Cov20={sc['coverage_20_pct']:.2f}%",
                flush=True,
            )
        best_gc_det = min(det_grid, key=lambda r: (r["conflicts_1"], r["fc2_flops_n1"]["mean_flops_per_instance"]))

        assoc_grid = []
        for temp in (0.3, 0.5, 0.8, 1.0, 2.0, 3.0):
            w_assoc = W.build_graph_coloring_assoc_wsp(
                n, cap=10000, temperature=temp, canonicalize_in_arch=True, device=device
            )
            res = W.solve_graph_coloring_assoc(w_assoc, edges_val, samples=20, seed=0, device=device)
            sc = score_graph_coloring_chains(A_val, res, aff_assoc, proper_sets=proper_val)
            assoc_grid.append({"temperature": temp, "cap": 10000, **sc})
            print(
                f"[DEV GC-{n} Assoc] T={temp} conf1={sc['conflicts_1']} conf20={sc['conflicts_n20_min_energy']} "
                f"Cov20={sc['coverage_20_pct']:.2f}% s={sc['worker_solve_seconds']:.3f}s",
                flush=True,
            )
        min_conf1 = min(r["conflicts_1"] for r in assoc_grid)
        best_gc_assoc = min(
            [r for r in assoc_grid if r["conflicts_1"] == min_conf1],
            key=lambda r: (-r["coverage_20_pct"], r["fc2_flops_n20"]["mean_flops_per_instance"]),
        )

        selections["graph_coloring"][str(n)] = {
            "val_graphs": int(len(edges_val)),
            "selected_det": {"passes": best_gc_det["passes"]},
            "selected_assoc": {"temperature": best_gc_assoc["temperature"], "cap": 10000, "theta": 0.45},
            "affine_flops": {"det": aff_det, "assoc": aff_assoc},
            "dev_grid_det": det_grid,
            "dev_grid_assoc": assoc_grid,
        }

    # --- 3. Generalized Queen+Leaper (N=13) Source-Only Timing Calibration (N=12 embedded calibration-data.npz ONLY) ---
    for bank, fam in FAMILIES.items():
        selections["queen_leaper_calibration"][f"b{bank}"] = calibrate_queen_leaper_bank(bank, fam, device)

    dev_sel_path = ARTIFACTS_DIR / "dev" / "dev_selections.json"
    save_json_atomic(dev_sel_path, selections)
    save_json_atomic(RESULTS_DIR / "dev_validation_summary.json", selections)
    print(f"Saved frozen dev selections to {dev_sel_path}", flush=True)
    return selections


def calibrate_queen_leaper_bank(bank: int, fam: dict[str, Any], device: str) -> dict[str, Any]:
    """Calibrate `selected_permutation_cap` strictly on `N=12` source `calibration-data.npz` (zero target access)."""
    leaper = fam["leaper"]
    b_dir = H3AD_ARTIFACTS / f"b{bank}"
    cal_npz_path = b_dir / "calibration-data.npz"
    cal_meta = json.loads((b_dir / "calibration-data.json").read_text())
    assert cal_meta["no_target_bank_labels_or_heldout_keys_accessed"] is True
    assert sha256_file(cal_npz_path) == cal_meta["data_sha256"]
    cal_inputs = np.load(cal_npz_path)["inputs"]
    init_streams: dict[int, np.ndarray] = {}
    for r in cal_meta["initial_states"]:
        p_init = ROOT / r["path"]
        assert sha256_file(p_init) == r["file_sha256"]
        init_streams[int(r["seed"])] = np.unpackbits(np.load(p_init)["boards"], axis=-1)[..., :169].astype(bool)

    aff_arb = W.calibrate_affine_chain_flops("leaper_arbitrary", 13, leaper=leaper, use_given_initial=True)
    aff_perm = W.calibrate_affine_chain_flops("leaper_perm", 13, leaper=leaper, use_given_initial=True)
    aff_det = W.calibrate_affine_chain_flops("nqueens_det", 13)

    w_warm = W.build_queen_leaper_wsp(13, leaper=leaper, cap=128, device=device)
    W.solve_arbitrary_swap(w_warm, cal_inputs, chains=20, seed=100, device=device, initial=init_streams[101])
    W.solve_permutation_swap(w_warm, cal_inputs, chains=20, seed=100, device=device, initial=init_streams[101])
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    def run_single_cal_trial(arm_kind: str, cap_val: int, seed_c: int) -> dict[str, Any]:
        w_c = W.build_queen_leaper_wsp(13, leaper=leaper, cap=cap_val, device=device)
        solver = W.solve_arbitrary_swap if arm_kind == "arbitrary" else W.solve_permutation_swap
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_s = time.perf_counter()
        z = solver(w_c, cal_inputs, chains=20, seed=seed_c, device=device, initial=init_streams[seed_c])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_s
        return {
            "seed": seed_c,
            "worker_solve_seconds": float(elapsed),
            "gpu_kernel_seconds": float(z["gpu_kernel_seconds"]),
            "stopped_chains": int(z["stopped"].sum()),
            "proposal_moves": int(z["moves"].sum()),
        }

    def summarize_cal_trials(arm_kind: str, cap_val: int, trials: list[dict[str, Any]]) -> dict[str, Any]:
        med_w = float(statistics.median(t["worker_solve_seconds"] for t in trials))
        med_k = float(statistics.median(t["gpu_kernel_seconds"] for t in trials))
        return {
            "kernel": arm_kind,
            "cap": int(cap_val),
            "median_worker_solve_seconds": med_w,
            "median_gpu_kernel_seconds": med_k,
            "trials": trials,
        }

    def time_arm_on_source(arm_kind: str, cap_val: int) -> dict[str, Any]:
        return summarize_cal_trials(
            arm_kind, cap_val, [run_single_cal_trial(arm_kind, cap_val, sc) for sc in (101, 102, 103)]
        )

    init_configs: list[tuple[str, int]] = [
        ("arbitrary", 20000),
        ("permutation", 12800),
        ("permutation", 19200),
        ("permutation", 25600),
        ("permutation", 32000),
        ("permutation", 44800),
        ("permutation", 64000),
        ("permutation", 96000),
    ]
    bucket_trials: dict[tuple[str, int], list[dict[str, Any]]] = {cfg: [] for cfg in init_configs}
    for idx_s, seed_c in enumerate((101, 102, 103)):
        ordered_cfgs = init_configs if idx_s % 2 == 0 else list(reversed(init_configs))
        for a_k, c_v in ordered_cfgs:
            tr = run_single_cal_trial(a_k, c_v, seed_c)
            bucket_trials[(a_k, c_v)].append(tr)

    ref_arb = summarize_cal_trials("arbitrary", 20000, bucket_trials[("arbitrary", 20000)])
    ref_t = ref_arb["median_worker_solve_seconds"]
    print(f"[CAL b{bank} {fam['name']}] arbitrary cap=20000 median={ref_t:.4f}s", flush=True)

    perm_candidates = []
    for _, p_cap in init_configs[1:]:
        rec = summarize_cal_trials("permutation", p_cap, bucket_trials[("permutation", p_cap)])
        perm_candidates.append(rec)
        ratio = rec["median_worker_solve_seconds"] / ref_t
        print(
            f"[CAL b{bank} {fam['name']}] perm cap={p_cap} median={rec['median_worker_solve_seconds']:.4f}s "
            f"ratio={ratio:.4f}",
            flush=True,
        )

    best_perm = min(perm_candidates, key=lambda v: (abs(v["median_worker_solve_seconds"] / ref_t - 1.0), v["cap"]))
    ratio = best_perm["median_worker_solve_seconds"] / ref_t
    interpolated = False
    for ref_step in range(3):
        if abs(ratio - 1.0) <= 0.05:
            break
        ordered = sorted(perm_candidates, key=lambda v: v["cap"])
        brackets = [
            (a, b)
            for a, b in zip(ordered[:-1], ordered[1:])
            if a["median_worker_solve_seconds"] <= ref_t <= b["median_worker_solve_seconds"]
        ]
        if brackets:
            lo_g, hi_g = brackets[0]
            dt = hi_g["median_worker_solve_seconds"] - lo_g["median_worker_solve_seconds"]
            raw_cap = lo_g["cap"] + (ref_t - lo_g["median_worker_solve_seconds"]) / max(dt, 1e-9) * (
                hi_g["cap"] - lo_g["cap"]
            )
        elif ref_t > ordered[-1]["median_worker_solve_seconds"]:
            a, b = ordered[0], ordered[-1]
            slope = max((b["median_worker_solve_seconds"] - a["median_worker_solve_seconds"]) / max(b["cap"] - a["cap"], 1), 1e-6)
            raw_cap = b["cap"] + (ref_t - b["median_worker_solve_seconds"]) / slope
        else:
            a, b = ordered[0], ordered[1]
            slope = max((b["median_worker_solve_seconds"] - a["median_worker_solve_seconds"]) / max(b["cap"] - a["cap"], 1), 1e-6)
            raw_cap = max(128.0, a["cap"] - (a["median_worker_solve_seconds"] - ref_t) / slope)

        interp_cap = max(128, int(round(raw_cap / 128.0) * 128))
        if interp_cap in [v["cap"] for v in perm_candidates]:
            break
        ref_rec = time_arm_on_source("permutation", interp_cap)
        perm_candidates.append(ref_rec)
        interpolated = True
        best_perm = min(perm_candidates, key=lambda v: (abs(v["median_worker_solve_seconds"] / ref_t - 1.0), v["cap"]))
        ratio = best_perm["median_worker_solve_seconds"] / ref_t
        print(
            f"[CAL b{bank} {fam['name']} REFINE-{ref_step + 1}] cap={interp_cap} "
            f"median={ref_rec['median_worker_solve_seconds']:.4f}s selected={best_perm['cap']} ratio={ratio:.4f}",
            flush=True,
        )

    return {
        "bank": bank,
        "family": fam["name"],
        "leaper": list(leaper),
        "calibration_hardware": hardware_info(device),
        "calibration_data_sha256": sha256_file(cal_npz_path),
        "no_target_data_accessed": True,
        "selected_arbitrary_cap": 20000,
        "selected_permutation_cap": int(best_perm["cap"]),
        "source_median_time_ratio": float(ratio),
        "matched_within_5pct": bool(abs(ratio - 1.0) <= 0.05),
        "interpolation_used": interpolated,
        "curve_caps_preregistered": [1024, 4096, 10240, 20000, 40960],
        "reference_arbitrary": ref_arb,
        "permutation_candidates": perm_candidates,
        "affine_flops": {"arbitrary": aff_arb, "permutation": aff_perm, "det": aff_det},
    }


def recalibrate_leaper_source_only(device: str) -> dict[str, Any]:
    """Re-run strictly source-only `N=12` `calibration-data.npz` timing calibration on an uncontended GPU (per #19)."""
    dev_sel_path = ARTIFACTS_DIR / "dev" / "dev_selections.json"
    if not dev_sel_path.exists():
        raise RuntimeError(f"Missing {dev_sel_path}; run `--stage dev` first!")
    selections = json.loads(dev_sel_path.read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    warmup_compiled_solvers(device)
    selections["build_timings"] = measure_build_timings(device)
    for bank, fam in FAMILIES.items():
        selections["queen_leaper_calibration"][f"b{bank}"] = calibrate_queen_leaper_bank(bank, fam, device)
    save_json_atomic(dev_sel_path, selections)
    save_json_atomic(RESULTS_DIR / "dev_validation_summary.json", selections)
    print(f"Updated source-only Queen+Leaper timing calibration in {dev_sel_path}", flush=True)
    return selections


# ---------------------------------------------------------------------------
# Stage 2: Post-Signoff Test Evaluation (`seeds = 0, 1, 2`)
# ---------------------------------------------------------------------------

def paired_bootstrap_diff(perm_ok: np.ndarray, arb_ok: np.ndarray, draws: int = 8000, seed: int = 20261114) -> dict[str, float]:
    """8,000-draw input-cluster paired bootstrap (`perm - arb` percentage points), matching `h3ad/analyze.py`."""
    # perm_ok, arb_ok: (3_seeds, 4096_inputs) bool/float in [0, 1]
    diff_per_input = 100.0 * (perm_ok.mean(axis=0) - arb_ok.mean(axis=0))  # (4096,)
    point = float(diff_per_input.mean())
    rng = np.random.default_rng(seed)
    n_in = len(diff_per_input)
    idx = rng.integers(0, n_in, size=(draws, n_in))
    boot = diff_per_input[idx].mean(axis=1)
    q025, q975 = np.quantile(boot, [0.025, 0.975])
    q05 = float(np.quantile(boot, 0.05))
    return {
        "gain_pct_points": point,
        "ci95_low": float(q025),
        "ci95_high": float(q975),
        "one_sided_95_lower_bound": q05,
        "draws": draws,
        "seed": seed,
    }


def run_test_stage(device: str, confirm_signoff: bool) -> dict[str, Any]:
    if not confirm_signoff:
        raise RuntimeError("Refusing to open test splits without explicit `--confirm-verifier-signoff`!")
    dev_sel_path = ARTIFACTS_DIR / "dev" / "dev_selections.json"
    if not dev_sel_path.exists():
        raise RuntimeError(f"Missing frozen dev selections at {dev_sel_path}; run `--stage dev` first!")

    dev_sel = json.loads(dev_sel_path.read_text())
    dev_sel_sha = sha256_file(dev_sel_path)
    torch.backends.cuda.matmul.allow_tf32 = False

    if device.startswith("cuda"):
        t_ctx0 = time.perf_counter()
        torch.zeros(1, device=device)
        torch.cuda.synchronize()
        cuda_context_s = time.perf_counter() - t_ctx0
    else:
        cuda_context_s = 0.0
    build_timings = measure_build_timings(device)
    compile_warmup_s = warmup_compiled_solvers(device)
    print(f"[TEST INIT] cuda_context={cuda_context_s:.3f}s compile_warmup={compile_warmup_s:.3f}s", flush=True)

    hw = hardware_info(device)

    # =========================================================================
    # 1. N-Queens Test (`n = 8, 10`, seeds `0, 1, 2`)
    # =========================================================================
    nq_summary: dict[str, Any] = {
        "benchmark": "nqueens_multi_solution",
        "dev_selections_sha256": dev_sel_sha,
        "cuda_context_seconds": cuda_context_s,
        "hardware": hw,
        "sizes": {},
    }
    for n in (8, 10):
        te_in, _, stored = NQ_DATA.load_split(n, "test")
        counts_npz = ROOT / "artifacts" / "multi_solution" / f"nq{n}-solution-counts.npz"
        c_data = dict(np.load(counts_npz))
        inverse = c_data["inverse"]
        S = c_data["counts"]
        assert np.array_equal(S[inverse], stored.reshape(-1))

        cfg_n = dev_sel["nqueens"][str(n)]
        aff_det = cfg_n["affine_flops"]["det"]
        aff_arb = cfg_n["affine_flops"]["arbitrary"]
        aff_perm = cfg_n["affine_flops"]["permutation"]

        w_det = W.build_nqueens_det_wsp(n, passes=cfg_n["selected_det"]["passes"], device=device)
        w_arb = W.build_nqueens_arbitrary_wsp(
            n, cap=cfg_n["selected_arbitrary"]["cap"], temperature=cfg_n["selected_arbitrary"]["temperature"], device=device
        )
        w_perm = W.build_nqueens_perm_wsp(
            n,
            cap=cfg_n["selected_permutation"]["cap"],
            temp_start=cfg_n["selected_permutation"]["temp_start"],
            end_ratio=cfg_n["selected_permutation"]["end_ratio"],
            device=device,
        )

        arms_nq = [
            (
                "wsp_det",
                "WSP-Det (rule-programmed deterministic Hopfield relaxation)",
                W.weight_disclosure(
                    w_det,
                    energy_keys=("W_attack", "bias"),
                    prior_disclosure="rule-programmed zero-training; unconstrained binary cells; deterministic single-mode coordinate relaxation",
                ),
                lambda s: W.solve_nqueens_det(w_det, te_in, chains=20, device=device),
                aff_det,
            ),
            (
                "wsp_arbitrary",
                "WSP-Assoc-Arbitrary (rule-programmed 20-chain focused heat-bath)",
                W.weight_disclosure(
                    w_arb,
                    energy_keys=("W_attack", "bias_scalar"),
                    prior_disclosure="rule-programmed zero-training; unconstrained binary cells (matched prior class to Hand-set CRAM)",
                ),
                lambda s: W.solve_nqueens_arbitrary(w_arb, te_in, chains=20, seed=s, device=device),
                aff_arb,
            ),
            (
                "wsp_permutation",
                "WSP-Assoc-Permutation (rule-programmed 20-chain rook-permutation 2-opt swaps)",
                W.weight_disclosure(
                    w_perm,
                    energy_keys=("W_attack",),
                    prior_disclosure="rule-programmed zero-training + row/column rook-permutation initialization and 2-opt swap proposal prior",
                ),
                lambda s: W.solve_permutation_swap(w_perm, te_in, chains=20, seed=s, device=device),
                aff_perm,
            ),
        ]

        size_res: dict[str, Any] = {
            "n": n,
            "test_rows": int(len(te_in)),
            "distinct_test_inputs": int(len(S)),
            "solution_counts_sha256": sha256_file(counts_npz),
            "arms": {},
        }
        for arm_id, label, disclosure, run_fn, aff in arms_nq:
            per_seed = []
            for seed in (0, 1, 2):
                res = run_fn(seed)
                sc = score_nqueens_chains(te_in, res, n, inverse, S, aff)
                rec_path = ARTIFACTS_DIR / "nqueens" / f"n{n}" / f"{arm_id}-s{seed}.npz"
                save_npz_atomic(
                    rec_path,
                    boards=np.packbits(res["boards"], axis=-1),
                    stopped=res["stopped"],
                    energy=res["energy"],
                    moves=res["moves"],
                )
                sc["seed"] = seed
                sc["records_npz"] = str(rec_path.relative_to(ROOT))
                sc["records_sha256"] = sha256_file(rec_path)
                per_seed.append(sc)
                print(
                    f"[TEST NQ-{n} {arm_id} s{seed}] N1={sc['acc_n1']:.4f}% N20={sc['acc_n20']:.4f}% "
                    f"Cov20={sc['cov20_per_puzzle']:.4f}% wrong={sc['wrong_stops']} t={sc['worker_solve_seconds']:.3f}s",
                    flush=True,
                )

            agg = summarize_across_seeds(
                per_seed,
                ("acc_n1", "acc_n20", "acc_n20_majority_vote", "cov20_per_puzzle", "cov20_per_row", "worker_solve_seconds"),
            )
            agg_flops_n1 = statistics.mean(r["fc2_flops_n1"]["mean_flops_per_instance"] for r in per_seed)
            agg_flops_n20 = statistics.mean(r["fc2_flops_n20"]["mean_flops_per_instance"] for r in per_seed)
            agg_flops_n20_cmp = statistics.mean(
                r["fc2_flops_n20"]["mean_flops_with_compare_per_instance"] for r in per_seed
            )
            size_res["arms"][arm_id] = {
                "label": label,
                "disclosure": disclosure,
                "metrics": agg,
                "mean_fc2_flops_n1_per_row": float(agg_flops_n1),
                "mean_fc2_flops_n20_per_row": float(agg_flops_n20),
                "mean_fc2_flops_with_compare_n20_per_row": float(agg_flops_n20_cmp),
                "pooled_wrong_stops": sum(int(r["wrong_stops"]) for r in per_seed),
                "pooled_budget_failures": sum(int(r["budget_failures"]) for r in per_seed),
                "per_seed": per_seed,
            }
        nq_summary["sizes"][str(n)] = size_res
    save_json_atomic(RESULTS_DIR / "nqueens_results.json", nq_summary)

    # =========================================================================
    # 2. Graph Coloring Test (`n = 8` [255], `n = 10` [192], seeds `0, 1, 2`)
    # =========================================================================
    gc_summary: dict[str, Any] = {
        "benchmark": "graph_coloring_multi_solution",
        "dev_selections_sha256": dev_sel_sha,
        "cuda_context_seconds": cuda_context_s,
        "hardware": hw,
        "sizes": {},
    }
    for n in (8, 10):
        x_te, y_te = GC_DATA.load_split(n, "test")
        edges_te, _ = GC_DATA.convert(x_te, y_te, n)
        A_te = GC_DATA.adjacency(edges_te, n)
        proper_te = GC_DATA.proper_canonical_sets(A_te)

        cfg_gc = dev_sel["graph_coloring"][str(n)]
        aff_det = cfg_gc["affine_flops"]["det"]
        aff_assoc = cfg_gc["affine_flops"]["assoc"]
        aff_assoc_nocanon = W.calibrate_affine_chain_flops("gc_assoc", n, canonicalize_in_arch=False)

        w_det = W.build_graph_coloring_det_wsp(n, passes=cfg_gc["selected_det"]["passes"], device=device)
        w_assoc = W.build_graph_coloring_assoc_wsp(
            n,
            cap=cfg_gc["selected_assoc"]["cap"],
            temperature=cfg_gc["selected_assoc"]["temperature"],
            theta=cfg_gc["selected_assoc"]["theta"],
            canonicalize_in_arch=True,
            device=device,
        )
        w_assoc_nocanon = W.build_graph_coloring_assoc_wsp(
            n,
            cap=cfg_gc["selected_assoc"]["cap"],
            temperature=cfg_gc["selected_assoc"]["temperature"],
            theta=cfg_gc["selected_assoc"]["theta"],
            canonicalize_in_arch=False,
            device=device,
        )

        arms_gc = [
            (
                "wsp_det",
                "WSP-Det (rule-programmed deterministic Potts relaxation)",
                W.weight_disclosure(
                    w_det,
                    energy_keys=("color_eye",),
                    prior_disclosure="rule-programmed zero-training; deterministic Gauss-Seidel Potts relaxation + in-arch tensor color canonicalizer",
                ),
                lambda s: W.solve_graph_coloring_det(w_det, edges_te, samples=20, device=device),
                aff_det,
            ),
            (
                "wsp_assoc",
                "WSP-Assoc (rule-programmed 20-chain Gumbel-Potts + in-arch tensor canonicalizer)",
                W.weight_disclosure(
                    w_assoc,
                    energy_keys=("color_eye",),
                    prior_disclosure="rule-programmed zero-training; 3-Potts edge-equality cost + pure-tensor first-appearance color canonicalizer inside arch.forward (matched rule scope to Hand-set CRAM)",
                ),
                lambda s: W.solve_graph_coloring_assoc(w_assoc, edges_te, samples=20, seed=s, device=device),
                aff_assoc,
            ),
            (
                "wsp_assoc_uncanonicalized",
                "WSP-Assoc-Uncanonicalized (ablation without color-order canonicalizer)",
                W.weight_disclosure(
                    w_assoc_nocanon,
                    energy_keys=("color_eye",),
                    prior_disclosure="rule-programmed zero-training; S3-symmetric 3-Potts edge-equality cost WITHOUT color-order canonicalization",
                ),
                lambda s: W.solve_graph_coloring_assoc(w_assoc_nocanon, edges_te, samples=20, seed=s, device=device),
                aff_assoc_nocanon,
            ),
        ]

        size_gc: dict[str, Any] = {
            "n": n,
            "test_graphs": int(len(edges_te)),
            "solutions_total": int(sum(len(s) for s in proper_te)),
            "arms": {},
        }
        for arm_id, label, disclosure, run_fn, aff in arms_gc:
            per_seed = []
            for seed in (0, 1, 2):
                res = run_fn(seed)
                sc = score_graph_coloring_chains(A_te, res, aff, proper_sets=proper_te)
                rec_path = ARTIFACTS_DIR / "graph_coloring" / f"n{n}" / f"{arm_id}-s{seed}.npz"
                save_npz_atomic(
                    rec_path,
                    boards=res["boards"].astype(np.uint8),
                    stopped=res["stopped"],
                    energy=res["energy"],
                    moves=res["moves"],
                )
                sc["seed"] = seed
                sc["records_npz"] = str(rec_path.relative_to(ROOT))
                sc["records_sha256"] = sha256_file(rec_path)
                per_seed.append(sc)
                print(
                    f"[TEST GC-{n} {arm_id} s{seed}] conf1={sc['conflicts_1']} conf20={sc['conflicts_n20_min_energy']} "
                    f"Cov20={sc['coverage_20_pct']:.4f}% (canon_diag={sc['coverage_canonicalized_20_pct']:.4f}%) "
                    f"wrong={sc['wrong_stops_all']} t={sc['worker_solve_seconds']:.4f}s",
                    flush=True,
                )

            agg = summarize_across_seeds(
                per_seed,
                ("coverage_20_pct", "coverage_canonicalized_20_pct", "proper_1_pct", "worker_solve_seconds"),
            )
            size_gc["arms"][arm_id] = {
                "label": label,
                "disclosure": disclosure,
                "n1_conflicts_per_seed": [int(r["conflicts_1"]) for r in per_seed],
                "n20_conflicts_min_energy_per_seed": [int(r["conflicts_n20_min_energy"]) for r in per_seed],
                "n20_conflicts_majority_per_seed": [int(r["conflicts_majority_20"]) for r in per_seed],
                "metrics": agg,
                "mean_fc2_flops_n1_per_graph": float(
                    statistics.mean(r["fc2_flops_n1"]["mean_flops_per_instance"] for r in per_seed)
                ),
                "mean_fc2_flops_n20_per_graph": float(
                    statistics.mean(r["fc2_flops_n20"]["mean_flops_per_instance"] for r in per_seed)
                ),
                "mean_fc2_flops_with_compare_n20_per_graph": float(
                    statistics.mean(r["fc2_flops_n20"]["mean_flops_with_compare_per_instance"] for r in per_seed)
                ),
                "pooled_wrong_stops": sum(int(r["wrong_stops_all"]) for r in per_seed),
                "pooled_budget_failures": sum(int(r["budget_failures_all"]) for r in per_seed),
                "per_seed": per_seed,
            }
        gc_summary["sizes"][str(n)] = size_gc
    save_json_atomic(RESULTS_DIR / "graph_coloring_results.json", gc_summary)

    # =========================================================================
    # 3. Generalized Queen+Leaper (`N = 13`, `b0` zebra & `b1` (2,5)-leaper, 4,096 inputs x 3 seeds)
    # =========================================================================
    ql_summary: dict[str, Any] = {
        "benchmark": "generalized_queen_leaper_n13",
        "dev_selections_sha256": dev_sel_sha,
        "cuda_context_seconds": cuda_context_s,
        "hardware": hw,
        "families": {},
    }
    for bank, fam in FAMILIES.items():
        leaper = fam["leaper"]
        b_dir = H3AD_ARTIFACTS / f"b{bank}"
        data_path = b_dir / "data.npz"
        data = dict(np.load(data_path))
        test_inputs = data["test_inputs"]
        all_solutions = data["all_solutions"]
        sol_counts = data["test_solution_counts"]
        sol_keys = {row.tobytes() for row in all_solutions}
        denom = np.minimum(sol_counts, 20)

        cal_cfg = dev_sel["queen_leaper_calibration"][f"b{bank}"]
        arb_cap = int(cal_cfg["selected_arbitrary_cap"])
        perm_cap = int(cal_cfg["selected_permutation_cap"])
        aff_arb = cal_cfg["affine_flops"]["arbitrary"]
        aff_perm = cal_cfg["affine_flops"]["permutation"]
        aff_det = cal_cfg["affine_flops"]["det"]

        initial_by_seed: dict[int, tuple[np.ndarray, str]] = {}
        for seed in (0, 1, 2):
            p_init = b_dir / "initial" / f"s{seed}.npz"
            arr = np.unpackbits(np.load(p_init)["boards"], axis=-1)[..., :169].astype(bool)
            initial_by_seed[seed] = (arr, sha256_file(p_init))

        # Untimed warmup on exact target shape before timed trials
        w_warm = W.build_queen_leaper_wsp(13, leaper=leaper, cap=64, device=device)
        W.solve_arbitrary_swap(w_warm, test_inputs, chains=20, seed=100, device=device, initial=initial_by_seed[0][0])
        W.solve_permutation_swap(w_warm, test_inputs, chains=20, seed=100, device=device, initial=initial_by_seed[0][0])
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        def eval_ql_run(arm_name: str, cap_val: int, seed: int, save_npz: bool = True) -> tuple[dict[str, Any], np.ndarray]:
            init_boards, init_sha = initial_by_seed[seed]
            if arm_name == "det":
                w_m = W.build_queen_leaper_det_wsp(13, leaper=leaper, passes=12, device=device)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_s = time.perf_counter()
                z = W.solve_nqueens_det(w_m, test_inputs, chains=20, device=device)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t_s
                aff_use = aff_det
            else:
                w_m = W.build_queen_leaper_wsp(13, leaper=leaper, cap=cap_val, device=device)
                solver = W.solve_arbitrary_swap if arm_name == "arbitrary" else W.solve_permutation_swap
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_s = time.perf_counter()
                z = solver(w_m, test_inputs, chains=20, seed=seed, device=device, initial=init_boards)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t_s
                aff_use = aff_arb if arm_name == "arbitrary" else aff_perm

            boards = z["boards"]
            stopped = z["stopped"]
            energy = z["energy"]
            moves = z["moves"]

            if not bool((boards >= test_inputs[:, None, :]).all()):
                raise AssertionError(f"Clue corruption in {arm_name} on bank {bank}!")
            if arm_name == "permutation":
                grid = boards.reshape(-1, 13, 13)
                if not bool((grid.sum(axis=1) == 1).all() and (grid.sum(axis=2) == 1).all()):
                    raise AssertionError("Permutation invariant violated!")

            valid = np.array(
                [[(row.tobytes() in sol_keys) and bool(row[given].all()) for row in samples] for given, samples in zip(test_inputs, boards)],
                dtype=bool,
            )
            wrong_stops = int((stopped & ~valid).sum())
            if wrong_stops != 0:
                raise AssertionError(f"Zero-wrong-stop invariant violated on {arm_name} bank {bank}: {wrong_stops}")

            orig_idx = energy.argmin(axis=1)
            sel_idx = np.where(stopped.any(axis=1), np.where(stopped, energy, np.inf).argmin(axis=1), orig_idx)
            row_ids = np.arange(len(test_inputs))
            cert_success = (stopped & valid).any(axis=1)
            n20_orig = valid[row_ids, orig_idx]
            n20_cert = valid[row_ids, sel_idx]
            found = np.array(
                [min(len({row.tobytes() for row in samples[ok]}), int(lim)) for samples, ok, lim in zip(boards, valid, denom)],
                dtype=np.int64,
            )
            fl_n20 = W.compute_trajectory_flops(moves, aff_use, chains_used=20)

            rec = {
                "bank": bank,
                "family": fam["name"],
                "arm": arm_name,
                "cap": int(cap_val),
                "seed": int(seed),
                "worker_solve_seconds": float(elapsed),
                "gpu_kernel_seconds": float(z["gpu_kernel_seconds"]),
                "certified_success_pct": 100.0 * float(cert_success.mean()),
                "certificate_accuracy_pct": 100.0 * float(n20_cert.mean()),
                "original_accuracy_pct": 100.0 * float(n20_orig.mean()),
                "coverage_pct": 100.0 * float(found.sum() / denom.sum()),
                "wrong_stops": wrong_stops,
                "capped_chains": int((~stopped).sum()),
                "halted_pct": 100.0 * float(stopped.mean()),
                "proposal_moves": int(moves.sum()),
                "accepted_swaps": int(z["accepted"].sum()) if "accepted" in z else 0,
                "fc2_flops_n20": fl_n20,
                "initial_file_sha256": init_sha,
            }
            if save_npz:
                npz_path = ARTIFACTS_DIR / "queen_leaper" / f"b{bank}" / f"{arm_name}-c{cap_val}-s{seed}.npz"
                save_npz_atomic(
                    npz_path,
                    boards=np.packbits(boards, axis=-1),
                    stopped=stopped,
                    energy=energy,
                    moves=moves,
                    valid=valid,
                    certified_success=cert_success,
                    found=found,
                )
                rec["records_npz"] = str(npz_path.relative_to(ROOT))
                rec["records_sha256"] = sha256_file(npz_path)
            return rec, cert_success

        # Primary source-calibrated matched-GPU-time comparison (interleaved across seeds 0, 1, 2)
        arb_primary = []
        perm_primary = []
        arb_cert_mat = []
        perm_cert_mat = []
        for seed in (0, 1, 2):
            order = [("arbitrary", arb_cap), ("permutation", perm_cap)] if seed % 2 == 0 else [("permutation", perm_cap), ("arbitrary", arb_cap)]
            res_pair: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
            for a_name, c_val in order:
                r_dict, c_vec = eval_ql_run(a_name, c_val, seed, save_npz=True)
                res_pair[a_name] = (r_dict, c_vec)
                print(
                    f"[TEST QL b{bank} {fam['name']} {a_name} cap={c_val} s{seed}] "
                    f"cert={r_dict['certified_success_pct']:.3f}% cov={r_dict['coverage_pct']:.3f}% "
                    f"wrong={r_dict['wrong_stops']} t={r_dict['worker_solve_seconds']:.4f}s",
                    flush=True,
                )
            arb_primary.append(res_pair["arbitrary"][0])
            arb_cert_mat.append(res_pair["arbitrary"][1])
            perm_primary.append(res_pair["permutation"][0])
            perm_cert_mat.append(res_pair["permutation"][1])

        det_rec, _ = eval_ql_run("det", 12, 0, save_npz=True)
        print(
            f"[TEST QL b{bank} {fam['name']} det] cert={det_rec['certified_success_pct']:.3f}% "
            f"cov={det_rec['coverage_pct']:.3f}% wrong={det_rec['wrong_stops']}",
            flush=True,
        )

        # Multi-cap time-vs-accuracy Pareto curve (`seeds 0, 1, 2`)
        curve_points: dict[str, list[dict[str, Any]]] = {"arbitrary": [], "permutation": []}
        for c_val in cal_cfg["curve_caps_preregistered"]:
            for a_name in ("arbitrary", "permutation"):
                if a_name == "arbitrary" and c_val == arb_cap:
                    seed_runs = arb_primary
                elif a_name == "permutation" and c_val == perm_cap:
                    seed_runs = perm_primary
                else:
                    seed_runs = [eval_ql_run(a_name, c_val, s, save_npz=False)[0] for s in (0, 1, 2)]
                agg_c = summarize_across_seeds(
                    seed_runs,
                    ("certified_success_pct", "coverage_pct", "worker_solve_seconds", "gpu_kernel_seconds"),
                )
                curve_points[a_name].append({
                    "cap": int(c_val),
                    "median_worker_solve_seconds": float(statistics.median(r["worker_solve_seconds"] for r in seed_runs)),
                    "median_gpu_kernel_seconds": float(statistics.median(r["gpu_kernel_seconds"] for r in seed_runs)),
                    "mean_fc2_flops_n20": float(
                        statistics.mean(r["fc2_flops_n20"]["mean_flops_per_instance"] for r in seed_runs)
                    ),
                    "mean_fc2_flops_with_compare_n20": float(
                        statistics.mean(r["fc2_flops_n20"]["mean_flops_with_compare_per_instance"] for r in seed_runs)
                    ),
                    "pooled_wrong_stops": sum(int(r["wrong_stops"]) for r in seed_runs),
                    "metrics": agg_c,
                    "per_seed": seed_runs,
                })
                print(
                    f"[CURVE QL b{bank} {a_name} cap={c_val}] "
                    f"cert={agg_c['certified_success_pct']['mean']:.3f}% "
                    f"med_t={curve_points[a_name][-1]['median_worker_solve_seconds']:.4f}s",
                    flush=True,
                )

        boot = paired_bootstrap_diff(np.stack(perm_cert_mat, axis=0), np.stack(arb_cert_mat, axis=0))
        med_arb_t = float(statistics.median(r["worker_solve_seconds"] for r in arb_primary))
        med_perm_t = float(statistics.median(r["worker_solve_seconds"] for r in perm_primary))

        w_ql_disc = W.build_queen_leaper_wsp(13, leaper=leaper, cap=arb_cap)
        ql_summary["families"][f"b{bank}"] = {
            "bank": bank,
            "family": fam["name"],
            "leaper": list(leaper),
            "test_inputs": int(len(test_inputs)),
            "all_solutions_count": int(len(all_solutions)),
            "data_npz_sha256": sha256_file(data_path),
            "source_calibration": {
                "selected_arbitrary_cap": arb_cap,
                "selected_permutation_cap": perm_cap,
                "source_median_time_ratio": cal_cfg["source_median_time_ratio"],
            },
            "matched_time_comparison": {
                "arbitrary": {
                    "disclosure": W.weight_disclosure(
                        w_ql_disc,
                        energy_keys=("W_attack",),
                        prior_disclosure="rule-programmed from queen+leaper offsets (0 training boards); fixed-cardinality arbitrary free-cell swap proposals from shared initial boards",
                    ),
                    "cap": arb_cap,
                    "median_worker_solve_seconds": med_arb_t,
                    "metrics": summarize_across_seeds(
                        arb_primary,
                        ("certified_success_pct", "certificate_accuracy_pct", "original_accuracy_pct", "coverage_pct", "worker_solve_seconds"),
                    ),
                    "mean_fc2_flops_n20": float(
                        statistics.mean(r["fc2_flops_n20"]["mean_flops_per_instance"] for r in arb_primary)
                    ),
                    "mean_fc2_flops_with_compare_n20": float(
                        statistics.mean(r["fc2_flops_n20"]["mean_flops_with_compare_per_instance"] for r in arb_primary)
                    ),
                    "pooled_wrong_stops": sum(int(r["wrong_stops"]) for r in arb_primary),
                    "per_seed": arb_primary,
                },
                "permutation": {
                    "disclosure": W.weight_disclosure(
                        w_ql_disc,
                        energy_keys=("W_attack",),
                        prior_disclosure="rule-programmed from queen+leaper offsets (0 training boards) + row/column permutation-preserving 2-opt swap proposal prior from shared initial boards",
                    ),
                    "cap": perm_cap,
                    "median_worker_solve_seconds": med_perm_t,
                    "metrics": summarize_across_seeds(
                        perm_primary,
                        ("certified_success_pct", "certificate_accuracy_pct", "original_accuracy_pct", "coverage_pct", "worker_solve_seconds"),
                    ),
                    "mean_fc2_flops_n20": float(
                        statistics.mean(r["fc2_flops_n20"]["mean_flops_per_instance"] for r in perm_primary)
                    ),
                    "mean_fc2_flops_with_compare_n20": float(
                        statistics.mean(r["fc2_flops_n20"]["mean_flops_with_compare_per_instance"] for r in perm_primary)
                    ),
                    "pooled_wrong_stops": sum(int(r["wrong_stops"]) for r in perm_primary),
                    "per_seed": perm_primary,
                },
                "deterministic_control": det_rec,
                "target_median_solve_time_ratio_perm_over_arb": float(med_perm_t / max(med_arb_t, 1e-9)),
                "paired_bootstrap_perm_minus_arb": boot,
            },
            "time_vs_accuracy_curves": curve_points,
        }
    save_json_atomic(RESULTS_DIR / "queen_leaper_results.json", ql_summary)

    master = {
        "status": "complete",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wsp_upstream_repo": "https://github.com/effl-lab/Weight-Space-Programming",
        "wsp_upstream_commit": UPSTREAM_COMMIT,
        "wsp_license": "MIT",
        "dev_selections_sha256": dev_sel_sha,
        "cuda_context_seconds": cuda_context_s,
        "compile_warmup_seconds": compile_warmup_s,
        "build_timings": build_timings,
        "hardware": hw,
        "files": {
            "nqueens": "results/wsp/multi/nqueens_results.json",
            "graph_coloring": "results/wsp/multi/graph_coloring_results.json",
            "queen_leaper": "results/wsp/multi/queen_leaper_results.json",
        },
        "nqueens": nq_summary,
        "graph_coloring": gc_summary,
        "queen_leaper": ql_summary,
    }
    save_json_atomic(RESULTS_DIR / "summary.json", master)
    print(f"All multi-solution WSP test results written atomically to {RESULTS_DIR}", flush=True)
    return master


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-solution Weight-Space Programming runner")
    parser.add_argument("--stage", choices=("dev", "recalibrate-leaper", "test"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confirm-verifier-signoff", action="store_true")
    args = parser.parse_args()
    if args.stage == "dev":
        run_dev_stage(args.device)
    elif args.stage == "recalibrate-leaper":
        recalibrate_leaper_source_only(args.device)
    else:
        run_test_stage(args.device, args.confirm_verifier_signoff)


if __name__ == "__main__":
    main()
