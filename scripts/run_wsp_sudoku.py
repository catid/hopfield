"""Runner, FullCounter (FC2) auditor, dev10k gate, and full test evaluator for WSP Sudoku-Extreme.

Strictly enforces the pre-experiment dev gate (`full-dev10k.npz` + `--verifier-approved`) before
touching any held-out slice of `upstream/data/full-test/full-all.npz`.

Subcommands:
  - `audit-counts`: Count FC2 FLOPs, verify exact affine step scaling (`max_abs_residual == 0`),
    audit AST static-tensor-program compliance (`static_problems == []`), compute weight metadata,
    and measure cold/warm `build()` times on GPU and 1-thread CPU.
  - `dev`: Run `WSP-Transformer`, `WSP-Det`, and 3-seed `WSP-Assoc` strictly on `full-dev10k.npz`
    (`0..9999`), write `artifacts/wsp/sudoku/dev/wsp-assoc-dev-s{0,1,2}.json` and
    `results/wsp/sudoku/dev10k-smoke-summary.json`, and project full-test wall-clock runtime.
  - `test-all` (requires `--verifier-approved` + passing dev records):
      1. `full-all[10000:422786]` (`412,786` unseen test puzzles): `WSP-Assoc` (`seeds 0,1,2`),
         `WSP-Det`, and `WSP-Transformer` -> `results/wsp/sudoku/wsp-sudoku-full-test-412786.json`
      2. `full-all[10000:20000]` (`10,000` puzzles):
         - Figure 2a anytime compute curves (`results/wsp/sudoku/wsp-sudoku-anytime-curves.json`)
         - Figure 3a 8-bin difficulty curve (`results/wsp/sudoku/wsp-sudoku-figure3a-difficulty.json`)
         - Figure 3b `k=0` 6-seed zero-training reference (`results/wsp/sudoku/wsp-sudoku-figure3b-zero-data.json`)
      3. `full-all[30000:40000]` (`10,000` puzzles, `seeds 0..5`):
         - `README.md` Sec 1 short-cap (`262,144` moves & FLOP-matched) + budget curve through
           `16,777,216` moves -> `results/wsp/sudoku/wsp-sudoku-count-budget-30k-40k.json`
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "reproduction" / "baselines" / "code" / "flops") not in sys.path:
    sys.path.insert(0, str(ROOT / "reproduction" / "baselines" / "code" / "flops"))

from fullcount import FullCounter  # noqa: E402

from hopfield_cram.wsp_contract import (  # noqa: E402
    atomic_write_json,
    check_wsp_static_tensor_program,
    measure_build_times,
    sha256_file,
    verify_fullcounter_affine,
    wsp_weight_metadata,
)
from hopfield_cram.wsp_sudoku import (  # noqa: E402
    N_CELLS,
    N_DIGITS,
    WSP_ASSOC_ARCH_SOURCE,
    WSP_ASSOC_PROP_STEP_SOURCE,
    WSP_ASSOC_STAGE1_SOURCE,
    WSP_DET_ARCH_SOURCE,
    WSP_TRANSFORMER_ARCH_SOURCE,
    WSPAssocSearch,
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
    wsp_det_forward,
    wsp_transformer_forward,
)

RESULTS_DIR = ROOT / "results" / "wsp" / "sudoku"
ARTIFACTS_DIR = ROOT / "artifacts" / "wsp" / "sudoku"

# Frozen zero-training WSP configurations (selected strictly on dev10k)
FROZEN_ASSOC = {
    "prop_steps": 16,
    "temp_hi": 0.52,
    "temp_lo": 0.38,
    "anneal_period": 524288,
    "theta": 4.0 / 9.0,
    "tau_tol": 1e-3,
    "C": 4,
    "check": 4096,
    "cap_flops": 4e11,
}
FROZEN_DET = {"prop_steps": 16, "relax_steps": 32, "eta": 0.5, "beta": 4.0}
FROZEN_TF = {"steps": 16}

TF_SWEEP_STEPS = [1, 2, 4, 8, 16, 24]
DET_SWEEP_SETTINGS = [(1, 0), (2, 0), (4, 0), (8, 0), (16, 0), (16, 8), (16, 16), (16, 32), (16, 64)]
COUNT_BUDGET_CAPS = [262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216]
FIG3A_BINS = np.asarray([0, 1, 2, 4, 8, 16, 32, 64, 10**9], dtype=np.int64)
FIG3A_BUDGETS = np.logspace(6, 12, 121)
FIG2A_GRID = 10.0 ** np.arange(6.0, 11.75 + 1e-9, 0.125)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_save_npz(arrays: dict[str, np.ndarray], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return sha256_file(path)


def flop_dist_stats(fl: np.ndarray) -> dict[str, float]:
    fl64 = np.asarray(fl, dtype=np.float64)
    return {
        "mean": float(fl64.mean()),
        "median": float(np.quantile(fl64, 0.5)),
        "p99": float(np.quantile(fl64, 0.99)),
        "p999": float(np.quantile(fl64, 0.999)),
        "max": float(fl64.max()),
    }


def det_flop_per_puzzle(prop_s: int, rel_s: int, det_counts: dict[str, Any], *, with_compare: bool = False) -> float:
    suf = "_with_compare" if with_compare else ""
    return float(
        det_counts[f"setup_plus_cert_total{suf}"]
        + prop_s * det_counts[f"per_prop_step_total{suf}"]
        + rel_s * det_counts[f"per_relax_step_total{suf}"]
    )


def tf_flop_per_puzzle(steps: int, tf_counts: dict[str, Any], *, with_compare: bool = False) -> float:
    suf = "_with_compare" if with_compare else ""
    return float(tf_counts[f"setup_total{suf}"] + steps * tf_counts[f"per_step_total{suf}"])


def assoc_t_cap(assoc_counts: dict[str, Any], cap_flops: float = 4e11) -> int:
    mv = int(assoc_counts["move_per_chain"]["total"])
    C = int(assoc_counts["C"])
    return int(cap_flops // (mv * C))


def run_audit_counts(device: str = "cuda") -> dict[str, Any]:
    counts = count_wsp_sudoku_flops(FullCounter)
    x1 = torch.zeros((1, N_CELLS), dtype=torch.int64)
    x1[0, 0] = 1
    step_grid = (0, 1, 2, 3, 5, 8, 13, 21)

    w_assoc_cpu = build_wsp_assoc_weights(device="cpu")
    peer_adj_cpu = w_assoc_cpu["peer_adj"]

    def _cert_check(logits: torch.Tensor) -> torch.Tensor:
        pred = logits.argmax(dim=-1)
        oh = F.one_hot(pred, N_DIGITS).float()
        conf = 0.5 * (oh * torch.matmul(peer_adj_cpu, oh)).sum(dim=(1, 2))
        is_clue = x1 > 0
        clue_ok = ((~is_clue) | (pred == (x1 - 1))).all(dim=1)
        return (conf == 0.0) & clue_ok

    def tf_at(s: int) -> dict[str, Any]:
        w = build_wsp_transformer_weights(steps=s, device="cpu")
        with torch.no_grad(), FullCounter() as fc:
            _ = _cert_check(wsp_transformer_forward(w, x1))
        return fc.summary()

    def det_prop_at(s: int) -> dict[str, Any]:
        w = build_wsp_det_weights(prop_steps=s, relax_steps=2, device="cpu")
        with torch.no_grad(), FullCounter() as fc:
            _ = _cert_check(wsp_det_forward(w, x1))
        return fc.summary()

    def det_rel_at(s: int) -> dict[str, Any]:
        w = build_wsp_det_weights(prop_steps=2, relax_steps=s, device="cpu")
        with torch.no_grad(), FullCounter() as fc:
            _ = _cert_check(wsp_det_forward(w, x1))
        return fc.summary()

    def assoc_s1_at(s: int) -> dict[str, Any]:
        with torch.no_grad(), FullCounter() as fc:
            is_clue = (x1 > 0).unsqueeze(-1)
            clue_oh = F.one_hot((x1 - 1).clamp(min=0), N_DIGITS).float() * is_clue.float()
            mb = torch.where(is_clue, clue_oh, torch.ones_like(clue_oh))
            for _ in range(s):
                m_prev = mb
                mb = wsp_assoc_prop_step_forward(w_assoc_cpu, x1, m_prev)
                _ = (mb == m_prev).all(dim=(1, 2))
                _ = (mb.sum(dim=-1) == 1.0).all(dim=1)
        return fc.summary()

    def assoc_move_at(s: int) -> dict[str, Any]:
        st = make_synthetic_assoc_chain_state(w_assoc_cpu, seed=0, device="cpu")
        st["tau"] = torch.tensor(-1e9)
        with torch.no_grad(), FullCounter() as fc:
            for k in range(s):
                st["k"] = torch.tensor(k, dtype=torch.int32)
                wsp_assoc_move_pytorch(st)
        return fc.summary()

    affine_checks = {
        "WSP-Transformer": verify_fullcounter_affine(tf_at, step_grid),
        "WSP-Det_prop": verify_fullcounter_affine(det_prop_at, step_grid),
        "WSP-Det_relax": verify_fullcounter_affine(det_rel_at, step_grid),
        "WSP-Assoc_stage1_prop": verify_fullcounter_affine(assoc_s1_at, step_grid),
        "WSP-Assoc_move": verify_fullcounter_affine(assoc_move_at, step_grid),
    }
    for name, aff in affine_checks.items():
        if aff["max_abs_residual_total"] != 0 or aff["max_abs_residual_total_with_compare"] != 0:
            raise RuntimeError(f"Nonzero affine FullCounter residual in {name}: {aff}")

    ast_checks = {
        "WSP-Transformer": check_wsp_static_tensor_program(WSP_TRANSFORMER_ARCH_SOURCE, "forward(w, x)"),
        "WSP-Det": check_wsp_static_tensor_program(WSP_DET_ARCH_SOURCE, "forward(w, x)"),
        "WSP-Assoc_prop_step": check_wsp_static_tensor_program(WSP_ASSOC_PROP_STEP_SOURCE, "forward(w, x, m)"),
        "WSP-Assoc_stage1": check_wsp_static_tensor_program(WSP_ASSOC_STAGE1_SOURCE, "forward(w, x)"),
        "WSP-Assoc_full": check_wsp_static_tensor_program(
            WSP_ASSOC_ARCH_SOURCE, "forward(w, x, u_init, u_cell, u_focus, u_digit)"
        ),
    }
    for name, probs in ast_checks.items():
        if probs:
            raise RuntimeError(f"AST contract violation in {name}: {probs}")

    cuda_dev = device if torch.cuda.is_available() else None
    build_timings = {
        "WSP-Transformer": measure_build_times(
            lambda dev: build_wsp_transformer_weights(steps=FROZEN_TF["steps"], device=dev), cuda_device=cuda_dev
        ),
        "WSP-Det": measure_build_times(
            lambda dev: build_wsp_det_weights(FROZEN_DET["prop_steps"], FROZEN_DET["relax_steps"], device=dev),
            cuda_device=cuda_dev,
        ),
        "WSP-Assoc": measure_build_times(
            lambda dev: build_wsp_assoc_weights(
                prop_steps=FROZEN_ASSOC["prop_steps"],
                temp_hi=FROZEN_ASSOC["temp_hi"],
                temp_lo=FROZEN_ASSOC["temp_lo"],
                anneal_period=FROZEN_ASSOC["anneal_period"],
                theta=FROZEN_ASSOC["theta"],
                tau_tol=FROZEN_ASSOC["tau_tol"],
                device=dev,
            ),
            cuda_device=cuda_dev,
        ),
    }

    metadata = {
        "WSP-Transformer": wsp_weight_metadata(
            build_wsp_transformer_weights(steps=FROZEN_TF["steps"], device="cpu"), index_keys={"steps"}
        ),
        "WSP-Det": wsp_weight_metadata(
            build_wsp_det_weights(FROZEN_DET["prop_steps"], FROZEN_DET["relax_steps"], device="cpu"),
            index_keys={"prop_steps", "relax_steps"},
        ),
        "WSP-Assoc": wsp_weight_metadata(
            w_assoc_cpu, index_keys={"peer_nbr", "color_cells", "prop_steps", "anneal_period"}
        ),
    }

    # Omit verbose sample_summary_at_s1 inside affine_checks to keep JSON crisp
    compact_affine = {
        k: {kk: vv for kk, vv in v.items() if kk != "sample_summary_at_s1"} for k, v in affine_checks.items()
    }

    record = {
        "utc": utc_now(),
        "convention": "FC2: baselines/flops/fullcount.py FullCounter (arithmetic + matmul + transc; compare & rng recorded separately)",
        "ast_static_checks": ast_checks,
        "weight_metadata": metadata,
        "build_timings": build_timings,
        "counted_flops": counts,
        "affine_extrapolation_checks": compact_affine,
        "source_sha256": {
            "src/hopfield_cram/wsp_contract.py": sha256_file(ROOT / "src/hopfield_cram/wsp_contract.py"),
            "src/hopfield_cram/wsp_sudoku.py": sha256_file(ROOT / "src/hopfield_cram/wsp_sudoku.py"),
            "reproduction/baselines/code/flops/fullcount.py": sha256_file(
                ROOT / "reproduction/baselines/code/flops/fullcount.py"
            ),
            "scripts/run_wsp_sudoku.py": sha256_file(Path(__file__)),
        },
    }
    out_path = RESULTS_DIR / "wsp-sudoku-opcounts-and-params.json"
    digest = atomic_write_json(record, out_path)
    print(f"[audit-counts] wrote {out_path} (sha256={digest[:16]}...)", flush=True)
    return record


def evaluate_det_sweep(
    x_tok: torch.Tensor,
    sol: np.ndarray,
    counts: dict[str, Any],
    device: str = "cuda",
) -> dict[str, Any]:
    dev = torch.device(device)
    peer_adj = build_wsp_det_weights(1, 0, device=dev)["peer_adj"]

    tf_rows: list[dict[str, Any]] = []
    for steps in TF_SWEEP_STEPS:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        w_tf = build_wsp_transformer_weights(steps=steps, device=dev)
        res = evaluate_deterministic_wsp(wsp_transformer_forward, w_tf, x_tok, peer_adj, batch_size=2048)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        wall_s = time.perf_counter() - t0

        pred_np = res["pred"].numpy()
        stopped_np = res["stopped"].numpy()
        exact_np = (pred_np == sol).all(axis=1)
        v_all = verify_sudoku_boards(pred_np, x_tok)
        wrong_stops = int((stopped_np & ~exact_np).sum())
        if stopped_np.any():
            assert int(v_all["valid_mask"][stopped_np].sum()) == int(stopped_np.sum())
        fl_tot = tf_flop_per_puzzle(steps, counts["WSP-Transformer"], with_compare=False)
        fl_cmp = tf_flop_per_puzzle(steps, counts["WSP-Transformer"], with_compare=True)
        tf_rows.append(
            {
                "steps": int(steps),
                "n": int(len(sol)),
                "solved": int(exact_np.sum()),
                "solved_pct": float(100.0 * exact_np.mean()),
                "halted": int(stopped_np.sum()),
                "halted_pct": float(100.0 * stopped_np.mean()),
                "wrong_stops": wrong_stops,
                "wrong_halts_pct_of_halted": (
                    float(100.0 * wrong_stops / int(stopped_np.sum())) if int(stopped_np.sum()) > 0 else 0.0
                ),
                "independent_rule_valid_count": int(v_all["n_valid"]),
                "clue_violations": int(v_all["n_clue_violations"]),
                "flop_per_puzzle_total": fl_tot,
                "flop_per_puzzle_with_compare": fl_cmp,
                "wall_s": float(wall_s),
            }
        )

    det_rows: list[dict[str, Any]] = []
    for prop_s, rel_s in DET_SWEEP_SETTINGS:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        w_d = build_wsp_det_weights(prop_steps=prop_s, relax_steps=rel_s, device=dev)
        res = evaluate_deterministic_wsp(wsp_det_forward, w_d, x_tok, peer_adj, batch_size=4096)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        wall_s = time.perf_counter() - t0

        pred_np = res["pred"].numpy()
        stopped_np = res["stopped"].numpy()
        exact_np = (pred_np == sol).all(axis=1)
        v_all = verify_sudoku_boards(pred_np, x_tok)
        wrong_stops = int((stopped_np & ~exact_np).sum())
        if stopped_np.any():
            assert int(v_all["valid_mask"][stopped_np].sum()) == int(stopped_np.sum())
        fl_tot = det_flop_per_puzzle(prop_s, rel_s, counts["WSP-Det"], with_compare=False)
        fl_cmp = det_flop_per_puzzle(prop_s, rel_s, counts["WSP-Det"], with_compare=True)
        det_rows.append(
            {
                "prop_steps": int(prop_s),
                "relax_steps": int(rel_s),
                "total_steps": int(prop_s + rel_s),
                "n": int(len(sol)),
                "solved": int(exact_np.sum()),
                "solved_pct": float(100.0 * exact_np.mean()),
                "halted": int(stopped_np.sum()),
                "halted_pct": float(100.0 * stopped_np.mean()),
                "wrong_stops": wrong_stops,
                "wrong_halts_pct_of_halted": (
                    float(100.0 * wrong_stops / int(stopped_np.sum())) if int(stopped_np.sum()) > 0 else 0.0
                ),
                "independent_rule_valid_count": int(v_all["n_valid"]),
                "clue_violations": int(v_all["n_clue_violations"]),
                "flop_per_puzzle_total": fl_tot,
                "flop_per_puzzle_with_compare": fl_cmp,
                "wall_s": float(wall_s),
            }
        )
    return {"WSP-Transformer": tf_rows, "WSP-Det": det_rows}


def run_dev_smoke(device: str = "cuda") -> dict[str, Any]:
    """Run strict pre-launch dev10k evaluation (`0..9999` ONLY; never opens `full-all.npz`)."""
    audit_path = RESULTS_DIR / "wsp-sudoku-opcounts-and-params.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else run_audit_counts(device=device)
    counts = audit["counted_flops"]

    sol, free, ratings, idx = load_sudoku_split("dev10k", 0, 10000)
    assert len(sol) == 10000 and int(idx[0]) == 0 and int(idx[-1]) == 9999
    x_tok = encode_sudoku_clues(sol, free)

    det_sweeps = evaluate_det_sweep(x_tok, sol, counts, device=device)

    # Run WSP-Assoc on seeds 0, 1, 2 on dev10k (cap = 16,777,216 moves on dev to verify fast 100% convergence;
    # note T_cap for 4e11 FLOPs is ~961M moves, and dev10k finishes 100% by <3M moves!)
    t_cap_paper = assoc_t_cap(counts["WSP-Assoc"], FROZEN_ASSOC["cap_flops"])
    w_assoc = build_wsp_assoc_weights(
        prop_steps=FROZEN_ASSOC["prop_steps"],
        temp_hi=FROZEN_ASSOC["temp_hi"],
        temp_lo=FROZEN_ASSOC["temp_lo"],
        anneal_period=FROZEN_ASSOC["anneal_period"],
        theta=FROZEN_ASSOC["theta"],
        tau_tol=FROZEN_ASSOC["tau_tol"],
        device=device,
    )

    assoc_seeds: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        # Cap at min(t_cap_paper, 16_777_216) for dev smoke (since 100% solve in <3M moves)
        ss = WSPAssocSearch(
            w_assoc,
            x_tok,
            C=FROZEN_ASSOC["C"],
            check=FROZEN_ASSOC["check"],
            max_moves=min(t_cap_paper, 16_777_216),
            seed=seed,
            device=device,
        )
        res = ss.run()
        pred_np = res["pred"].numpy()
        stopped_np = res["stopped"].numpy()
        exact_np = (pred_np == sol).all(axis=1)
        ver = verify_sudoku_boards(pred_np, x_tok)
        wrong_stops = int((stopped_np & ~exact_np).sum())
        assert wrong_stops == 0, f"Dev seed {seed} produced {wrong_stops} wrong stops!"
        assert ver["n_clue_violations"] == 0

        fl_tot = compute_wsp_assoc_puzzle_flops(
            res["steps_used"],
            res["stage2_entered"],
            counts["WSP-Assoc"],
            prop_steps_used=res["prop_steps_used"],
            with_compare=False,
        )
        fl_cmp = compute_wsp_assoc_puzzle_flops(
            res["steps_used"],
            res["stage2_entered"],
            counts["WSP-Assoc"],
            prop_steps_used=res["prop_steps_used"],
            with_compare=True,
        )

        seed_rec = {
            "seed": seed,
            "split": "dev10k",
            "n": int(len(sol)),
            "config": FROZEN_ASSOC,
            "T_cap_paper": int(t_cap_paper),
            "solved": int(exact_np.sum()),
            "solved_pct": float(100.0 * exact_np.mean()),
            "stopped": int(stopped_np.sum()),
            "halted_pct": float(100.0 * stopped_np.mean()),
            "stage1_solved": int(res["stage1_solved"].sum().item()),
            "wrong_stops": wrong_stops,
            "budget_failures": int((~stopped_np).sum()),
            "unstopped_but_right": int((~stopped_np & exact_np).sum()),
            "independent_rule_valid": int(ver["n_valid"]),
            "clue_violations": int(ver["n_clue_violations"]),
            "moves": {
                "mean": float(res["steps_used"].double().mean().item()),
                "median": float(np.quantile(res["steps_used"].numpy(), 0.5)),
                "p99": float(np.quantile(res["steps_used"].numpy(), 0.99)),
                "max": int(res["steps_used"].max().item()),
            },
            "prop_steps_used_mean": float(res["prop_steps_used"].double().mean().item()),
            "fc2_flops_total": flop_dist_stats(fl_tot),
            "fc2_flops_with_compare": flop_dist_stats(fl_cmp),
            "wall_s": float(res["seconds"]),
            "gpu": torch.cuda.get_device_name(torch.device(device)) if torch.cuda.is_available() else "cpu",
        }
        dev_out = ARTIFACTS_DIR / "dev" / f"wsp-assoc-dev-s{seed}.json"
        dev_sha = atomic_write_json(seed_rec, dev_out)
        seed_rec["dev_record_path"] = str(dev_out.relative_to(ROOT))
        seed_rec["dev_record_sha256"] = dev_sha
        assoc_seeds.append(seed_rec)
        print(
            f"[dev s{seed}] solved={seed_rec['solved']}/10000 ({seed_rec['solved_pct']:.2f}%), "
            f"wrong={wrong_stops}, mean_FLOPs={seed_rec['fc2_flops_total']['mean']/1e6:.2f}M, "
            f"wall={seed_rec['wall_s']:.2f}s",
            flush=True,
        )

    mean_dev_wall = float(statistics.mean(r["wall_s"] for r in assoc_seeds))
    proj_412k_per_seed_s = mean_dev_wall * (412786.0 / 10000.0)
    summary = {
        "utc": utc_now(),
        "split": "dev10k (upstream/data/full-test/full-dev10k.npz, rows 0..9999)",
        "test_splits_touched": False,
        "WSP-Assoc_3seed": {
            "config": FROZEN_ASSOC,
            "per_seed": assoc_seeds,
            "solved_pct_mean": float(statistics.mean(r["solved_pct"] for r in assoc_seeds)),
            "wrong_stops_total": int(sum(r["wrong_stops"] for r in assoc_seeds)),
            "fc2_mean_flops": float(statistics.mean(r["fc2_flops_total"]["mean"] for r in assoc_seeds)),
            "fc2_with_compare_mean_flops": float(
                statistics.mean(r["fc2_flops_with_compare"]["mean"] for r in assoc_seeds)
            ),
            "mean_wall_s_per_10k_seed": mean_dev_wall,
            "projected_412786_wall_s_per_seed": proj_412k_per_seed_s,
            "projected_412786_3seeds_total_min": (3.0 * proj_412k_per_seed_s) / 60.0,
        },
        "WSP-Det_sweep": det_sweeps["WSP-Det"],
        "WSP-Transformer_sweep": det_sweeps["WSP-Transformer"],
        "opcounts_record": "results/wsp/sudoku/wsp-sudoku-opcounts-and-params.json",
        "opcounts_sha256": sha256_file(RESULTS_DIR / "wsp-sudoku-opcounts-and-params.json"),
    }
    out_summary = RESULTS_DIR / "dev10k-smoke-summary.json"
    sha = atomic_write_json(summary, out_summary)
    print(f"[dev] wrote {out_summary} (sha256={sha[:16]}...)", flush=True)
    return summary


def enforce_dev_and_verifier_gate(verifier_approved: bool) -> dict[int, str]:
    """Refuse to open `full-all.npz` unless `--verifier-approved` is set AND all 3 dev seeds passed."""
    if not verifier_approved:
        raise RuntimeError(
            "TEST GATE BLOCKED: Do not run test splits without `--verifier-approved` after @verifier sign-off!"
        )
    dev_shas: dict[int, str] = {}
    for seed in (0, 1, 2):
        p = ARTIFACTS_DIR / "dev" / f"wsp-assoc-dev-s{seed}.json"
        if not p.exists():
            raise RuntimeError(f"TEST GATE BLOCKED: Missing dev record {p}")
        rec = json.loads(p.read_text())
        if (
            rec.get("n") != 10000
            or rec.get("solved", 0) < 9999
            or rec.get("wrong_stops", 1) != 0
            or rec.get("clue_violations", 1) != 0
            or rec.get("config") != FROZEN_ASSOC
        ):
            raise RuntimeError(f"TEST GATE BLOCKED: Dev record {p} failed acceptance criteria: {rec}")
        dev_shas[seed] = sha256_file(p)
    return dev_shas


def build_fig3a_bins(
    rat_first10k: np.ndarray,
    solved_10k: np.ndarray,
    fl_10k: np.ndarray,
    det_10k_exact: np.ndarray,
    det_fl_tot: float,
    tf_10k_exact: np.ndarray,
    tf_fl_tot: float,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute Figure 3a 8-bin difficulty stats on `first10k` (`(s, 10000)` arrays)."""
    bin_counts = [
        int(((rat_first10k >= lo) & (rat_first10k < hi)).sum())
        for lo, hi in zip(FIG3A_BINS[:-1], FIG3A_BINS[1:])
    ]
    assoc_bins: list[dict[str, Any]] = []
    det_bins: list[dict[str, Any]] = []
    tf_bins: list[dict[str, Any]] = []
    n_seeds = int(solved_10k.shape[0])
    for lo_b, hi_b in zip(FIG3A_BINS[:-1], FIG3A_BINS[1:]):
        mask = (rat_first10k >= lo_b) & (rat_first10k < hi_b)
        fl_bin = fl_10k[:, mask].reshape(-1)
        assoc_bins.append(
            {
                "solved_pct": [float(100.0 * solved_10k[s, mask].mean()) for s in range(n_seeds)] if mask.any() else None,
                "flop_median": float(np.median(fl_bin)) if mask.any() else None,
                "flop_p10": float(np.percentile(fl_bin, 10)) if mask.any() else None,
                "flop_p90": float(np.percentile(fl_bin, 90)) if mask.any() else None,
                "flop_mean": float(fl_bin.mean()) if mask.any() else None,
            }
        )
        det_bins.append(
            {
                "solved_pct": [float(100.0 * det_10k_exact[mask].mean())] if mask.any() else None,
                "flop_median": det_fl_tot if mask.any() else None,
                "flop_p10": det_fl_tot if mask.any() else None,
                "flop_p90": det_fl_tot if mask.any() else None,
                "flop_mean": det_fl_tot if mask.any() else None,
            }
        )
        tf_bins.append(
            {
                "solved_pct": [float(100.0 * tf_10k_exact[mask].mean())] if mask.any() else None,
                "flop_median": tf_fl_tot if mask.any() else None,
                "flop_p10": tf_fl_tot if mask.any() else None,
                "flop_p90": tf_fl_tot if mask.any() else None,
                "flop_mean": tf_fl_tot if mask.any() else None,
            }
        )
    return bin_counts, assoc_bins, det_bins, tf_bins


def build_anytime_curve(
    solved: np.ndarray,
    flop: np.ndarray,
    record_caps: np.ndarray,
    caps_grid: np.ndarray,
) -> list[dict[str, Any]]:
    """Exact anytime compute curve matching `upstream/experiments/compute_curve_export.py:curve`.

    Below `record_caps`, unstopped puzzles count strictly as unsolved (`flop <= b`).
    """
    rows: list[dict[str, Any]] = []
    for b in caps_grid:
        rate = (solved & ((flop <= b) | (b >= record_caps[:, None]))).mean(axis=1)
        used = np.minimum(flop, b)
        rows.append(
            {
                "cap": float(b),
                "solved_per_seed": rate.tolist(),
                "solved_mean": float(rate.mean()),
                "solved_sd": float(rate.std(ddof=1)) if len(rate) > 1 else 0.0,
                "mean_flop_per_seed": used.mean(axis=1).tolist(),
                "mean_flop": float(used.mean()),
                "p99_flop_per_seed": np.percentile(used, 99, axis=1).tolist(),
                "p99_flop": float(np.percentile(used, 99, axis=1).mean()),
            }
        )
    return rows


def run_test_all(device: str = "cuda", *, verifier_approved: bool = False) -> None:
    dev_shas = enforce_dev_and_verifier_gate(verifier_approved)
    audit_path = RESULTS_DIR / "wsp-sudoku-opcounts-and-params.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else run_audit_counts(device=device)
    counts = audit["counted_flops"]
    dev_t = torch.device(device)
    gpu_name = torch.cuda.get_device_name(dev_t) if dev_t.type == "cuda" else "cpu"

    # =========================================================================
    # PART A: Full 412,786 Unseen Test Split (`full-all[10000:422786]`)
    # =========================================================================
    print("[test-all] Loading full-all[10000:422786] (412,786 test puzzles)...", flush=True)
    sol_full, free_full, ratings_full, idx_full = load_sudoku_split("full-all", 10000, 412786)
    assert len(sol_full) == 412786 and int(idx_full[0]) == 10000 and int(idx_full[-1]) == 422785
    x_full = encode_sudoku_clues(sol_full, free_full)

    t_cap = assoc_t_cap(counts["WSP-Assoc"], FROZEN_ASSOC["cap_flops"])
    t_end = -(-t_cap // FROZEN_ASSOC["check"]) * FROZEN_ASSOC["check"]
    w_assoc = build_wsp_assoc_weights(
        prop_steps=FROZEN_ASSOC["prop_steps"],
        temp_hi=FROZEN_ASSOC["temp_hi"],
        temp_lo=FROZEN_ASSOC["temp_lo"],
        anneal_period=FROZEN_ASSOC["anneal_period"],
        theta=FROZEN_ASSOC["theta"],
        tau_tol=FROZEN_ASSOC["tau_tol"],
        device=device,
    )

    assoc_per_seed_summaries: list[dict[str, Any]] = []
    solved_stack: list[np.ndarray] = []
    stopped_stack: list[np.ndarray] = []
    fl_tot_stack: list[np.ndarray] = []
    fl_cmp_stack: list[np.ndarray] = []
    record_caps_list: list[float] = []
    npz_records: dict[str, str] = {}

    for seed in (0, 1, 2):
        npz_path = ARTIFACTS_DIR / "test" / f"wsp-assoc-test412k-s{seed}.npz"
        meta_path = ARTIFACTS_DIR / "test" / f"wsp-assoc-test412k-s{seed}-meta.json"
        if npz_path.exists() and meta_path.exists():
            print(f"[test-all] Reusing completed {npz_path}", flush=True)
            z = np.load(npz_path)
            s_meta = json.loads(meta_path.read_text())
            exact_np = z["solved"].astype(bool)
            stopped_np = z["stopped"].astype(bool)
            steps_np = z["steps_used"].astype(np.int64)
            prop_np = z["prop_steps_used"].astype(np.int64)
            s2_np = z["stage2_entered"].astype(bool)
            valid_np = z["valid_sudoku"].astype(bool)
            clue_viol = int(s_meta["clue_violations"])
            wall_s = float(s_meta["seconds"])
            npz_sha = sha256_file(npz_path)
        else:
            print(f"[test-all] Running WSP-Assoc on 412,786 test puzzles (seed={seed}, T_cap={t_cap})...", flush=True)
            ss = WSPAssocSearch(
                w_assoc,
                x_full,
                C=FROZEN_ASSOC["C"],
                check=FROZEN_ASSOC["check"],
                max_moves=t_end,
                seed=seed,
                device=device,
            )
            res = ss.run()
            pred_np = res["pred"].numpy()
            stopped_np = res["stopped"].numpy()
            exact_np = (pred_np == sol_full).all(axis=1)
            steps_np = res["steps_used"].numpy().astype(np.int64)
            prop_np = res["prop_steps_used"].numpy().astype(np.int64)
            s2_np = res["stage2_entered"].numpy().astype(bool)
            ver = verify_sudoku_boards(pred_np, x_full)
            valid_np = ver["valid_mask"]
            clue_viol = int(ver["n_clue_violations"])
            wall_s = float(res["seconds"])
            assert int((stopped_np & ~exact_np).sum()) == 0, f"Wrong stop on test seed {seed}!"
            assert int((stopped_np & ~valid_np).sum()) == 0, f"Invalid Sudoku certificate on test seed {seed}!"
            assert clue_viol == 0

            npz_sha = atomic_save_npz(
                {
                    "puzzle": idx_full,
                    "solved": exact_np,
                    "stopped": stopped_np,
                    "steps_used": steps_np,
                    "prop_steps_used": prop_np,
                    "stage2_entered": s2_np,
                    "valid_sudoku": valid_np,
                },
                npz_path,
            )
            atomic_write_json({"seed": seed, "seconds": wall_s, "clue_violations": clue_viol, "npz_sha256": npz_sha}, meta_path)

        npz_records[str(npz_path.relative_to(ROOT))] = npz_sha
        fl_tot = compute_wsp_assoc_puzzle_flops(
            steps_np, s2_np, counts["WSP-Assoc"], prop_steps_used=prop_np, with_compare=False
        )
        fl_cmp = compute_wsp_assoc_puzzle_flops(
            steps_np, s2_np, counts["WSP-Assoc"], prop_steps_used=prop_np, with_compare=True
        )
        rec_cap = float(
            compute_wsp_assoc_puzzle_flops(
                np.array([t_end]),
                np.array([True]),
                counts["WSP-Assoc"],
                prop_steps_used=np.array([FROZEN_ASSOC["prop_steps"]]),
                with_compare=False,
            )[0]
        )
        record_caps_list.append(rec_cap)
        solved_stack.append(exact_np)
        stopped_stack.append(stopped_np)
        fl_tot_stack.append(fl_tot)
        fl_cmp_stack.append(fl_cmp)

        wrong_stops = int((stopped_np & ~exact_np).sum())
        assoc_per_seed_summaries.append(
            {
                "seed": seed,
                "record": str(npz_path.relative_to(ROOT)),
                "record_sha256": npz_sha,
                "dev_record_sha256": dev_shas[seed],
                "n": 412786,
                "solved": int(exact_np.sum()),
                "stopped": int(stopped_np.sum()),
                "stage1_solved": int((~s2_np).sum()),
                "wrong_stops": wrong_stops,
                "wrong_halts_pct_of_halted": (
                    float(100.0 * wrong_stops / int(stopped_np.sum())) if int(stopped_np.sum()) > 0 else 0.0
                ),
                "budget_failures": int((~stopped_np).sum()),
                "unstopped_but_right": int((~stopped_np & exact_np).sum()),
                "independent_rule_valid": int(valid_np.sum()),
                "clue_violations": clue_viol,
                "solved_percent": float(100.0 * exact_np.mean()),
                "halted_percent": float(100.0 * stopped_np.mean()),
                "paper_flops": flop_dist_stats(fl_tot),
                "paper_flops_with_compare": flop_dist_stats(fl_cmp),
                "move_count": {
                    "mean": float(steps_np.mean()),
                    "median": float(np.quantile(steps_np, 0.5)),
                    "p99": float(np.quantile(steps_np, 0.99)),
                    "max": int(steps_np.max()),
                    "T_cap": int(t_cap),
                    "rounded_endpoint": int(t_end),
                },
                "prop_steps_used_mean": float(prop_np.mean()),
                "paper_cap": rec_cap,
                "seconds": round(wall_s, 2),
                "gpu": gpu_name,
            }
        )
        print(
            f"[test-412k s{seed}] solved={int(exact_np.sum())}/412786 ({100.0*exact_np.mean():.4f}%), "
            f"wrong={wrong_stops}, mean_FLOPs={fl_tot.mean()/1e6:.2f}M, wall={wall_s:.2f}s",
            flush=True,
        )

    # Evaluate deterministic WSP-Det (16, 32) and WSP-Transformer (T=16) on full 412,786 test puzzles!
    print("[test-all] Running WSP-Det (16, 32) and WSP-Transformer (T=16) on full 412,786 test puzzles...", flush=True)
    peer_adj_dev = w_assoc["peer_adj"]
    if dev_t.type == "cuda":
        torch.cuda.synchronize(dev_t)
    t0_det = time.perf_counter()
    w_det_f = build_wsp_det_weights(FROZEN_DET["prop_steps"], FROZEN_DET["relax_steps"], device=device)
    res_det_full = evaluate_deterministic_wsp(wsp_det_forward, w_det_f, x_full, peer_adj_dev, batch_size=8192)
    if dev_t.type == "cuda":
        torch.cuda.synchronize(dev_t)
    wall_det_full = time.perf_counter() - t0_det

    det_pred_full = res_det_full["pred"].numpy()
    det_stop_full = res_det_full["stopped"].numpy()
    det_exact_full = (det_pred_full == sol_full).all(axis=1)
    det_ver_full = verify_sudoku_boards(det_pred_full, x_full)
    assert int((det_stop_full & ~det_exact_full).sum()) == 0
    det_fl_tot = det_flop_per_puzzle(FROZEN_DET["prop_steps"], FROZEN_DET["relax_steps"], counts["WSP-Det"], with_compare=False)
    det_fl_cmp = det_flop_per_puzzle(FROZEN_DET["prop_steps"], FROZEN_DET["relax_steps"], counts["WSP-Det"], with_compare=True)

    if dev_t.type == "cuda":
        torch.cuda.synchronize(dev_t)
    t0_tf = time.perf_counter()
    w_tf_f = build_wsp_transformer_weights(steps=FROZEN_TF["steps"], device=device)
    res_tf_full = evaluate_deterministic_wsp(wsp_transformer_forward, w_tf_f, x_full, peer_adj_dev, batch_size=4096)
    if dev_t.type == "cuda":
        torch.cuda.synchronize(dev_t)
    wall_tf_full = time.perf_counter() - t0_tf

    tf_pred_full = res_tf_full["pred"].numpy()
    tf_stop_full = res_tf_full["stopped"].numpy()
    tf_exact_full = (tf_pred_full == sol_full).all(axis=1)
    tf_ver_full = verify_sudoku_boards(tf_pred_full, x_full)
    assert int((tf_stop_full & ~tf_exact_full).sum()) == 0
    tf_fl_tot = tf_flop_per_puzzle(FROZEN_TF["steps"], counts["WSP-Transformer"], with_compare=False)
    tf_fl_cmp = tf_flop_per_puzzle(FROZEN_TF["steps"], counts["WSP-Transformer"], with_compare=True)

    solved_arr = np.stack(solved_stack, axis=0)  # (3, 412786)
    stopped_arr = np.stack(stopped_stack, axis=0)
    fl_tot_arr = np.stack(fl_tot_stack, axis=0)
    fl_cmp_arr = np.stack(fl_cmp_stack, axis=0)

    pooled_tot = fl_tot_arr.reshape(-1)
    pooled_cmp = fl_cmp_arr.reshape(-1)
    seed_means_tot = [float(x.mean()) for x in fl_tot_stack]
    seed_means_cmp = [float(x.mean()) for x in fl_cmp_stack]
    seed_p99s_tot = [float(np.quantile(x, 0.99)) for x in fl_tot_stack]
    seed_solved_pct = [float(100.0 * x.mean()) for x in solved_stack]
    seed_halted_pct = [float(100.0 * x.mean()) for x in stopped_stack]

    full_test_record = {
        "utc": utc_now(),
        "dataset": "upstream/data/full-test/full-all.npz",
        "dataset_sha256": sha256_file(ROOT / "upstream/data/full-test/full-all.npz"),
        "rows": [10000, 422786],
        "unique_test_puzzles": 412786,
        "op_count_convention": "FC2 (baselines/flops/fullcount.py FullCounter)",
        "models": {
            "WSP-Assoc": {
                "method": "WSP-Assoc (zero-training Weight-Space Program: Stage 1 deductive tensor domain reduction + Stage 2 C=4 focused Gumbel-Potts)",
                "params_learned": 0,
                "training_examples": 0,
                "fit_time_display": "none",
                "weight_metadata": audit["weight_metadata"]["WSP-Assoc"],
                "build_timings": audit["build_timings"]["WSP-Assoc"],
                "config": FROZEN_ASSOC,
                "seeds": [0, 1, 2],
                "seed_puzzle_evaluations": int(solved_arr.size),
                "counts_pooled_over_seeds": {
                    "solved": int(solved_arr.sum()),
                    "stopped": int(stopped_arr.sum()),
                    "wrong_stops": int((stopped_arr & ~solved_arr).sum()),
                    "budget_failures": int((~stopped_arr).sum()),
                    "unstopped_but_right": int((~stopped_arr & solved_arr).sum()),
                },
                "solved_percent": {
                    "mean": float(statistics.mean(seed_solved_pct)),
                    "sd": float(statistics.stdev(seed_solved_pct)),
                    "per_seed": seed_solved_pct,
                },
                "halted_percent": {
                    "mean": float(statistics.mean(seed_halted_pct)),
                    "sd": float(statistics.stdev(seed_halted_pct)),
                    "per_seed": seed_halted_pct,
                },
                "wrong_halts_pct_of_halted": 0.0,
                "paper_mean_flops": {
                    "mean": float(statistics.mean(seed_means_tot)),
                    "sd": float(statistics.stdev(seed_means_tot)),
                    "per_seed": seed_means_tot,
                },
                "paper_mean_flops_with_compare": {
                    "mean": float(statistics.mean(seed_means_cmp)),
                    "sd": float(statistics.stdev(seed_means_cmp)),
                    "per_seed": seed_means_cmp,
                },
                "paper_p99_flops_mean_of_seed_quantiles": {
                    "mean": float(statistics.mean(seed_p99s_tot)),
                    "sd": float(statistics.stdev(seed_p99s_tot)),
                    "per_seed": seed_p99s_tot,
                },
                "paper_pooled_distribution": flop_dist_stats(pooled_tot),
                "paper_pooled_distribution_with_compare": flop_dist_stats(pooled_cmp),
                "per_seed": assoc_per_seed_summaries,
                "source_records": npz_records,
            },
            "WSP-Det": {
                "method": "WSP-Det (zero-training deterministic task-native Weight-Space Program)",
                "params_learned": 0,
                "training_examples": 0,
                "fit_time_display": "none",
                "weight_metadata": audit["weight_metadata"]["WSP-Det"],
                "build_timings": audit["build_timings"]["WSP-Det"],
                "config": FROZEN_DET,
                "deterministic": True,
                "n": 412786,
                "solved": int(det_exact_full.sum()),
                "solved_percent": float(100.0 * det_exact_full.mean()),
                "halted": int(det_stop_full.sum()),
                "halted_percent": float(100.0 * det_stop_full.mean()),
                "wrong_stops": int((det_stop_full & ~det_exact_full).sum()),
                "wrong_halts_pct_of_halted": 0.0,
                "independent_rule_valid": int(det_ver_full["n_valid"]),
                "clue_violations": int(det_ver_full["n_clue_violations"]),
                "paper_mean_flops": det_fl_tot,
                "paper_mean_flops_with_compare": det_fl_cmp,
                "seconds": round(wall_det_full, 2),
                "gpu": gpu_name,
            },
            "WSP-Transformer": {
                "method": "WSP-Transformer (zero-training 2-layer looped standard Transformer programmed in weight space)",
                "params_learned": 0,
                "training_examples": 0,
                "fit_time_display": "none",
                "weight_metadata": audit["weight_metadata"]["WSP-Transformer"],
                "build_timings": audit["build_timings"]["WSP-Transformer"],
                "config": FROZEN_TF,
                "deterministic": True,
                "n": 412786,
                "solved": int(tf_exact_full.sum()),
                "solved_percent": float(100.0 * tf_exact_full.mean()),
                "halted": int(tf_stop_full.sum()),
                "halted_percent": float(100.0 * tf_stop_full.mean()),
                "wrong_stops": int((tf_stop_full & ~tf_exact_full).sum()),
                "wrong_halts_pct_of_halted": 0.0,
                "independent_rule_valid": int(tf_ver_full["n_valid"]),
                "clue_violations": int(tf_ver_full["n_clue_violations"]),
                "paper_mean_flops": tf_fl_tot,
                "paper_mean_flops_with_compare": tf_fl_cmp,
                "seconds": round(wall_tf_full, 2),
                "gpu": gpu_name,
            },
        },
        "opcounts_record": "results/wsp/sudoku/wsp-sudoku-opcounts-and-params.json",
        "opcounts_sha256": sha256_file(audit_path),
    }
    full_test_out = RESULTS_DIR / "wsp-sudoku-full-test-412786.json"
    full_test_sha = atomic_write_json(full_test_record, full_test_out)
    print(f"[test-412k] wrote {full_test_out} (sha256={full_test_sha[:16]}...)", flush=True)

    # =========================================================================
    # PART B: `full-all[10000:20000]` (Figure 2a Anytime, Figure 3a Difficulty, Figure 3b k=0)
    # =========================================================================
    print("[test-all] Building Figure 2a, 3a, and 3b exports on full-all[10000:20000]...", flush=True)
    x_first10k = x_full[:10000]
    sol_first10k = sol_full[:10000]
    rat_first10k = ratings_full[:10000]
    det_sweeps_10k = evaluate_det_sweep(x_first10k, sol_first10k, counts, device=device)

    rec_caps_np = np.asarray(record_caps_list, dtype=np.float64)
    caps_grid = np.unique(np.concatenate([FIG2A_GRID[FIG2A_GRID < rec_caps_np.min()], [rec_caps_np.max()]]))
    anytime_sets = {
        "unseen-412786": build_anytime_curve(solved_arr, fl_tot_arr, rec_caps_np, caps_grid),
        "first10k": build_anytime_curve(solved_arr[:, :10000], fl_tot_arr[:, :10000], rec_caps_np, caps_grid),
    }

    # Compare WSP-Assoc against published baselines on `first10k` (same rule as scripts/render_sudoku_compute.py)
    baseline_path = ROOT / "upstream/results/compute-sweep/sudoku-extreme.json"
    baseline_data = json.loads(baseline_path.read_text())
    methods = sorted({r["method"] for r in baseline_data["summary"]})
    wsp_vs_baselines: list[dict[str, Any]] = []
    for method in methods:
        settings = [r for r in baseline_data["summary"] if r["method"] == method and r["seeds"] == [0, 1, 2]]
        top_acc = max(r["solve_rate_mean"] for r in settings)
        top = min((r for r in settings if r["solve_rate_mean"] == top_acc), key=lambda r: r["mean_flop"])
        elig = [r for r in anytime_sets["first10k"] if r["solved_mean"] >= top_acc]
        match = min(elig, key=lambda r: r["mean_flop"]) if elig else None
        wsp_vs_baselines.append(
            {
                "method": method,
                "paper_main_figure_comparison": method != "TRMP-fast",
                "baseline_best_setting": top,
                "wsp_assoc_matching_point": match,
                "baseline_mean_flops_over_wsp_assoc": (top["mean_flop"] / match["mean_flop"]) if match else None,
            }
        )

    anytime_out = RESULTS_DIR / "wsp-sudoku-anytime-curves.json"
    atomic_write_json(
        {
            "utc": utc_now(),
            "figure": "2a",
            "op_counts": "FC2: baselines/flops/fullcount.py FullCounter",
            "record_caps": rec_caps_np.tolist(),
            "n_puzzles": {"unseen-412786": 412786, "first10k": 10000},
            "strict_cap_rule": "Below record_cap, puzzles with flop > b count strictly as unsolved (solved & (flop <= b)).",
            "WSP-Assoc_sets": anytime_sets,
            "WSP-Det_first10k_sweep": det_sweeps_10k["WSP-Det"],
            "WSP-Transformer_first10k_sweep": det_sweeps_10k["WSP-Transformer"],
            "wsp_assoc_vs_published_baselines": wsp_vs_baselines,
            "source_full_test_json": str(full_test_out.relative_to(ROOT)),
            "source_full_test_sha256": full_test_sha,
        },
        anytime_out,
    )

    # Figure 3a: 8-bin difficulty curve on full-all[10000:20000]
    det_10k_exact = det_exact_full[:10000]
    tf_10k_exact = tf_exact_full[:10000]
    bin_counts, assoc_bins, det_bins, tf_bins = build_fig3a_bins(
        rat_first10k,
        solved_arr[:, :10000],
        fl_tot_arr[:, :10000],
        det_10k_exact,
        det_fl_tot,
        tf_10k_exact,
        tf_fl_tot,
    )
    assert sum(bin_counts) == 10000

    fig3a_assoc = {
        "status": "fresh zero-training WSP evaluation",
        "seeds": 3,
        "solved_pct": [float(100.0 * solved_arr[s, :10000].mean()) for s in range(3)],
        "mean_flop": [float(fl_tot_arr[s, :10000].mean()) for s in range(3)],
        "wrong_stops": [int((stopped_arr[s, :10000] & ~solved_arr[s, :10000]).sum()) for s in range(3)],
        "budget_failures": [int((~stopped_arr[s, :10000]).sum()) for s in range(3)],
        "bins": assoc_bins,
        "solved_within_budget_pct": [
            [float(100.0 * (solved_arr[s, :10000] & (fl_tot_arr[s, :10000] <= b)).mean()) for b in FIG3A_BUDGETS]
            for s in range(3)
        ],
        "hardest_to_easiest_median_compute": float(assoc_bins[-1]["flop_median"] / assoc_bins[0]["flop_median"]),
    }
    fig3a_out = RESULTS_DIR / "wsp-sudoku-figure3a-difficulty.json"
    atomic_write_json(
        {
            "utc": utc_now(),
            "figure": "3a",
            "subset": {"full_all_rows": [10000, 20000], "n": 10000, "rating_mean": float(rat_first10k.mean())},
            "bins": list(zip(FIG3A_BINS[:-1].tolist(), FIG3A_BINS[1:].tolist())),
            "bin_counts": bin_counts,
            "budgets": FIG3A_BUDGETS.tolist(),
            "models": {
                "WSP-Assoc": fig3a_assoc,
                "WSP-Det": {
                    "solved_pct": float(100.0 * det_10k_exact.mean()),
                    "flop_per_puzzle": det_fl_tot,
                    "wrong_stops": 0,
                    "bins": det_bins,
                },
                "WSP-Transformer": {
                    "solved_pct": float(100.0 * tf_10k_exact.mean()),
                    "flop_per_puzzle": tf_fl_tot,
                    "wrong_stops": 0,
                    "bins": tf_bins,
                },
            },
            "source_full_test_sha256": full_test_sha,
        },
        fig3a_out,
    )

    # Figure 3b: Full 6-seed (`0..5`) zero-training (`k=0`) reference run directly on `x_first10k`
    # so all 6 seeds share identical batch compaction dynamics on `full-all[10000:20000]`.
    fig3b_seed_rows: list[dict[str, Any]] = []
    for s in range(6):
        ss_extra = WSPAssocSearch(
            w_assoc,
            x_first10k,
            C=FROZEN_ASSOC["C"],
            check=FROZEN_ASSOC["check"],
            max_moves=t_end,
            seed=s,
            device=device,
        )
        r_ex = ss_extra.run()
        p_ex = r_ex["pred"].numpy()
        st_ex = r_ex["stopped"].numpy()
        ex_ok = (p_ex == sol_first10k).all(axis=1)
        v_ex = verify_sudoku_boards(p_ex, x_first10k)
        assert int((st_ex & ~ex_ok).sum()) == 0 and v_ex["n_clue_violations"] == 0
        fl_ex = compute_wsp_assoc_puzzle_flops(
            r_ex["steps_used"], r_ex["stage2_entered"], counts["WSP-Assoc"], prop_steps_used=r_ex["prop_steps_used"]
        )
        fig3b_seed_rows.append(
            {
                "seed": s,
                "n": 10000,
                "solved": int(ex_ok.sum()),
                "accuracy_pct": float(100.0 * ex_ok.mean()),
                "wrong_stops": 0,
                "mean_flop": float(fl_ex.mean()),
                "wall_s": float(r_ex["seconds"]),
            }
        )
    accs_6 = [r["accuracy_pct"] for r in fig3b_seed_rows]
    fig3b_out = RESULTS_DIR / "wsp-sudoku-figure3b-zero-data.json"
    atomic_write_json(
        {
            "utc": utc_now(),
            "figure": "3b",
            "subset": {"full_all_rows": [10000, 20000], "n": 10000},
            "k_training_puzzles": 0,
            "models": {
                "WSP-Assoc": {
                    "k": 0,
                    "n_measured": 6,
                    "seeds": [0, 1, 2, 3, 4, 5],
                    "accuracy_pct_median": float(statistics.median(accs_6)),
                    "accuracy_pct_mean": float(statistics.mean(accs_6)),
                    "accuracy_pct_min": float(min(accs_6)),
                    "accuracy_pct_max": float(max(accs_6)),
                    "wrong_stops_total": 0,
                    "per_seed": fig3b_seed_rows,
                },
                "WSP-Det": {
                    "k": 0,
                    "deterministic": True,
                    "accuracy_pct": float(100.0 * det_10k_exact.mean()),
                    "solved": int(det_10k_exact.sum()),
                    "wrong_stops": 0,
                    "mean_flop": det_fl_tot,
                },
                "WSP-Transformer": {
                    "k": 0,
                    "deterministic": True,
                    "accuracy_pct": float(100.0 * tf_10k_exact.mean()),
                    "solved": int(tf_10k_exact.sum()),
                    "wrong_stops": 0,
                    "mean_flop": tf_fl_tot,
                },
            },
        },
        fig3b_out,
    )

    # =========================================================================
    # PART C: `full-all[30000:40000]` (README Sec 1 Count-Energy 6-Seed Budget Curve)
    # =========================================================================
    print("[test-all] Running 6-seed budget curve on full-all[30000:40000] (up to 16,777,216 moves)...", flush=True)
    x_30_40 = x_full[20000:30000]  # since x_full starts at 10000, index 20000:30000 == full-all[30000:40000]!
    sol_30_40 = sol_full[20000:30000]
    idx_30_40 = idx_full[20000:30000]
    assert int(idx_30_40[0]) == 30000 and int(idx_30_40[-1]) == 39999 and len(sol_30_40) == 10000

    max_budget_cap = max(COUNT_BUDGET_CAPS)  # 16,777,216
    cb_rows: list[dict[str, Any]] = []
    cb_solved_by_cap: dict[int, list[np.ndarray]] = {c: [] for c in COUNT_BUDGET_CAPS}
    cb_flop_matched_by_cap: dict[int, list[np.ndarray]] = {c: [] for c in COUNT_BUDGET_CAPS}

    # Count-Energy's FC2 FLOP budget at cap `c` moves (from cram-hand-sudoku.json: 142811 + c*4*124 + ceil(c/4096)*2)
    count_energy_flop_caps = {c: 142811.0 + c * 4.0 * 124.0 + math.ceil(c / 4096.0) * 2.0 for c in COUNT_BUDGET_CAPS}

    for seed in range(6):
        ss_cb = WSPAssocSearch(
            w_assoc,
            x_30_40,
            C=FROZEN_ASSOC["C"],
            check=FROZEN_ASSOC["check"],
            max_moves=max_budget_cap,
            seed=seed,
            device=device,
        )
        r_cb = ss_cb.run()
        p_cb = r_cb["pred"].numpy()
        st_cb = r_cb["stopped"].numpy()
        ex_cb = (p_cb == sol_30_40).all(axis=1)
        steps_cb = r_cb["steps_used"].numpy().astype(np.int64)
        prop_cb = r_cb["prop_steps_used"].numpy().astype(np.int64)
        s2_cb = r_cb["stage2_entered"].numpy().astype(bool)
        v_cb = verify_sudoku_boards(p_cb, x_30_40)
        assert int((st_cb & ~ex_cb).sum()) == 0 and v_cb["n_clue_violations"] == 0

        fl_cb = compute_wsp_assoc_puzzle_flops(
            steps_cb, s2_cb, counts["WSP-Assoc"], prop_steps_used=prop_cb, with_compare=False
        )
        for cap_m in COUNT_BUDGET_CAPS:
            # Strict stopping rule: halted on a valid solution within `cap_m` moves!
            s_mask = st_cb & ex_cb & (steps_cb <= cap_m)
            used_m = np.minimum(steps_cb, cap_m)
            fl_cap_val = count_energy_flop_caps[cap_m]
            s_flop_mask = st_cb & ex_cb & (fl_cb <= fl_cap_val)
            cb_solved_by_cap[cap_m].append(s_mask)
            cb_flop_matched_by_cap[cap_m].append(s_flop_mask)
            cb_rows.append(
                {
                    "seed": seed,
                    "cap": int(cap_m),
                    "n": 10000,
                    "solved": int(s_mask.sum()),
                    "accuracy_pct": float(100.0 * s_mask.mean()),
                    "stopped": int((st_cb & (steps_cb <= cap_m)).sum()),
                    "wrong_stops": 0,
                    "mean_moves": float(used_m.mean()),
                    "flop_matched_cap_fc2": fl_cap_val,
                    "flop_matched_solved": int(s_flop_mask.sum()),
                    "flop_matched_accuracy_pct": float(100.0 * s_flop_mask.mean()),
                    "full_trajectory_wall_s": float(r_cb["seconds"]),
                }
            )
        print(
            f"[count-budget 30k..40k s{seed}] @262k={100.0*cb_solved_by_cap[262144][-1].mean():.2f}% "
            f"(FLOP-matched={100.0*cb_flop_matched_by_cap[262144][-1].mean():.2f}%), "
            f"@16.7M={100.0*cb_solved_by_cap[16777216][-1].mean():.2f}%, wall={r_cb['seconds']:.2f}s",
            flush=True,
        )

    cb_grouped: list[dict[str, Any]] = []
    for cap_m in COUNT_BUDGET_CAPS:
        s_mats = np.stack(cb_solved_by_cap[cap_m], axis=0)  # (6, 10000)
        f_mats = np.stack(cb_flop_matched_by_cap[cap_m], axis=0)
        per_seed_s = [int(x.sum()) for x in s_mats]
        per_seed_f = [int(x.sum()) for x in f_mats]
        joint_s = int(s_mats.all(axis=0).sum())
        joint_f = int(f_mats.all(axis=0).sum())
        cb_grouped.append(
            {
                "cap": int(cap_m),
                "unique_puzzles": 10000,
                "mean_accuracy_pct": float(100.0 * np.mean(per_seed_s) / 10000.0),
                "per_seed_solved": per_seed_s,
                "joint_solved": joint_s,
                "joint_accuracy_pct": float(100.0 * joint_s / 10000.0),
                "flop_matched_cap_fc2": count_energy_flop_caps[cap_m],
                "flop_matched_mean_accuracy_pct": float(100.0 * np.mean(per_seed_f) / 10000.0),
                "flop_matched_per_seed_solved": per_seed_f,
                "flop_matched_joint_solved": joint_f,
                "wrong_stops": 0,
            }
        )

    det_30_40 = det_exact_full[20000:30000]
    tf_30_40 = tf_exact_full[20000:30000]
    cb_out = RESULTS_DIR / "wsp-sudoku-count-budget-30k-40k.json"
    atomic_write_json(
        {
            "status": "validated",
            "utc": utc_now(),
            "subset": {"full_all_rows": [30000, 40000], "n": 10000},
            "seeds": [0, 1, 2, 3, 4, 5],
            "C": 4,
            "rows": cb_rows,
            "grouped": cb_grouped,
            "deterministic_controls_on_30k_40k": {
                "WSP-Det": {
                    "config": FROZEN_DET,
                    "solved": int(det_30_40.sum()),
                    "accuracy_pct": float(100.0 * det_30_40.mean()),
                    "wrong_stops": 0,
                    "flop_per_puzzle_fc2": det_fl_tot,
                },
                "WSP-Transformer": {
                    "config": FROZEN_TF,
                    "solved": int(tf_30_40.sum()),
                    "accuracy_pct": float(100.0 * tf_30_40.mean()),
                    "wrong_stops": 0,
                    "flop_per_puzzle_fc2": tf_fl_tot,
                },
            },
        },
        cb_out,
    )
    print(f"[test-all] All Sudoku WSP test benchmarks completed cleanly! Wrote {cb_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_aud = sub.add_parser("audit-counts", help="Run FullCounter (FC2), AST, metadata, and build-timing audit")
    p_aud.add_argument("--device", default="cuda")

    p_dev = sub.add_parser("dev", help="Run pre-launch smoke & profiling strictly on full-dev10k.npz")
    p_dev.add_argument("--device", default="cuda")

    p_test = sub.add_parser("test-all", help="Run full 412,786 + 10k test evaluations (requires --verifier-approved)")
    p_test.add_argument("--device", default="cuda")
    p_test.add_argument(
        "--verifier-approved",
        action="store_true",
        help="Confirm @verifier has signed off on code, tests, FullCounter, and dev10k smoke run",
    )

    args = parser.parse_args()
    if args.cmd == "audit-counts":
        run_audit_counts(device=args.device)
    elif args.cmd == "dev":
        run_dev_smoke(device=args.device)
    elif args.cmd == "test-all":
        run_test_all(device=args.device, verifier_approved=args.verifier_approved)


if __name__ == "__main__":
    main()
