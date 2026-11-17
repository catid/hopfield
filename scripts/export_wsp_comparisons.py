"""Consolidate certified Weight-Space Programming (WSP) results with frozen CRAM/published tables.

Strictly read-only on results/reproduction/*, artifacts/*, and results/research/*.
Verifies all 66 frozen reproduction export-manifest SHA-256 digests remain intact and writes
results/wsp/wsp-comparison-tables.json atomically with allow_nan=False.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/wsp/wsp-comparison-tables.json"


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def verify_frozen_export_manifest(sources: dict[str, str]) -> int:
    manifest_rel = "results/reproduction/export-manifest.json"
    manifest_path = ROOT / manifest_rel
    sources[manifest_rel] = sha256_path(manifest_path)
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(entries) == 66, f"Expected 66 frozen entries, got {len(entries)}"
    for item in entries:
        pub = ROOT / item["public"]
        src = ROOT / item["source"]
        assert sha256_path(pub) == item["sha256"], f"Frozen public hash changed: {item['public']}"
        if src.exists():
            assert sha256_path(src) == item["sha256"], f"Frozen source hash changed: {item['source']}"
    return len(entries)


def main() -> None:
    sources: dict[str, str] = {}

    def load_json(rel: str):
        p = ROOT / rel
        sources[rel] = sha256_path(p)
        return json.loads(p.read_text(encoding="utf-8"))

    frozen_checked = verify_frozen_export_manifest(sources)

    # Read-only frozen reproduction & research references
    main_comp = load_json("results/reproduction/main-comparison.json")
    halt_comp = load_json("results/reproduction/halting-comparison.json")
    nq_ref = load_json("results/reproduction/multi_solution/nqueens-paper-results.json")
    gc_ref = load_json("results/reproduction/multi_solution/graph-coloring-paper-results.json")
    sud_comp = load_json("results/reproduction/figures/sudoku-compute-comparison.json")
    sud_hand = load_json("results/reproduction/sudoku_audit/hand-full-test-audit.json")
    sud_learn = load_json("results/reproduction/sudoku_audit/learned-full-test-audit.json")
    fig3a_ref = load_json("results/reproduction/difficulty/figure3a.json")
    fig3b_ref = load_json("results/reproduction/data_efficiency/collection.json")
    count_bud = load_json("results/research/foureyes/count_budget/summary.json")
    learn_bud = load_json("results/research/foureyes/budget/summary.json")
    short_bud = load_json("results/research/ripper/finaltest.json")
    maze_3coh = load_json("results/research/work/endpoint-cost-three-cohorts/provenance.json")
    h3ad_ref = load_json("results/research/loki/h3ad/summary.json")

    # Certified WSP outputs
    wsp_sud_full = load_json("results/wsp/sudoku/wsp-sudoku-full-test-412786.json")
    wsp_sud_ops = load_json("results/wsp/sudoku/wsp-sudoku-opcounts-and-params.json")
    wsp_sud_any = load_json("results/wsp/sudoku/wsp-sudoku-anytime-curves.json")
    wsp_sud_3a = load_json("results/wsp/sudoku/wsp-sudoku-figure3a-difficulty.json")
    wsp_sud_3b = load_json("results/wsp/sudoku/wsp-sudoku-figure3b-zero-data.json")
    wsp_sud_cb = load_json("results/wsp/sudoku/wsp-sudoku-count-budget-30k-40k.json")

    wsp_maze_all = load_json("results/wsp/maze/wsp_maze_complete_summary.json")
    wsp_maze_u = load_json("results/wsp/maze/maze_unique_summary.json")
    wsp_maze_h = load_json("results/wsp/maze/maze_hard_summary.json")
    wsp_maze_180 = load_json("results/wsp/maze/maze_180x180_cohorts_summary.json")

    wsp_multi_sum = load_json("results/wsp/multi/summary.json")
    wsp_nq = load_json("results/wsp/multi/nqueens_results.json")
    wsp_gc = load_json("results/wsp/multi/graph_coloring_results.json")
    wsp_ql = load_json("results/wsp/multi/queen_leaper_results.json")

    load_json("results/wsp/figures/sudoku-compute-wsp.json")
    load_json("results/wsp/figures/figure3a-wsp.json")
    load_json("results/wsp/figures/figure3b-wsp.json")

    # 1. Main comparison rows (Tables 2 & 8 + WSP rows)
    wsp_main_rows = []
    for key, label in [
        ("WSP-Transformer", "WSP-Transformer (rule-programmed 2L Transformer)"),
        ("WSP-Det", "WSP-Det (rule-programmed deterministic)"),
        ("WSP-Assoc", "WSP-Assoc (rule-programmed associative + C=4 search)"),
    ]:
        m = wsp_sud_full["models"][key]
        wm = m["weight_metadata"]
        bt = wsp_sud_ops["build_timings"][key]
        if m.get("deterministic"):
            acc_disp = f"{m['solved_percent']:.6f} (deterministic)"
            fl_mean = m["paper_mean_flops"]
            fl_cmp = m["paper_mean_flops_with_compare"]
            seeds = [0]
        else:
            acc_disp = f"{m['solved_percent']['mean']:.6f} ± {m['solved_percent']['sd']:.6f}"
            fl_mean = m["paper_mean_flops"]["mean"]
            fl_cmp = m["paper_mean_flops_with_compare"]["mean"]
            seeds = m["seeds"]
        wsp_main_rows.append({
            "table": 2,
            "task": "Sudoku-Extreme",
            "model": label,
            "short_model": key,
            "provenance": "wsp_rule_programmed_local",
            "training_examples": 0,
            "params_learned": 0,
            "distinct_programmed_coefficients": wm["distinct_programmed_coefficients"],
            "materialized_entries": wm["materialized_entries"],
            "nnz": wm["nnz"],
            "params_display": f"0 learned ({wm['distinct_programmed_coefficients']} distinct; {wm['materialized_entries']:,} entries / {wm['nnz']:,} nnz)",
            "accuracy_display": acc_disp,
            "flops_fc2_mean": fl_mean,
            "flops_fc2_with_compare_mean": fl_cmp,
            "flops_display": f"{fl_mean:.8g}",
            "fit_display": "none",
            "build_cpu_1t_warm_ms": 1e3 * bt["cpu_1thread"]["warm_median_s"],
            "build_gpu_warm_ms": 1e3 * bt["gpu"]["warm_median_s"],
            "distinct_test_inputs": 412786,
            "seeds": seeds,
        })

    for task_name, tbl_num, summary_obj in [
        ("Maze-Unique", 2, wsp_maze_u),
        ("Maze-Hard", 8, wsp_maze_h),
    ]:
        hl = wsp_maze_all["headline"][task_name]
        for key in ("WSP-DeadEnd", "WSP-Wavefront", "WSP-Potential"):
            h = hl[key]
            wm = summary_obj["models"][key]["weight_metadata"]
            wsp_main_rows.append({
                "table": tbl_num,
                "task": task_name,
                "model": f"{key} (rule-programmed)",
                "short_model": key,
                "provenance": "wsp_rule_programmed_local",
                "training_examples": 0,
                "params_learned": 0,
                "distinct_programmed_coefficients": wm["distinct_nonzero_coefficients"],
                "materialized_entries": wm["materialized_entries"],
                "nnz": wm["nonzero_entries"],
                "params_display": f"0 learned ({wm['distinct_nonzero_coefficients']} distinct; {wm['materialized_entries']} entries / {wm['nonzero_entries']} nnz)",
                "accuracy_display": f"{h['solved_pct_mean']:.6f} ± {h['solved_pct_sd']:.6f}",
                "flops_fc2_mean": h["fc2_mean_total"],
                "flops_fc2_with_compare_mean": h["fc2_mean_total_with_compare"],
                "flops_display": f"{h['fc2_mean_total']:.8g}",
                "fit_display": "none" if tbl_num == 2 else None,
                "build_cpu_1t_warm_ms": 1e3 * h["weight_build_cpu_1thread_warm_median_s"],
                "build_gpu_warm_ms": 1e3 * h["weight_build_gpu_warm_median_s"],
                "inference_gpu_warm_ms": 1e3 * h["inference_gpu_warm_median_s"],
                "distinct_test_inputs": 1000,
                "seeds": [0, 1, 2],
            })

    # 2. Halting comparison rows (Table 4 + WSP)
    wsp_halting_rows = []
    for key in ("WSP-Transformer", "WSP-Det", "WSP-Assoc"):
        m = wsp_sud_full["models"][key]
        if m.get("deterministic"):
            n_eval = m["n"]
            halted = m["halted"]
            wrong = m["wrong_stops"]
            solved = m["solved"]
            h_pct = m["halted_percent"]
            s_pct = m["solved_percent"]
        else:
            n_eval = m["seed_puzzle_evaluations"]
            c = m["counts_pooled_over_seeds"]
            halted = c["stopped"]
            wrong = c["wrong_stops"]
            solved = c["solved"]
            h_pct = m["halted_percent"]["mean"]
            s_pct = m["solved_percent"]["mean"]
        wsp_halting_rows.append({
            "task": "Sudoku-Extreme",
            "method": f"{key} (zero-violation certificate)",
            "provenance": "wsp_rule_programmed_local",
            "n": n_eval,
            "distinct_inputs": 412786,
            "halted": halted,
            "wrong_halts": wrong,
            "solved": solved,
            "halted_percent": h_pct,
            "wrong_halts_percent": m["wrong_halts_pct_of_halted"],
            "solved_percent": s_pct,
        })
    for task_name in ("Maze-Unique", "Maze-Hard"):
        hl = wsp_maze_all["headline"][task_name]
        for key in ("WSP-DeadEnd", "WSP-Wavefront", "WSP-Potential"):
            h = hl[key]
            pc = h["pooled_3seed_counts"]
            wsp_halting_rows.append({
                "task": task_name,
                "method": f"{key} (fixed-point stop)",
                "provenance": "wsp_rule_programmed_local",
                "n": pc["evaluations"],
                "distinct_inputs": 1000,
                "halted": pc["halted"],
                "wrong_halts": pc["wrong_halts"],
                "solved": pc["solved"],
                "halted_percent": h["halted_pct_mean"],
                "wrong_halts_percent": h["wrong_halts_pct_of_halted"],
                "solved_percent": h["solved_pct_mean"],
            })

    # 3. Multi-solution N-Queens & Graph Coloring
    wsp_nqueens_rows = []
    for sz in ("8", "10"):
        sv = wsp_nq["sizes"][sz]
        for arm_key in ("wsp_det", "wsp_arbitrary", "wsp_permutation"):
            av = sv["arms"][arm_key]
            d = av["disclosure"]
            met = av["metrics"]
            wsp_nqueens_rows.append({
                "size": int(sz),
                "arm": arm_key,
                "label": av["label"],
                "params_learned": d["params_learned"],
                "distinct_programmed_coefficients": d["distinct_programmed_coefficients"],
                "energy_entries": d["energy_entries"],
                "energy_nonzero": d["energy_nonzero"],
                "materialized_entries": d["materialized_entries"],
                "materialized_nonzero": d["materialized_nonzero"],
                "fit_display": d["fit_display"],
                "prior_disclosure": d["prior_disclosure"],
                "acc_n1": met["acc_n1"],
                "acc_n20": met["acc_n20"],
                "cov20_per_puzzle": met["cov20_per_puzzle"],
                "mean_fc2_flops_n1_per_row": av["mean_fc2_flops_n1_per_row"],
                "mean_fc2_flops_n20_per_row": av["mean_fc2_flops_n20_per_row"],
                "mean_fc2_flops_with_compare_n20_per_row": av["mean_fc2_flops_with_compare_n20_per_row"],
                "pooled_wrong_stops": av["pooled_wrong_stops"],
                "pooled_budget_failures": av["pooled_budget_failures"],
            })

    wsp_gc_rows = []
    for sz in ("8", "10"):
        sv = wsp_gc["sizes"][sz]
        for arm_key in ("wsp_det", "wsp_assoc", "wsp_assoc_uncanonicalized"):
            av = sv["arms"][arm_key]
            d = av["disclosure"]
            met = av["metrics"]
            wsp_gc_rows.append({
                "vertices": int(sz),
                "arm": arm_key,
                "label": av["label"],
                "params_learned": d["params_learned"],
                "distinct_programmed_coefficients": d["distinct_programmed_coefficients"],
                "energy_entries": d["energy_entries"],
                "energy_nonzero": d["energy_nonzero"],
                "materialized_entries": d["materialized_entries"],
                "materialized_nonzero": d["materialized_nonzero"],
                "fit_display": d["fit_display"],
                "prior_disclosure": d["prior_disclosure"],
                "n1_conflicts_per_seed": av["n1_conflicts_per_seed"],
                "n20_conflicts_per_seed": av["n20_conflicts_min_energy_per_seed"],
                "coverage_20_pct": met["coverage_20_pct"],
                "coverage_canonicalized_20_pct": met["coverage_canonicalized_20_pct"],
                "mean_fc2_flops_n1_per_graph": av["mean_fc2_flops_n1_per_graph"],
                "mean_fc2_flops_n20_per_graph": av["mean_fc2_flops_n20_per_graph"],
                "mean_fc2_flops_with_compare_n20_per_graph": av["mean_fc2_flops_with_compare_n20_per_graph"],
                "pooled_wrong_stops": av["pooled_wrong_stops"],
                "pooled_budget_failures": av["pooled_budget_failures"],
            })

    # 4. 180x180 Dense Maze 3-cohort comparison
    wsp_180_cohorts = {}
    for cid in ("H2bo", "H2bp", "H2bq"):
        c = wsp_maze_180["cohorts"][cid]
        t = c["timing_4_balanced_pairs"]
        wsp_180_cohorts[cid] = {
            "cohort": c["cohort"],
            "label": c["label"],
            "accuracy": c["accuracy"],
            "fc2_flops_per_unique_maze": c["fc2_flops_per_unique_maze"],
            "matched_3seed_B32_median_seconds": t["matched_3seed_B32_median_seconds"],
            "matched_3seed_B32_complete_seconds": t["matched_3seed_B32_complete_seconds"],
            "unique_1seed_B256_median_seconds": t["unique_1seed_B256_median_seconds"],
            "comparison_vs_published_LP96_12288": t["comparison_vs_published_LP96_12288"],
            "comparison_vs_published_BP_geometry_LP96_12288": t["comparison_vs_published_BP_geometry_LP96_12288"],
        }

    # 5. N=13 Queen+Leaper matched-time comparison
    wsp_ql_summary = {}
    for bk in ("b0", "b1"):
        fv = wsp_ql["families"][bk]
        mtc = fv["matched_time_comparison"]
        h3_fam = h3ad_ref["family_results"][fv["bank"]]
        wsp_ql_summary[fv["family"]] = {
            "bank": fv["bank"],
            "leaper": fv["leaper"],
            "test_inputs": fv["test_inputs"],
            "all_solutions_count": fv["all_solutions_count"],
            "wsp_source_calibration": fv["source_calibration"],
            "wsp_det": {
                "cap": mtc["deterministic_control"]["cap"],
                "certified_success_pct": mtc["deterministic_control"]["certified_success_pct"],
                "coverage_pct": mtc["deterministic_control"]["coverage_pct"],
                "wrong_stops": mtc["deterministic_control"]["wrong_stops"],
                "mean_fc2_flops": mtc["deterministic_control"]["fc2_flops_n20"]["mean_flops_per_instance"],
            },
            "wsp_arbitrary": {
                "cap": mtc["arbitrary"]["cap"],
                "certified_success_pct": mtc["arbitrary"]["metrics"]["certified_success_pct"],
                "coverage_pct": mtc["arbitrary"]["metrics"]["coverage_pct"],
                "median_worker_solve_seconds": mtc["arbitrary"]["median_worker_solve_seconds"],
                "mean_fc2_flops_n20": mtc["arbitrary"]["mean_fc2_flops_n20"],
                "pooled_wrong_stops": mtc["arbitrary"]["pooled_wrong_stops"],
            },
            "wsp_permutation": {
                "cap": mtc["permutation"]["cap"],
                "certified_success_pct": mtc["permutation"]["metrics"]["certified_success_pct"],
                "coverage_pct": mtc["permutation"]["metrics"]["coverage_pct"],
                "median_worker_solve_seconds": mtc["permutation"]["median_worker_solve_seconds"],
                "mean_fc2_flops_n20": mtc["permutation"]["mean_fc2_flops_n20"],
                "pooled_wrong_stops": mtc["permutation"]["pooled_wrong_stops"],
            },
            "wsp_target_time_ratio_perm_over_arb": mtc["target_median_solve_time_ratio_perm_over_arb"],
            "wsp_paired_bootstrap_perm_minus_arb": mtc["paired_bootstrap_perm_minus_arb"],
            "h3ad_bp_adamw_reference": {
                "rows": h3_fam["rows"],
                "gain_pp": h3_fam["primary"]["gain_pp"],
                "target_time_ratio": h3_fam["target_time_ratio"],
            },
        }

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wsp_reference": {
            "paper": "Weight-Space Programming Doesn't Need Transformers (Hyunjong Ok & Jaeho Lee, EFFL Lab, POSTECH, 2026)",
            "upstream_repo": "https://github.com/effl-lab/Weight-Space-Programming",
            "upstream_commit": "02c31abd3b34d84854cc80d9717717c45365c719",
            "license": "MIT",
            "static_ast_contract": "All WSP forward(w, x) programs pass upstream wsp/free_model.py:static_problems == []",
        },
        "frozen_reproduction_manifest_entries_verified": frozen_checked,
        "scope_and_disclosure": {
            "wsp_paradigm": "Rule-programmed weights in weight space (params_learned = 0, training_examples = 0, Fit time = 'none'); grouped conceptually with Hand-set CRAM as zero-training rule-informed programs, never conflated with data-learned models (CRAM, Count Energy, EBR, TRM, etc.) that infer constraints from solved examples without supplied rule descriptions.",
            "flops_convention": "Paper FullCounter (FC2) arithmetic + transcendental FLOPs across all chains and all capped/unsolved instances; comparison-inclusive totals (total_with_compare) also recorded.",
            "hardware_note": "All local WSP timings and CRAM fresh timings measured on NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (Compute Capability 12.0) and AMD Ryzen Threadripper PRO 9985WX; published neural baseline wall times use H200.",
        },
        "tables": {
            "main_comparison_wsp_rows": wsp_main_rows,
            "main_comparison_existing_rows_count": len(main_comp["rows"]),
            "halting_comparison_wsp_rows": wsp_halting_rows,
            "halting_comparison_existing_local_rows": halt_comp["local"],
            "multi_solution_nqueens_wsp_rows": wsp_nqueens_rows,
            "multi_solution_graph_coloring_wsp_rows": wsp_gc_rows,
            "multi_solution_build_timings": wsp_multi_sum["build_timings"],
            "sudoku_compute_ratios_wsp_assoc_vs_published": wsp_sud_any["wsp_assoc_vs_published_baselines"],
            "sudoku_full_test_reference": {
                "learned_cram_mean_flops": sud_learn["paper_mean_flops"]["mean"],
                "learned_cram_solved_pct": sud_learn["solved_percent"],
                "hand_cram_mean_flops": sud_hand["paper_mean_flops"]["mean"],
                "hand_cram_solved_pct": sud_hand["solved_percent"],
                "wsp_assoc_mean_flops": wsp_sud_full["models"]["WSP-Assoc"]["paper_mean_flops"]["mean"],
                "hand_over_wsp_assoc_flops_ratio": sud_hand["paper_mean_flops"]["mean"] / wsp_sud_full["models"]["WSP-Assoc"]["paper_mean_flops"]["mean"],
                "learned_over_wsp_assoc_flops_ratio": sud_learn["paper_mean_flops"]["mean"] / wsp_sud_full["models"]["WSP-Assoc"]["paper_mean_flops"]["mean"],
            },
            "sudoku_figure3a_difficulty_summary": {
                k: {
                    "solved_pct": v.get("solved_pct"),
                    "mean_flop": v.get("mean_flop"),
                    "hardest_to_easiest_median_compute": v.get("hardest_to_easiest_median_compute"),
                }
                for k, v in wsp_sud_3a["models"].items()
            },
            "sudoku_figure3b_zero_data_k0": wsp_sud_3b["models"],
            "sudoku_count_budget_30k_40k": {
                "wsp_assoc_grouped": wsp_sud_cb["grouped"],
                "wsp_deterministic_controls": wsp_sud_cb["deterministic_controls_on_30k_40k"],
                "count_energy_16m": count_bud["grouped"][-1],
                "transformed_learned_16m": learn_bud["grouped"][-1],
                "original_cram_short_262k": next(r for r in short_bud["grouped"] if r["k"] == 10 and r["arm"] == "original_general"),
            },
            "maze_hard_test_106_audit": wsp_maze_h["maze_106_audit"],
            "maze_180x180_three_cohorts": wsp_180_cohorts,
            "queen_leaper_n13_matched_time": wsp_ql_summary,
        },
        "source_sha256": sources,
    }
    atomic_write_json(OUT, payload)
    print(json.dumps({
        "output": str(OUT.relative_to(ROOT)),
        "frozen_manifest_verified": frozen_checked,
        "main_wsp_rows": len(wsp_main_rows),
        "halting_wsp_rows": len(wsp_halting_rows),
        "nq_wsp_rows": len(wsp_nqueens_rows),
        "gc_wsp_rows": len(wsp_gc_rows),
        "sources_hashed": len(sources),
    }))


if __name__ == "__main__":
    main()
