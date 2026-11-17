"""Render 3-panel headline Weight-Space Programming (WSP) comparison figure for README.md.

Strictly read-only on results/reproduction/* and results/research/*.
Writes assets/figures/wsp-comparison.{png,svg} and provenance JSON.
"""
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/figures"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    inputs: dict[str, str] = {}

    def load(rel: str):
        p = ROOT / rel
        inputs[rel] = sha(p)
        return json.loads(p.read_text(encoding="utf-8"))

    sud_full = load("results/wsp/sudoku/wsp-sudoku-full-test-412786.json")
    sud_any = load("results/wsp/sudoku/wsp-sudoku-anytime-curves.json")
    sud_comp = load("results/reproduction/figures/sudoku-compute-comparison.json")
    sud_learn = load("results/reproduction/sudoku_audit/learned-full-test-audit.json")
    sud_hand = load("results/reproduction/sudoku_audit/hand-full-test-audit.json")
    count_bud = load("results/research/foureyes/count_budget/summary.json")
    maze_all = load("results/wsp/maze/wsp_maze_complete_summary.json")
    nq_wsp = load("results/wsp/multi/nqueens_results.json")
    gc_wsp = load("results/wsp/multi/graph_coloring_results.json")
    nq_ref = load("results/reproduction/multi_solution/nqueens-paper-results.json")
    gc_ref = load("results/reproduction/multi_solution/graph-coloring-paper-results.json")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "text.color": "#17243a",
        "axes.labelcolor": "#273449",
        "xtick.color": "#273449",
        "ytick.color": "#273449",
        "axes.edgecolor": "#b7c1cf",
        "svg.hashsalt": "hopfield-wsp-headline-v1",
    })

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.1), gridspec_kw={"width_ratios": [1.08, 1.15, 1.07]})

    # ------------------------------------------------------------------
    # Panel A: Sudoku-Extreme Full Test (412,786 puzzles) Architecture & Paradigm
    # ------------------------------------------------------------------
    ax_a = axes[0]
    rows_a = [
        ("TRM (pub., 5.0M)", 85.73, "411B FLOPs", "#8997a9", ""),
        ("EBR (pub., 1.8M)", 99.85, "337B FLOPs", "#c7423e", ""),
        ("Count energy (10k, 0 opt)", count_bud["grouped"][-1]["mean_accuracy_pct"], "no optimizer", "#16887d", ""),
        ("Learned CRAM (7.7K nnz)", sud_learn["solved_percent"]["mean"], "2.03B FLOPs", "#1676c4", ""),
        ("WSP-Transformer (k=0, 614K)", sud_full["models"]["WSP-Transformer"]["solved_percent"], "1.71B FLOPs", "#c44e52", "//"),
        ("WSP-Det (k=0, 3 coeffs)", sud_full["models"]["WSP-Det"]["solved_percent"], "15.4M FLOPs", "#d97724", "//"),
        ("Hand-set CRAM (k=0, 1 coeff)", sud_hand["solved_percent"]["mean"], "108.5M FLOPs", "#5c768d", "//"),
        ("WSP-Assoc (k=0, 7 coeffs)", sud_full["models"]["WSP-Assoc"]["solved_percent"]["mean"], "23.8M FLOPs", "#6a4c93", "//"),
    ]
    y_pos = np.arange(len(rows_a))
    for yi, (lbl, val, note, col, hatch) in enumerate(rows_a):
        b = ax_a.barh(yi, val, height=0.62, color=col, alpha=0.88, edgecolor="#17243a", linewidth=0.55, zorder=3)
        if hatch:
            b[0].set_hatch(hatch)
        val_str = f"{val:.3f}%" if 99.99 < val < 100.0 else f"{val:.2f}%"
        ax_a.text(val + 1.5, yi, f"{val_str} · {note}", va="center", ha="left", fontsize=8.3, weight="bold", color="#17243a")

    ax_a.axhline(3.5, color="#9aa6b8", linestyle="--", linewidth=1.0, zorder=2)
    ax_a.text(181, 0.35, "Data-learned\n(0 rules)", ha="right", va="center", fontsize=8.1, color="#273449", style="italic")
    ax_a.text(181, 4.55, "Rule-programmed\n(k=0 boards)", ha="right", va="center", fontsize=8.1, color="#5a377a", style="italic")
    ax_a.set_yticks(y_pos, [r[0] for r in rows_a], fontsize=8.8)
    ax_a.set_xlim(0, 184)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.set_xlabel("Sudoku-Extreme solved (%)")
    ax_a.set_title("A   Sudoku-Extreme (412,786 test)", loc="left", fontsize=11.8, pad=10)
    ax_a.grid(axis="x", alpha=0.18, zorder=0)

    # ------------------------------------------------------------------
    # Panel B: Sudoku-Extreme Compute Frontier (10,000 test puzzles, FC2 GFLOPs)
    # ------------------------------------------------------------------
    ax_b = axes[1]
    base_colors = {
        "EBR": "#c7423e", "TRM": "#12916e", "HRM": "#916740", "GRAM": "#a16c34",
        "EqR": "#cb73a4", "FRM": "#e49a00", "TRMP": "#7863ba",
    }
    for method, frontier in sud_comp["baseline_frontiers"].items():
        if method == "TRMP-fast":
            continue
        xs = np.asarray([r["mean_flop"] for r in frontier]) / 1e9
        ys = 100.0 * np.asarray([r["solve_rate_mean"] for r in frontier])
        ax_b.plot(xs, ys, marker="s", ms=3.0, lw=1.15, alpha=0.72, color=base_colors[method], label=f"{method} (pub.)")

    fresh_cram = sud_comp["fresh_cram"]
    ax_b.plot(
        np.asarray([r["mean_flop"] for r in fresh_cram]) / 1e9,
        100.0 * np.asarray([r["solved_mean"] for r in fresh_cram]),
        marker="o", ms=3.4, lw=2.0, color="#1676c4", label="CRAM (learned)", zorder=5,
    )

    wsp_curve = sud_any["WSP-Assoc_sets"]["first10k"]
    ax_b.plot(
        np.asarray([r["mean_flop"] for r in wsp_curve]) / 1e9,
        100.0 * np.asarray([r["solved_mean"] for r in wsp_curve]),
        marker="D", ms=3.8, lw=2.3, linestyle="--", color="#6a4c93", label="WSP-Assoc (k=0)", zorder=6,
    )

    det_sweep = sud_any["WSP-Det_first10k_sweep"]
    ax_b.plot(
        np.asarray([r["flop_per_puzzle_total"] for r in det_sweep]) / 1e9,
        np.asarray([r["solved_pct"] for r in det_sweep]),
        marker="^", ms=4.2, lw=1.5, linestyle=":", color="#d97724", label="WSP-Det (k=0)", zorder=5,
    )

    tf_sweep = sud_any["WSP-Transformer_first10k_sweep"]
    ax_b.plot(
        np.asarray([r["flop_per_puzzle_total"] for r in tf_sweep]) / 1e9,
        np.asarray([r["solved_pct"] for r in tf_sweep]),
        marker="v", ms=4.2, lw=1.5, linestyle=":", color="#c44e52", label="WSP-Transformer (k=0)", zorder=5,
    )

    ebr_match = next(r for r in sud_any["wsp_assoc_vs_published_baselines"] if r["method"] == "EBR")
    x0 = ebr_match["wsp_assoc_matching_point"]["mean_flop"] / 1e9
    x1 = ebr_match["baseline_best_setting"]["mean_flop"] / 1e9
    yp = 100.0 * ebr_match["baseline_best_setting"]["solve_rate_mean"]
    ax_b.annotate("", xy=(x1, yp), xytext=(x0, yp), arrowprops={"arrowstyle": "<->", "color": "#222222", "ls": "--", "lw": 1.1})
    ax_b.text(np.sqrt(x0 * x1), 101.8, f"{ebr_match['baseline_mean_flops_over_wsp_assoc']:,.0f}× fewer FLOPs vs EBR @ 99.86%",
              ha="center", va="bottom", fontsize=8.0, weight="bold", color="#6a4c93")

    ax_b.set_xscale("log")
    ax_b.set_xlim(1.5e-3, 2e7)
    ax_b.set_ylim(-2, 111)
    ax_b.set_yticks([0, 20, 40, 60, 80, 100])
    ax_b.set_xlabel("Mean counted FC2 GFLOPs / puzzle (all chains)")
    ax_b.set_ylabel("Solved puzzles (%)")
    ax_b.set_title("B   Compute vs accuracy (10k test)", loc="left", fontsize=11.8, pad=10)
    ax_b.legend(loc="lower right", ncol=2, frameon=True, facecolor="white", framealpha=0.92, edgecolor="none", fontsize=7.0, columnspacing=0.55, handletextpad=0.30)
    ax_b.grid(alpha=0.18, which="major")

    # ------------------------------------------------------------------
    # Panel C: Mazes & Multi-Solution CSPs (Local vs Geodesic & Mode Collapse vs Assoc)
    # ------------------------------------------------------------------
    ax_c = axes[2]
    nq10_learn = next(r for r in nq_ref if r["task"] == "nqueens-10" and r["kind"] == "fresh")["metrics"]["cov20_per_puzzle"]["mean"]
    gc10_learn = next(r for r in gc_ref if r["task"] == "graph-coloring-10" and r["kind"] == "learned")["metrics"]["cov20"]["mean"]
    gc10_hand = next(r for r in gc_ref if r["task"] == "graph-coloring-10" and r["kind"] == "hand")["metrics"]["cov20"]["mean"]

    groups_c = [
        ("Maze-Hard\n(Solved %)", [
            ("WSP-DeadEnd (k=0)", maze_all["headline"]["Maze-Hard"]["WSP-DeadEnd"]["solved_pct_mean"], "#d97724", "//"),
            ("EBR (pub.)", 88.30, "#c7423e", ""),
            ("CRAM+z (learned)", 99.90, "#1676c4", ""),
            ("WSP-Wavefront (k=0)", maze_all["headline"]["Maze-Hard"]["WSP-Wavefront"]["solved_pct_mean"], "#6a4c93", "//"),
        ]),
        ("10-Queens\n(Cov@20 %)", [
            ("WSP-Det (k=0)", nq_wsp["sizes"]["10"]["arms"]["wsp_det"]["metrics"]["cov20_per_puzzle"]["mean"], "#d97724", "//"),
            ("EBR (pub.)", 92.480378, "#c7423e", ""),
            ("CRAM (learned)", nq10_learn, "#1676c4", ""),
            ("WSP-Assoc (k=0)", nq_wsp["sizes"]["10"]["arms"]["wsp_arbitrary"]["metrics"]["cov20_per_puzzle"]["mean"], "#6a4c93", "//"),
        ]),
        ("Graph Color 10\n(Cov@20 %)", [
            ("WSP-Det (k=0)", gc_wsp["sizes"]["10"]["arms"]["wsp_det"]["metrics"]["coverage_20_pct"]["mean"], "#d97724", "//"),
            ("CRAM (learned)", gc10_learn, "#1676c4", ""),
            ("Hand CRAM (k=0)", gc10_hand, "#5c768d", "//"),
            ("WSP-Assoc (k=0)", gc_wsp["sizes"]["10"]["arms"]["wsp_assoc"]["metrics"]["coverage_20_pct"]["mean"], "#6a4c93", "//"),
        ]),
    ]

    offs = np.linspace(-0.32, 0.32, 4)
    bw = 0.17
    for gi, (_, bars_spec) in enumerate(groups_c):
        for bi, (_, val, col, hatch) in enumerate(bars_spec):
            b = ax_c.bar(gi + offs[bi], val, width=bw, color=col, alpha=0.88, edgecolor="#17243a", linewidth=0.55, zorder=3)
            if hatch:
                b[0].set_hatch(hatch)
            ax_c.text(gi + offs[bi], val + 1.6, f"{val:.1f}", ha="center", va="bottom", fontsize=6.8, weight="bold")

    ax_c.set_xticks([0, 1, 2], [g[0] for g in groups_c], fontsize=9.4)
    ax_c.set_ylim(0, 136)
    ax_c.set_yticks([0, 25, 50, 75, 100])
    ax_c.set_ylabel("Metric score (%)")
    ax_c.set_title("C   Loopy mazes & multi-solution diversity", loc="left", fontsize=11.8, pad=10)
    ax_c.grid(axis="y", alpha=0.18, zorder=0)

    c_legend = [
        Patch(facecolor="#d97724", edgecolor="#17243a", hatch="//", label="WSP-Det / DeadEnd (k=0)"),
        Patch(facecolor="#6a4c93", edgecolor="#17243a", hatch="//", label="WSP-Assoc / Wavefront (k=0)"),
        Patch(facecolor="#5c768d", edgecolor="#17243a", hatch="//", label="Hand-set CRAM (k=0)"),
        Patch(facecolor="#1676c4", edgecolor="#17243a", label="CRAM / CRAM+z (learned)"),
        Patch(facecolor="#c7423e", edgecolor="#17243a", label="EBR (published)"),
    ]
    ax_c.legend(handles=c_legend, loc="upper left", ncol=2, frameon=False, fontsize=7.3, columnspacing=0.8, handletextpad=0.4)

    fig.suptitle(
        "Weight-Space Programming doesn't need Transformers—and combinatorial reasoning needs associative search",
        x=0.04,
        y=0.975,
        ha="left",
        fontsize=15.4,
        weight="bold",
    )
    fig.text(
        0.04,
        0.015,
        "All WSP models pass upstream wsp.free_model.static_problems == [] with 0 learned parameters and 0 training boards (hatched; rule-informed, grouped with Hand-set CRAM).\n"
        "Programming a 2-layer Transformer (10.52%, 1.71B FLOPs) or deterministic relaxation (14.16%) plateaus on hard CSPs; structured associative WSP + stochastic search solves 100.000% of 412,786 Sudokus (23.8M FLOPs).\n"
        "On Maze-Hard, WSP-Wavefront & WSP-Potential both solve 1,000/1,000 (including Test Maze 106); on N-Queens & Graph Coloring, 20-chain stochastic WSP eliminates deterministic mode collapse.",
        fontsize=8.9,
        color="#526075",
        linespacing=1.35,
    )
    fig.subplots_adjust(left=0.115, right=0.99, top=0.83, bottom=0.20, wspace=0.23)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"wsp-comparison.{ext}", dpi=190, bbox_inches="tight", facecolor="white",
                    metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)

    prov = {
        "script": "scripts/plot_wsp_headline.py",
        "script_sha256": sha(Path(__file__)),
        "inputs_sha256": inputs,
        "outputs_sha256": {
            f"assets/figures/wsp-comparison.{ext}": sha(OUT / f"wsp-comparison.{ext}")
            for ext in ("png", "svg")
        },
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
    }
    atomic_write_text(OUT / "wsp-comparison-provenance.json", json.dumps(prov, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
