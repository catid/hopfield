"""Render WSP-extended versions of Paper Figures 2a, 3a, and 3b without mutating frozen artifacts.

Strictly read-only on results/reproduction/* and artifacts/*.
Writes:
  - assets/figures/wsp-figure2a-compute.{png,pdf,svg} + results/wsp/figures/sudoku-compute-wsp.json
  - assets/figures/wsp-figure3a-difficulty.{png,pdf,svg} + results/wsp/figures/figure3a-wsp.json
  - assets/figures/wsp-figure3b-data-efficiency.{png,pdf,svg} + results/wsp/figures/figure3b-wsp.json
  - assets/figures/wsp-paper-figures-provenance.json
"""
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_OUT = ROOT / "assets/figures"
JSON_OUT = ROOT / "results/wsp/figures"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def render_figure2a(inputs: dict[str, str], outputs: dict[str, str]) -> None:
    comp_rel = "results/reproduction/figures/sudoku-compute-comparison.json"
    wsp_rel = "results/wsp/sudoku/wsp-sudoku-anytime-curves.json"
    inputs[comp_rel] = sha(ROOT / comp_rel)
    inputs[wsp_rel] = sha(ROOT / wsp_rel)

    comp = json.loads((ROOT / comp_rel).read_text(encoding="utf-8"))
    wsp = json.loads((ROOT / wsp_rel).read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "savefig.dpi": 190,
        "pdf.fonttype": 42,
        "svg.hashsalt": "hopfield-wsp-fig2a-v1",
    })
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 8.2), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.05, 1.45]})
    colors = {
        "EBR": "#c7423e", "TRM": "#12916e", "HRM": "#916740", "GRAM": "#a16c34",
        "EqR": "#cb73a4", "FRM": "#e49a00", "TRMP": "#7863ba",
    }

    wsp_curve = wsp["WSP-Assoc_sets"]["first10k"]
    det_sweep = wsp["WSP-Det_first10k_sweep"]
    tf_sweep = wsp["WSP-Transformer_first10k_sweep"]
    fresh_cram = comp["fresh_cram"]

    for ax in axes:
        for method, frontier in comp["baseline_frontiers"].items():
            if method == "TRMP-fast":
                continue
            x = np.asarray([r["mean_flop"] for r in frontier]) / 1e9
            y = 100.0 * np.asarray([r["solve_rate_mean"] for r in frontier])
            low = 100.0 * np.asarray([min(r["per_seed_solve_rate"]) for r in frontier])
            high = 100.0 * np.asarray([max(r["per_seed_solve_rate"]) for r in frontier])
            ax.errorbar(x, y, yerr=np.maximum(0, np.stack([y - low, high - y])),
                        color=colors[method], marker="s", ms=3.2, lw=1.2, alpha=0.78,
                        elinewidth=0.65, capsize=1.5, label=f"{method} (published)")

        xc = np.asarray([r["mean_flop"] for r in fresh_cram]) / 1e9
        yc = 100.0 * np.asarray([r["solved_mean"] for r in fresh_cram])
        lc = 100.0 * np.asarray([min(r["solved_per_seed"]) for r in fresh_cram])
        hc = 100.0 * np.asarray([max(r["solved_per_seed"]) for r in fresh_cram])
        ax.errorbar(xc, yc, yerr=np.maximum(0, np.stack([yc - lc, hc - yc])),
                    color="#1676c4", marker="o", ms=3.5, lw=2.2, elinewidth=1.0, capsize=2.0,
                    label="CRAM (fresh, learned)", zorder=5)

        xw = np.asarray([r["mean_flop"] for r in wsp_curve]) / 1e9
        yw = 100.0 * np.asarray([r["solved_mean"] for r in wsp_curve])
        lw_err = 100.0 * np.asarray([min(r["solved_per_seed"]) for r in wsp_curve])
        hw_err = 100.0 * np.asarray([max(r["solved_per_seed"]) for r in wsp_curve])
        ax.errorbar(xw, yw, yerr=np.maximum(0, np.stack([yw - lw_err, hw_err - yw])),
                    color="#6a4c93", marker="D", ms=3.8, lw=2.3, ls="--", elinewidth=1.0, capsize=2.0,
                    label="WSP-Assoc (k=0, rules)", zorder=6)

        ax.plot(np.asarray([r["flop_per_puzzle_total"] for r in det_sweep]) / 1e9,
                np.asarray([r["solved_pct"] for r in det_sweep]),
                marker="^", ms=4.2, lw=1.5, ls=":", color="#d97724", label="WSP-Det (k=0, det.)", zorder=5)
        ax.plot(np.asarray([r["flop_per_puzzle_total"] for r in tf_sweep]) / 1e9,
                np.asarray([r["solved_pct"] for r in tf_sweep]),
                marker="v", ms=4.2, lw=1.5, ls=":", color="#c44e52", label="WSP-Transformer (k=0, det.)", zorder=5)

        ax.set_xscale("log")
        ax.set_ylabel("Solved puzzles (%)")
        ax.grid(alpha=0.18, which="both")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylim(98.0, 100.42)
    axes[0].set_yticks([98.0, 98.5, 99.0, 99.5, 100.0])
    axes[1].set_ylim(0, 101.5)
    axes[1].set_xlim(1.5e-3, 2.5e6)
    axes[1].axhspan(98, 100, color="#6a4c93", alpha=0.06)
    axes[1].set_xlabel("Mean counted FC2 GFLOPs per puzzle (all search chains included)")

    ebr_wsp = next(r for r in wsp["wsp_assoc_vs_published_baselines"] if r["method"] == "EBR")
    x0 = ebr_wsp["wsp_assoc_matching_point"]["mean_flop"] / 1e9
    x1 = ebr_wsp["baseline_best_setting"]["mean_flop"] / 1e9
    ypos = 100.0 * ebr_wsp["baseline_best_setting"]["solve_rate_mean"]
    axes[0].annotate("", xy=(x1, ypos), xytext=(x0, ypos),
                     arrowprops={"arrowstyle": "<->", "color": "#6a4c93", "ls": "--", "lw": 1.3})
    axes[0].text(np.sqrt(x0 * x1), 100.10,
                 f"WSP-Assoc: {ebr_wsp['baseline_mean_flops_over_wsp_assoc']:,.0f}× fewer FLOPs vs EBR at matched mean accuracy (k=0 rules)",
                 ha="center", va="bottom", fontsize=9.2, color="#6a4c93", weight="bold")

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="none", ncols=2, fontsize=7.6, columnspacing=0.8)
    fig.suptitle("Figure 2a (extended): Sudoku-Extreme compute frontier with Weight-Space Programming", fontsize=13.6, weight="bold")
    axes[0].set_title("Top 2% of solve rates · error bars span seeds 0–2 · unstopped boards below cap counted unsolved", fontsize=9.8)
    fig.supxlabel("10,000 held-out test puzzles (full-all[10000:20000]); WSP uses 0 training examples and supplied Sudoku rules; neural baselines are published.", fontsize=8.8)

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        out_p = FIG_OUT / f"wsp-figure2a-compute.{ext}"
        fig.savefig(out_p, bbox_inches="tight", facecolor="white", metadata={"Date": None} if ext == "svg" else None)
        outputs[str(out_p.relative_to(ROOT))] = sha(out_p)
    plt.close(fig)

    atomic_write_json(JSON_OUT / "sudoku-compute-wsp.json", {
        "figure": "2a-wsp-extended",
        "subset": comp["subset"],
        "wsp_assoc_vs_published_baselines": wsp["wsp_assoc_vs_published_baselines"],
        "fresh_cram_vs_published_baselines": comp["comparisons"],
        "source_sha256": {comp_rel: inputs[comp_rel], wsp_rel: inputs[wsp_rel]},
    })


def render_figure3a(inputs: dict[str, str], outputs: dict[str, str]) -> None:
    ref_rel = "results/reproduction/difficulty/figure3a.json"
    wsp_rel = "results/wsp/sudoku/wsp-sudoku-figure3a-difficulty.json"
    inputs[ref_rel] = sha(ROOT / ref_rel)
    inputs[wsp_rel] = sha(ROOT / wsp_rel)

    ref = json.loads((ROOT / ref_rel).read_text(encoding="utf-8"))
    wsp = json.loads((ROOT / wsp_rel).read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.8,
        "savefig.dpi": 190,
        "pdf.fonttype": 42,
        "svg.hashsalt": "hopfield-wsp-fig3a-v1",
    })
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.85, 1.15]})
    x = np.arange(8)

    series = [
        ("WSP-Assoc (k=0, rules)", wsp["models"]["WSP-Assoc"], "#6a4c93", "-.", "D"),
        ("WSP-Det (k=0, det.)", wsp["models"]["WSP-Det"], "#d97724", ":", "^"),
        ("WSP-Transformer (k=0, det.)", wsp["models"]["WSP-Transformer"], "#c44e52", ":", "v"),
        ("CRAM fresh (learned)", ref["fresh"]["CRAM"], "#1b9e77", "-", "o"),
        ("FRM published", ref["published_reference"]["FRM"], "#e49a00", "--", "s"),
        ("EBR published", ref["published_reference"]["EBR"], "#7570b3", "--", "s"),
    ]

    for label, model, color, style, marker in series:
        bins = model["bins"]
        meds = [b["flop_median"] for b in bins]
        axes[0].plot(x, meds, linestyle=style, marker=marker, ms=4.2, lw=1.9, color=color, label=label)
        if "flop_p10" in bins[0] and bins[0]["flop_p10"] is not None:
            axes[0].fill_between(x, [b["flop_p10"] for b in bins], [b["flop_p90"] for b in bins], color=color, alpha=0.10)
        acc = np.asarray([b["solved_pct"] for b in bins])
        if acc.ndim == 2:
            axes[1].plot(x, np.median(acc, axis=1), linestyle=style, marker=marker, ms=4.0, lw=1.8, color=color)
            axes[1].fill_between(x, acc.min(1), acc.max(1), color=color, alpha=0.10)
        else:
            axes[1].plot(x, acc, linestyle=style, marker=marker, ms=4.0, lw=1.8, color=color)

    axes[0].set_yscale("log")
    axes[0].set_ylim(1.5e6, 2.5e14)
    axes[0].set_ylabel("FC2 FLOPs per puzzle (median, p10–p90)")
    axes[0].legend(fontsize=8.1, ncol=2, loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="none")
    axes[1].set_ylabel("Solved (%)")
    axes[1].set_ylim(-3, 105)
    axes[1].set_xlabel("Sudoku difficulty rating (backtracking guesses)")
    axes[1].set_xticks(x, ["0", "1", "2–3", "4–7", "8–15", "16–31", "32–63", "64+"])
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Figure 3a (extended): Difficulty scaling of CRAM, published baselines, and WSP", fontsize=12.6, weight="bold")
    fig.tight_layout()

    for ext in ("png", "pdf", "svg"):
        out_p = FIG_OUT / f"wsp-figure3a-difficulty.{ext}"
        fig.savefig(out_p, bbox_inches="tight", facecolor="white", metadata={"Date": None} if ext == "svg" else None)
        outputs[str(out_p.relative_to(ROOT))] = sha(out_p)
    plt.close(fig)

    atomic_write_json(JSON_OUT / "figure3a-wsp.json", {
        "figure": "3a-wsp-extended",
        "subset": ref["subset"],
        "bins": ref["bins"],
        "bin_counts": ref["bin_counts"],
        "wsp_models": wsp["models"],
        "cram_fresh": ref["fresh"]["CRAM"],
        "source_sha256": {ref_rel: inputs[ref_rel], wsp_rel: inputs[wsp_rel]},
    })


def render_figure3b(inputs: dict[str, str], outputs: dict[str, str]) -> None:
    col_rel = "results/reproduction/data_efficiency/collection.json"
    wsp_rel = "results/wsp/sudoku/wsp-sudoku-figure3b-zero-data.json"
    inputs[col_rel] = sha(ROOT / col_rel)
    inputs[wsp_rel] = sha(ROOT / wsp_rel)

    col = json.loads((ROOT / col_rel).read_text(encoding="utf-8"))
    wsp = json.loads((ROOT / wsp_rel).read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.8,
        "savefig.dpi": 190,
        "pdf.fonttype": 42,
        "svg.hashsalt": "hopfield-wsp-fig3b-v1",
    })
    fig, ax = plt.subplots(figsize=(8.8, 5.5))

    points = col["points"]
    x = [p["k"] for p in points]
    y = [p["accuracy_pct_median"] for p in points]
    low = [p["accuracy_pct_min"] for p in points]
    high = [p["accuracy_pct_max"] for p in points]

    ax.plot(x, y, "o-", color="#1b9e77", lw=2.1, ms=4.5, label="CRAM fresh (learned, seed42 ref.)", zorder=5)
    ax.fill_between(x, low, high, color="#1b9e77", alpha=0.14)

    for method, color in [("TRM", "#d95f02"), ("EBR", "#7570b3"), ("FRM", "#e7298a"), ("CRAM", "#777777")]:
        pp = [p for p in col["published_reference"]["points"] if p["method"] == method]
        ax.plot([p["k"] for p in pp],
                [p["accuracy_pct_median"] if p["n_seeds"] == 6 else np.nan for p in pp],
                "--", color=color, lw=1.3, label=f"{method} published (learned)")

    # Zero-training WSP horizontal reference lines (k=0 training examples, explicit rule-programmed label)
    w_assoc = wsp["models"]["WSP-Assoc"]["accuracy_pct_median"]
    w_det = wsp["models"]["WSP-Det"]["accuracy_pct"]
    w_tf = wsp["models"]["WSP-Transformer"]["accuracy_pct"]

    ax.axhline(w_assoc, color="#6a4c93", ls="-.", lw=2.1,
               label=f"WSP-Assoc (k=0 rules, {w_assoc:.1f}%)", zorder=4)
    ax.axhline(w_det, color="#d97724", ls=":", lw=1.7,
               label=f"WSP-Det (k=0 rules, {w_det:.2f}%)", zorder=4)
    ax.axhline(w_tf, color="#c44e52", ls=":", lw=1.5,
               label=f"WSP-Transformer (k=0 rules, {w_tf:.2f}%)", zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Unique training puzzles k (each with 99 stored symmetry variants; WSP uses k=0 + rules)")
    ax.set_ylabel("Solved on 10,000 test puzzles (%)")
    ax.set_ylim(-3, 105)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.2, loc="center right", frameon=True, facecolor="white", edgecolor="#e0e6ef", framealpha=0.92)
    ax.set_title("Figure 3b (extended): Data efficiency vs zero-training Weight-Space Programming", fontsize=12.2, weight="bold")

    fig.text(
        0.5,
        0.012,
        "Data-learned curves show 6-seed medians without supplied Sudoku rules. Horizontal WSP lines use k=0 training puzzles with explicit Sudoku rules compiled into weights.",
        ha="center",
        fontsize=8.4,
        color="#526075",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    for ext in ("png", "pdf", "svg"):
        out_p = FIG_OUT / f"wsp-figure3b-data-efficiency.{ext}"
        fig.savefig(out_p, bbox_inches="tight", facecolor="white", metadata={"Date": None} if ext == "svg" else None)
        outputs[str(out_p.relative_to(ROOT))] = sha(out_p)
    plt.close(fig)

    atomic_write_json(JSON_OUT / "figure3b-wsp.json", {
        "figure": "3b-wsp-extended",
        "subset": wsp["subset"],
        "wsp_zero_data_k0": wsp["models"],
        "cram_fresh_points": col["points"],
        "source_sha256": {col_rel: inputs[col_rel], wsp_rel: inputs[wsp_rel]},
    })


def main() -> None:
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    render_figure2a(inputs, outputs)
    render_figure3a(inputs, outputs)
    render_figure3b(inputs, outputs)
    atomic_write_json(FIG_OUT / "wsp-paper-figures-provenance.json", {
        "script": "scripts/plot_wsp_paper_figures.py",
        "script_sha256": sha(Path(__file__)),
        "inputs_sha256": inputs,
        "outputs_sha256": outputs,
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
    })
    print(json.dumps({"rendered_outputs": len(outputs)}))


if __name__ == "__main__":
    main()
