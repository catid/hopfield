"""Figure 2a: fresh CRAM anytime records and explicitly imported baseline curves."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figures"
fresh_path = ROOT / "artifacts/sudoku_audit/learned-anytime-curves.json"
baseline_path = ROOT / "upstream/results/compute-sweep/sudoku-extreme.json"
published_cram_path = ROOT / "upstream/results/cram-anytime/sudoku-extreme.json"
fresh = json.loads(fresh_path.read_text())["sets"]["first10k"]
baseline = json.loads(baseline_path.read_text())
published_cram = json.loads(published_cram_path.read_text())["sets"]["first10k"]
assert all(r["n"] == 10000 for r in baseline["rows"])
methods = sorted({r["method"] for r in baseline["summary"]})
comparisons = []
frontiers = {}
for method in methods:
    settings = [r for r in baseline["summary"] if r["method"] == method and r["seeds"] == [0, 1, 2]]
    top_accuracy = max(r["solve_rate_mean"] for r in settings)
    top = min((r for r in settings if r["solve_rate_mean"] == top_accuracy), key=lambda r:r["mean_flop"])
    eligible = [r for r in fresh if r["solved_mean"] >= top_accuracy]
    match = min(eligible, key=lambda r:r["mean_flop"]) if eligible else None
    published_eligible = [r for r in published_cram if r["solved_mean"] >= top_accuracy]
    published_match = min(published_eligible, key=lambda r:r["mean_flop"]) if published_eligible else None
    comparisons.append({"method": method, "baseline_provenance": "Published imported sweep, not locally reproduced",
        "paper_main_figure_comparison": method != "TRMP-fast",
        "baseline_best_setting": top, "fresh_cram_matching_point": match,
        "published_cram_matching_point": published_match,
        "baseline_mean_flops_over_published_cram": top["mean_flop"] / published_match["mean_flop"] if published_match else None,
        "baseline_mean_flops_over_fresh_cram": top["mean_flop"] / match["mean_flop"] if match else None})
    frontier, best = [], -1.
    for r in sorted(settings, key=lambda r:(r["mean_flop"], -r["solve_rate_mean"])):
        if r["solve_rate_mean"] > best:
            raw = [q for q in baseline["rows"] if all(q[k] == r[k] for k in ("method", "knob", "setting"))]
            assert sorted(q["seed"] for q in raw) == [0, 1, 2]
            rates = [q["solve_rate"] for q in sorted(raw, key=lambda q:q["seed"])]
            assert np.isclose(np.mean(rates), r["solve_rate_mean"], atol=1e-12, rtol=0)
            frontier.append(dict(r, per_seed_solve_rate=rates))
            best = r["solve_rate_mean"]
    frontiers[method] = frontier

ebr_validation = next(r for r in comparisons if r["method"] == "EBR")
published_ebr_ratio = ebr_validation["baseline_mean_flops_over_published_cram"]
assert round(published_ebr_ratio) == 282
assert ebr_validation["published_cram_matching_point"]["cap"] == ebr_validation["fresh_cram_matching_point"]["cap"]
meta = {"figure": "2a", "subset": "full-all rows10000..19999; 10000 held-out test puzzles",
    "seeds": [0, 1, 2], "fresh_cram_source": str(fresh_path.relative_to(ROOT)),
    "fresh_cram_source_sha256": hashlib.sha256(fresh_path.read_bytes()).hexdigest(),
    "baseline_source": str(baseline_path.relative_to(ROOT)),
    "baseline_source_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    "published_cram_source": str(published_cram_path.relative_to(ROOT)),
    "published_cram_source_sha256": hashlib.sha256(published_cram_path.read_bytes()).hexdigest(),
    "published_curve_validation": {"method": "EBR", "reported_rounded_ratio": 282,
        "recomputed_ratio": published_ebr_ratio, "matches_reported_rounding": True,
        "same_cap_as_fresh": True,
        "interpretation": "Applying the identical matching rule to the released CRAM curve reproduces the reported 282x after rounding. The fresh 275.4x difference comes from the measured CRAM trajectories and mean counts at the same cap, not the matching rule."},
    "baseline_status": "Published reference only; no local baseline training or inference",
    "scope_note": "Figure2a displays the seven main-table baselines. TRMP-fast is the separately footnoted lattice model with its handwritten search; its imported comparison is retained in JSON but excluded from the main figure and main seven-baseline minimum-ratio claim.",
    "bp_adamw_control": "Not measured; outside the confirmed CRAM-only scope",
    "comparison_rule": "Each baseline's maximum mean accuracy across its published fixed sweep; ties choose lower mean FLOPs. Match the lowest-mean-FLOP point of the frozen CRAM cap grid at least as accurate. This is a post-hoc curve comparison, not model/test hyperparameter tuning. Ratios depend on grid resolution and imported reference conventions.",
    "curve_rule": "Connected points form the nondominated frontier of the fixed published settings for each baseline. Error bars span minimum to maximum seed accuracy. Fresh CRAM caps use saved first-success counts without rerunning inference; below maximum cap unsolved returned boards are conservatively counted as failures.",
    "comparisons": comparisons, "baseline_frontiers": frontiers, "fresh_cram": fresh}
(OUT / "sudoku-compute-comparison.json").write_text(json.dumps(meta, indent=2)+"\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 190, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True, layout="constrained",
                         gridspec_kw={"height_ratios": [1, 1.5]})
colors = {"EBR":"#c7423e", "TRM":"#12916e", "HRM":"#916740", "GRAM":"#a16c34", "EqR":"#cb73a4",
          "FRM":"#e49a00", "TRMP":"#7863ba", "TRMP-fast":"#8d8d8d"}
for ax in axes:
    for method, frontier in frontiers.items():
        if method == "TRMP-fast":
            continue
        x = np.asarray([r["mean_flop"] for r in frontier])/1e9
        y = 100*np.asarray([r["solve_rate_mean"] for r in frontier])
        low = 100*np.asarray([min(r["per_seed_solve_rate"]) for r in frontier])
        high = 100*np.asarray([max(r["per_seed_solve_rate"]) for r in frontier])
        ax.errorbar(x, y, yerr=np.maximum(0, np.stack([y-low, high-y])), color=colors[method], marker="s", ms=3.4,
                    lw=1.25, alpha=.8, elinewidth=.65, capsize=1.5, label=f"{method} (published)")
    x = np.asarray([r["mean_flop"] for r in fresh])/1e9
    y = 100*np.asarray([r["solved_mean"] for r in fresh])
    low = 100*np.asarray([min(r["solved_per_seed"]) for r in fresh])
    high = 100*np.asarray([max(r["solved_per_seed"]) for r in fresh])
    ax.errorbar(x, y, yerr=np.maximum(0, np.stack([y-low, high-y])), color="#1676c4", marker="o", ms=3.5,
                lw=2.3, elinewidth=1, capsize=2, label="CRAM (fresh)", zorder=5)
    ax.set_xscale("log")
    ax.set_ylabel("Solved puzzles (%)")
    ax.grid(alpha=.18, which="both")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylim(98, 100.05)
axes[1].set_ylim(0, 101)
axes[1].set_xlim(.025, max(r["mean_flop"] for method, rows in frontiers.items() if method != "TRMP-fast" for r in rows)/1e9*1.4)
axes[1].axhspan(98, 100, color="#1676c4", alpha=.08)
axes[1].set_xlabel("Mean counted GFLOPs per puzzle, all search chains included")
ebr = next(r for r in comparisons if r["method"] == "EBR")
match = ebr["fresh_cram_matching_point"]
if match:
    ypos = 100*ebr["baseline_best_setting"]["solve_rate_mean"]
    x0, x1 = match["mean_flop"]/1e9, ebr["baseline_best_setting"]["mean_flop"]/1e9
    axes[0].annotate("", xy=(x1, ypos), xytext=(x0, ypos), arrowprops={"arrowstyle":"<->", "color":"#222222", "ls":"--"})
    axes[0].text(np.sqrt(x0*x1), ypos-.18,
        f"{ebr['baseline_mean_flops_over_fresh_cram']:.1f}× at matched mean accuracy", ha="center", va="top", fontsize=10)
handles, labels = axes[1].get_legend_handles_labels()
order = [labels.index("CRAM (fresh)")] + [i for i,label in enumerate(labels) if label != "CRAM (fresh)"]
axes[1].legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", frameon=False, ncols=2, fontsize=9)
fig.suptitle("Sudoku-Extreme compute: fresh CRAM with published baseline references", fontsize=14)
axes[0].set_title("Top 2% of solve rates · error bars span three seeds", fontsize=10)
fig.supxlabel("10,000 held-out puzzles; baseline curves are imported and have not been locally reproduced.", fontsize=9)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"sudoku-compute-fresh.{ext}", bbox_inches="tight", facecolor="white")
plt.close(fig)
for c in comparisons:
    print(c["method"], "accuracy", 100*c["baseline_best_setting"]["solve_rate_mean"],
          "ratio", c["baseline_mean_flops_over_fresh_cram"])
