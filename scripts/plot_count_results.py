"""Plot audited count-energy, learned CRAM, and zero-training WSP controls on Sudoku-Extreme."""
from pathlib import Path
import hashlib
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/figures"
SOURCES = [
    "results/research/foureyes/count_budget/summary.json",
    "results/research/foureyes/budget/summary.json",
    "results/research/ripper/finaltest.json",
    "results/wsp/sudoku/wsp-sudoku-count-budget-30k-40k.json",
]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    count, learned, short, wsp = [json.loads((ROOT / s).read_text(encoding="utf-8")) for s in SOURCES]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "text.color": "#17243a",
        "axes.labelcolor": "#273449",
        "xtick.color": "#273449",
        "ytick.color": "#273449",
        "axes.edgecolor": "#b7c1cf",
        "svg.hashsalt": "hopfield-count-wsp-v2",
    })

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), gridspec_kw={"width_ratios": [1.12, 1.28]})

    # Left panel: Short cap (262,144 moves/chain or fixed deterministic pass)
    original = next(r for r in short["grouped"] if r["k"] == 10 and r["arm"] == "original_general")
    wsp_det_pct = wsp["deterministic_controls_on_30k_40k"]["WSP-Det"]["accuracy_pct"]
    wsp_assoc_262k = wsp["grouped"][0]["mean_accuracy_pct"]

    bar_labels = [
        "Original\nCRAM\n(data)",
        "Transformed\nlearned\n(data)",
        "Count\nenergy\n(data)",
        "WSP-Det\n(rules,\nk=0)",
        "WSP-Assoc\n(rules,\nk=0)",
    ]
    values = [
        original["mean_accuracy_pct"],
        learned["grouped"][0]["mean_accuracy_pct"],
        count["grouped"][0]["mean_accuracy_pct"],
        wsp_det_pct,
        wsp_assoc_262k,
    ]
    colors = ["#8997a9", "#447bb7", "#16887d", "#d97724", "#7b4ea3"]
    hatches = ["", "", "", "//", "//"]

    bars = axes[0].bar(np.arange(len(values)), values, color=colors, width=0.62, zorder=3, edgecolor="#17243a", linewidth=0.6)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)

    axes[0].axvline(2.5, color="#9aa6b8", linestyle="--", linewidth=1.0, zorder=2)
    axes[0].text(1.0, 103.5, "Learned / counted from boards", ha="center", fontsize=9.8, color="#273449", weight="bold")
    axes[0].text(3.5, 103.5, "Rule-programmed (k=0)", ha="center", fontsize=9.8, color="#5a377a", weight="bold")

    axes[0].set(
        ylim=(0, 110),
        ylabel="Solved puzzles (%)",
        xticks=np.arange(len(values)),
        xticklabels=bar_labels,
    )
    axes[0].tick_params(axis="x", labelsize=9.6)
    axes[0].set_title("A   At 262,144 moves / chain (or det. pass)", loc="left", fontsize=13.2, pad=14)
    for i, v in enumerate(values):
        axes[0].text(i, v + 2.0, f"{v:.2f}%", ha="center", weight="bold", fontsize=10.3)
    axes[0].grid(axis="y", alpha=0.18, zorder=0)

    # Right panel: Budget curve up to 16,777,216 moves/chain using symlog so 0 unsolved is exact and visible
    axes[1].set_xscale("log")
    axes[1].set_yscale("symlog", linthresh=1.0, linscale=0.65)

    series = [
        (learned["grouped"], "Transformed learned (data)", "#447bb7", "o", "-"),
        (count["grouped"], "No-optimizer counts (data)", "#16887d", "s", "-"),
        (wsp["grouped"], "WSP-Assoc (rule-programmed, k=0)", "#7b4ea3", "D", "--"),
    ]
    for grp, label, color, marker, ls in series:
        xs = [r["cap"] for r in grp]
        ys = [10000.0 - float(np.mean(r["per_seed_solved"])) for r in grp]
        axes[1].plot(xs, ys, marker=marker, linestyle=ls, color=color, lw=2.2, ms=6, label=label, zorder=4)

    axes[1].axhline(0, color="#7b4ea3", alpha=0.25, lw=1.0, zorder=1)
    axes[1].set(
        xlabel="Moves per chain (4 chains × 6 search seeds)",
        ylabel="Unsolved puzzles per 10,000\n(mean across six search seeds; symlog scale)",
        ylim=(-0.18, 3500),
    )
    axes[1].set_yticks([0, 1, 10, 100, 1000])
    axes[1].set_yticklabels(["0", "1", "10", "100", "1,000"])
    axes[1].set_title("B   Longer search budget, fixed energies / weights", loc="left", fontsize=13.2, pad=14)
    axes[1].legend(frameon=False, fontsize=9.8, loc="upper right")
    axes[1].grid(which="major", alpha=0.18)

    last_count = count["grouped"][-1]
    axes[1].annotate(
        f"Counts @ 16.8M: {last_count['mean_accuracy_pct']:.4f}%\n(4.33 unsolved / 10k)",
        xy=(last_count["cap"], 10000.0 - float(np.mean(last_count["per_seed_solved"]))),
        xytext=(2.6e6, 115),
        arrowprops={"arrowstyle": "->", "color": "#16887d", "lw": 1.1},
        fontsize=9.5,
        weight="bold",
        color="#16887d",
    )
    axes[1].annotate(
        "WSP-Assoc @ ≥4.19M: 100.0000%\n(0 / 60,000 unsolved)",
        xy=(4194304, 0.0),
        xytext=(3.0e5, 0.65),
        arrowprops={"arrowstyle": "->", "color": "#7b4ea3", "lw": 1.1},
        fontsize=9.5,
        weight="bold",
        color="#7b4ea3",
    )

    fig.suptitle(
        "Sudoku from simple counts vs rule-programmed Weight-Space Programming",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=18.5,
        weight="bold",
    )
    fig.text(
        0.06,
        0.015,
        "10,000 reserved official-test puzzles (full-all[30000:40000]) · four chains · six search seeds · zero wrong stops across count and WSP arms.\n"
        "Data arms infer constraints from 10 base solutions × 100 symmetry variants (0 supplied rules). WSP arms compile Sudoku rules directly into weights (k=0 training boards, hatched).\n"
        "Stage 1 deduction in WSP-Assoc reduces domains before Gumbel-Potts search; deterministic WSP-Det (14.06%) plateaus on bifurcation puzzles without stochastic search.",
        fontsize=9.4,
        color="#526075",
        linespacing=1.38,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.81, bottom=0.23, wspace=0.31)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"count-energy.{ext}", dpi=190, bbox_inches="tight", facecolor="white",
                    metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)

    prov = {
        "script": "scripts/plot_count_results.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in SOURCES},
        "outputs_sha256": {
            f"assets/figures/count-energy.{ext}": hashlib.sha256((OUT / f"count-energy.{ext}").read_bytes()).hexdigest()
            for ext in ("png", "svg")
        },
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
    }
    atomic_write_text(OUT / "count-provenance.json", json.dumps(prov, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
