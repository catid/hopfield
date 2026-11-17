"""Render 3-cohort 180x180 dense-maze complete-cost comparison with WSP-Wavefront.

Strictly read-only on results/research/work/* (preserving sealed H2bo/H2bp/H2bq figures).
Writes assets/figures/wsp-maze-endpoint-cost.{png,svg} and provenance JSON.
"""
from fractions import Fraction
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
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "text.color": "#17243a",
        "axes.labelcolor": "#273449",
        "xtick.color": "#273449",
        "ytick.color": "#273449",
        "axes.edgecolor": "#b7c1cf",
        "svg.hashsalt": "hopfield-wsp-maze-three-cohorts-v1",
    })

    inputs: dict[str, str] = {}
    wsp_rel = "results/wsp/maze/maze_180x180_cohorts_summary.json"
    wsp_path = ROOT / wsp_rel
    inputs[wsp_rel] = sha(wsp_path)
    wsp_data = json.loads(wsp_path.read_text(encoding="utf-8"))

    cohort_specs = [
        ("H2bo", "bp-first-fresh-confirmation", "Corner endpoints (H2bo)"),
        ("H2bp", "bp-first-endpoint-shift", "Uniform open-cell endpoints (H2bp)"),
        ("H2bq", "bp-first-far-placement", "Opposite-region endpoints (H2bq)"),
    ]

    cohorts = []
    for hid, name, title in cohort_specs:
        d = ROOT / "results/research/work" / name
        for fn in ("audit.json", "protocol.json"):
            p = d / fn
            inputs[str(p.relative_to(ROOT))] = sha(p)
        audit = json.loads((d / "audit.json").read_text(encoding="utf-8"))
        assert audit["status"] == "passed" and audit["fresh_independent_inputs"] == 4096

        cascade_savings = []
        for pair in range(4):
            times = {}
            for arm in ("LP96", "BP_geometry_LP96"):
                rp = d / "runs" / f"pair{pair}-{arm}" / "result.json"
                inputs[str(rp.relative_to(ROOT))] = sha(rp)
                rec = json.loads(rp.read_text(encoding="utf-8"))
                total = rec["model_readiness_seconds"] + sum(r["timing"]["answer_seconds"] for r in rec["rows"])
                assert abs(total - rec["complete_answer_seconds"]) < 1e-8
                times[arm] = total
            sav = 1 - Fraction(times["BP_geometry_LP96"]) / Fraction(times["LP96"])
            cascade_savings.append(100.0 * float(sav))
        cascade_med = 100.0 * float(Fraction(audit["cost_endpoint"]["exact_median_fraction"]))

        wc = wsp_data["cohorts"][hid]["timing_4_balanced_pairs"]["comparison_vs_published_LP96_12288"]
        wsp_savings = [100.0 * float(p["saving"]) for p in wc["pairs"]]
        wsp_med = 100.0 * float(wc["median_saving"])
        wsp_speedup = float(wc["median_speedup_ratio"])
        wsp_med_s = float(wsp_data["cohorts"][hid]["timing_4_balanced_pairs"]["matched_3seed_B32_median_seconds"])

        cohorts.append({
            "id": hid,
            "title": title,
            "cascade_savings": cascade_savings,
            "cascade_med": cascade_med,
            "wsp_savings": wsp_savings,
            "wsp_med": wsp_med,
            "wsp_speedup": wsp_speedup,
            "wsp_med_s": wsp_med_s,
        })

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.9), sharey=True)
    x = np.arange(4)
    w = 0.36

    for ax, c in zip(axes, cohorts):
        c_col = "#087f8c" if c["cascade_med"] >= 0 else "#be4d25"
        w_col = "#6a4c93"

        ax.axhline(0, color="#444444", linewidth=1.0, zorder=2)
        b1 = ax.bar(x - w / 2, c["cascade_savings"], width=w, color=c_col, alpha=0.86, zorder=3, edgecolor="#17243a", linewidth=0.5)
        b2 = ax.bar(x + w / 2, c["wsp_savings"], width=w, color=w_col, alpha=0.88, hatch="//", zorder=3, edgecolor="#17243a", linewidth=0.5)

        ax.axhline(c["cascade_med"], color=c_col, linestyle="--", linewidth=1.4, zorder=2)
        ax.axhline(c["wsp_med"], color=w_col, linestyle=":", linewidth=1.6, zorder=2)

        for xi, cv in zip(x - w / 2, c["cascade_savings"]):
            ypos = cv + 1.8 if cv >= 0 else cv - 2.2
            va = "bottom" if cv >= 0 else "top"
            ax.text(xi, ypos, f"{cv:+.1f}%", ha="center", va=va, fontsize=8.5, color=c_col, weight="bold")
        for xi, wv in zip(x + w / 2, c["wsp_savings"]):
            ax.text(xi, wv + 1.8, f"+{wv:.1f}%", ha="center", va="bottom", fontsize=8.5, color=w_col, weight="bold")

        casc_str = (
            f"Cascade median: +{c['cascade_med']:.2f}% saving"
            if c["cascade_med"] >= 0
            else f"Cascade median: {-c['cascade_med']:.2f}% slower"
        )
        wsp_str = f"WSP-Wavefront: +{c['wsp_med']:.2f}% ({c['wsp_speedup']:.1f}×, {c['wsp_med_s']:.1f}s)"
        ax.text(0.02, 0.985, wsp_str, transform=ax.transAxes, va="top", fontsize=9.3, color=w_col, weight="bold")
        ax.text(0.02, 0.915, casc_str, transform=ax.transAxes, va="top", fontsize=9.3, color=c_col, weight="bold")

        ax.set_title(c["title"], loc="left", fontsize=11.5, pad=10)
        ax.set_xticks(x, ["Pair 0", "Pair 1", "Pair 2", "Pair 3"], fontsize=9.5)
        ax.set_ylim(-28, 128)
        ax.grid(axis="y", alpha=0.18, zorder=0)

    axes[0].set_ylabel("Complete answer-cost saving vs standalone LP96 (%)")
    legend_handles = [
        Patch(facecolor="#087f8c", edgecolor="#17243a", label="BP-first geometry cascade (learned, corner/far saving)"),
        Patch(facecolor="#be4d25", edgecolor="#17243a", label="BP-first geometry cascade (learned, uniform slowdown)"),
        Patch(facecolor="#6a4c93", edgecolor="#17243a", hatch="//", label="WSP-Wavefront (rule-programmed, k=0, 12,288 B=32 evals)"),
    ]
    fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.045, 0.905), ncol=3, frameon=False, fontsize=9.5)

    fig.suptitle(
        "Dense 180×180 mazes (80% loops): learned cascade routing vs rule-programmed WSP-Wavefront",
        x=0.048,
        y=0.975,
        ha="left",
        fontsize=15.0,
        weight="bold",
    )
    fig.text(
        0.048,
        0.015,
        "Each cohort: 4,096 fresh 180×180 mazes · 12,288 model-input evaluations (B=32) · 4 balanced paired replays · LP96, Cascade, and WSP-Wavefront each solve 4,096/4,096 (0 wrong halts).\n"
        "Complete costs charge weight/model readiness, H2D transfer, GPU solve, acceptance checks, and D2H return. WSP-Wavefront compiles 5 geodesic/tie weights (k=0) and remains fast on uniform endpoints.\n"
        "Hardware disclosure: WSP measured on local RTX PRO 6000 Blackwell Max-Q; archived LP96/Cascade timings ran on Work RTX PRO 6000 Blackwell (single-seed B=256 WSP takes 1.83–3.68s).",
        fontsize=8.9,
        color="#526075",
        linespacing=1.35,
    )
    fig.subplots_adjust(left=0.052, right=0.99, top=0.80, bottom=0.20, wspace=0.10)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"wsp-maze-endpoint-cost.{ext}", dpi=180, bbox_inches="tight", facecolor="white",
                    metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)

    prov = {
        "script": "scripts/plot_wsp_maze_endpoint_cost.py",
        "script_sha256": sha(Path(__file__)),
        "inputs_sha256": inputs,
        "outputs_sha256": {
            f"assets/figures/wsp-maze-endpoint-cost.{ext}": sha(OUT / f"wsp-maze-endpoint-cost.{ext}")
            for ext in ("png", "svg")
        },
        "cohorts": cohorts,
    }
    atomic_write_text(OUT / "wsp-maze-endpoint-cost-provenance.json", json.dumps(prov, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
