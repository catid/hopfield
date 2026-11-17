"""Render N=13 generalized queen+leaper comparison: learned BP AdamW (H3ad) vs zero-training WSP.

Strictly read-only on results/research/loki/h3ad/* (preserving sealed H3ad figures).
Writes assets/figures/wsp-queen-families.{png,svg} and provenance JSON.
"""
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    h3_rel = "results/research/loki/h3ad/summary.json"
    wsp_rel = "results/wsp/multi/queen_leaper_results.json"
    h3 = json.loads((ROOT / h3_rel).read_text(encoding="utf-8"))
    wsp = json.loads((ROOT / wsp_rel).read_text(encoding="utf-8"))
    assert h3["status"] == "validated"

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
        "svg.hashsalt": "hopfield-wsp-queen-families-v1",
    })

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.2), gridspec_kw={"width_ratios": [1.22, 1.0]})

    fam_labels = ["Queen + zebra (2,3)\n(4,096 fresh N=13 inputs)", "Queen + (2,5) leaper\n(4,096 fresh N=13 inputs)"]
    arm_specs = [
        ("BP AdamW, arbitrary (N=12 fit)", "#c44e52", ""),
        ("BP AdamW, permutation (N=12 fit)", "#19876b", ""),
        ("WSP-Det (rules, k=0)", "#d97724", "//"),
        ("WSP-Assoc, arbitrary (rules, k=0)", "#4c72b0", "//"),
        ("WSP-Assoc, permutation (rules, k=0)", "#6a4c93", "//"),
    ]

    offsets = np.linspace(-0.32, 0.32, len(arm_specs))
    bar_w = 0.145

    for i, bk in enumerate(("b0", "b1")):
        hf = h3["family_results"][i]
        wf = wsp["families"][bk]["matched_time_comparison"]

        bp_arb = hf["rows"][0]
        bp_perm = hf["rows"][1]
        w_det = wf["deterministic_control"]
        w_arb = wf["arbitrary"]
        w_perm = wf["permutation"]

        cert_vals = [
            bp_arb["certified_success_pct"],
            bp_perm["certified_success_pct"],
            w_det["certified_success_pct"],
            w_arb["metrics"]["certified_success_pct"]["mean"],
            w_perm["metrics"]["certified_success_pct"]["mean"],
        ]
        cov_vals = [
            bp_arb["coverage_pct"],
            bp_perm["coverage_pct"],
            w_det["coverage_pct"],
            w_arb["metrics"]["coverage_pct"]["mean"],
            w_perm["metrics"]["coverage_pct"]["mean"],
        ]

        axes[0].plot([i - 0.42, i + 0.42], [100.0, 100.0], ls="--", color="#222222", lw=1.2, zorder=2)

        for j, ((_, col, hatch), cv, covv) in enumerate(zip(arm_specs, cert_vals, cov_vals)):
            b_cert = axes[0].bar(i + offsets[j], cv, width=bar_w, color=col, alpha=0.88, zorder=3,
                                 edgecolor="#17243a", linewidth=0.55)
            b_cov = axes[1].bar(i + offsets[j], covv, width=bar_w, color=col, alpha=0.88, zorder=3,
                                edgecolor="#17243a", linewidth=0.55)
            if hatch:
                b_cert[0].set_hatch(hatch)
                b_cov[0].set_hatch(hatch)
            axes[0].text(i + offsets[j], cv + 1.8, f"{cv:.1f}%", ha="center", va="bottom", fontsize=8.3, weight="bold")
            axes[1].text(i + offsets[j], covv + 1.8, f"{covv:.1f}%", ha="center", va="bottom", fontsize=8.3, weight="bold")

    axes[0].set_ylim(0, 114)
    axes[0].set_ylabel("Certified valid completion (%)")
    axes[0].set_title("A   Certified completion at matched GPU solve time", loc="left", fontsize=12.6, pad=12)

    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Valid solution coverage Cov@20 (%)")
    axes[1].set_title("B   Multi-solution diversity (Cov@20 across 20 chains)", loc="left", fontsize=12.6, pad=12)

    for ax in axes:
        ax.set_xlim(-0.52, 1.52)
        ax.set_xticks([0, 1], fam_labels, fontsize=10.2)
        ax.grid(axis="y", alpha=0.18, zorder=0)

    handles = [
        Patch(facecolor=col, edgecolor="#17243a", hatch=hatch, label=lbl)
        for lbl, col, hatch in arm_specs
    ] + [Line2D([0], [0], ls="--", color="#222222", lw=1.2, label="Exact F-capacity ceiling (100%)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.048, 0.895), ncol=3, frameon=False, fontsize=9.2)

    fig.suptitle(
        "Permutation proposals elevate both learned energies and rule-programmed WSP on N=13 queen variants",
        x=0.052,
        y=0.975,
        ha="left",
        fontsize=15.0,
        weight="bold",
    )
    fig.text(
        0.052,
        0.015,
        "Shared N=13 cohorts (4,096 inputs/family, 20 chains × 3 seeds, identical initial permutation boards, zero wrong stops across all arms).\n"
        "Learned BP AdamW fits complete N=12 universes (~21.5s matched GPU solve). WSP compiles attack offsets directly (k=0 boards; ~5.0s matched GPU solve, time ratios 1.007× and 1.039×).\n"
        "Permutation swaps raise WSP certified completion from 86.73%→100.00% (+13.27 pp) and 90.23%→100.00% (+9.77 pp) while nearly doubling Cov@20 (50.9%→92.6%, 38.7%→87.8%).",
        fontsize=9.0,
        color="#526075",
        linespacing=1.35,
    )
    fig.subplots_adjust(left=0.058, right=0.985, top=0.77, bottom=0.20, wspace=0.20)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"wsp-queen-families.{ext}", dpi=185, bbox_inches="tight", facecolor="white",
                    metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)

    prov = {
        "script": "scripts/plot_wsp_queen_families.py",
        "script_sha256": sha(Path(__file__)),
        "inputs_sha256": {
            h3_rel: sha(ROOT / h3_rel),
            wsp_rel: sha(ROOT / wsp_rel),
        },
        "outputs_sha256": {
            f"assets/figures/wsp-queen-families.{ext}": sha(OUT / f"wsp-queen-families.{ext}")
            for ext in ("png", "svg")
        },
    }
    atomic_write_text(OUT / "wsp-queen-families-provenance.json", json.dumps(prov, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
