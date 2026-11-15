"""Plot the audited H4be comparison, including prior BP context."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/research/foureyes/repeat_diversity/summary.json"
PARENT = ROOT / "results/research/foureyes/fresh_pooled_donor_corrected/summary.json"
DEST = ROOT / "assets/figures"


def main():
    data = json.loads(SOURCE.read_text())
    prior = json.loads(PARENT.read_text())
    groups = {r["arm"]: r for r in data["grouped"]}
    bp = next(r for r in prior["grouped"] if r["arm"] == "bp")
    plt.rcParams.update({"font.size":12, "axes.spines.top":False,
                         "axes.spines.right":False, "svg.fonttype":"none",
                         "svg.hashsalt":"repeat-diversity-h4be"})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 8), gridspec_kw={"width_ratios":[1, 1.2]})
    fig.subplots_adjust(left=.07, right=.97, top=.72, bottom=.35, wspace=.34)
    fig.suptitle("Repeated measurements recover 256/256 graphs", x=.04, y=.95,
                 ha="left", fontsize=23, fontweight="bold")
    fig.text(.04, .87, "600 colorings × 3 noisy responses versus 1,800 colorings × 1 response.", fontsize=15)
    fig.text(.04, .81, "Examined graph cohort, new noise: three gains and zero losses at equal label budgets.", fontsize=12)
    totals = [bp["success"], groups["baseline"]["success"], groups["repeat"]["success"]]
    colors = ["#87919a", "#bd8c61", "#287b83"]
    axes[0].bar(range(3), totals, color=colors, width=.65)
    axes[0].set_xticks(range(3), ["BP AdamW 32\nprior context", "1,800 × 1\nprior LP", "600 × 3\nnew LP"])
    axes[0].set_ylim(0, 282)
    axes[0].set_ylabel("Exact recovered graphs / 256")
    axes[0].axhline(245, color="#456268", ls="--", lw=1)
    for i, n in enumerate(totals):
        axes[0].text(i, n+6, str(n), ha="center", fontweight="bold")
    x = np.arange(4)
    for offset, arm, color, label in zip([-.18, .18], ["baseline", "repeat"], colors[1:],
                                       ["1,800 distinct colorings", "600 repeated colorings"]):
        y = [r["exact"] for r in groups[arm]["families"]]
        axes[1].bar(x+offset, y, width=.34, color=color, label=label)
        for pos, n in zip(x+offset, y):
            axes[1].text(pos, n+1, str(n), ha="center", fontsize=10)
    axes[1].set_xticks(x, ["Cycle", "Cycle²", "Cycle³", "Nine cliques"])
    axes[1].set_ylim(0, 72)
    axes[1].set_ylabel("Exact recovered graphs / 64")
    axes[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.04), ncol=1, fontsize=10)
    for ax in axes:
        ax.grid(axis="y", alpha=.14)
        ax.set_axisbelow(True)
    fig.text(.04, .24, "Both frozen targets pass: ≥245 exact recoveries and ≤4 paired regressions.", fontsize=12)
    fig.text(.04, .18, "All banks retained; every selected result is integral and certified.\n"
             "Same inferred degree and first 600 noisy responses in both LP arms.", fontsize=11.5, linespacing=1.5)
    fig.text(.04, .10, "This is synthetic graph recovery, not puzzle accuracy or fresh-graph confirmation.\n"
             "BP used the prior 1,800-row dataset; BP on repeated rows is unmeasured. Compute is not matched.",
             fontsize=10.5, color="#444", linespacing=1.5)
    fig.text(.04, .035, "Replay checks all 512 selected LP certificates and regenerates 512 latent archives byte-for-byte.",
             fontsize=10.5, color="#444")
    for ext in ["png", "svg"]:
        fig.savefig(DEST / f"repeat-diversity.{ext}", dpi=180, facecolor="white",
                    metadata={"Date":None} if ext == "svg" else None)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = dict(input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE, PARENT, Path(__file__)]},
                   output_sha256={f"repeat-diversity.{ext}":sha(DEST / f"repeat-diversity.{ext}") for ext in ["png", "svg"]})
    (DEST / "repeat-diversity-provenance.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
