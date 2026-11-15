"""Matched observation-budget graph recovery from audited H4ba/H4bb outputs."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/figures"
matplotlib.rcParams["svg.hashsalt"] = "h4bb-pooled-donor"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [ROOT / f"results/research/foureyes/{name}/summary.json"
             for name in ("larger_donor", "pooled_donor")]
    parent, pooled = [json.loads(p.read_text()) for p in paths]
    old = {r["index"]: r for r in parent["rows"]}
    rows = pooled["rows"]
    assert len(old) == len(rows) == 256
    assert all(r["parent_exact"] == (old[r["index"]]["classification"] == "success") for r in rows)
    gains = sum(r["classification"] == "success" and not r["parent_exact"] for r in rows)
    losses = sum(r["classification"] != "success" and r["parent_exact"] for r in rows)
    families = ("cycle81", "power2", "power3", "cliques9")
    counts = [[sum(r["family"] == f and (r["parent_exact"] if arm == 0 else r["classification"] == "success")
                   for r in rows) for f in families] for arm in (0, 1)]
    assert [sum(c) for c in counts] == [229, 254]
    assert (gains, losses) == (25, 0)
    colors = ["#8496a0", "#148b91"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    totals = [sum(c) for c in counts]
    ax.bar([0, 1], totals, width=.58, color=colors)
    for x, n in enumerate(totals):
        ax.text(x, n + 4, f"{n}/256", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1], ["Donor + masked target", "Pool all counts"])
    ax.set_ylim(0, 280)
    ax.set_ylabel("Exact graphs / 256 examined banks")
    ax.set_title(f"Same 1,800 observations: {gains} gains, {losses} losses", fontsize=11)
    ax = axes[1]
    x = np.arange(4)
    for i, (c, label) in enumerate(zip(counts, ["Donor + masked target", "Pooled counts"])):
        positions = x + (i - .5) * .35
        ax.bar(positions, c, width=.33, color=colors[i], label=label)
        for pos, n in zip(positions, c):
            ax.text(pos, n + .8, str(n), ha="center", fontsize=10)
    ax.set_xticks(x, ["Cycle", "Square", "Cube", "Nine cliques"])
    ax.set_ylim(0, 72)
    ax.set_ylabel("Exact graphs / 64 per family")
    ax.set_title("All four synthetic graph families", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Pooling the same noisy data recovers 254/256 graphs", x=.02, ha="left", fontsize=15)
    fig.text(.02, .085, "Both fixed targets pass: at least 245 recoveries, at most 4 losses. Two nine-clique failures remain.", fontsize=10)
    fig.text(.02, .035, "Examined synthetic banks; degree inherited from the 600-row prefix. Graph recovery, not puzzle solving. BP AdamW not run.", fontsize=9)
    fig.tight_layout(rect=[0, .14, 1, .93])
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"pooled-donor.{ext}", dpi=160,
                    metadata={"Date": None} if ext == "svg" else {"Software": "Matplotlib"})
    plt.close(fig)
    provenance = dict(renderer_sha256=sha(Path(__file__)),
                      source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
                      scope="Same 1800 observed rows, all 256 examined banks; inherited target degree; no equal-compute claim.")
    (OUT / "pooled-donor-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
