"""Plot the completed matched preconditioning ablation's diagonal costs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figures"
data_path = ROOT / "artifacts/multi_solution/nqueens-preconditioning-results.json"
data = json.loads(data_path.read_text())
assert data["complete"]
(OUT / "nqueens-preconditioning-fresh.json").write_text(json.dumps({
    "source": str(data_path.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "curve_rule": "Mean diagonal offset contrast over two signs for each offset, then mean and sample SD across seeds 0, 1, 2.",
    "rows": [{"n": r["n"], "method": r["method"], "arm": r["arm"],
              "preconditioned": r["preconditioned"],
              "per_seed": [{"seed": s["seed"], "diagonal_offset_contrast": s["diagonal_offset_contrast"]}
                           for s in r["per_seed"]]} for r in data["rows"]]
}, indent=2)+"\n")
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "savefig.dpi":190, "pdf.fonttype":42})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
for ax, n in zip(axes, (8, 10)):
    for preconditioned, color in [(True, "#087e8b"), (False, "#c94a50")]:
        row = next(r for r in data["rows"] if r["n"] == n and r["preconditioned"] == preconditioned)
        contrasts = np.array([[d["contrast_mean"] for d in seed["diagonal_offset_contrast"]] for seed in row["per_seed"]])
        ax.errorbar(np.arange(1, n), contrasts.mean(axis=0), yerr=contrasts.std(axis=0, ddof=1),
                    marker="o", color=color, lw=1.7, ms=4, capsize=2,
                    label="With preconditioning" if preconditioned else "Without preconditioning")
    ax.set(xlabel="Diagonal offset distance", ylabel="Learned interaction contrast",
           title=f"{n}×{n}: {'locally selected MPL/table' if n==8 else 'main PL/Ising recipe'}")
    ax.set_xticks(range(1, n))
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
fig.suptitle("N-Queens: offset-frequency normalization preserves long-diagonal interactions", fontsize=13)
fig.supxlabel("Mean ± sample SD over three fresh seeds. Each matched pair changes only preconditioning; size-8 recipe differs from Table 3.", fontsize=9)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"nqueens-preconditioning-fresh.{ext}", bbox_inches="tight", facecolor="white")
plt.close(fig)
