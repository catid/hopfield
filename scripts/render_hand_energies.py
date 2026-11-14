"""Render Figure 5 hand-set energy arrays from their CPU source audit."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figures"
z = np.load(OUT / "hand-energies-fresh.npz")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 200, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1.45, 1, 1, 1]})

focal = 40
table = np.einsum("j,ab->jab", z["sudoku"][focal], np.eye(9)).reshape(9, 9, 9, 9).transpose(0, 2, 1, 3).reshape(81, 81)
table[36:45, 36:45] = np.nan
red = plt.get_cmap("Reds").copy()
red.set_bad("#4b545c")
ax = axes[0]
im = ax.imshow(table, cmap=red, vmin=0, vmax=1.4, interpolation="nearest")
ax.text(40, 40, "i", color="white", ha="center", va="center", fontsize=13)
for at in np.arange(8.5, 81, 9):
    ax.axhline(at, color="#aaaaaa", lw=.5)
    ax.axvline(at, color="#aaaaaa", lw=.5)
for at in (26.5, 53.5):
    ax.axhline(at, color="#444444", lw=1.2)
    ax.axvline(at, color="#444444", lw=1.2)
ax.set_xticks(np.arange(4, 81, 9), labels=range(1, 10))
ax.set_yticks(np.arange(4, 81, 9), labels=range(1, 10))
ax.set_title("(a) Sudoku: 1 value", fontsize=12)
ax.set(xlabel="Neighbor cell column", ylabel="Neighbor cell row")
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=.75, pad=.05, label="Same scale as learned positive costs")

ax = axes[1]
matrix = z["maze"]
ax.imshow(matrix, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{matrix[i,j]:+.1f}", ha="center", va="center", fontsize=13,
                color="white" if abs(matrix[i,j]) > .5 else "#222222")
ax.set_title("(b) Maze-Unique: 3 values", fontsize=12)
ax.set_xlabel("On-minus-off field\nCenter = constant +1.5", labelpad=10)
ax.set_xticks([])
ax.set_yticks([])

ax = axes[2]
nqfocal = 27
matrix = z["nqueens"][nqfocal].reshape(8, 8).copy()
matrix[3, 3] = np.nan
ax.imshow(matrix, cmap=red, vmin=0, vmax=1.4, interpolation="nearest")
ax.text(3, 3, "i", color="white", ha="center", va="center", fontsize=13)
ax.set_xticks(np.arange(-.5, 8), minor=True)
ax.set_yticks(np.arange(-.5, 8), minor=True)
ax.grid(which="minor", color="#bbbbbb", lw=.5)
ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
ax.set_title("(c) N-Queens: 2 values", fontsize=12)
ax.set_xlabel("Attack pair: +1\nEach queen unary cost: −0.5", labelpad=10)

ax = axes[3]
ax.imshow(z["gc"], cmap=red, vmin=0, vmax=1.4)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{z['gc'][i,j]:.0f}", ha="center", va="center", fontsize=15,
                color="white" if i == j else "#444444")
ax.set_xticks(range(3), labels=["A", "B", "C"])
ax.set_yticks(range(3), labels=["A", "B", "C"])
ax.set_title("(d) Graph coloring: 1 value", fontsize=12)
ax.set_xlabel("Colors at an edge\nEvery non-edge table is zero", labelpad=10)
for ax in axes:
    for spine in ax.spines.values():
        spine.set_color("#777777")
fig.suptitle("Figure 5 reproduction · hand-set energies · no learned parameters", fontsize=15)
fig.supxlabel("Values are reconstructed from the unchanged control implementations and checked against the task rules.", fontsize=10)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"hand-energies-fresh.{ext}", bbox_inches="tight", facecolor="white")
plt.close(fig)
