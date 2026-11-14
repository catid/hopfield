"""Render the fresh learned-coupling inspection using isolated matplotlib."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figures"
d = json.loads((OUT / "learned-couplings-fresh.json").read_text())
z = np.load(OUT / "learned-couplings-fresh.npz")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 200, "pdf.fonttype": 42})


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


fig, axes = plt.subplots(1, 3, figsize=(14.7, 4.9), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
matrix = z["maze_normalized"]
ax = axes[0]
im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.3, vmax=1.3)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center", fontsize=14,
                weight="bold" if abs(matrix[i,j]) > .2 else "normal", color="white" if abs(matrix[i,j]) > .6 else "#222222")
ax.set_xticks(np.arange(-.5, 3), minor=True)
ax.set_yticks(np.arange(-.5, 3), minor=True)
ax.grid(which="minor", color="white", lw=3)
ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
ax.set_title("Maze-Unique: on-minus-off field\nNormalized by mean cardinal coupling", fontsize=11)
ax.set_xlabel("Center = constant, not a self-coupling\nNegative prefers on; positive prefers off", labelpad=12)

S = z["sudoku_pair_tables"]
focal = d["sudoku"]["focal_cell"]
table = S[focal].reshape(9, 9, 9, 9).transpose(0, 2, 1, 3).reshape(81, 81).copy()
ri, ci = divmod(focal, 9)
table[ri * 9:(ri + 1) * 9, ci * 9:(ci + 1) * 9] = np.nan
colors = plt.get_cmap("RdBu_r").copy()
colors.set_bad("#535b62")
norm = TwoSlopeNorm(vmin=-.1, vcenter=0, vmax=1.4)
ax = axes[1]
im = ax.imshow(table, cmap=colors, norm=norm, interpolation="nearest")
ax.text(ci * 9 + 4, ri * 9 + 4, "i", ha="center", va="center", color="white", fontsize=12)
for at in np.arange(8.5, 81, 9):
    ax.axhline(at, color="#aaaaaa", lw=.4)
    ax.axvline(at, color="#aaaaaa", lw=.4)
for at in (26.5, 53.5):
    ax.axhline(at, color="#444444", lw=1.2)
    ax.axvline(at, color="#444444", lw=1.2)
ax.set_xticks(np.arange(4, 81, 9), labels=range(1, 10))
ax.set_yticks(np.arange(4, 81, 9), labels=range(1, 10))
ax.set(xlabel="Neighbor cell column", ylabel="Neighbor cell row")
ax.set_title("Sudoku: learned 9×9 digit-pair costs\nFocal cell i at row 5, column 5", fontsize=11)

ax = axes[2]
im2 = ax.imshow(z["sudoku_pair_mean_same_digit"], cmap=colors, norm=norm, interpolation="nearest")
ax.set_title("Sudoku: every pair of 81 cells\nMean same-digit cost", fontsize=11)
ax.set(xlabel="Cell index (row-major)", ylabel="Cell index (row-major)")
for at in np.arange(8.5, 81, 9):
    ax.axhline(at, color="#777777", lw=.3)
    ax.axvline(at, color="#777777", lw=.3)
fig.colorbar(im2, ax=axes[1:], label="Energy cost (blue: negative, red: positive)", shrink=.8, pad=.015,
             ticks=[-.1, 0, .5, 1, 1.4])
fig.suptitle("Figure 2b reproduction · fresh seed-0 learned weights", fontsize=15)
fig.supxlabel("Sudoku positive support: all 810 rule pairs, no extras. Raw weights also contain 426 negative entries across 253 non-rule pairs.", fontsize=10)
save(fig, "learned-couplings-fresh")

diag = np.diagonal(S, axis1=2, axis2=3)
i, j = np.triu_indices(81, 1)
rule = z["sudoku_rule_graph"][i, j]
entry = diag[i, j]
positive = entry[rule].ravel()
extra = entry[~rule].ravel()
extra = extra[extra != 0]
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), layout="constrained")
axes[0].hist(positive, bins=36, color="#bf454b", edgecolor="white", lw=.4)
axes[0].set(title="7,290 positive rule-conflict entries", xlabel="Same-digit pair cost", ylabel="Coefficient count")
axes[1].hist(extra, bins=32, color="#236ba1", edgecolor="white", lw=.4)
axes[1].set(title="426 extra negative entries on non-rule pairs", xlabel="Same-digit pair cost", ylabel="Coefficient count")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
fig.suptitle("Sudoku seed 0: nonuniform and additional learned costs", fontsize=14)
save(fig, "sudoku-coupling-distributions-fresh")
