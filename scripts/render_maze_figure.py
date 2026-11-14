"""Render fresh Figure 6 data with the task-owned matplotlib environment."""
from pathlib import Path
import json
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figures"
d = json.loads((OUT / "maze-hard-figure-fresh.json").read_text())
tok = np.asarray(d["tokens"])
label = np.asarray(d["label"], bool)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 180, "pdf.fonttype": 42})


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


energy = np.asarray(d["energy"] + d["energy_after_stop"])
fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
ax.plot(np.arange(len(energy)), energy, color="#087e8b", lw=2)
ax.axvline(d["descent_steps"], color="#ae3e42", ls="--", lw=1.2,
           label=f"Stopping check: step {d['descent_steps']}")
ax.set(xlabel="Damped descent step", ylabel="Answer energy, including entropy",
       title="Maze-Hard: fresh seed-0 model, test maze 0")
ax.legend(frameon=False, loc="upper right")
ax.grid(alpha=.18)
ax.text(.98, .32, "144 inference steps + 48 validation steps\nRounded solution remains unchanged",
        ha="right", va="top", transform=ax.transAxes, fontsize=9, color="#555555")
fig.supxlabel("Energy is nonincreasing through the stop; post-stop float32 fluctuations ≤ 4.89×10⁻⁴.",
              fontsize=9, color="#555555")
save(fig, "maze-hard-energy-fresh")

fig, axes = plt.subplots(2, 4, figsize=(15.8, 8.5), layout="constrained")
cmap = ListedColormap(["#202a34", "#f5f5f2", "#008c9b", "#ed9465"])


def frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def endpoints(ax):
    for token, text, color in [(2, "S", "#a32443"), (3, "G", "#6336a7")]:
        i, j = np.argwhere(tok == token)[0]
        ax.scatter([j], [i], s=60, c=color, marker="s", edgecolor="white", lw=.5, zorder=5)
        ax.text(j, i, text, color="white", ha="center", va="center", fontsize=6, weight="bold", zorder=6)


def board(ax, value, title):
    value = np.asarray(value, bool)
    colors = np.where(tok == 0, 0, 1)
    colors[value & label] = 2
    colors[value & ~label] = 3
    ax.imshow(colors, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    frame(ax)
    endpoints(ax)
    ax.set_title(title, fontsize=11)


logz = np.array([[np.nan if v is None else v for v in row] for row in d["log_z"]])
ax = axes[0, 0]
hidden = plt.get_cmap("viridis").copy()
hidden.set_bad("#202a34")
im = ax.imshow(logz, cmap=hidden, interpolation="nearest")
frame(ax)
endpoints(ax)
ax.set_title(f"Hidden log z: {d['hidden_steps']} settle steps", fontsize=11)
fig.colorbar(im, ax=ax, shrink=.72, pad=.02)

ax = axes[0, 1]
ax.imshow(np.where(tok > 0, 1., np.nan), cmap=ListedColormap(["#f5f5f2"]), interpolation="nearest")
ax.set_facecolor("#202a34")
offset = np.asarray(d["pointer_offset"])
rr, cc = np.nonzero(tok > 0)
ax.quiver(cc, rr, offset[rr, cc, 1], offset[rr, cc, 0], color="#087e8b", angles="xy",
          scale_units="xy", scale=1.15, width=.004, headwidth=3.5, headlength=4)
frame(ax)
endpoints(ax)
ax.set_title("Largest-attention neighbor", fontsize=11)

for ax, step in zip([axes[0, 2], axes[0, 3], axes[1, 0], axes[1, 1]], (0, 16, 32, 64)):
    board(ax, d["boards"][str(step)], f"Rounded answer: step {step}")
board(axes[1, 2], d["final"], f"Stopped answer: step {d['descent_steps']}")
board(axes[1, 3], d["label"], "Exact target (matched)")
fig.suptitle("Maze-Hard Figure 6 reproduction · fresh seed 0 · held-out maze 0", fontsize=16)
fig.legend(handles=[Patch(color="#202a34", label="Wall"), Patch(color="#f5f5f2", ec="#bbbbbb", label="Unselected open cell"),
                    Patch(color="#008c9b", label="Selected target cell"), Patch(color="#ed9465", label="Selected non-target cell")],
           loc="outside lower center", ncols=4, frameon=False)
save(fig, "maze-hard-panels-fresh")
(OUT / "plotting-environment.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),
    "matplotlib":matplotlib.__version__,"numpy":np.__version__,"source":"maze-hard-figure-fresh.json",
    "command":"artifacts/plotting-venv/bin/python scripts/render_maze_figure.py"},indent=2)+"\n")
