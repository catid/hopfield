"""Plot only audited six-seed CRAM points, keeping partial exports explicit."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/data_efficiency"


def main():
    source = OUT / "collection.json"
    d = json.loads(source.read_text())
    complete = d["figure_complete"]
    assert complete == (d["measured_jobs"] == d["expected_jobs"] == 102)
    points = d["points"]
    assert any(p["n_measured"] == 6 for p in points), "No complete six-seed point to plot"
    x = [p["k"] for p in points]
    y = [p["accuracy_pct_median"] if p["n_measured"] == 6 else np.nan for p in points]
    low = [p["accuracy_pct_min"] if p["n_measured"] == 6 else np.nan for p in points]
    high = [p["accuracy_pct_max"] if p["n_measured"] == 6 else np.nan for p in points]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "o-", color="#1b9e77", label="CRAM fresh (seed42 reference)", markersize=4, zorder=5)
    ax.fill_between(x, low, high, color="#1b9e77", alpha=.12)
    for method, color in [("TRM", "#d95f02"), ("EBR", "#7570b3"), ("FRM", "#e7298a"), ("CRAM", "#777777")]:
        pp = [p for p in d["published_reference"]["points"] if p["method"] == method]
        assert all(p["n_seeds"] in (0, 6) for p in pp)
        ax.plot([p["k"] for p in pp], [p["accuracy_pct_median"] if p["n_seeds"] == 6 else np.nan for p in pp],
                "--", color=color, label=f"{method} published", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Unique training puzzles (each with 99 stored variants)")
    ax.set_ylabel("Solved on 10,000 test puzzles (%)")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="lower right")
    title = "Figure 3b: data-efficiency protocol recreation" if complete else f"INCOMPLETE preview: {d['measured_jobs']}/102 measured runs"
    ax.set_title(title)
    fig.text(.5, .01, "Lines: median of six seeds. Fresh band: seed range. Published baselines were not rerun.\n"
             "Fresh and published curves use different sampled training references; missing points remain gaps.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, 1))
    stem = "figure3b" if complete else "figure3b-preview"
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=180)
    plt.close(fig)
    metadata = {"status": "complete" if complete else "incomplete_preview", "figure_complete": complete,
                "measured_jobs": d["measured_jobs"], "expected_jobs": 102,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "plotted_k": [p["k"] for p in points if p["n_measured"] == 6],
                "missing_k": [p["k"] for p in points if p["n_measured"] < 6],
                "scope": d["scope"], "baseline_status": "published only"}
    (OUT / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
