"""Rebuild Figure 3a from fresh full zero-stop records and labeled published baselines."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"
BINS = np.asarray([0, 1, 2, 4, 8, 16, 32, 64, 10**9])
BUDGETS = np.logspace(6, 12, 121)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true", help="Also save standalone PNG/PDF figures; requires matplotlib")
    a = p.parse_args()
    sys.path.insert(0, str(UP))
    from cram import opcount
    out = ROOT / "artifacts/difficulty"
    data_path = UP / "data/full-test/full-all.npz"
    rating = np.load(data_path)["ratings"][10000:20000]
    published_path = UP / "results/difficulty/sudoku-difficulty.json"
    published = json.loads(published_path.read_text())
    bin_counts = [int(((rating >= lo) & (rating < hi)).sum()) for lo, hi in zip(BINS[:-1], BINS[1:])]
    assert bin_counts == published["bin_counts"]
    assert sum(bin_counts) == 10000
    records, sources = [], []
    for seed in range(3):
        path = out / f"full/test/cram0-hb-s{seed}.json"
        rec = json.loads(path.read_text())
        sig, pp = rec["signature"], rec["per_puzzle"]
        assert sig["lo"] == 10000 and sig["n"] == 412786 and sig["test"] == "zero"
        assert sig["seed"] == seed and sig["scale"] == "hbar" and sig["rel"] == [.40, .31, .45]
        assert sig["C"] == 4 and sig["cap"] == 4e11 and sig["check"] == 4096
        assert np.array_equal(np.asarray(pp["puzzle"]), np.arange(10000, 422786))
        model_path = out / f"full/full-aug-s{seed}/model.pt"
        assert sig["model_sha256"] == sha(model_path)
        dev = json.loads((out / f"full/dev/cram0-hb-s{seed}.json").read_text())
        assert dev["signature"]["test"] == "zero" and dev["summary"]["solved"] >= 9999
        assert dev["summary"]["wrong_stops"] == 0 and dev["signature"]["model_sha256"] == sig["model_sha256"]
        solved = np.asarray(pp["solved"], bool)[:10000]
        stopped = np.asarray(pp["stopped"], bool)[:10000]
        raw = np.asarray(pp["flops"], float)[:10000]
        flop = opcount.cram0_sudoku(raw, sig)
        assert np.isfinite(flop).all() and (flop > 0).all()
        records.append(dict(solved=solved, stopped=stopped, flop=flop))
        sources.append(dict(seed=seed, path=str(path.relative_to(ROOT)), sha256=sha(path),
                            model_sha256=sig["model_sha256"], full_summary=rec["summary"]))
    fresh = {"status": "fresh reproduction", "seeds": 3,
             "solved_pct": [float(100 * r["solved"].mean()) for r in records],
             "mean_flop": [float(r["flop"].mean()) for r in records],
             "wrong_stops": [int((r["stopped"] & ~r["solved"]).sum()) for r in records],
             "budget_failures": [int((~r["stopped"]).sum()) for r in records], "bins": []}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mask = (rating >= lo) & (rating < hi)
        fl = np.concatenate([r["flop"][mask] for r in records])
        fresh["bins"].append({
            "solved_pct": [float(100 * r["solved"][mask].mean()) for r in records] if mask.any() else None,
            "flop_median": float(np.median(fl)) if mask.any() else None,
            "flop_p10": float(np.percentile(fl, 10)) if mask.any() else None,
            "flop_p90": float(np.percentile(fl, 90)) if mask.any() else None,
            "flop_mean": float(fl.mean()) if mask.any() else None})
    fresh["solved_within_budget_pct"] = [[float(100 * (r["solved"] & (r["flop"] <= b)).mean())
                                         for b in BUDGETS] for r in records]
    fresh["hardest_to_easiest_median_compute"] = fresh["bins"][-1]["flop_median"] / fresh["bins"][0]["flop_median"]
    result = {"figure": "3a", "subset": {"full_all_rows": [10000, 20000], "n": 10000,
              "rating_mean": float(rating.mean())}, "test_protocol": "Full 412786 zero-stop search per seed; exact published scheduling",
              "bins": list(zip(BINS[:-1].tolist(), BINS[1:].tolist())), "bin_counts": bin_counts,
              "budgets": BUDGETS.tolist(), "fresh": {"CRAM": fresh},
              "published_reference": {name: {"status": "published; not reproduced locally", **published["models"][name]}
                                      for name in ("CRAM", "FRM", "EBR")},
              "source_records": sources,
              "provenance": {"wrapper_sha256": sha(__file__), "dataset_sha256": sha(data_path),
                             "published_source": str(published_path.relative_to(ROOT)),
                             "published_sha256": sha(published_path),
                             "count_convention": "FullCounter FC2 via pinned opcount.cram0_sudoku; includes start, moves and checks"}}
    (out / "figure3a.json").write_text(json.dumps(result, indent=2) + "\n")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        lines = [("CRAM fresh", fresh, "#1b9e77", "-"),
                 ("CRAM published", published["models"]["CRAM"], "#999999", ":"),
                 ("FRM published", published["models"]["FRM"], "#d95f02", "--"),
                 ("EBR published", published["models"]["EBR"], "#7570b3", "--")]
        x = np.arange(8)
        for label, model, color, style in lines:
            bins = model["bins"]
            axes[0].plot(x, [b["flop_median"] for b in bins], style, color=color, label=label)
            axes[0].fill_between(x, [b["flop_p10"] for b in bins], [b["flop_p90"] for b in bins], color=color, alpha=.1)
            accuracy = np.asarray([b["solved_pct"] for b in bins])
            axes[1].plot(x, np.median(accuracy, axis=1), style, color=color)
            axes[1].fill_between(x, accuracy.min(1), accuracy.max(1), color=color, alpha=.1)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("FLOPs per puzzle (median)")
        axes[0].legend(fontsize=8)
        axes[1].set_ylabel("Solved (%)")
        axes[1].set_xlabel("Sudoku difficulty rating")
        axes[1].set_xticks(x, ["0", "1", "2–3", "4–7", "8–15", "16–31", "32–63", "64+"])
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle("Figure 3a: fresh CRAM and published reference curves")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"figure3a.{ext}", dpi=180)
        plt.close(fig)
    print(json.dumps({"solved_pct": fresh["solved_pct"], "mean_flop": fresh["mean_flop"],
                      "hardest_to_easiest_median_compute": fresh["hardest_to_easiest_median_compute"]}))


if __name__ == "__main__":
    main()
