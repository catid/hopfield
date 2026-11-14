"""Independently validate complete fresh Sudoku records and reconstruct paper FLOPs.

No inference and no writes to upstream source/results. Requires all three seeds
for the requested method; incomplete runs raise rather than masquerade as results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"
OUT = ROOT / "artifacts/sudoku_audit"
sys.path[:0] = [str(UP), str(UP / "experiments")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    x = np.asarray(values, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None, "per_seed": x.tolist()}


def distribution(x):
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p99": float(np.quantile(x, .99)), "p999": float(np.quantile(x, .999)), "max": float(x.max())}


def audit(method, seeds=(0, 1, 2)):
    OUT.mkdir(exist_ok=True)
    learned = method == "learned"
    base = UP / "experiments/runs/CRAM-S/full" if learned else UP / "experiments/runs/HAND-E1"
    if learned:
        counter_path = ROOT / "artifacts/sudoku_operation_counts.json"
        counters = json.loads(counter_path.read_text())
        overhead = counters["sudoku_general_overheads_s0"]
        start, check = overhead["init_per_puzzle"]["total"], overhead["check_per_puzzle"]["total"]
        from cram import opcount
        published_components = opcount.components()["cram0_sudoku"]
        assert start == published_components["start"]["total"]
        assert check == published_components["check"]["total"]
    else:
        counter_path = OUT / "cram-hand-sudoku.json"
        counters = json.loads(counter_path.read_text())
        start = counters["counted"]["start_per_puzzle"]["total"]
        check = counters["counted"]["check_per_puzzle"]["total"]
    rows, all_paper = [], []
    for seed in seeds:
        path = base / (f"test/cram0-hb-fit-s{seed}.json" if learned else f"sudoku-test-s{seed}.json")
        devpath = base / (f"dev/cram0-hb-fit-s{seed}.json" if learned else f"sudoku-dev-s{seed}.json")
        rec, dev = json.loads(path.read_text()), json.loads(devpath.read_text())
        sig, pp = rec["signature"], rec["per_puzzle"]
        assert sig["seed"] == seed and sig["C"] == 4 and sig["check"] == 4096 and sig["cap"] == 4e11
        assert sig["lo"] == 10000 and sig["n"] == 412786
        ids = np.asarray(pp["puzzle"])
        assert np.array_equal(ids, np.arange(10000, 422786)), "full test IDs are incomplete or misordered"
        assert len(np.unique(ids)) == 412786
        assert dev["summary"]["n"] == dev["summary"]["solved"] == 10000
        assert dev["summary"]["wrong_stops"] == dev["summary"]["budget_failures"] == 0
        for key, value in sig.items():
            if key not in {"data", "lo", "n"}:
                assert dev["signature"][key] == value, (seed, key)
        model_path = None
        if learned:
            assert sig["data"] == "full-all" and sig["split"] == seed and sig["test"] == "fit"
            assert sig["model"] == "full-aug" and sig["scale"] == "hbar"
            assert sig["rel"] == [.4, .31, .45] and sig["anneal"][2] == 524288
            model_path = base / f"full-aug-s{seed}/model.pt"
            assert sig["model_sha256"] == sha(model_path)
            move = counters[f"sudoku_general_move_D{sig['D']}"]["total"]
            assert move == published_components["move"][str(sig["D"])]["total"]
        else:
            assert sig["data"] == "test" and sig["restart"] == 0 and sig["focused"] and sig["sparse"]
            assert sig["test"] == "zero" and sig["tau"] == 0 and sig["fl_move"] == 124.0
            move = counters["counted"]["move_per_chain"]["total"]
        raw = np.asarray(pp["flops"], float)
        solved, stopped = np.asarray(pp["solved"], bool), np.asarray(pp["stopped"], bool)
        assert raw.shape == solved.shape == stopped.shape == ids.shape
        assert np.isfinite(raw).all() and (raw >= 0).all()
        counts = {"n": len(raw), "solved": int(solved.sum()), "stopped": int(stopped.sum()),
                  "wrong_stops": int((stopped & ~solved).sum()), "budget_failures": int((~stopped).sum()),
                  "unstopped_but_right": int((~stopped & solved).sum())}
        assert all(rec["summary"][k] == v for k, v in counts.items())
        raw_stats = distribution(raw)
        assert all(np.isclose(rec["summary"][f"{k}_flop"], v, rtol=1e-12, atol=1e-6) for k, v in raw_stats.items())
        steps = raw / (sig["fl_move"] * sig["C"])
        assert np.allclose(steps, np.rint(steps), rtol=0, atol=1e-6), "count reconstruction assumes no board restarts"
        steps = np.rint(steps)
        paper = steps * sig["C"] * move + start + np.ceil(steps / sig["check"]) * check
        if learned:
            assert np.array_equal(paper, opcount.cram0_sudoku(raw, sig)), "independent reconstruction disagrees"
        else:
            expected = next(r for r in counters["per_seed"] if r["seed"] == seed)
            assert expected["new_mean"] == float(paper.mean())
        end_steps = int(np.ceil(pp["T_cap"] / sig["check"])) * sig["check"]
        cap = end_steps * sig["C"] * move + start + np.ceil(end_steps / sig["check"]) * check
        assert steps.max() <= end_steps and paper.max() <= cap
        rows.append({"seed": seed, "record": str(path.relative_to(ROOT)), "record_sha256": sha(path),
                     "dev_record_sha256": sha(devpath), "signature": sig, **counts,
                     "solved_percent": 100 * float(solved.mean()), "raw_proxy_flops": raw_stats,
                     "paper_flops": distribution(paper),
                     "move_count": {"mean": float(steps.mean()), "max": int(steps.max()),
                                    "T_cap": pp["T_cap"], "rounded_endpoint": end_steps},
                     "paper_cap": float(cap), "counted_move_per_chain": move,
                     "counted_start_per_puzzle": start, "counted_check_per_puzzle": check,
                     "seconds": pp["seconds"], "resumes": rec["resumes"],
                     "gpu": rec["gpu"], "ids_complete_unique": True,
                     "summary_recomputed_from_per_puzzle_flags": True,
                     "all_test_settings_match_passed_dev": True})
        all_paper.append(paper)
    reference_path = UP / "results/op-counts.json"
    paper_row = next(r for r in json.loads(reference_path.read_text())["cells"]
                     if r["benchmark"] == "sudoku-extreme" and r["row"] == ("cram-0-fit" if learned else "hand-energy"))
    paper_mean = float(np.mean([r["paper_flops"]["mean"] for r in rows]))
    complete = list(seeds) == [0, 1, 2]
    result = {"method": "trained CRAM" if learned else "hand-set CRAM", "seeds": list(seeds),
              "complete": complete, "unique_test_puzzles": 412786, "seed_puzzle_evaluations": 412786 * len(seeds),
              "counts_pooled_over_seeds": {k: sum(r[k] for r in rows) for k in
                  ["solved", "stopped", "wrong_stops", "budget_failures", "unstopped_but_right"]},
              "solved_percent": stats([r["solved_percent"] for r in rows]),
              "raw_proxy_mean_flops": stats([r["raw_proxy_flops"]["mean"] for r in rows]),
              "paper_mean_flops": stats([r["paper_flops"]["mean"] for r in rows]),
              "paper_p99_flops_mean_of_seed_quantiles": stats([r["paper_flops"]["p99"] for r in rows]),
              "paper_pooled_distribution": distribution(np.concatenate(all_paper)),
              "counter_record": str(counter_path.relative_to(ROOT)), "counter_record_sha256": sha(counter_path),
              "published_reference": {"file": str(reference_path.relative_to(ROOT)), "sha256": sha(reference_path),
                                      "row": paper_row["row"], "paper_mean_flops": paper_row["new"],
                                      "provenance": "Imported published result, not a local measurement."},
              "paper_mean_flops_relative_difference_percent": 100 * (paper_mean / paper_row["new"] - 1),
              "formula": "steps = raw_flops / (C * fl_move); paper = steps * C * move + start + ceil(steps / 4096) * check",
              "convention": "Arithmetic-only FullCounter: multiply-add 2, elementary arithmetic/transcendentals 1; comparison/selection/mask/movement/indexing/RNG excluded. Logical counts charge the first passing move, although kernels finish the current check chunk.",
              "scope": "Recomputes aggregate correctness flags and counts from saved records, not independent board verification. Seed evaluations reuse the same 412786 puzzles; only seeds0,1,2 together are the complete three-seed result.",
              "per_seed": rows}
    target = OUT / (f"{method}-full-test-audit.json" if complete else f"{method}-test-audit-s{'-'.join(map(str, seeds))}.json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    if learned and complete:
        import compute_curve_export as C
        C.OUT = OUT / "learned-anytime-curves.json"
        C.main()
    print(json.dumps({"out": str(target), "solved_percent": result["solved_percent"],
                      "pooled_counts": result["counts_pooled_over_seeds"], "paper_mean_flops": result["paper_mean_flops"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["learned", "hand"])
    parser.add_argument("--seeds", type=int, choices=[0, 1, 2], nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    assert sorted(set(args.seeds)) == args.seeds
    audit(args.method, args.seeds)
