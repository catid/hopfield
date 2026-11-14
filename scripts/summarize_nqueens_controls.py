"""Summarize completed preconditioning controls from saved chains, no inference."""
from pathlib import Path
import hashlib
import json
import sys
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"
OUT = ROOT / "artifacts/multi_solution"
sys.path.insert(0, str(UP))
from cram.nqueens_data import load_split, valid


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reference_path = UP / "results/nqueens-results.json"
    reference = json.loads(reference_path.read_text())["offset_energy"]["sizes"]
    rows = []
    for n in (8, 10):
        inputs, labels, stored = load_split(n, "test")
        counts = np.load(OUT / f"nq{n}-solution-counts.npz")
        inv, total = counts["inverse"], counts["counts"]
        caps = np.minimum(total, 20)
        assert np.array_equal(total[inv], stored.reshape(-1))
        head = reference[str(n)]["headline"]
        method, arm = head["method"], head["arm"]
        for precond in (True, False):
            paper = reference[str(n)]["rows"][f"{method}-{arm}"] if precond else reference[str(n)]["ablation_no_precond"]
            per_seed = []
            for seed in (0, 1, 2):
                relative = f"final/{method}-{arm}-s{seed}" if precond else f"ablation/{method}-{arm}-noprecond-s{seed}"
                folder = UP / f"experiments/runs/NQ/nq11/n{n}" / relative
                if not (folder / "final.json").exists():
                    continue
                rec = json.loads((folder / "final.json").read_text())
                assert rec["selection"]["config"]["precond"] == precond
                record = np.load(folder / "records.npz")
                boards = np.unpackbits(record["boards"], axis=-1)[..., :n*n].astype(bool)
                ok = valid(boards, n) & (boards >= inputs[:, None]).all(axis=-1)
                keep = np.argmin(record["energy"], axis=1)
                found = np.array([len({b.tobytes() for b in boards[k, ok[k]]}) for k in range(len(inputs))])
                clipped = np.minimum(found, caps[inv])
                found_u = np.bincount(inv, weights=clipped, minlength=len(total)) / np.bincount(inv, minlength=len(total))
                wrong = int((record["stopped"] & ~ok).sum())
                capped = int((~record["stopped"]).sum())
                assert wrong == rec["test"]["wrong_stops"] and capped == rec["test"]["budget_failures"]
                info = rec["fit"]
                off, contrast = np.asarray(info["offsets"]["offsets"]), np.asarray(info["offsets"]["contrast"])
                diagonal = []
                for distance in range(1, n):
                    mask = (off[:, 0] == distance) & (np.abs(off[:, 1]) == distance)
                    assert mask.sum() == 2
                    diagonal.append({"distance": distance, "unordered_pairs_per_offset": (n-distance)**2,
                        "directed_pairs_per_offset": 2*(n-distance)**2,
                        "contrast_mean": float(contrast[mask].mean()), "contrast_both_signs": contrast[mask].tolist()})
                per_seed.append({"seed": seed, "source": str(folder.relative_to(ROOT)),
                    "model_sha256": sha(folder / "model.pt"), "records_sha256": sha(folder / "records.npz"),
                    "acc_n1": 100*float(ok[:, 0].mean()),
                    "acc_n20_lowest_energy": 100*float(ok[np.arange(len(inputs)), keep].mean()),
                    "cov20_per_puzzle": 100*float(found_u.sum()/caps.sum()),
                    "diagnostic_acc_majority20": 100*rec["test"]["acc_majority20"],
                    "diagnostic_gram_row_mean_cov20": 100*rec["test"]["cov20"],
                    "wrong_stops_20chains": wrong, "cap_failures_20chains": capped,
                    "fit_cold_seconds": info.get("cold_seconds", info["fit_seconds"]),
                    "parameters_learned": info["params_learned"], "parameters_nonzero": info["params_nonzero"],
                    "selection": rec["selection"], "diagonal_offset_contrast": diagonal})
            if not per_seed:
                continue
            metrics = {}
            keys = ["acc_n1", "acc_n20_lowest_energy", "cov20_per_puzzle", "diagnostic_acc_majority20",
                    "diagnostic_gram_row_mean_cov20", "wrong_stops_20chains", "cap_failures_20chains"]
            for key in keys:
                values = [r[key] for r in per_seed]
                metrics[key] = {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values)>1 else None,
                                "per_seed": values}
            mapping = {"acc_n1": "acc_n1", "diagnostic_acc_majority20": "acc_majority20",
                       "diagnostic_gram_row_mean_cov20": "cov20", "wrong_stops_20chains": "wrong_stops_20chains",
                       "cap_failures_20chains": "cap_failures_20chains"}
            rows.append({"n": n, "method": method, "arm": arm, "preconditioned": precond,
                "seeds": [r["seed"] for r in per_seed], "complete_seeds": len(per_seed)==3,
                "test_rows": len(inputs), "distinct_test_inputs": len(total), "metrics": metrics,
                "paper_reference": {key: paper[src] for key, src in mapping.items()},
                "mean_differences_against_paper": {key: metrics[key]["mean"]-paper[src]["mean"] for key,src in mapping.items()},
                "metric_scope": "Published ablation records retain majority-vote N20 and GRAM row-mean Cov20; compare these only with matching diagnostic columns. Lowest-energy N20 and per-puzzle coverage are additionally recomputed from saved chains.",
                "size8_scope": "Matched local MPL/table control and ablation. The Table3 headline instead transfers the n10 PL/Ising recipe.",
                "per_seed": per_seed})
    target = OUT / "nqueens-preconditioning-results.json"
    target.write_text(json.dumps({"reference_file": str(reference_path.relative_to(ROOT)),
        "reference_sha256": sha(reference_path), "rows": rows,
        "complete": len(rows)==4 and all(r["complete_seeds"] for r in rows)}, indent=2)+"\n")
    for r in rows:
        print(json.dumps({k:r[k] for k in ["n","method","arm","preconditioned","seeds","complete_seeds","metrics"]}))


if __name__ == "__main__":
    main()
