"""Frozen Table-3 free-pair CRAM fit and one test pass, with timing on seed 0.

Uses the unchanged official fit/score routines. The selected settings come from
the published validation record, never the local test set. Only seed 0 repeats
seven full fits for the table's warm timing; seeds 1/2 are quality replications.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"
sys.path[:0] = [str(UP), str(UP / "experiments")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(event, **kw):
    print(json.dumps({"event": event, "unix": time.time(), **kw}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, choices=[8, 10], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    args = parser.parse_args()
    os.chdir(UP / "experiments")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    import nqueens_free as N
    from cram import cram_nqueens as Q

    reference = UP / "results/nqueens-results.json"
    published = json.loads(reference.read_text())["training_methods"]["sizes"][str(args.n)]
    method = published["selected_method"]
    selection = published["rows"][method]["choice"]
    dest = N.ROOT / f"n{args.n}/final/{method}-s{args.seed}"
    meta_path = ROOT / f"artifacts/multi_solution/free-nq{args.n}-s{args.seed}-provenance.json"
    if (dest / "final.json").exists():
        log("already_complete_no_rescore", path=str(dest / "final.json"))
        return
    start = time.time()
    context = N.cuda_context()
    provenance = {
        "seed": args.seed, "n": args.n, "method": method,
        "upstream_commit": subprocess.check_output(["git", "-C", str(UP), "rev-parse", "HEAD"], text=True).strip(),
        "source_reference": str(reference.relative_to(ROOT)), "source_reference_sha256": sha(reference),
        "recipe_provenance": "Imported published validation-selected recipe; no tuning or selection on local test results.",
        "selection": selection, "wrapper_sha256": sha(Path(__file__).resolve()),
        "source_fit_sha256": sha(UP / "experiments/nqueens_free.py"),
        "source_objectives_sha256": sha(UP / "experiments/training_methods.py"),
        "source_search_sha256": sha(UP / "cram/cram_nqueens.py"),
        "host": platform.node(), "gpu": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "timing_protocol": "One cold fit all seeds; seven reinitialized warm fits only seed 0, whose median is the paper table timing. No warm timing is claimed for other seeds.",
        "start_unix": start,
    }
    N.save(meta_path, provenance)
    cfg = dict(selection["config"], momentum=.5)
    if (dest / "fit.json").exists():
        info = N.read(dest / "fit.json")
        model = torch.load(dest / "model.pt", weights_only=False)
        log("resume_completed_fit_for_single_test", n=args.n, seed=args.seed)
    else:
        tokens, labels = Q.load_tokens(args.n, "train")
        log("cold_fit_start", n=args.n, seed=args.seed, method=method, config=cfg, rows=len(labels))
        model, info = N.fit(cfg, tokens, labels, args.n, seed=args.seed)
        assert not info["diverged"] and N.usable(model)
        log("cold_fit_complete", seconds=info["fit_seconds"], parameters=info["params_nonzero"])
        warm, comparisons = [], []
        for repeat in range(7 if args.seed == 0 else 0):
            other, other_info = N.fit(cfg, tokens, labels, args.n, seed=args.seed)
            assert not other_info["diverged"] and N.usable(other)
            comparison = {k: {"exact_equal": bool(torch.equal(model[k], other[k])),
                "max_absolute_difference": float((model[k].double() - other[k].double()).abs().max())}
                for k in ("W", "B", "S")}
            warm.append(other_info["fit_seconds"])
            comparisons.append(comparison)
            log("warm_fit_complete", repeat=repeat, seconds=other_info["fit_seconds"], comparison=comparison)
        info.update(cold_seconds=info["fit_seconds"], warm_seconds=statistics.median(warm) if warm else None,
                    warm_repeats=warm, warm_model_comparisons=comparisons,
                    cuda_context_seconds=context, hostname=platform.node(), hardware=N.hardware(),
                    timing_protocol=provenance["timing_protocol"])
        N.save_torch(dest / "model.pt", model)
        N.save(dest / "fit.json", info)
    test_inputs, test_labels, _ = Q.load_split(args.n, "test")
    log("single_test_start", n=args.n, seed=args.seed, rows=len(test_inputs), chains=20)
    result = N.score(model, test_inputs, test_labels, args.n, chains=20,
                     temperature=selection["temperature"], cap=selection["cap"], seed=args.seed,
                     stop=selection["stop"], records=dest / "records.npz")
    N.save(dest / "final.json", dict(method=method, seed=args.seed, selection=selection, fit=info, test=result))
    provenance.update(completed_unix=time.time(), model_sha256=sha(dest / "model.pt"),
                      records_sha256=sha(dest / "records.npz"), final_sha256=sha(dest / "final.json"),
                      total_seconds=time.time() - start)
    N.save(meta_path, provenance)
    log("complete", n=args.n, seed=args.seed, total_seconds=provenance["total_seconds"], test=result)


if __name__ == "__main__":
    main()
