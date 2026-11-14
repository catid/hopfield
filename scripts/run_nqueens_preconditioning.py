"""Reproduce the published N-Queens preconditioning ablation and its matched control.

The size-8 published local headline is MPL/table, while size-10 is PL/Ising.
Preserve this distinction; the Table-3 size-8 main row transfers the size-10 recipe.
Only one cold fit is needed for each quality-only ablation/control run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
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
    parser.add_argument("--precond", choices=["on", "off"], default="off")
    args = parser.parse_args()
    os.chdir(UP / "experiments")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    import nqueens_offset as O
    import nqueens_free as N
    from cram import cram_nqueens as Q

    reference = UP / "results/nqueens-results.json"
    block = json.loads(reference.read_text())["offset_energy"]["sizes"][str(args.n)]
    headline = block["headline"]
    method, arm = headline["method"], headline["arm"]
    selection = block["rows"][f"{method}-{arm}"]["choice"]
    selection = dict(selection, config=dict(selection["config"], precond=args.precond == "on"))
    if args.precond == "off":
        dest = O.ROOT / f"n{args.n}/ablation/{method}-{arm}-noprecond-s{args.seed}"
    else:
        dest = O.ROOT / f"n{args.n}/final/{method}-{arm}-s{args.seed}"
    meta_path = ROOT / f"artifacts/multi_solution/precond-{args.precond}-nq{args.n}-s{args.seed}-provenance.json"
    if (dest / "final.json").exists():
        log("already_complete_no_rescore", path=str(dest / "final.json"))
        return
    start = time.time()
    context = N.cuda_context()
    meta = {"n": args.n, "seed": args.seed, "precond": args.precond, "selection": selection,
            "headline": headline, "start_unix": start, "host": platform.node(),
            "gpu": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "source_reference": str(reference.relative_to(ROOT)), "source_reference_sha256": sha(reference),
            "source_commit": subprocess.check_output(["git", "-C", str(UP), "rev-parse", "HEAD"], text=True).strip(),
            "wrapper_sha256": sha(Path(__file__).resolve()),
            "fit_source_sha256": sha(UP / "experiments/nqueens_offset.py"),
            "search_source_sha256": sha(UP / "cram/cram_nqueens.py"),
            "protocol": "Published local headline config, with only precond toggled. One cold fit and one test pass; model saved additionally to official ablation output. No tuning on test data.",
            "size8_scope": "MPL/table local headline control and ablation; distinct from the transferred PL/Ising Table-3 headline."}
    N.save(meta_path, meta)
    if (dest / "fit.json").exists():
        model = torch.load(dest / "model.pt", weights_only=False)
        info = N.read(dest / "fit.json")
        log("resume_completed_fit_for_single_test", n=args.n, seed=args.seed)
    else:
        tokens, labels = Q.load_tokens(args.n, "train")
        log("cold_fit_start", n=args.n, seed=args.seed, precond=args.precond, config=selection["config"])
        model, info = O.fit(selection["config"], tokens, labels, args.n, seed=args.seed)
        info.update(cold_seconds=info["fit_seconds"], cuda_context_seconds=context,
                    hostname=platform.node(), hardware=N.hardware(), warm_seconds=None, warm_repeats=[])
        N.save_torch(dest / "model.pt", model)
        N.save(dest / "fit.json", info)
        log("cold_fit_complete", seconds=info["fit_seconds"], nonzero_parameters=info["params_nonzero"])
    test_inputs, test_labels, _ = Q.load_split(args.n, "test")
    log("single_test_start", n=args.n, seed=args.seed, precond=args.precond, rows=len(test_inputs), chains=O.CHAINS)
    result = N.score(model, test_inputs, test_labels, args.n, chains=O.CHAINS,
                     temperature=selection["temperature"], cap=selection["cap"], seed=args.seed,
                     stop=selection["stop"], records=dest / "records.npz")
    N.save(dest / "final.json", dict(method=method, arm=arm, seed=args.seed,
                                      selection=selection, fit=info, test=result))
    meta.update(completed_unix=time.time(), total_seconds=time.time() - start,
                model_sha256=sha(dest / "model.pt"), records_sha256=sha(dest / "records.npz"),
                final_sha256=sha(dest / "final.json"))
    N.save(meta_path, meta)
    log("complete", n=args.n, seed=args.seed, precond=args.precond, total_seconds=meta["total_seconds"], test=result)


if __name__ == "__main__":
    main()
