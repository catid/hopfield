"""Fit the frozen successful Maze-Hard recipe on 900 mazes; score 100 held out.

Uses the published fit/evaluate functions, retaining predictions for independent
BFS validation. No hyperparameter selection or test-file access is permitted.
"""
import argparse
import dataclasses
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from run_maze_reproduction import path_audit, patched, save, sha256, utc

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"
LOADED_SCRIPT_SHA = sha256(__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Use a new output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / f"artifacts/gpu{args.gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def query(argv):
        return subprocess.check_output(argv, text=True, timeout=20).strip()
    gpu = query(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used", "--format=csv,noheader,nounits"])
    gpu_uuid = gpu.split(",")[1].strip()
    owners = query(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"])
    assert not any(gpu_uuid in line for line in owners.splitlines()), "GPU has an existing compute owner"
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CRAM_DATA=str(UP / "data"),
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    data_csv_reads = []
    def guard_csv(event, event_args):
        if event != "open" or not isinstance(event_args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(event_args[0])).resolve()
        if path.suffix == ".csv" and (UP / "data").resolve() in path.parents:
            if path.name != "train.csv":
                raise RuntimeError(f"Development run cannot open {path}")
            data_csv_reads.append(str(path.relative_to(ROOT)))
    sys.addaudithook(guard_csv)
    sys.path[:0] = [str(UP), str(UP / "experiments")]
    import numpy as np
    import torch
    import triton
    import maze_hard_latent as H
    from cram import maze_h3 as M
    torch.set_num_threads(1)
    recipe_path = UP / "experiments/recipes/maze-hard-latent.json"
    recipe = H.Recipe(**json.loads(recipe_path.read_text())["recipe"])
    assert recipe.updates == 4000
    source_paths = sorted((UP / "cram").glob("*.py")) + sorted((UP / "experiments").glob("*.py"))
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    tok, labels = M.load("train")
    fit_rows, dev_rows = M.dev_split()
    assert tok.shape == labels.shape == (1000, 30, 30)
    assert len(fit_rows) == 900 and len(dev_rows) == 100
    assert not np.intersect1d(fit_rows, dev_rows).size
    assert np.array_equal(np.sort(np.concatenate([fit_rows, dev_rows])), np.arange(1000))
    train_path = M.load.__globals__["DATA"] / "train.csv"
    record = {"status": "running", "started_utc": utc(), "seed": args.seed,
        "recipe": dataclasses.asdict(recipe), "recipe_path": str(recipe_path.relative_to(ROOT)),
        "recipe_sha256": sha256(recipe_path), "fit_rows": fit_rows.tolist(),
        "development_rows": dev_rows.tolist(), "train_csv_sha256": sha256(train_path),
        "source_commit": query(["git", "-C", str(UP), "rev-parse", "HEAD"]),
        "source_sha256": sources, "wrapper_sha256": LOADED_SCRIPT_SHA,
        "independent_audit_source_sha256": sha256(ROOT / "scripts/run_maze_reproduction.py"),
        "hostname": platform.node(), "python": sys.version, "torch": torch.__version__,
        "cuda": torch.version.cuda, "triton": triton.__version__, "gpu_before": gpu,
        "gpu_uuid": gpu_uuid, "cuda_visible_devices": str(args.gpu), "pid": os.getpid(),
        "argv": sys.argv, "test_data_used": False, "tuning": False,
        "protocol": "Frozen successful recipe; fused Triton fit; PyTorch hidden solve at development evaluation; one development pass; no zero-temperature diagnostic",
        "timing_note": "One cold fit; prediction capture adds evaluation transfer overhead; no hardware-matched speed claim"}
    save(args.out / "provenance.json", record)
    print(json.dumps({"event": "fit_start", "seed": args.seed, "gpu": args.gpu, "utc": utc()}), flush=True)
    started = time.monotonic()
    try:
        context_start = time.perf_counter()
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        context_s = time.perf_counter() - context_start
        model, times, curve = H.fit_recipe(tok[fit_rows], labels[fit_rows], args.seed, recipe,
            ckpt=args.out / "rolling.pt", log=args.out / "train.jsonl", triton=True)
        assert [row["u"] for row in curve] == list(range(50, 4001, 50))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        model.triton_settle = False
        torch.save(model.state_dict(), args.out / "model.pt")
        save(args.out / "train.json", {"times": times, "curve": curve, "cuda_context_s": context_s,
                                      "parameters": H.count_params(model, recipe), "latent": H.latent_summary(model)})
        print(json.dumps({"event": "fit_complete", "seed": args.seed, "times": times, "utc": utc()}), flush=True)
        captured = []
        original_descend = M.descend
        def capture(*a, **kw):
            predictions, steps, stopped = original_descend(*a, **kw)
            captured.append({"predictions": predictions.cpu().numpy(), "steps": steps.cpu().numpy(),
                             "stopped": stopped.cpu().numpy()})
            return predictions, steps, stopped
        with patched(M, "descend", capture):
            result = H.evaluate(model, tok[dev_rows], labels[dev_rows])
        arrays = {k: np.concatenate([batch[k] for batch in captured]) for k in captured[0]}
        arrays.update(inputs=tok[dev_rows], labels=labels[dev_rows], development_rows=dev_rows)
        np.savez_compressed(args.out / "development_predictions.npz", **arrays)
        audit = path_audit(arrays["inputs"], arrays["predictions"], arrays["labels"])
        exact = np.all(arrays["predictions"].astype(bool) == arrays["labels"], axis=(1, 2))
        assert audit["n"] == result["n"] == 100
        assert int((exact & arrays["stopped"]).sum()) == result["solved"]
        assert int((~exact & arrays["stopped"]).sum()) == result["wrong_stops"]
        assert int((~arrays["stopped"]).sum()) == result["budget_failures"]
        assert np.array_equal(result["per_maze"]["exact"], exact & arrays["stopped"])
        assert np.array_equal(result["per_maze"]["stopped"], arrays["stopped"])
        save(args.out / "development.json", result)
        save(args.out / "independent_development_audit.json", audit)
        assert all(sha256(ROOT / p) == digest for p, digest in sources.items())
        record.update(status="complete", completed_utc=utc(), wall_seconds=time.monotonic() - started,
            data_csv_reads=sorted(set(data_csv_reads)), times=times,
            result={k: result[k] for k in ("n", "solved", "wrong_stops", "budget_failures")},
            independent_audit=audit, output_sha256={name: sha256(args.out / name) for name in
                ("model.pt", "train.json", "train.jsonl", "development.json", "development_predictions.npz", "independent_development_audit.json")})
        print(json.dumps({"event": "complete", "seed": args.seed, "result": record["result"], "audit": audit}), flush=True)
    except BaseException as error:
        record.update(status="failed", error=repr(error), finished_utc=utc())
        raise
    finally:
        save(args.out / "provenance.json", record)


if __name__ == "__main__":
    main()
