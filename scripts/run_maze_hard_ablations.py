"""Run frozen, explicitly reconstructed Maze-Hard development ablations.

The release omits the original failed-ablation recipes and records. These jobs
change exactly one field of the released successful recipe, without tuning.
Only the training CSV is opened; the official test set is never evaluated.
"""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"
OUT = ROOT / "artifacts/maze_hard_ablations"
CHANGES = {"token-source": {"source": "token"}, "edge-gates": {"coupling": "gate"}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=CHANGES, required=True)
    p.add_argument("--seed", type=int, choices=range(3), required=True)
    p.add_argument("--preflight", action="store_true", help="Materialize provenance without fitting or evaluating")
    a = p.parse_args()
    if a.variant == "edge-gates" and a.seed != 0:
        p.error("The paper reports one 100-maze gate ablation; only seed 0 is in the frozen scope")
    os.environ.setdefault("CRAM_DATA", str(UP / "data"))
    sys.path[:0] = [str(UP), str(UP / "experiments")]
    import numpy as np
    import torch
    import maze_hard_latent as H

    torch.set_num_threads(1)
    H.ROOT = OUT
    source = UP / "experiments/recipes/maze-hard-latent.json"
    recipe = dict(json.loads(source.read_text())["recipe"], **CHANGES[a.variant])
    r = H.Recipe(**recipe)
    model = H.build(r)
    fit_ids, dev_ids = H.m3.dev_split()
    destination = OUT / a.variant
    destination.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "prepared" if a.preflight else "running", "variant": a.variant, "seed": a.seed,
        "scope": "Single-field controlled reconstruction; exact published failed-ablation recipe is unavailable",
        "published_result": {"solved": 0, "n": 300 if a.variant == "token-source" else 100},
        "recipe": recipe, "changed_fields": CHANGES[a.variant],
        "recipe_source": str(source.relative_to(ROOT)), "recipe_source_sha256": sha(source),
        "fit_rows": fit_ids.tolist(), "development_rows": dev_ids.tolist(),
        "train_csv_sha256": sha(H.m3.load.__globals__["DATA"] / "train.csv"),
        "parameter_counts": H.count_params(model, r),
        "triton_forward_requested": True,
        "adjoint_backend": "pytorch fallback" if a.variant == "token-source" else "fused triton",
        "test_data_used": False, "tuning": False,
        "source_commit": "62bbe830c7d6f87f2ba18f30a28039258b986454",
        "source_sha256": {str(f.relative_to(UP)): sha(f) for f in
            [UP / "experiments/maze_hard_latent.py", UP / "cram/cram_maze_hard_graph.py",
             UP / "cram/cram_maze_hard_simple.py", UP / "cram/maze_learned.py"]},
        "wrapper_sha256": sha(__file__), "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "argv": sys.argv,
    }
    assert len(fit_ids) == 900 and len(dev_ids) == 100 and not np.intersect1d(fit_ids, dev_ids).size
    path = destination / f"provenance-s{a.seed}.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    if a.preflight:
        print(json.dumps({"path": str(path), "recipe": recipe, "parameters": protocol["parameter_counts"]}))
        return
    start = time.perf_counter()
    try:
        H.train(H.Train(name=a.variant, seed=a.seed, recipe=r, triton=True))
        rec = json.loads((destination / f"dev-s{a.seed}.json").read_text())
        assert rec["recipe"] == recipe and rec["dev100"]["n"] == 100
        protocol.update(status="complete", elapsed_seconds=time.perf_counter() - start,
                        result=rec["dev100"], model_sha256=sha(destination / f"model-s{a.seed}.pt"))
    except BaseException as e:
        protocol.update(status="failed", elapsed_seconds=time.perf_counter() - start, error=repr(e))
        raise
    finally:
        path.write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
