"""Frozen zero-stop development gate and Figure 3a test search.

full runs the published 412,786-puzzle scheduling, from which the figure selects
the first 10,000. first10k is an explicit subset protocol recreation: compaction
uses active-batch positions in the RNG and is not prefix-equivalent to full.
All records remain separate from Table 2's fitted-threshold results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, choices=range(3), required=True)
    p.add_argument("--test-size", choices=["full", "first10k"], required=True)
    p.add_argument("--preflight", action="store_true", help="Stage frozen model and provenance without GPU inference")
    a = p.parse_args()
    out = ROOT / "artifacts/difficulty" / a.test_size
    out.mkdir(parents=True, exist_ok=True)
    model = UP / f"experiments/runs/CRAM-S/full/full-aug-s{a.seed}/model.pt"
    staged = out / f"full-aug-s{a.seed}/model.pt"
    staged.parent.mkdir(exist_ok=True)
    if staged.exists():
        assert sha(staged) == sha(model), "staged model differs from the fresh main fit"
    else:
        shutil.copy2(model, staged)
    os.environ.setdefault("CRAM_DATA", str(UP / "data"))
    sys.path[:0] = [str(UP), str(UP / "experiments")]
    import sudoku as S
    S.ROOT = out
    protocol = {
        "status": "prepared" if a.preflight else "running", "seed": a.seed,
        "test_size": a.test_size, "test_rows": [10000, 422786 if a.test_size == "full" else 20000],
        "figure_rows": [10000, 20000], "stop": "zero", "relative_search": [.40, .31, .45],
        "search": S.FROZEN,
        "scope": "Published full search scheduling" if a.test_size == "full" else
                 "Same settings and figure rows; subset RNG/compaction scheduling differs from published full search",
        "prefix_equivalence_claim": False,
        "model_sha256": sha(model), "fresh_model_source": str(model.relative_to(ROOT)),
        "source_commit": "62bbe830c7d6f87f2ba18f30a28039258b986454",
        "source_sha256": {str(f.relative_to(UP)): sha(f) for f in
            [UP / "experiments/sudoku.py", UP / "cram/cram_sudoku_general.py", UP / "cram/cram_sudoku_search.py"]},
        "wrapper_sha256": sha(__file__), "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "argv": sys.argv,
    }
    path = out / f"provenance-s{a.seed}.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    if a.preflight:
        print(json.dumps({"path": str(path), "model_sha256": protocol["model_sha256"], "scope": protocol["scope"]}))
        return
    settings = dict(scale="hbar", hi=.40, lo_t=.31, theta=.45, stop="zero")
    start = time.perf_counter()
    try:
        dev = S.search(a.seed, "dev10k", 0, 10000, out / f"dev/cram0-hb-s{a.seed}.json", **settings)
        sig, summary = dev["signature"], dev["summary"]
        assert sig["model_sha256"] == sha(model) and sig["test"] == "zero" and sig["rel"] == [.40, .31, .45]
        assert summary["solved"] >= 9999 and summary["wrong_stops"] == 0, "zero-stop development gate failed"
        n = 412786 if a.test_size == "full" else 10000
        rec = S.search(a.seed, "full-all", 10000, n, out / f"test/cram0-hb-s{a.seed}.json", **settings)
        assert rec["signature"]["model_sha256"] == sha(model)
        protocol.update(status="complete", elapsed_seconds=time.perf_counter() - start,
                        development_summary=summary, test_summary=rec["summary"])
    except BaseException as e:
        protocol.update(status="failed", elapsed_seconds=time.perf_counter() - start, error=repr(e))
        raise
    finally:
        path.write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
