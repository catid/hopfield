"""Replay H4be from public evidence; reconstruct missing latent arrays exactly.

Only temporary paths and historical receipt timestamps replace audit globals.
The frozen scientific sources, fitted models and observations stay unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import numpy as np
import audit_repeat_diversity as audit
import repeat_diversity_source as generation
from repeat_diversity_common import (
    ROOT, HERE, PROTOCOL, PARENT_PROTOCOL, PARENT, PUBLIC, N,
    read, sha, npz, repeats, phase_files, aggregate,
)
from fresh_pooled_donor_source import source

MANIFEST = ROOT / "results/research/foureyes/repeat_diversity_public_manifest.json"
RAW_PREFIX = "artifacts/research/20261114/foureyes/fresh_pooled_donor/raw/"


def one(index):
    p = read(PROTOCOL)
    saved = read(PUBLIC / "audits" / f"b{index:03d}.json")
    orders = npz(PARENT / "orders.npz")
    hot = orders["hot_cells"][index]
    arrays, receipt = source(index, read(PARENT_PROTOCOL), orders["orders"][index], hot)
    record = read(PARENT / "raw" / f"b{index:03d}.json")
    assert receipt == record["sampler_receipt"]
    noise = repeats(arrays["clean"][:600], hot,
                    [p["seeds"][key] + index for key in ["repeat1", "repeat2"]])
    with tempfile.TemporaryDirectory(prefix="h4be-public-bank-") as temporary:
        base = Path(temporary)
        current, parent = base / "current", base / "parent"
        current.mkdir()
        (parent / "raw").mkdir(parents=True)
        for item in PUBLIC.iterdir():
            if item.name != "noise":
                (current / item.name).symlink_to(item, target_is_directory=item.is_dir())
        (current / "noise").mkdir()
        raw = parent / "raw" / f"b{index:03d}.npz"
        np.savez_compressed(raw, **arrays)
        assert sha(raw) == record["raw_sha256"] == p["input_sha256"][RAW_PREFIX + raw.name]
        latent = current / "noise" / raw.name
        np.savez_compressed(latent, **noise)
        seal = read(PUBLIC / "source-seal.json")
        assert sha(latent) == seal["files"]["noise/" + raw.name]
        with patch.object(audit, "OUT", current), \
             patch.object(audit, "PARENT_RAW", parent), \
             patch.object(generation, "PARENT_RAW", parent), \
             patch.object(audit.time, "monotonic_ns", return_value=saved["audit_started_monotonic_ns"]):
            replay = audit.audit_bank(index, p)
        assert replay == saved, index
    return replay


def main(skip_manifest=False):
    started = time.perf_counter()
    if not skip_manifest:
        for rel, digest in read(MANIFEST)["sha256"].items():
            assert sha(ROOT / rel) == digest, rel
    p = read(PROTOCOL)
    unavailable = []
    for rel, digest in (p["source_sha256"] | p["input_sha256"]).items():
        if rel.startswith(RAW_PREFIX) and rel.endswith(".npz"):
            assert digest == read(PARENT / "raw" / Path(rel).with_suffix(".json").name)["raw_sha256"]
        elif not (ROOT / rel).exists():
            assert rel == "upstream/experiments/sudoku.py", rel
            unavailable.append(rel)
        else:
            assert sha(ROOT / rel) == digest, rel
    for phase in ["source", "fit"]:
        seal = read(PUBLIC / f"{phase}-seal.json")
        assert seal["phase"] == phase and seal["protocol_sha256"] == sha(PROTOCOL)
        assert set(seal["files"]) == phase_files(phase)
        for rel, digest in seal["files"].items():
            if rel.startswith("noise/"):
                assert digest == read(PUBLIC / "source_audit" / Path(rel).with_suffix(".json").name)["noise_sha256"]
            else:
                assert sha(PUBLIC / rel) == digest, rel
    saved, runtime = [read(PUBLIC / name) for name in ["summary.json", "runtime.json"]]
    assert saved["status"] == "validated" and saved["protocol_sha256"] == sha(PROTOCOL)
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, range(N)))
    assert rows == saved["rows"]
    for key, value in aggregate(rows).items():
        assert saved[key] == value, key
    assert saved["accounting"] == p["accounting"]
    assert saved["new_LPs"] == p["accounting"]["new_LPs"] == N
    assert saved["new_BP_fits"] == p["accounting"]["new_BP_fits"] == 0
    assert runtime["status"] == "complete" and runtime["error"] is None
    assert runtime["protocol_sha256"] == sha(PROTOCOL) and runtime["returncodes"] == [0] * 24
    assert runtime["wall_seconds"] < p["budget"]["complete_wall_seconds"] == 120
    assert runtime["aggregate_CPU_seconds"] < p["budget"]["aggregate_CPU_seconds"] == 600
    assert runtime["samples"][-1]["counters"] == {"source": N, "fit": N, "audit": N}
    assert runtime["samples"][-1]["phase"] == "summaries_written"
    return dict(status="passed", banks=N, regenerated_original_raw_archives=N,
                regenerated_repeat_noise_archives=N, new_scientific_draws=0,
                **aggregate(rows), replay_wall_seconds=time.perf_counter()-started,
                unavailable_training_only_source=unavailable,
                scope="Byte-identical source/noise regeneration, independent provenance and count replay, both selected LP certificates and mutations, paired outcomes and timing receipts. No LP solving or BP retraining; BP is prior distinct-source context only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-manifest", action="store_true", help="Packaging bootstrap only.")
    print(json.dumps(main(parser.parse_args().skip_manifest), indent=2))
