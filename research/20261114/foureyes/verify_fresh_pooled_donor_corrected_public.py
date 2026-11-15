"""Public H4bd replay, including exact regeneration of every latent source file.

The original frozen audit is unchanged; the correction changes only its BP
NumPy-version assertion. Historical audit timestamps are reused solely for
receipt comparison. Replay writes only disposable temporary directories.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import numpy as np
import audit_fresh_pooled_donor_corrected as audit
from fresh_pooled_donor_common import ROOT, HERE, PROTOCOL, N, sha, read, files_for
from fresh_pooled_donor_source import source

PUBLIC = ROOT/"results/research/foureyes/fresh_pooled_donor_corrected"
CORRECTED = HERE/"fresh-pooled-donor-corrected.json"
MANIFEST = ROOT/"results/research/foureyes/fresh_pooled_donor_corrected_public_manifest.json"
ORIGINAL_PREFIX = "artifacts/research/20261114/foureyes/fresh_pooled_donor/"
GATE_PREFIX = "artifacts/research/20261114/foureyes/fresh_pooled_donor_prefreeze/bp_gate/"


def bound_path(rel):
    if rel.startswith(ORIGINAL_PREFIX):
        name = rel[len(ORIGINAL_PREFIX):]
        return PUBLIC/("original-incomplete-runtime.json" if name == "runtime.json" else name)
    if rel.startswith(GATE_PREFIX):
        return PUBLIC/"prefreeze"/Path(rel).name
    return ROOT/rel


def one(index):
    p = read(PROTOCOL)
    saved = read(PUBLIC/"audits"/f"b{index:03d}.json")
    with np.load(PUBLIC/"orders.npz", allow_pickle=False) as z:
        order, hot = z["orders"][index], z["hot_cells"][index]
    # Recreate precisely the originally drawn arrays, without a new seed,
    # fit, observation acquisition, or access to the private archive.
    arrays, receipt = source(index, p, order, hot)
    raw_record = read(PUBLIC/"raw"/f"b{index:03d}.json")
    assert receipt == raw_record["sampler_receipt"]
    with tempfile.TemporaryDirectory(prefix="h4bd-replay-bank-") as temporary:
        base = Path(temporary)
        for item in PUBLIC.iterdir():
            if item.name != "raw":
                (base/item.name).symlink_to(item, target_is_directory=item.is_dir())
        (base/"raw").mkdir()
        raw = base/"raw"/f"b{index:03d}.npz"
        np.savez_compressed(raw, **arrays)
        assert sha(raw) == raw_record["raw_sha256"]
        raw.with_suffix(".json").symlink_to(PUBLIC/"raw"/raw.with_suffix(".json").name)
        with patch.object(audit, "OUT", base), patch.object(audit.time, "monotonic_ns", return_value=saved["audit_started_monotonic_ns"]):
            replay = audit.audit_bank(index, p)
        assert replay == saved, index
    return replay


def main(skip_manifest=False):
    started = time.perf_counter()
    if not skip_manifest:
        manifest = read(MANIFEST)
        for rel, digest in manifest["sha256"].items():
            assert sha(ROOT/rel) == digest, rel
    p, corrected = read(PROTOCOL), read(CORRECTED)
    unavailable = []
    for rel, digest in (p["source_sha256"] | p["input_sha256"] | corrected["source_sha256"] | corrected["parent_sha256"]).items():
        path = bound_path(rel)
        if not path.exists():
            assert rel == "upstream/experiments/sudoku.py", rel
            unavailable.append(rel)
        else:
            assert sha(path) == digest, rel
    old = (HERE/"audit_fresh_pooled_donor.py").read_text()
    needle = 'assert env["numpy"] == "2.3.4" and env["physical_GPU"]'
    assert old.count(needle) == 1
    assert old.replace(needle, needle.replace("2.3.4", "2.5.4")) == Path(audit.__file__).read_text()
    for phase in ["source", "grid", "donor", "final"]:
        seal = read(PUBLIC/f"{phase}-seal.json")
        assert seal["protocol_sha256"] == sha(PROTOCOL) and seal["phase"] == phase
        assert set(seal["files"]) == files_for(phase)
        for rel, digest in seal["files"].items():
            path = PUBLIC/rel
            if rel.startswith("raw/") and rel.endswith(".npz"):
                # Each missing latent NPZ is regenerated and hash-checked by
                # one(), then subjected to the independent source audit.
                assert digest == read(path.with_suffix(".json"))["raw_sha256"]
            else:
                assert sha(path) == digest, rel
    saved = read(PUBLIC/"summary.json")
    assert saved["status"] == "validated_corrected_audit" and saved["original_run_status"] == "incomplete"
    assert saved["protocol_sha256"] == sha(CORRECTED) and saved["original_protocol_sha256"] == sha(PROTOCOL)
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, range(N)))
    assert rows == saved["rows"]
    for key, value in audit.aggregate(rows).items():
        assert saved[key] == value, key
    original, runtime = [read(PUBLIC/name) for name in ["original-incomplete-runtime.json", "runtime.json"]]
    assert original["status"] == "incomplete" and original["error"] == "AssertionError('owned_worker_failure')"
    assert original["returncodes"] == [0]*36+[1]*8 and original["protocol_sha256"] == sha(PROTOCOL)
    assert runtime["status"] == "complete" and runtime["error"] is None and runtime["returncodes"] == [0]*8
    assert runtime["protocol_sha256"] == sha(CORRECTED)
    assert runtime["wall_seconds"] < corrected["budget"]["wall_seconds"] == 120
    assert runtime["aggregate_CPU_seconds"] < corrected["budget"]["aggregate_CPU_seconds"] == 600
    assert runtime["new_observations"] == runtime["new_LPs"] == runtime["new_BP_fits"] == 0
    assert runtime["samples"][-1]["audited_banks"] == N
    assert original["samples"][-1]["counters"]["bp"] == original["samples"][-1]["counters"]["masked"] == N
    return dict(status="passed", banks=N, regenerated_latent_sources=N, new_scientific_draws=0,
                **audit.aggregate(rows), replay_wall_seconds=time.perf_counter()-started,
                unavailable_training_only_source=unavailable,
                scope="All sources/noise regenerated byte-identically, independent source audits, counts, degree grid, exact LP proofs/mutations, BP checkpoint transforms/history/provenance and paired endpoints replayed. BP optimization is not rerun. Original failed audit and separate successful correction retained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-manifest", action="store_true", help="Packaging bootstrap only; regular verification requires the public manifest.")
    print(json.dumps(main(parser.parse_args().skip_manifest), indent=2))
