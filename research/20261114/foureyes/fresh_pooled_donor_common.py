"""H4bc immutable bindings, phase seals and conservative endpoint arithmetic."""
from pathlib import Path
import json
import sys
import time
from weak_weight_study import ROOT, HERE, RIPPER, sha, save
sys.path.insert(0, str(RIPPER))
OUT = ROOT / "artifacts/research/20261114/foureyes/fresh_pooled_donor"
PUBLIC = ROOT / "results/research/foureyes/fresh_pooled_donor"
PROTOCOL = HERE / "fresh-pooled-donor.json"
PROPOSAL = HERE / "fresh-pooled-donor-proposal.json"
CPU_PYTHON = ROOT / "artifacts/degree-lp-venv/bin/python"
GPU_PYTHON = ROOT / "upstream/.venv/bin/python"
FAMILIES = ["cycle81", "power2", "power3", "cliques9"]
N = 256


def read(path):
    return json.loads(Path(path).read_text())


def protocol():
    p = read(PROTOCOL)
    assert p["status"] == "frozen_before_fresh_draws"
    for name, digest in (p["source_sha256"] | p["input_sha256"]).items():
        assert sha(ROOT / name) == digest, name
    return p


def files_for(phase):
    if phase == "source":
        return {f"{folder}/b{i:03d}.{ext}" for i in range(N)
                for folder, extensions in [("observed", ["npz"]), ("raw", ["npz", "json"]), ("source_audit", ["json"])]
                for ext in extensions} | {"orders.npz", "orders-seal.json"}
    names = {"grid": ["grid"], "donor": ["donor", "pooled"], "final": ["grid", "donor", "pooled", "masked", "bp"]}[phase]
    return {f"fits/b{i:03d}/{name}.{ext}" for i in range(N) for name in names for ext in ["npz", "json"]}


def seal(phase):
    target = OUT / f"{phase}-seal.json"
    assert not target.exists()
    files = files_for(phase)
    assert all((OUT / name).is_file() for name in files)
    if phase != "source":
        assert not list((OUT / "audits").glob("*.json"))
    record = dict(phase=phase, protocol_sha256=sha(PROTOCOL), sealed_monotonic_ns=time.monotonic_ns(),
                  files={name: sha(OUT / name) for name in sorted(files)})
    save(target, record)
    return record


def check_seal(phase, folder=OUT):
    record = read(folder / f"{phase}-seal.json")
    assert record["phase"] == phase and record["protocol_sha256"] == sha(PROTOCOL)
    assert set(record["files"]) == files_for(phase)
    for name, digest in record["files"].items():
        assert sha(folder / name) == digest, name
    return record


def observations(index):
    import numpy as np
    file = OUT / "observed" / f"b{index:03d}.npz"
    source = read(OUT / "source_audit" / f"b{index:03d}.json")
    assert sha(file) == source["observed_sha256"]
    with np.load(file, allow_pickle=False) as z:
        assert z.files == ["noisy"]
        rows = z["noisy"]
    assert rows.shape == (1800, 81) and rows.dtype == np.uint8 and (rows < 9).all()
    return rows


def bounds(pooled, masked, minimum=245, maximum=4):
    statuses = {"success", "resolved_failure_nonexact", "resolved_failure_fractional", "unresolved"}
    assert len(pooled) == len(masked) and set(pooled + masked) <= statuses
    exact = pooled.count("success")
    uncertain = pooled.count("unresolved")
    loss_low = sum(a.startswith("resolved_failure") and b == "success" for a, b in zip(pooled, masked))
    loss_high = sum(a != "success" and b in ["success", "unresolved"] for a, b in zip(pooled, masked))
    return dict(exact_recovery=dict(lower=exact, upper=exact+uncertain, required=minimum,
                                   decision="supported" if exact >= minimum else "falsified" if exact+uncertain < minimum else "inconclusive"),
                regressions=dict(lower=loss_low, upper=loss_high, maximum=maximum,
                                 decision="supported" if loss_high <= maximum else "falsified" if loss_low > maximum else "inconclusive"))
