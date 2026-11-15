"""H4bc source closure and final freeze; never acquires cohort observations."""
import ast
import datetime
from fresh_pooled_donor_common import *


def source_paths():
    files = set(HERE.glob("*fresh_pooled_donor*.py"))
    pending = list(files)
    while pending:
        path = pending.pop()
        for node in ast.walk(ast.parse(path.read_text())):
            imports = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            for module in imports:
                name = (module or "").split(".")[0]+".py"
                for directory in [HERE, RIPPER]:
                    candidate = directory/name
                    if candidate.is_file() and candidate not in files:
                        files.add(candidate)
                        pending.append(candidate)
    # Dynamic upstream import in the inherited BP implementation.
    for rel in read(HERE/"fresh_curvature.json")["source_sha256"]:
        files.add(ROOT/rel)
    return sorted(files)


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate = read(HERE/"fresh-pooled-donor-preflight.json")
    seeds = read(HERE/"fresh-pooled-donor-seed-audit.json")
    assert gate["status"] == seeds["status"] == "passed" and gate["cohort_draws"] == 0
    assert gate["proposal_sha256"] == sha(PROPOSAL) and gate["seed_audit_sha256"] == sha(HERE/"fresh-pooled-donor-seed-audit.json")
    for rel, h in gate["source_sha256"].items():
        assert sha(ROOT/rel) == h, rel
    for rel, h in seeds["frozen_protocol_sha256"].items():
        assert sha(ROOT/rel) == h, rel
    assert set(gate["source_sha256"]) == {str(p.relative_to(ROOT)) for p in source_paths()}
    design = read(PROPOSAL)
    solver = read(HERE/"pooled-donor.json")["solver"]
    options = {k: v for k, v in solver.items() if k not in ["highspy", "numpy", "random_seed_base"]}
    inputs = [PROPOSAL, HERE/"fresh-pooled-donor-preflight.json", HERE/"fresh-pooled-donor-seed-audit.json"]
    bp = ROOT/"artifacts/research/20261114/foureyes/fresh_pooled_donor_prefreeze/bp_gate"
    for gpu in range(4):
        record = read(bp/f"gpu{gpu}.json")
        rel = str((bp/f"gpu{gpu}.json").relative_to(ROOT))
        assert gate["BP_gate_sha256"][rel] == sha(bp/f"gpu{gpu}.json")
        assert record["status"] == "passed" and record["source_sha256"] == sha(HERE/"fresh_pooled_donor_bp.py")
        assert record["worker_wall_seconds"] < 30 and record["full_updates"] == 32 and record["cohort_draws"] == 0
        inputs.append(bp/f"gpu{gpu}.json")
    p = dict(design, status="frozen_before_fresh_draws", frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             solver_options=options, source_sha256=gate["source_sha256"],
             input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs})
    save(PROTOCOL, p)
    print("FROZEN", sha(PROTOCOL), flush=True)


if __name__ == "__main__":
    freeze()
