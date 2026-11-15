"""H4be closure and freeze without new scientific noise draws."""
import ast
import datetime
from repeat_diversity_common import *


def source_paths():
    files=set(HERE.glob("*repeat_diversity*.py"));pending=list(files)
    while pending:
        path=pending.pop()
        for node in ast.walk(ast.parse(path.read_text())):
            imports=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            for module in imports:
                name=(module or "").split(".")[0]+".py"
                for folder in [HERE,RIPPER]:
                    candidate=folder/name
                    if candidate.is_file() and candidate not in files:files.add(candidate);pending.append(candidate)
    return sorted(files)


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate=read(HERE/"repeat-diversity-preflight.json")
    seeds=read(HERE/"repeat-diversity-seed-audit.json")
    assert gate["status"]==seeds["status"]=="passed" and gate["scientific_noise_draws"]==0
    assert gate["proposal_sha256"]==sha(PROPOSAL) and gate["seed_audit_sha256"]==sha(HERE/"repeat-diversity-seed-audit.json")
    assert set(gate["source_sha256"])=={str(p.relative_to(ROOT)) for p in source_paths()}
    for rel,digest in (gate["source_sha256"]|seeds["frozen_protocol_sha256"]).items():assert sha(ROOT/rel)==digest,rel
    design=read(PROPOSAL)
    for rel,digest in design["parent_sha256"].items():assert sha(ROOT/rel)==digest,rel
    assert read(PARENT/"runtime.json")["status"]=="complete"
    assert read(PARENT/"summary.json")["status"]=="validated_corrected_audit"
    parentp=read(PARENT_PROTOCOL)
    inputs={str(path.relative_to(ROOT)):sha(path) for path in
        [PROPOSAL,HERE/"repeat-diversity-preflight.json",HERE/"repeat-diversity-seed-audit.json",PARENT_PROTOCOL]}
    inputs.update(design["parent_sha256"])
    for phase in ["source","grid","donor","final"]:
        file=PARENT/f"{phase}-seal.json";inputs[str(file.relative_to(ROOT))]=sha(file)
    for path in [PARENT/"orders.npz",PARENT/"runtime.json",*sorted((PARENT/"source_audit").glob("*.json")),
                 *sorted((PARENT/"raw").glob("*.json")),*sorted((PARENT/"audits").glob("*.json")),
                 *sorted((PARENT_RAW/"raw").glob("*.npz"))]:
        inputs[str(path.relative_to(ROOT))]=sha(path)
    degrees=[]
    for i in range(N):
        file=PARENT/"fits"/f"b{i:03d}"/"grid.json";inputs[str(file.relative_to(ROOT))]=sha(file)
        degrees.append(read(file)["selected_degree"])
    assert len(degrees)==256 and all(type(d) is int and 1<=d<=39 for d in degrees)
    solver=dict(read(HERE/"pooled-donor.json")["solver"],random_seed_base=design["seeds"]["solver_base"])
    # Every parent scientific implementation stays bound without modifying it.
    inputs.update(parentp["source_sha256"])
    record=dict(design,status="frozen_before_new_repeat_noise_and_fits",frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                solver=solver,inherited_degrees=degrees,source_sha256=gate["source_sha256"],input_sha256=inputs)
    save(PROTOCOL,record);print("FROZEN",sha(PROTOCOL),flush=True)


if __name__=="__main__":freeze()
