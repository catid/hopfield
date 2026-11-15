"""H4be source/identity/proof/endpoint/lifecycle gates, no scientific noise."""
from itertools import product
import json
import tempfile
import time
from unittest.mock import patch
from repeat_diversity_common import *
from freeze_repeat_diversity import source_paths
from larger_donor_math import counts,independent_counts
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_pooled_donor import audit_fit
from preflight_repeat_diversity_lifecycle import gate as lifecycle


def rejected(fn):
    try:fn()
    except (AssertionError,ValueError,FileNotFoundError):return
    raise AssertionError("Invalid synthetic fixture accepted")


def seed_audit(design):
    explicit=set();files={};solver_bases=set();large_bases=set()
    def visit(value,context="",is_seed=False):
        if isinstance(value,dict):
            for key,item in value.items():visit(item,context+"/"+key,is_seed or "seed" in key.lower())
        elif isinstance(value,list):
            for item in value:visit(item,context,is_seed)
        elif type(value) is int and is_seed:
            explicit.add(value)
            if "solver" in context.lower() and "base" in context.lower() and 0<value<2**31:solver_bases.add(value)
            if value>=10**10:large_bases.add(value)
    for file in sorted((ROOT/"research").rglob("*.json")):
        if any(word in file.name for word in ["proposal","candidate","preflight","seed-audit"]):continue
        try:record=read(file)
        except (ValueError,UnicodeDecodeError):continue
        if isinstance(record,dict) and "frozen_utc" in record:
            files[str(file.relative_to(ROOT))]=sha(file);visit(record)
    reserved=RIPPER/"diversity-after-correlation-proposal.json"
    if reserved.exists():files[str(reserved.relative_to(ROOT))]=sha(reserved);visit(read(reserved))
    new=set().union(*(set(range(design["seeds"][k],design["seeds"][k]+256)) for k in ["repeat1","repeat2","solver_base"]))
    assert len(new)==768 and not new&explicit
    base=design["seeds"]["solver_base"]
    assert 0<base<=base+255<2**31
    assert all(base+255<old or base>old+100000 for old in solver_bases)
    for key in ["repeat1","repeat2"]:
        current=design["seeds"][key]
        assert all(current+255<old or current>old+4096 for old in large_bases)
    result=dict(status="passed",frozen_protocol_sha256=files,historical_explicit_seed_values=len(explicit),
        conservative_solver_intervals=[[v,v+100000] for v in sorted(solver_bases)],
        conservative_large_acquisition_intervals=[[v,v+4096] for v in sorted(large_bases)],
        new_seed_ranges={k:[v,v+255] for k,v in design["seeds"].items()},new_seed_count=768,collisions=[],
        scope="All explicit seed fields in current frozen_utc research JSONs excluding draft/proposal/candidate/gate files, plus approved H1ci reservation. Conservatively reserve100000 offsets after named solver bases and4096 after large acquisition fields. Not exhaustive undocumented streams or PRNG independence.")
    save(HERE/"repeat-diversity-seed-audit.json",result);return result


def main():
    started=time.perf_counter()
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    target=HERE/"repeat-diversity-preflight.json";assert not target.exists()
    design=read(PROPOSAL);seed_audit(design)
    fixtures=[];clean=((np.arange(600)[:,None]+np.arange(81)[None,:])%9).astype(np.uint8)
    for i,hot in enumerate([np.arange(18),np.arange(63,81),np.arange(0,54,3),np.sort(np.random.default_rng(99341).choice(81,18,replace=False))]):
        seeds=[99350+2*i,99351+2*i];before=time.perf_counter();arrays=repeats(clean,hot,seeds)
        independent_repeats(clean,hot,seeds,arrays)
        first=np.roll(clean,1,axis=1);saved=first.copy();rows=assemble(first,arrays["r1_noisy"],arrays["r2_noisy"])
        assert np.array_equal(rows[:600],first) and np.array_equal(first,saved)
        assert np.array_equal(counts(rows),independent_counts(rows))
        bad={k:v.copy() for k,v in arrays.items()};bad["r1_noisy"][0,0]=(bad["r1_noisy"][0,0]+1)%9
        rejected(lambda:independent_repeats(clean,hot,seeds,bad))
        assert rows.size==145800 and arrays["r1_noisy"].size+arrays["r2_noisy"].size==97200
        fixtures.append(dict(rows=1800,new_response_rows=1200,reused_rows=600,wall_seconds=time.perf_counter()-before))
    rejected(lambda:assemble(clean.astype(np.int64),clean,clean))
    rejected(lambda:assemble(clean[:599],clean,clean))
    rejected(lambda:repeats(clean,np.arange(18)[::-1],seeds))
    bad=clean.copy();bad[0,0]=9;rejected(lambda:repeats(bad,np.arange(18),seeds))
    statuses=["success","resolved_failure_nonexact","resolved_failure_fractional","unresolved"]
    for new,old in product(statuses,repeat=2):
        result=bounds([new]+["success"]*255,[old]+["success"]*255)["regressions"]
        assert result["lower"]==int(new.startswith("resolved_failure") and old=="success")
        assert result["upper"]==int(new!="success" and old in ["success","unresolved"])
    assert bounds(["success"]*244+["resolved_failure_nonexact"]*12,["success"]*256)["exact_recovery"]["decision"]=="falsified"
    assert bounds(["success"]*245+["unresolved"]*11,["resolved_failure_nonexact"]*256)["exact_recovery"]["decision"]=="supported"
    with tempfile.TemporaryDirectory(prefix="h4be-proof-gate-") as directory:
        ii,jj=np.triu_indices(81,1);truth=(jj-ii==1)|((ii==0)&(jj==80))
        cost=np.where(truth,1,10).astype(np.int64);log=Path(directory)/"proof.log"
        x,y,meta=solve(81,2,ii,jj,cost,99370-2026111502,log)
        cert,proof=attempt_certificate(81,2,ii,jj,cost,x,y)
        solver=dict(read(HERE/"pooled-donor.json")["solver"],random_seed_base=99370)
        result=audit_fit(dict(cost=cost,primal_values=x,node_dual_values=y,**proof),dict(solver=meta,certificate=cert),cost,2,0,dict(solver=solver),truth)
        assert result["certified"] and result["classification"]=="success" and result["true_gap_twice"]==0
        # Two half-valued triangles joined by an integer matching, plus a
        # 75-cycle, give a literal feasible fractional degree-two witness.
        matrix=np.zeros((81,81),np.int64)
        for vertices in [[0,1,2],[3,4,5],list(range(6,81))]:
            for left,right in zip(vertices,vertices[1:]+vertices[:1]):matrix[min(left,right),max(left,right)]=1 if len(vertices)==3 else 2
        for left,right in [(0,3),(1,4),(2,5)]:matrix[left,right]=2
        x=matrix[ii,jj]/2;y=np.zeros(81,np.float64);cost=np.zeros(3240,np.int64)
        cert,proof=attempt_certificate(81,2,ii,jj,cost,x,y)
        assert cert["certified"] and not cert["integral"]
        meta=dict(meta,solver_objective=0.)
        model=dict(cost=cost,primal_values=x,node_dual_values=y,**proof)
        result=audit_fit(model,dict(solver=meta,certificate=cert),cost,2,0,dict(solver=solver),truth)
        assert result["classification"]=="resolved_failure_fractional" and result["true_gap_twice"]==0
        unknown=dict(model,**{key:np.empty(0,np.int64) for key in ["primal_twice","node_dual_twice","upper_dual_twice"]})
        result=audit_fit(unknown,dict(solver=meta,certificate=dict(certified=False)),cost,2,0,dict(solver=solver),truth)
        assert result["classification"]=="unresolved" and result["true_gap_twice"] is None
    # Actual phase-seal implementations reject absence, overwrites and mutation.
    import repeat_diversity_common as common
    with tempfile.TemporaryDirectory(prefix="h4be-seal-gate-") as directory:
        base=Path(directory);(base/"audits").mkdir();proto=base/"protocol.json";proto.write_text('{}\n')
        with patch.object(common,"OUT",base),patch.object(common,"PROTOCOL",proto):
            rejected(lambda:common.seal("source"))
            for rel in common.phase_files("source"):
                file=base/rel;file.parent.mkdir(exist_ok=True);file.write_text('synthetic\n')
            common.seal("source");common.check_seal("source",base)
            rejected(lambda:common.seal("source"))
            file=base/"observed/b000.npz";file.write_text('altered\n');rejected(lambda:common.check_seal("source",base))
    life=lifecycle()
    result=dict(status="passed",scientific_noise_draws=0,scientific_fits=0,synthetic_fullshape_noise_fixtures=fixtures,
        exact_query_counts=True,first_response_unchanged=True,independent_counts=True,paired_uncertainty_combinations=16,
        actual_synthetic_LPs=1,synthetic_fractional_witnesses=1,synthetic_unresolved_fixtures=1,
        synthetic_certificate_mutations=6,phase_missing_overwrite_mutation_rejected=True,
        lifecycle=life,proposal_sha256=sha(PROPOSAL),seed_audit_sha256=sha(HERE/"repeat-diversity-seed-audit.json"),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source_paths()},wall_seconds=time.perf_counter()-started)
    save(target,result);print(json.dumps({k:v for k,v in result.items() if k!="source_sha256"},indent=2))


if __name__=="__main__":main()
