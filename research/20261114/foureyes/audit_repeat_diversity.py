"""H4be independent source/noise/count/selected-proof replay after the fit seal."""
import argparse
import os
import time
from repeat_diversity_common import *
from repeat_diversity_source import original
from larger_donor_math import independent_counts
from audit_objective_gap import reconstruct
from audit_pooled_donor import audit_fit


def audit_bank(index,p):
    started=time.monotonic_ns()
    final=read(OUT/"fit-seal.json")
    source=read(OUT/"source-seal.json")
    assert final["protocol_sha256"]==source["protocol_sha256"]==sha(PROTOCOL)
    assert started>final["sealed_monotonic_ns"]>source["sealed_monotonic_ns"]
    raw,hot=original(index)
    arrays=npz(OUT/"noise"/f"b{index:03d}.npz")
    independent_repeats(raw["clean"][:600],hot,[p["seeds"][k]+index for k in ["repeat1","repeat2"]],arrays)
    rows=npz(OUT/"observed"/f"b{index:03d}.npz")["noisy"]
    # Independent assembly, with every original/reacquired row identity kept.
    expected=np.empty((1800,81),np.uint8)
    expected[:600]=raw["noisy"][:600]
    expected[600:1200]=arrays["r1_noisy"]
    expected[1200:]=arrays["r2_noisy"]
    assert rows.dtype==np.uint8 and np.array_equal(rows,expected)
    generation=read(OUT/"source_audit"/f"b{index:03d}.json")
    assert generation["status"]=="passed" and generation["index"]==index
    assert generation["protocol_sha256"]==sha(PROTOCOL)
    for key,path in [("observed_sha256",OUT/"observed"/f"b{index:03d}.npz"),("noise_sha256",OUT/"noise"/f"b{index:03d}.npz"),
                     ("original_source_audit_sha256",PARENT/"source_audit"/f"b{index:03d}.json"),
                     ("original_observed_sha256",PARENT/"observed"/f"b{index:03d}.npz"),
                     ("original_raw_sha256",PARENT_RAW/"raw"/f"b{index:03d}.npz")]:
        assert generation[key]==sha(path)
    assert generation["source_started_monotonic_ns"]<=generation["source_finished_monotonic_ns"]<source["sealed_monotonic_ns"]
    assert generation["new_noisy_board_responses"]==1200 and generation["reused_noisy_board_responses"]==600
    assert generation["new_noisy_cell_responses"]==97200 and generation["candidate_standalone_noisy_cell_responses"]==145800
    assert generation["first_response_byte_values_unchanged"] and generation["original_full_source_independently_replayed"] and generation["new_repeats_independently_replayed"]
    rec=read(OUT/"fits"/f"b{index:03d}.json")
    model=npz(OUT/"fits"/f"b{index:03d}.npz")
    assert rec["index"]==index and rec["arm"]=="repeat" and rec["protocol_sha256"]==sha(PROTOCOL)
    assert rec["source_seal_sha256"]==sha(OUT/"source-seal.json") and rec["observed_sha256"]==generation["observed_sha256"]
    assert rec["models_sha256"]==sha(OUT/"fits"/f"b{index:03d}.npz") and rec["log_sha256"]==sha(OUT/"fits"/f"b{index:03d}.log")
    assert source["sealed_monotonic_ns"]<rec["fit_started_monotonic_ns"]<=rec["fit_completed_monotonic_ns"]<final["sealed_monotonic_ns"]
    degree=rec["degree"]
    assert degree==p["inherited_degrees"][index]==read(PARENT/"fits"/f"b{index:03d}"/"grid.json")["selected_degree"]
    order=npz(PARENT/"orders.npz")["orders"][index]
    matrix=reconstruct(order,FAMILIES[index%4]);ii,jj=np.triu_indices(81,1);truth=matrix[ii,jj]
    # This cohort's inherited degrees were already fixed and audited. Reject
    # a provenance change instead of silently changing the old comparison.
    assert (matrix.sum(0)==degree).all()
    repeated=audit_fit(model,rec,independent_counts(rows),degree,index,p,truth)
    parentp=read(PARENT_PROTOCOL)
    folder=PARENT/"fits"/f"b{index:03d}"
    parentrec=read(folder/"pooled.json");parentmodel=npz(folder/"pooled.npz")
    parentseal=read(PARENT/"final-seal.json")
    for ext in ["json","npz"]:
        rel=f"fits/b{index:03d}/pooled.{ext}"
        assert sha(PARENT/rel)==parentseal["files"][rel]
    assert parentrec["protocol_sha256"]==sha(PARENT_PROTOCOL) and parentrec["degree"]==degree
    assert parentrec["models_sha256"]==sha(folder/"pooled.npz") and parentrec["log_sha256"]==sha(folder/"pooled.log")
    config=dict(p["solver"],random_seed_base=parentp["seeds"]["solver"]+44*index+42)
    baseline=audit_fit(parentmodel,parentrec,independent_counts(raw["noisy"]),degree,0,{"solver":config},truth)
    inherited=read(PARENT/"audits"/f"b{index:03d}.json")["methods"]["pooled"]
    for key in ["classification","integral","missing_true","extra"]:assert baseline[key]==inherited[key]
    assert baseline["true_gap_twice"]==inherited["truth_gap_twice"]
    clean=raw["clean"][:600]
    same=[int(np.count_nonzero((left==right).all(1))) for left,right in
        [(rows[:600],rows[600:1200]),(rows[:600],rows[1200:]),(rows[600:1200],rows[1200:])]]
    blocks=[rows[:600],rows[600:1200],rows[1200:]]
    masks=[block!=clean for block in blocks]
    assert all((mask.sum(1)==9).all() for mask in masks)
    overlaps=[(masks[a]&masks[b]).sum(1) for a,b in [(0,1),(0,2),(1,2)]]
    return dict(index=index,family=FAMILIES[index%4],degree=degree,repeat=repeated,baseline=baseline,
        underlying_source_instances=600,unique_underlying_colorings=len(np.unique(clean,axis=0)),
        standalone_noisy_response_instances=1800,unique_noisy_responses=len(np.unique(rows,axis=0)),
        baseline_unique_underlying_colorings=len(np.unique(raw["clean"],axis=0)),
        baseline_unique_noisy_responses=len(np.unique(raw["noisy"],axis=0)),
        pairwise_identical_response_rows=same,
        pairwise_equal_response_cell_counts=[int((blocks[a]==blocks[b]).sum()) for a,b in [(0,1),(0,2),(1,2)]],
        pairwise_actual_corruption_overlap_totals=[int(v.sum()) for v in overlaps],
        pairwise_actual_corruption_overlap_histograms=[np.bincount(v,minlength=10).tolist() for v in overlaps],
        postfit_overlap_pair_order=["first_repeat1","first_repeat2","repeat1_repeat2"],new_noisy_cell_responses=97200,
        protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),audit_started_monotonic_ns=started,
        fit_seal_sha256=sha(OUT/"fit-seal.json"),source_seal_sha256=sha(OUT/"source-seal.json"))


def worker(worker):
    p=protocol(False);check_seal("fit");check_seal("source")
    for i in range(worker,N,8):
        result=audit_bank(i,p)
        save(OUT/"audits"/f"b{i:03d}.json",result)
        save(OUT/"progress"/f"audit{worker}.json",dict(completed_banks=(i-worker)//8+1,pid=os.getpid()))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--worker",type=int,required=True,choices=range(8));worker(p.parse_args().worker)
