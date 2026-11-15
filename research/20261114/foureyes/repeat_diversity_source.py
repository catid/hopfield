"""H4be generation/audit namespace; fitting never receives clean or noise arrays."""
import argparse
import os
import time
from repeat_diversity_common import *


def original(index):
    parentp=read(PARENT_PROTOCOL)
    raw=PARENT_RAW/"raw"/f"b{index:03d}.npz"
    record=read(PARENT/"raw"/f"b{index:03d}.json")
    prior=read(PARENT/"source_audit"/f"b{index:03d}.json")
    observed=PARENT/"observed"/f"b{index:03d}.npz"
    assert sha(raw)==record["raw_sha256"]==prior["raw_sha256"]
    assert sha(observed)==prior["observed_sha256"]
    a=npz(raw);orders=npz(PARENT/"orders.npz")
    from fresh_pooled_donor_source import audit_source
    check=audit_source(index,parentp,a,record["sampler_receipt"],orders["orders"][index],orders["hot_cells"][index])
    for key,value in check.items():assert prior[key]==value
    assert np.array_equal(a["noisy"],npz(observed)["noisy"])
    return a,orders["hot_cells"][index]


def source_worker(worker):
    p=protocol(False)
    for i in range(worker,N,8):
        started=time.monotonic_ns()
        raw,hot=original(i)
        seeds=[p["seeds"][key]+i for key in ["repeat1","repeat2"]]
        arrays=repeats(raw["clean"][:600],hot,seeds)
        independent_repeats(raw["clean"][:600],hot,seeds,arrays)
        rows=assemble(raw["noisy"][:600],arrays["r1_noisy"],arrays["r2_noisy"])
        noise_file=OUT/"noise"/f"b{i:03d}.npz"
        observed=OUT/"observed"/f"b{i:03d}.npz"
        assert not noise_file.exists() and not observed.exists()
        np.savez_compressed(noise_file,**arrays)
        np.savez_compressed(observed,noisy=rows)
        save(OUT/"source_audit"/f"b{i:03d}.json",dict(status="passed",index=i,protocol_sha256=sha(PROTOCOL),
            observed_sha256=sha(observed),noise_sha256=sha(noise_file),
            original_source_audit_sha256=sha(PARENT/"source_audit"/f"b{i:03d}.json"),
            original_observed_sha256=sha(PARENT/"observed"/f"b{i:03d}.npz"),
            original_raw_sha256=sha(PARENT_RAW/"raw"/f"b{i:03d}.npz"),
            new_noisy_board_responses=1200,reused_noisy_board_responses=600,new_noisy_cell_responses=97200,
            candidate_standalone_noisy_cell_responses=145800,first_response_byte_values_unchanged=True,
            original_full_source_independently_replayed=True,new_repeats_independently_replayed=True,
            source_started_monotonic_ns=started,source_finished_monotonic_ns=time.monotonic_ns()))
        save(OUT/"progress"/f"source{worker}.json",dict(completed_banks=(i-worker)//8+1,pid=os.getpid()))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--worker",type=int,required=True,choices=range(8));source_worker(p.parse_args().worker)
