"""H4be production fit: noisy responses and inherited degree only."""
import argparse
import os
import time
from repeat_diversity_common import *
from larger_donor_math import counts
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate


def fit_worker(worker):
    p=protocol(False)
    source=check_seal("source")
    ii,jj=np.triu_indices(81,1)
    for i in range(worker,N,8):
        started=time.monotonic_ns();assert started>source["sealed_monotonic_ns"]
        observed=OUT/"observed"/f"b{i:03d}.npz"
        assert sha(observed)==source["files"][f"observed/b{i:03d}.npz"]
        arrays=npz(observed);assert set(arrays)=={"noisy"}
        rows=arrays["noisy"];assert rows.shape==(1800,81) and rows.dtype==np.uint8 and (rows<9).all()
        degree=p["inherited_degrees"][i];assert type(degree) is int and 1<=degree<=39
        cost=counts(rows)
        folder=OUT/"fits";log=folder/f"b{i:03d}.log"
        assert not any((folder/f"b{i:03d}.{ext}").exists() for ext in ["log","json","npz"])
        x,y,meta=solve(81,degree,ii,jj,cost,p["seeds"]["solver_base"]-2026111502+i,log)
        if meta["solver_optimal"] and meta["value_valid"] and meta["dual_valid"]:
            cert,proof=attempt_certificate(81,degree,ii,jj,cost,x,y)
        else:
            cert=dict(certified=False,reason="solver_status_or_validity")
            proof={key:np.empty(0,np.int64) for key in ["primal_twice","node_dual_twice","upper_dual_twice"]}
        np.savez_compressed(folder/f"b{i:03d}.npz",cost=cost,primal_values=x,node_dual_values=y,**proof)
        save(folder/f"b{i:03d}.json",dict(index=i,arm="repeat",degree=degree,solver=meta,certificate=cert,
            protocol_sha256=sha(PROTOCOL),source_seal_sha256=sha(OUT/"source-seal.json"),
            observed_sha256=sha(observed),models_sha256=sha(folder/f"b{i:03d}.npz"),log_sha256=sha(log),
            fit_started_monotonic_ns=started,fit_completed_monotonic_ns=time.monotonic_ns()))
        save(OUT/"progress"/f"fit{worker}.json",dict(completed_banks=(i-worker)//8+1,pid=os.getpid()))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--worker",type=int,required=True,choices=range(8));fit_worker(p.parse_args().worker)
