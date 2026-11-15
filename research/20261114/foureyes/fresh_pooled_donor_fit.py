"""H4bc production fits read only noisy arrays and previously sealed fits."""
import argparse
import os
import time
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import numpy as np
from fresh_pooled_donor_common import *
from larger_donor_math import counts

PROOF = ["primal_twice", "node_dual_twice", "upper_dual_twice"]


def lp(cost, degree, index, slot, folder, name, p):
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate
    ii, jj = np.triu_indices(81, 1)
    seed = p["seeds"]["solver"] + 44*index + slot
    log = folder / f"{name}.log"
    x, y, solver = solve(81, degree, ii, jj, cost, seed-2026111502, log)
    if solver["solver_optimal"] and solver["value_valid"] and solver["dual_valid"]:
        certificate, proof = attempt_certificate(81, degree, ii, jj, cost, x, y)
    else:
        certificate = dict(certified=False, reason="solver_status_or_validity")
        proof = {k: np.empty(0, np.int64) for k in PROOF}
    return dict(primal_values=x, node_dual_values=y, **proof), dict(degree=degree, slot=slot,
                solver=solver, certificate=certificate, log_sha256=sha(log))


def selected_degree(records):
    if len(records) != 40 or not all(r["certificate"]["certified"] for r in records):
        return None, None
    objectives = [0] + [r["certificate"]["primal_objective_twice"] for r in records]
    second = [objectives[i+1]-2*objectives[i]+objectives[i-1] for i in range(1, 40)]
    degree = max(range(1, 40), key=lambda i: (second[i-1], -i))
    return degree, second


def write_fit(folder, name, arrays, record, index, started):
    assert not (folder/f"{name}.npz").exists() and not (folder/f"{name}.json").exists()
    np.savez_compressed(folder/f"{name}.npz", **arrays)
    record.update(index=index, arm=name, protocol_sha256=sha(PROTOCOL),
                  fit_started_monotonic_ns=started, fit_completed_monotonic_ns=time.monotonic_ns(),
                  models_sha256=sha(folder/f"{name}.npz"))
    save(folder/f"{name}.json", record)


def skip(folder, name, index, started, degree, reason, parent_seal):
    write_fit(folder, name, dict(unexecuted=np.array(True)),
              dict(executed=False, degree=degree, reason=reason, parent_seal_sha256=parent_seal,
                   certificate=dict(certified=False)), index, started)


def grid_bank(index, p):
    folder = OUT/"fits"/f"b{index:03d}"
    folder.mkdir(exist_ok=True)
    started = time.monotonic_ns()
    cost = counts(observations(index)[:600])
    arrays = dict(cost=cost, d00_primal_twice=np.zeros(3240, np.int64),
                  d00_node_dual_twice=np.zeros(81, np.int64), d00_upper_dual_twice=np.zeros(3240, np.int64))
    records = []
    for degree in range(1, 41):
        proof, record = lp(cost, degree, index, degree, folder, f"d{degree:02d}", p)
        arrays.update({f"d{degree:02d}_{key}": value for key, value in proof.items()})
        records.append(record)
    selected, second = selected_degree(records)
    write_fit(folder, "grid", arrays, dict(records=records, selected_degree=selected, second_difference=second,
              parent_seal_sha256=sha(OUT/"source-seal.json")), index, started)


def donor_bank(index, p):
    folder = OUT/"fits"/f"b{index:03d}"
    degree = read(folder/"grid.json")["selected_degree"]
    parent_seal = sha(OUT/"grid-seal.json")
    rows = observations(index)
    for name, slot, data in [("donor", 41, rows[600:]), ("pooled", 42, rows)]:
        started = time.monotonic_ns()
        if degree is None:
            skip(folder, name, index, started, degree, "required_grid_uncertified", parent_seal)
            continue
        cost = counts(data)
        arrays, record = lp(cost, degree, index, slot, folder, name, p)
        write_fit(folder, name, dict(cost=cost, **arrays), dict(executed=True, **record,
                  parent_seal_sha256=parent_seal), index, started)


def masked_bank(index, p):
    from conflict_mask_math import detector
    from corruption_censor_math import score
    folder = OUT/"fits"/f"b{index:03d}"
    donor = read(folder/"donor.json")
    degree = donor["degree"]
    started = time.monotonic_ns()
    parent_seal = sha(OUT/"donor-seal.json")
    if not donor["certificate"]["certified"]:
        skip(folder, "masked", index, started, degree, "required_donor_uncertified", parent_seal)
        return
    with np.load(folder/"donor.npz", allow_pickle=False) as z:
        primal = z["primal_twice"]
    target = observations(index)[:600]
    scores, mask = detector(target, primal)
    objective = score(target, mask)
    arrays, record = lp(objective["cost"], degree, index, 43, folder, "masked", p)
    write_fit(folder, "masked", dict(**objective, scores=scores, mask=mask, **arrays),
              dict(executed=True, **record, parent_seal_sha256=parent_seal), index, started)


def worker(phase, worker_id):
    p = protocol()
    check_seal({"grid": "source", "donor": "grid", "masked": "donor"}[phase])
    for index in range(worker_id, N, 8):
        globals()[phase+"_bank"](index, p)
        save(OUT/"progress"/f"{phase}{worker_id}.json", dict(completed_banks=(index-worker_id)//8+1, pid=os.getpid()))


def bp_worker(gpu):
    import signal
    import resource
    from weak_weight_study import setup
    from fresh_pooled_donor_bp import train, CONFIG
    started = time.monotonic()
    p = protocol()
    check_seal("source")
    lock, torch, _, *_ = setup(gpu)
    import noisy_pl_long as base
    _, _, CF = base.dependencies()
    device = torch.cuda.get_device_properties(0)
    environment = dict(torch=str(torch.__version__), CUDA_runtime=torch.version.cuda,
                       numpy=np.__version__, device_name=device.name,
                       device_capability=[device.major, device.minor], device_total_memory=device.total_memory,
                       physical_GPU=gpu, TF32=bool(torch.backends.cuda.matmul.allow_tf32),
                       PID=os.getpid(), hostname=os.uname().nodename)
    def timeout(signum, frame):
        raise TimeoutError("Frozen 30-second BP job cap")
    previous = signal.signal(signal.SIGALRM, timeout)
    try:
        for index in range(gpu, N, 4):
            assert time.monotonic()-started < 600
            folder = OUT/"fits"/f"b{index:03d}"
            folder.mkdir(exist_ok=True)
            job_start = time.monotonic_ns()
            save(OUT/f"bp-active{gpu}.json", dict(status="running", index=index, gpu=gpu, pid=os.getpid(),
                 job_started_monotonic_ns=job_start, job_cap_seconds=30))
            signal.setitimer(signal.ITIMER_REAL, 30)
            try:
                arrays, record = train(torch, np, CF, observations(index), p["seeds"]["adamw_augmentation"]+index)
                record.update(executed=True, status="finite", config=CONFIG, gpu=gpu, environment=environment,
                              parent_seal_sha256=sha(OUT/"source-seal.json"))
                write_fit(folder, "bp", arrays, record, index, job_start)
                assert (time.monotonic_ns()-job_start)/1e9 < 30, "BP_job_write_deadline"
                save(OUT/f"bp-active{gpu}.json", dict(status="idle", index=index, gpu=gpu, pid=os.getpid()))
            except BaseException as exc:
                save(folder/"bp-failure.json", dict(status="unresolved", error=repr(exc), index=index, gpu=gpu, environment=environment, whole_study_incomplete=True, no_primary_decision=True))
                raise
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            save(OUT/"progress"/f"bp{gpu}.json", dict(completed_banks=(index-gpu)//4+1, pid=os.getpid(),
                 wall_seconds=time.monotonic()-started, CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime))
    finally:
        signal.signal(signal.SIGALRM, previous)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["grid", "donor", "masked", "bp"])
    parser.add_argument("--worker", type=int, required=True)
    args = parser.parse_args()
    if args.phase == "bp":
        assert args.worker in range(4)
        bp_worker(args.worker)
    else:
        assert args.worker in range(8)
        worker(args.phase, args.worker)
