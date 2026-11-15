"""H4bc independent source/count/proof reconstruction after the final seal."""
import argparse
import os
import time
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import numpy as np
from fresh_pooled_donor_common import *
from audit_degree_prior import check_certificate
from larger_donor_math import independent_counts
from conflict_mask_math import independent_detector
from corruption_censor_math import independent_score
from fresh_pooled_donor_bp import independent_transform, CONFIG


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def certificate(arrays, record, cost, degree, index, slot, p, prefix=""):
    ii, jj = np.triu_indices(81, 1)
    assert record["degree"] == degree and record["slot"] == slot
    meta = record["solver"]
    assert meta["highs_version"] == "1.12.0" and meta["numpy"] == "2.3.4"
    for name, value in p["solver_options"].items():
        assert meta["solver_options"][name] == value
    assert meta["solver_options"]["random_seed"] == p["seeds"]["solver"]+44*index+slot
    proof = [arrays[prefix+key] for key in ["primal_twice", "node_dual_twice", "upper_dual_twice"]]
    valid = check_certificate(81, degree, ii, jj, cost, *proof)
    assert valid == record["certificate"]["certified"]
    if not valid:
        return None
    assert meta["solver_optimal"] and meta["value_valid"] and meta["dual_valid"]
    assert arrays[prefix+"primal_values"].dtype == arrays[prefix+"node_dual_values"].dtype == np.float64
    assert np.array_equal(arrays[prefix+"primal_values"]*2, proof[0])
    assert np.max(abs(arrays[prefix+"node_dual_values"]*2-proof[1])) <= 1e-6
    objective = sum(int(c)*int(v) for c, v in zip(cost, proof[0], strict=True))
    cert = record["certificate"]
    assert objective == cert["primal_objective_twice"] == cert["dual_objective_twice"]
    assert cert["primal_rounding_used"] is False and cert["integral"] == bool(np.isin(proof[0], [0, 2]).all())
    assert abs(meta["solver_objective"]-objective/2) < 1e-5
    for which in range(3):
        bad = [a.copy() for a in proof]
        bad[which][0] += 1
        assert not check_certificate(81, degree, ii, jj, cost, *bad)
    return objective


def outcome(primal, truth, degree=None, cost=None, objective=None):
    result = dict(classification="unresolved", integral=None, missing_true=None, extra=None,
                  truth_degree_feasible=None, truth_gap_twice=None)
    if primal.size == 0:
        return result
    assert primal.shape == truth.shape and primal.dtype == np.int64 and np.isin(primal, [0, 1, 2]).all()
    integral = bool(np.isin(primal, [0, 2]).all())
    missing = int(np.count_nonzero(truth & (primal != 2))) if integral else None
    extra = int(np.count_nonzero(~truth & (primal == 2))) if integral else None
    result.update(integral=integral, missing_true=missing, extra=extra,
                  classification=("success" if missing == extra == 0 else "resolved_failure_nonexact") if integral else "resolved_failure_fractional")
    if degree is not None:
        ii, jj = np.triu_indices(81, 1)
        degrees = np.bincount(np.concatenate([ii[truth], jj[truth]]), minlength=81)
        feasible = bool((degrees == degree).all())
        result["truth_degree_feasible"] = feasible
        if feasible:
            assert objective is not None and cost is not None
            gap = 2*sum(map(int, cost[truth]))-objective
            assert gap >= 0
            result["truth_gap_twice"] = gap
    return result


def audit_bank(index, p):
    started = time.monotonic_ns()
    final_seal = read(OUT/"final-seal.json")
    assert started > final_seal["sealed_monotonic_ns"]
    source = read(OUT/"source_audit"/f"b{index:03d}.json")
    assert source["status"] == "passed" and source["protocol_sha256"] == sha(PROTOCOL)
    raw_file = OUT/"raw"/f"b{index:03d}.npz"
    observed_file = OUT/"observed"/f"b{index:03d}.npz"
    assert sha(raw_file) == source["raw_sha256"] and sha(observed_file) == source["observed_sha256"]
    raw_record = read(raw_file.with_suffix(".json"))
    order_seal = read(OUT/"orders-seal.json")
    source_seal = read(OUT/"source-seal.json")
    assert raw_record["raw_sha256"] == sha(raw_file)
    assert order_seal["protocol_sha256"] == source_seal["protocol_sha256"] == sha(PROTOCOL)
    assert order_seal["orders_sha256"] == sha(OUT/"orders.npz")
    assert order_seal["sealed_monotonic_ns"] < raw_record["source_started_monotonic_ns"] <= raw_record["source_finished_monotonic_ns"] < source_seal["sealed_monotonic_ns"]
    raw = npz(raw_file)
    rows = npz(observed_file)["noisy"]
    assert np.array_equal(rows, raw["noisy"]) and rows.dtype == np.uint8
    orders = npz(OUT/"orders.npz")
    from fresh_pooled_donor_source import audit_source
    replay = audit_source(index, p, raw, read(raw_file.with_suffix(".json"))["sampler_receipt"],
                          orders["orders"][index], orders["hot_cells"][index])
    for key, value in replay.items():
        assert source[key] == value
    family = FAMILIES[index % 4]
    from audit_objective_gap import reconstruct
    ii, jj = np.triu_indices(81, 1)
    matrix = reconstruct(orders["orders"][index], family)
    truth = matrix[ii, jj]
    folder = OUT/"fits"/f"b{index:03d}"
    records, arrays = {}, {}
    for arm in ["grid", "donor", "pooled", "masked", "bp"]:
        record = read(folder/f"{arm}.json")
        assert record["index"] == index and record["arm"] == arm and record["protocol_sha256"] == sha(PROTOCOL)
        assert record["models_sha256"] == sha(folder/f"{arm}.npz")
        assert record["fit_started_monotonic_ns"] <= record["fit_completed_monotonic_ns"] < final_seal["sealed_monotonic_ns"]
        parent = {"grid": "source", "donor": "grid", "pooled": "grid", "masked": "donor", "bp": "source"}[arm]
        assert record["parent_seal_sha256"] == sha(OUT/f"{parent}-seal.json")
        assert record["fit_started_monotonic_ns"] > read(OUT/f"{parent}-seal.json")["sealed_monotonic_ns"]
        records[arm], arrays[arm] = record, npz(folder/f"{arm}.npz")
    grid, rec = arrays["grid"], records["grid"]
    target_cost = independent_counts(rows[:600])
    assert grid["cost"].dtype == np.int64 and np.array_equal(grid["cost"], target_cost)
    assert len(rec["records"]) == 40
    assert check_certificate(81, 0, ii, jj, target_cost, *[grid["d00_"+k] for k in ["primal_twice", "node_dual_twice", "upper_dual_twice"]])
    values = [0]
    certified = fractional = attempted = 0
    for degree, r in enumerate(rec["records"], 1):
        assert r["log_sha256"] == sha(folder/f"d{degree:02d}.log")
        value = certificate(grid, r, target_cost, degree, index, degree, p, f"d{degree:02d}_")
        values.append(value)
        attempted += 1
        if value is not None:
            certified += 1
            fractional += not np.isin(grid[f"d{degree:02d}_primal_twice"], [0, 2]).all()
    if all(v is not None for v in values):
        second = [values[d+1]+values[d-1]-2*values[d] for d in range(1, 40)]
        chosen = second.index(max(second))+1
    else:
        second, chosen = None, None
    assert rec["selected_degree"] == chosen and rec["second_difference"] == second
    results = {}
    for arm, slot, data in [("donor", 41, rows[600:]), ("pooled", 42, rows), ("masked", 43, rows[:600])]:
        r, a = records[arm], arrays[arm]
        assert r["degree"] == chosen
        allowed = chosen is not None and (arm != "masked" or records["donor"]["certificate"]["certified"])
        assert r["executed"] == allowed
        if not allowed:
            assert set(a) == {"unexecuted"} and a["unexecuted"].dtype == bool and a["unexecuted"].item()
            assert r["certificate"]["certified"] is False
            results[arm] = outcome(np.empty(0, np.int64), truth)
            continue
        if arm == "masked":
            scores, mask = independent_detector(data, arrays["donor"]["primal_twice"])
            counts = independent_score(data, mask)
            for key, value in dict(scores=scores, mask=mask, **counts).items():
                assert a[key].dtype == value.dtype and np.array_equal(a[key], value)
            cost = counts["cost"]
        else:
            cost = independent_counts(data)
            assert a["cost"].dtype == np.int64 and np.array_equal(a["cost"], cost)
        assert r["log_sha256"] == sha(folder/f"{arm}.log")
        value = certificate(a, r, cost, chosen, index, slot, p)
        attempted += 1
        certified += value is not None
        results[arm] = outcome(a["primal_twice"] if value is not None else np.empty(0, np.int64), truth,
                               chosen, cost, value)
    bp, bpr = arrays["bp"], records["bp"]
    env = bpr["environment"]
    assert env["numpy"] == "2.5.4" and env["physical_GPU"] == bpr["gpu"] == index % 4
    assert env["hostname"] == "foureyes" and env["TF32"] is False and env["CUDA_runtime"]
    assert env["device_name"] and env["torch"] and env["device_total_memory"] > 0
    assert bp["W"].dtype == bp["B"].dtype == np.float32 and bp["W"].shape == (3240, 9, 9) and bp["B"].shape == (81, 9)
    assert np.isfinite(bp["W"]).all() and np.isfinite(bp["B"]).all()
    support, coefficients, eligible = independent_transform(np, bp["W"])
    assert np.array_equal(support, bp["support"]) and np.max(abs(coefficients-bp["coefficients"])) < 1e-12
    assert eligible == bpr["transform"]["eligible"] and bpr["status"] == "finite" and bpr["config"] == CONFIG
    assert bpr["augmentation_seed"] == p["seeds"]["adamw_augmentation"]+index and bpr["updates"] == len(bpr["history"]) == 32
    assert abs(bpr["history"][0]-np.log(9)) < 1e-6 and all(np.isfinite(v) and v >= 0 for v in bpr["history"])
    assert 0 < bpr["optimizer_loop_seconds"] <= bpr["train_and_transform_wall_seconds"] <= (bpr["fit_completed_monotonic_ns"]-bpr["fit_started_monotonic_ns"])/1e9 < 30
    results["bp"] = dict(outcome(2*support.astype(np.int64), truth), eligible=eligible)
    return dict(index=index, family=family, selected_degree=chosen, actual_degree=int(matrix.sum(0)[0]),
                methods=results, attempted_LPs=attempted, certified_LPs=certified,
                intermediate_fractional_LPs=int(fractional), mutations_rejected=3*certified,
                protocol_sha256=sha(PROTOCOL), auditor_sha256=sha(__file__), audit_started_monotonic_ns=started,
                final_seal_sha256=sha(OUT/"final-seal.json"), analytic_zero_witness=True)


def aggregate(rows):
    assert [r["index"] for r in rows] == list(range(N))
    statuses = ["success", "resolved_failure_nonexact", "resolved_failure_fractional", "unresolved"]
    grouped = []
    for arm in ["bp", "pooled", "masked", "donor"]:
        values = [r["methods"][arm]["classification"] for r in rows]
        grouped.append(dict(arm=arm, **{k: values.count(k) for k in statuses},
                            families=[dict(family=f, exact=sum(r["family"] == f and r["methods"][arm]["classification"] == "success" for r in rows)) for f in FAMILIES]))
    pooled = [r["methods"]["pooled"]["classification"] for r in rows]
    masked = [r["methods"]["masked"]["classification"] for r in rows]
    return dict(grouped=grouped, primary=bounds(pooled, masked), all256_banks_retained=True,
                paired_gains=sum(a == "success" and b.startswith("resolved_failure") for a, b in zip(pooled, masked)),
                paired_losses=sum(b == "success" and a.startswith("resolved_failure") for a, b in zip(pooled, masked)),
                attempted_LPs=sum(r["attempted_LPs"] for r in rows), certificates_checked=sum(r["certified_LPs"] for r in rows),
                mutations_rejected=sum(r["mutations_rejected"] for r in rows), analytic_zero_witnesses=N)


def worker(worker_id):
    p = protocol()
    check_seal("final")
    for index in range(worker_id, N, 8):
        result = audit_bank(index, p)
        save(OUT/"audits"/f"b{index:03d}.json", result)
        save(OUT/"progress"/f"audit{worker_id}.json", dict(completed_banks=(index-worker_id)//8+1, pid=os.getpid()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True, choices=range(8))
    worker(parser.parse_args().worker)
