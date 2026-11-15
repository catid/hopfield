"""Fresh synthetic H4bc source production and independent provenance replay."""
import argparse
import os
import time
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import numpy as np
from fresh_pooled_donor_common import *


def source(index, p, order, hot):
    family = FAMILIES[index % 4]
    if family in ["cycle81", "cliques9"]:
        from varying_degree_sampling import sample
    else:
        from cycle_power_sampling import sample
    from regularity_intensity_study import corrupt
    clean, receipt = sample(order, family, p["seeds"]["coloring"]+index, rows=1800)
    rng = np.random.default_rng(p["seeds"]["noise"]+index)
    free = np.ones((1800, 81), bool)
    uniforms = 1-rng.random((1800, 81))
    offsets = rng.integers(1, 9, (1800, 81), dtype=np.int64)
    noisy, cells = corrupt(np, clean, free, uniforms, offsets, hot, "hot18_weight32")
    return dict(clean=clean, noisy=noisy, cells=cells, uniforms=uniforms, offsets=offsets), receipt


def audit_source(index, p, arrays, receipt, order, hot):
    from audit_regularity_intensity import literal_corruption
    family = FAMILIES[index % 4]
    if family in ["cycle81", "cliques9"]:
        from audit_varying_degree_controls import independent_colorings, independent_graph
    else:
        from audit_cycle_power_controls import independent_colorings, independent_graph
    expected_order = np.random.default_rng(p["seeds"]["graph"]+index).permutation(81)
    expected_hot = np.sort(np.random.default_rng(p["seeds"]["heatmap"]+index).choice(81, 18, replace=False))
    assert order.dtype == hot.dtype == np.int64
    assert np.array_equal(order, expected_order) and np.array_equal(hot, expected_hot)
    clean, reference = independent_colorings(order, family, p["seeds"]["coloring"]+index, rows=1800)
    assert reference == receipt and np.array_equal(clean, arrays["clean"]) and arrays["clean"].dtype == np.uint8
    truth = independent_graph(order, family)
    ii, jj = np.triu_indices(81, 1)
    assert (clean[:, ii[truth]] != clean[:, jj[truth]]).all()
    rng = np.random.default_rng(p["seeds"]["noise"]+index)
    uniform = 1-rng.random((1800, 81))
    offsets = rng.integers(1, 9, (1800, 81), dtype=np.int64)
    assert arrays["uniforms"].dtype == uniform.dtype and np.array_equal(arrays["uniforms"], uniform)
    assert arrays["offsets"].dtype == offsets.dtype and np.array_equal(arrays["offsets"], offsets)
    noisy, cells = literal_corruption(clean, np.ones((1800, 81), bool), uniform, offsets, hot, "hot18_weight32")
    assert arrays["noisy"].dtype == np.uint8 and np.array_equal(arrays["noisy"], noisy)
    assert arrays["cells"].dtype == np.int64 and np.array_equal(arrays["cells"], cells)
    assert ((clean != noisy).sum(1) == 9).all()
    return dict(status="passed", index=index, rows=1800, generated_candidates=receipt["generated_candidates"],
                clean_duplicate_instances=1800-len(np.unique(clean, axis=0)),
                noisy_duplicate_instances=1800-len(np.unique(noisy, axis=0)),
                provenance_only=True, fitted_outputs_inspected=False)


def orders(p):
    assert not (OUT / "orders.npz").exists()
    order = np.stack([np.random.default_rng(p["seeds"]["graph"]+i).permutation(81) for i in range(N)])
    hot = np.stack([np.sort(np.random.default_rng(p["seeds"]["heatmap"]+i).choice(81, 18, replace=False)) for i in range(N)])
    np.savez_compressed(OUT / "orders.npz", orders=order, hot_cells=hot)
    save(OUT / "orders-seal.json", dict(protocol_sha256=sha(PROTOCOL), orders_sha256=sha(OUT/"orders.npz"),
                                       sealed_monotonic_ns=time.monotonic_ns(), before_colorings_or_noise=True))


def worker(worker_id):
    p = protocol()
    seal = read(OUT / "orders-seal.json")
    assert seal["protocol_sha256"] == sha(PROTOCOL) and sha(OUT/"orders.npz") == seal["orders_sha256"]
    with np.load(OUT / "orders.npz", allow_pickle=False) as z:
        graph_orders, heatmaps = z["orders"], z["hot_cells"]
    for index in range(worker_id, N, 8):
        started = time.monotonic_ns()
        assert started > seal["sealed_monotonic_ns"]
        arrays, receipt = source(index, p, graph_orders[index], heatmaps[index])
        result = audit_source(index, p, arrays, receipt, graph_orders[index], heatmaps[index])
        raw = OUT / "raw" / f"b{index:03d}.npz"
        observed = OUT / "observed" / f"b{index:03d}.npz"
        assert not raw.exists() and not observed.exists()
        np.savez_compressed(raw, **arrays)
        np.savez_compressed(observed, noisy=arrays["noisy"])
        save(raw.with_suffix(".json"), dict(index=index, sampler_receipt=receipt, raw_sha256=sha(raw),
                                           source_started_monotonic_ns=started, source_finished_monotonic_ns=time.monotonic_ns()))
        result.update(protocol_sha256=sha(PROTOCOL), raw_sha256=sha(raw), observed_sha256=sha(observed),
                      auditor_sha256=sha(__file__))
        save(OUT / "source_audit" / f"b{index:03d}.json", result)
        save(OUT / "progress" / f"source{worker_id}.json", dict(completed_banks=(index-worker_id)//8+1, pid=os.getpid()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True, choices=range(8))
    worker(parser.parse_args().worker)
