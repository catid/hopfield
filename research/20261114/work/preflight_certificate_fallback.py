"""Boundaries and data isolation for label-free numerical fallback routing."""
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
from certificate_fallback_policy import accepted, chosen_model


def check():
    start = time.perf_counter()
    fields = list(itertools.product([False, True], repeat=5))
    margins = [-math.inf, math.nan, math.inf, -1e-4,
               np.nextafter(-1e-6, -math.inf), -1e-6,
               np.nextafter(-1e-6, math.inf), 0., 1.]
    rows = [(f, m) for f in fields for m in margins]
    flags = np.array([r[0] for r in rows], dtype=bool)
    delta = np.array([r[1] for r in rows])
    actual = accepted(*(flags[:, i] for i in range(5)), delta)
    expected = np.array([
        all(f[:3]) and not any(f[3:]) and math.isfinite(m) and m >= -1e-6
        for f, m in rows
    ])
    assert np.array_equal(actual, expected)
    assert np.array_equal(chosen_model(actual), np.array([0 if x else 1 for x in expected]))
    assert list(inspect.signature(accepted).parameters) == [
        'hidden_good', 'finite', 'stopped', 'native_cap', 'block_cap', 'margin']
    # Shape mismatches must not broadcast into unexpected routing populations.
    rejected = False
    try:
        accepted(flags[:, 0], flags[:1, 1], flags[:, 2], flags[:, 3], flags[:, 4], delta)
    except ValueError:
        rejected = True
    assert rejected
    return {
        'status': 'passed', 'synthetic_certificate_cases': len(rows),
        'exact_tolerance_boundary_checked': True,
        'nonfinite_margins_rejected': True,
        'shape_broadcast_rejected': True,
        'interface_has_no_labels_graph_or_indices': True,
        'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ['certificate_fallback_policy.py', 'preflight_certificate_fallback.py']},
        'seconds': time.perf_counter()-start,
        'scope': 'Synthetic routing preflight only; no source outcomes or GPU trajectories.'
    }


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
