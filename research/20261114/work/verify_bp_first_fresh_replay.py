"""H2bo replay/subset trace comparison against independently audited pair0."""
import numpy as np
import json
from audit_bp_first_geometry import sha, trace_by_input, terminal_by_input
from audit_bp_first_fresh_states import numeric_error


def check_trace_pair(actual_directory, row, source_directory, control_row, subset_indices, size):
    selected = set(map(int, subset_indices))
    expected_arms = ['connectivity_veto', 'original_lexical16', 'unified16', 'unified48', 'unified64', 'unified96']
    if not len(subset_indices):
        assert not row['trace_files']
        return 0, 0, 0.
    assert len(row['trace_files']) == 6
    candidates = active_states = 0
    maximum = 0.
    for arm, filename in zip(expected_arms, row['trace_files']):
        actual_path = actual_directory / filename
        assert sha(actual_path) == row['trace_sha256'][filename]
        source_name = control_row['trace_files'][expected_arms.index(arm)]
        source_path = source_directory / source_name
        assert sha(source_path) == control_row['trace_sha256'][source_name]
        with np.load(actual_path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        with np.load(source_path, allow_pickle=False) as z:
            b = {k: z[k] for k in z.files}
        ap, bp = json.loads(a['trace_utf8'].tobytes()), json.loads(b['trace_utf8'].tobytes())
        assert ap['arm'] == bp['arm'] == arm
        assert ap['protocol_sha256'] == bp['protocol_sha256'] and ap['offset'] == bp['offset']
        actual_owners = subset_indices[a['valid_prefix_indices']]
        original_owners = b['valid_prefix_indices']
        assert np.array_equal(actual_owners, original_owners[np.isin(original_owners, subset_indices)])
        assert np.array_equal(a['initial_terminal'], b['initial_terminal'][subset_indices])
        for field, ao, bo in [('initial_terminal_checks', subset_indices, np.arange(size)),
                              ('final_terminal_checks', actual_owners, original_owners)]:
            assert terminal_by_input(ap[field], ao, selected) == terminal_by_input(bp[field], bo, selected)
        aa = trace_by_input(ap, a, actual_owners, selected)
        bb = trace_by_input(bp, b, original_owners, selected)
        assert aa.keys() == bb.keys()
        for owner in aa:
            assert len(aa[owner]) == len(bb[owner])
            for x, y in zip(aa[owner], bb[owner]):
                assert x[0:2] == y[0:2] and x[3:] == y[3:]
                assert len(x[2]) == len(y[2])
                for c, d in zip(x[2], y[2]):
                    assert c[0] == d[0] and c[2] == d[2]
                    error = numeric_error(c[1], d[1], 1e-8)
                    maximum = max(maximum, error)
                    candidates += 1
                active_states += 1
        positions = np.flatnonzero(np.isin(original_owners, subset_indices))
        for field in ['pre_terminal_stopped', 'pre_terminal_block_cap']:
            expected = b[field][positions] if len(b[field]) else np.zeros(0, bool)
            assert np.array_equal(a[field], expected)
    return candidates, active_states, maximum

