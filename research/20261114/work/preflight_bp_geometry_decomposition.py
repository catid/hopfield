"""Synthetic-only gate: never reads any source cohort outcome or state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np
from bp_geometry_checks import simple_path, classify, partition, endpoint
from audit_connectivity_screen import path_valid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate():
    started = time.perf_counter()
    tokens = np.ones((4, 4), dtype=np.uint8)
    tokens[0, 0] = 2
    tokens[3, 3] = 3
    tokens[0, 3] = tokens[3, 0] = 0
    free = np.flatnonzero(tokens.ravel() == 1)
    assert len(free) == 12
    valid = 0
    for mask in range(1 << 12):
        board = tokens >= 2
        board.ravel()[free] = [(mask >> bit) & 1 for bit in range(12)]
        actual = simple_path(tokens, board)
        assert actual == path_valid(tokens, board), mask
        valid += int(actual)
    t = np.ones((5, 5), dtype=np.uint8)
    t[2, 0] = 2
    t[2, 4] = 3
    straight = np.zeros_like(t, dtype=bool)
    straight[2, :] = True
    long = np.zeros_like(straight)
    long[0, :] = True
    long[:3, 0] = long[:3, 4] = True
    checks = [classify(t, straight), classify(t, long)]
    assert checks[0] == {'graph_valid_path': 1, 'graph_valid_shortest': 1,
                         'shortest_edges': 4, 'selected_path_edges': 4, 'extra_edges': 0}
    assert checks[1] == {'graph_valid_path': 1, 'graph_valid_shortest': 0,
                         'shortest_edges': 4, 'selected_path_edges': 8, 'extra_edges': 4}
    negatives = []
    for cell, selected in [((2, 0), False), ((2, 4), False), ((2, 2), False), ((1, 2), True)]:
        b = straight.copy()
        b[cell] = selected
        negatives.append((t, b))
    wall = t.copy()
    wall[2, 2] = 0
    negatives.append((wall, straight))
    detached = np.ones((7, 7), dtype=np.uint8)
    detached[6, 0] = 2
    detached[6, 6] = 3
    b = np.zeros_like(detached, dtype=bool)
    b[6, :] = True
    b[:2, :2] = True
    negatives.append((detached, b))
    for x, y in negatives:
        assert not simple_path(x, y) and not path_valid(x, y)
    malformed = [(t, straight[:2]), (t, straight.astype(int)),
                 (t.ravel(), straight.ravel()), (np.ones_like(t), straight),
                 (np.where(t == 1, 4, t), straight)]
    for x, y in malformed:
        try:
            simple_path(x, y)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed input accepted')
    assert [partition(*x) for x in [(0, 0), (1, 0), (1, 1)]] == ['invalid_geometry', 'valid_longer', 'valid_shortest']
    for x in [(0, 1), (2, 0), (0, -1)]:
        try:
            partition(*x)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed graph partition accepted')
    assert endpoint(458, 510)['decision'] == 'falsified'
    assert endpoint(459, 510)['decision'] == 'supported'
    assert endpoint(510, 510)['decision'] == 'supported'
    for n, d in [(0, 509), (-1, 510), (511, 510)]:
        try:
            endpoint(n, d)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed denominator accepted')
    analyzer = Path(__file__).with_name('analyze_bp_geometry_decomposition.py')
    with tempfile.TemporaryDirectory(prefix='h2bm-synthetic-lifecycle-') as temporary:
        tmp = Path(temporary)
        target = tmp / 'failed-output'
        command = [sys.executable, str(analyzer), str(tmp / 'missing-protocol'), str(tmp / 'missing-source'), str(target)]
        failure = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert failure.returncode != 0
        incomplete = target / 'incomplete.json'
        receipt = json.loads(incomplete.read_text())
        assert receipt['status'] == 'incomplete' and receipt['error_type'] == 'FileNotFoundError'
        initial_hash = sha(incomplete)
        refused = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert refused.returncode != 0 and 'FileExistsError' in refused.stderr
        assert sha(incomplete) == initial_hash
        fifo = tmp / 'blocked-protocol'
        os.mkfifo(fifo)
        target = tmp / 'watchdog-output'
        child = subprocess.Popen([sys.executable, str(analyzer), str(fifo), str(tmp / 'missing-source'), str(target)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not target.exists():
                assert child.poll() is None and time.monotonic() < deadline
                time.sleep(0.01)
            # The runner installs its handler immediately after creating output.
            time.sleep(0.05)
            child.send_signal(signal.SIGALRM)
            child.communicate(timeout=5)
            assert child.returncode != 0
            receipt = json.loads((target / 'incomplete.json').read_text())
            assert receipt['status'] == 'incomplete' and receipt['error_type'] == 'TimeoutError'
            assert receipt['wall_budget_seconds'] == 120
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
    return {'status': 'passed', 'synthetic_only': True, 'truth_table_boards': 4096,
            'truth_table_valid_paths': valid, 'explicit_shortest_and_longer_paths': checks,
            'explicit_invalid_geometry_cases': len(negatives), 'malformed_array_cases': len(malformed),
            'partition_checks': 6, 'endpoint_boundary_and_denominator_checks': 6,
            'incomplete_receipt_preserved': True, 'existing_output_refused': True,
            'watchdog_signal_incomplete_receipt': True,
            'seconds': time.perf_counter() - started,
            'source_sha256': {p.name: sha(p) for p in [Path(__file__), analyzer, Path(__file__).with_name('bp_geometry_checks.py'), Path(__file__).with_name('audit_connectivity_screen.py')]}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = gate()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
