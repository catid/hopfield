"""Independent public H2bm reconstruction; no analysis or proposed walker imports."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from audit_connectivity_screen import path_valid
from verify_hard_negative_feasibility import valid_shortest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def verify(directory, source):
    started = time.perf_counter()
    protocol = json.loads((directory / 'protocol.json').read_text())
    result = json.loads((directory / 'analysis.json').read_text())
    gate = json.loads((directory / 'synthetic-gate.json').read_text())
    assert result['status'] == gate['status'] == 'passed'
    assert result['protocol_sha256'] == sha(directory / 'protocol.json')
    assert protocol['status'] == 'frozen_before_first_geometry_aggregate'
    assert protocol['synthetic_gate_sha256'] == sha(directory / 'synthetic-gate.json')
    assert protocol['freeze_reads_real_outcome_fields'] is False
    assert gate['synthetic_only'] and gate['truth_table_boards'] == 4096
    assert gate['incomplete_receipt_preserved'] and gate['existing_output_refused'] and gate['watchdog_signal_incomplete_receipt']
    for name, digest in protocol['analysis_source_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name, digest in protocol['sealed_source_sha256'].items():
        assert sha(source / name) == digest
    assert sha(source / 'public-verification.json') == protocol['source_public_verification_sha256']
    original = read_csv(source / 'per-model-input.csv')
    rows = read_csv(directory / 'per-model-input.csv')
    assert sha(directory / 'per-model-input.csv') == result['per_model_input_sha256']
    assert len(rows) == len(original) == 12288
    expected_categories = []
    for old, new in zip(original, rows):
        extra = dict(new)
        category = extra.pop('geometry_partition')
        assert extra == old
        valid, shortest = int(old['graph_valid_path']), int(old['graph_valid_shortest'])
        assert valid in (0, 1) and shortest in (0, 1) and shortest <= valid
        expected = 'valid_shortest' if shortest else 'valid_longer' if valid else 'invalid_geometry'
        assert category == expected
        expected_categories.append(expected)
    counts = Counter(expected_categories)
    assert result['partition'] == {k: counts[k] for k in ['valid_shortest', 'invalid_geometry', 'valid_longer']}
    assert counts['valid_shortest'] == 11778 and counts['invalid_geometry'] + counts['valid_longer'] == 510
    target = result['sole_endpoint']
    assert target['invalid_geometry_failures'] == counts['invalid_geometry']
    assert target['all_shortestness_failures'] == 510 and target['minimum_percent'] == 90
    assert target['observed_percent'] == 100 * counts['invalid_geometry'] / 510
    assert target['decision'] == ('supported' if counts['invalid_geometry'] >= 459 else 'falsified')
    bycase = {(int(r['input_index']), int(r['model_seed'])): r for r in rows}
    assert set(bycase) == {(i, seed) for i in range(4096) for seed in range(3)}
    for seed in range(3):
        selected = [r for r in rows if int(r['model_seed']) == seed]
        values = Counter(r['geometry_partition'] for r in selected)
        assert result['per_seed_partition'][str(seed)] == {k: values[k] for k in result['partition']}
    expected_controls = []
    for field in ['bp_own_accept', 'dual_accept', 'graph_valid_path']:
        good = [r for r in rows if r['graph_valid_shortest'] == '1']
        bad = [r for r in rows if r['graph_valid_shortest'] == '0']
        expected_controls.append({'source_field': field,
            'accepted_shortest': sum(int(r[field]) for r in good),
            'accepted_wrong': sum(int(r[field]) for r in bad),
            'rejected_wrong': sum(1 - int(r[field]) for r in bad)})
    for old, new in zip(result['controls_in_bp_first_order'], expected_controls):
        assert {k: old[k] for k in new} == new
    d = source / 'public-fixtures'
    manifest = json.loads((d / 'manifest.json').read_text())
    fixed = {(r['input_index'], r['model_seed']) for r in manifest['fixed_cases']}
    assert fixed == {(i * 256, seed) for i in range(16) for seed in range(3)}
    negative = {(r['input_index'], r['model_seed']) for r in manifest['negative_or_retention_loss_cases']}
    assert negative == {k for k, r in bycase.items() if r['graph_valid_shortest'] == '0'}
    checks = read_csv(directory / 'fixture-checks.csv')
    assert sha(directory / 'fixture-checks.csv') == result['fixture_checks_sha256']
    records = manifest['fixed_cases'] + manifest['negative_or_retention_loss_cases']
    assert len(checks) == len(records) == 558
    geometry_cache = {}
    label_cache = {}
    for record, measured in zip(records, checks):
        file = d / record['file']
        assert sha(file) == record['sha256'] == measured['fixture_sha256']
        assert measured['fixture'] == record['file']
        assert (int(measured['input_index']), int(measured['model_seed'])) == (record['input_index'], record['model_seed'])
        with np.load(file, allow_pickle=False) as z:
            tokens, board, label = z['tokens'][0], z['board'][0], z['labels'][0]
        token_key = hashlib.sha256(tokens.tobytes()).hexdigest()
        if token_key not in label_cache:
            assert valid_shortest(tokens, label)
            label_cache[token_key] = int(label.sum()) - 1
        key = (token_key, np.packbits(board).tobytes())
        if key not in geometry_cache:
            geometry_cache[key] = (int(path_valid(tokens, board)), int(valid_shortest(tokens, board)))
        valid, shortest = geometry_cache[key]
        row = bycase[record['input_index'], record['model_seed']]
        assert valid == int(row['graph_valid_path']) == int(measured['graph_valid_path'])
        assert shortest == int(row['graph_valid_shortest']) == int(measured['graph_valid_shortest'])
        distance = label_cache[token_key]
        assert int(measured['shortest_edges']) == distance
        if valid:
            edges = int(board.sum()) - 1
            assert int(measured['selected_path_edges']) == edges and int(measured['extra_edges']) == edges - distance
        else:
            assert measured['selected_path_edges'] == measured['extra_edges'] == ''
    assert result['wall_seconds'] <= 120 and result['peak_cpu_rss_kib'] <= 2 * 1024**2
    assert result['process_cpu_seconds'] >= 0
    assert result['new_scientific_evaluations'] == result['fresh_inputs'] == result['gpu_seconds'] == 0
    return {'status': 'passed', 'source_model_rows': 12288, 'partition': result['partition'],
            'sole_endpoint': target, 'all_failure_fixtures': 510, 'fixed_control_fixtures': 48,
            'fixture_records': 558, 'distinct_geometry_reconstructions': len(geometry_cache),
            'shortest_labels_independently_checked': len(label_cache),
            'protocol_sha256': sha(directory / 'protocol.json'), 'source_sha256': sha(Path(__file__)),
            'seconds': time.perf_counter() - started,
            'scope': 'Complete sealed scalar cohort and all failed plus fixed-control full boards. Independent degree/connectivity and shortest-path verifier; proposed endpoint walker and analysis implementation are not imported. Positive full states outside fixtures inherit H2bl audit. Source-only result; no new accuracy or screen-cost evidence.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.source), indent=2))
