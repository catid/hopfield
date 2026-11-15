"""Frozen, source-only H2bm partition and independently reconstructed fixtures."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np
from bp_geometry_checks import classify, endpoint, partition


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bit(row, field):
    value = row[field]
    if value not in ('0', '1'):
        raise ValueError(f'Nonbinary source field {field}')
    return int(value)


def write_json(path, result):
    path.write_text(json.dumps(result, indent=2) + '\n')


def analyze(protocol_path, source, output):
    started = time.perf_counter()
    cpu_started = time.process_time()
    p = json.loads(protocol_path.read_text())
    assert p['status'] == 'frozen_before_first_geometry_aggregate'
    assert p['hypothesis_id'] == 'H2bm'
    assert p['budgets']['work_analysis_wall_seconds'] == 120
    here = Path(__file__).parent
    for name, digest in p['analysis_source_sha256'].items():
        assert sha(here / name) == digest, name
    for name, digest in p['sealed_source_sha256'].items():
        assert sha(source / name) == digest, name
    a = json.loads((source / 'audit.json').read_text())
    assert a['status'] == 'passed' and a['rows_sha256'] == sha(source / 'per-model-input.csv')
    assert a['protocol_sha256'] == sha(source / 'protocol.json')
    d = source / 'public-fixtures'
    m = json.loads((d / 'manifest.json').read_text())
    assert m['audit_sha256'] == sha(source / 'audit.json')
    assert m['protocol_sha256'] == a['protocol_sha256']
    assert m['source_audit_sha256'] == sha(d / 'source-audit.json')
    with (source / 'per-model-input.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12288
    bycase = {(int(r['input_index']), int(r['model_seed'])): r for r in rows}
    assert len(bycase) == 12288 and set(bycase) == {(i, s) for i in range(4096) for s in range(3)}
    flags = ['graph_valid_path', 'graph_valid_shortest', 'hidden_good', 'BP_finite',
             'BP_stopped', 'clamps_ok', 'bp_own_accept', 'lp_cross_accept', 'dual_accept']
    for (i, seed), row in bycase.items():
        assert row['source_job'] == f'shard{i // 256:02d}-s{seed}'
        for field in flags:
            bit(row, field)
        assert bit(row, 'dual_accept') == (bit(row, 'bp_own_accept') & bit(row, 'lp_cross_accept'))
        row['geometry_partition'] = partition(bit(row, 'graph_valid_path'), bit(row, 'graph_valid_shortest'))
    assert sum(bit(r, 'graph_valid_shortest') for r in rows) == 11778
    fixed = {(r['input_index'], r['model_seed']) for r in m['fixed_cases']}
    assert fixed == {(shard * 256, seed) for shard in range(16) for seed in range(3)}
    negatives = {(r['input_index'], r['model_seed']) for r in m['negative_or_retention_loss_cases']}
    assert negatives == {k for k, r in bycase.items() if not bit(r, 'graph_valid_shortest')}
    assert len(negatives) == 510
    fixture_rows = []
    cache = {}
    all_records = m['fixed_cases'] + m['negative_or_retention_loss_cases']
    for index, record in enumerate(all_records):
        path = d / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as z:
            tokens = z['tokens'][0]
            board = z['board'][0]
        key = hashlib.sha256(tokens.tobytes() + np.packbits(board).tobytes()).hexdigest()
        if key not in cache:
            cache[key] = classify(tokens, board)
        result = cache[key]
        source_row = bycase[record['input_index'], record['model_seed']]
        for field in ['graph_valid_path', 'graph_valid_shortest']:
            assert result[field] == bit(source_row, field), record['file']
        if source_row['geometry_partition'] == 'valid_longer':
            assert result['extra_edges'] > 0
        fixture_rows.append({'input_index': record['input_index'], 'model_seed': record['model_seed'],
                             'fixture': record['file'], 'fixture_sha256': record['sha256'], **result})
        if (index + 1) % 64 == 0 or index + 1 == len(all_records):
            print(json.dumps({'event': 'geometry_fixtures_checked', 'checked': index + 1,
                              'required': len(all_records)}), flush=True)
    def counts(selected):
        c = Counter(r['geometry_partition'] for r in selected)
        return {k: c[k] for k in ['valid_shortest', 'invalid_geometry', 'valid_longer']}
    totals = counts(rows)
    assert totals['valid_shortest'] == 11778 and totals['invalid_geometry'] + totals['valid_longer'] == 510
    tables = []
    for label, field in [('BP AdamW own numerical acceptance', 'bp_own_accept'),
                         ('BP plus original LP certificate, archived dual', 'dual_accept'),
                         ('Simple-path geometry on the BP source board', 'graph_valid_path')]:
        good = [r for r in rows if bit(r, 'graph_valid_shortest')]
        bad = [r for r in rows if not bit(r, 'graph_valid_shortest')]
        tables.append({'control': label, 'source_field': field,
                       'accepted_shortest': sum(bit(r, field) for r in good),
                       'accepted_wrong': sum(bit(r, field) for r in bad),
                       'rejected_wrong': sum(1 - bit(r, field) for r in bad)})
    assert tables[0]['accepted_shortest'] == 11778 and tables[0]['accepted_wrong'] == 510
    assert tables[1]['accepted_shortest'] == 11778 and tables[1]['accepted_wrong'] == 509
    with (output / 'per-model-input.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'fixture-checks.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fixture_rows[0]))
        writer.writeheader()
        writer.writerows(fixture_rows)
    elapsed = time.perf_counter() - started
    cpu_elapsed = time.process_time() - cpu_started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert elapsed <= p['budgets']['work_analysis_wall_seconds']
    assert peak <= p['budgets']['peak_cpu_rss_gib'] * 1024**2
    result = {'status': 'passed', 'hypothesis': 'H2bm', 'protocol_sha256': sha(protocol_path),
              'source_rows': 12288, 'source_inputs': 4096, 'fixed_model_seeds': [0, 1, 2],
              'partition': totals, 'sole_endpoint': endpoint(totals['invalid_geometry'], 510),
              'per_seed_partition': {str(seed): counts([r for r in rows if int(r['model_seed']) == seed]) for seed in range(3)},
              'controls_in_bp_first_order': tables,
              'shortest_implies_valid_geometry_invariant': {'checked_shortest_rows': 11778, 'violations': 0,
                  'meaning': 'Definitional consistency only, not an efficacy endpoint or new accuracy evidence.'},
              'numerical_source_accounting': {field: sum(1 - bit(r, field) for r in rows)
                  for field in ['hidden_good', 'BP_finite', 'BP_stopped', 'clamps_ok', 'bp_own_accept']},
              'independent_fixture_check': {'records': len(fixture_rows), 'fixed_controls': len(m['fixed_cases']),
                  'all_failure_cases': len(negatives), 'unique_geometry_calculations': len(cache),
                  'mismatches': 0, 'complete_failure_supplement': True,
                  'method': 'Independent endpoint walk for geometry and full-grid BFS for shortest distance.'},
              'wall_seconds': elapsed, 'process_cpu_seconds': cpu_elapsed,
              'peak_cpu_rss_kib': peak, 'cpu_processes': 1, 'gpu_seconds': 0,
              'new_scientific_evaluations': 0, 'fresh_inputs': 0,
              'per_model_input_sha256': sha(output / 'per-model-input.csv'),
              'fixture_checks_sha256': sha(output / 'fixture-checks.csv'),
              'analysis_sha256': sha(Path(__file__)),
              'scope': 'Analysis of sealed H2bl fields and existing H2bh source boards. Finite source decomposition only. No fresh accuracy, population claim, executed cascade, conditional rescue or geometry-screen cost inference. Remaining positive full states inherit the complete H2bl independent audit.'}
    write_json(output / 'analysis.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('protocol', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    # Refuse all existing output directories, including incomplete earlier runs.
    # The caller may not silently replace either scientific results or failures.
    args.output.mkdir(exist_ok=False, parents=True)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    def watchdog(signum, frame):
        raise TimeoutError('Frozen 120-second analysis wall budget exhausted')
    signal.signal(signal.SIGALRM, watchdog)
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        analyze(args.protocol, args.source, args.output)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt = {'status': 'incomplete', 'error_type': type(error).__name__,
                   'error': str(error), 'pid': os.getpid(),
                   'wall_seconds': time.perf_counter() - wall_started,
                   'process_cpu_seconds': time.process_time() - cpu_started,
                   'wall_budget_seconds': 120, 'new_scientific_evaluations': 0}
        write_json(args.output / 'incomplete.json', receipt)
        print(json.dumps(receipt), flush=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
