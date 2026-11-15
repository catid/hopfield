"""Complete labels-last H2bq direct-state, replay, outcome and cost audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
from audit_bp_first_fresh_states import job_check
from audit_bp_first_fresh_replays import one_shard
from audit_bp_first_fresh_costs import verify_costs
from audit_bp_first_geometry import one_shard as source_shard
from bp_first_fresh_statistics import absolute, noninferiority


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cohorts(root, directory, ph):
    cohorts = []
    for gpu in [0, 1]:
        for kind in ['gate', 'timing']:
            worker = json.loads((directory / f'{kind}-worker{gpu}.json').read_text())
            assert worker['status'] == 'complete' and worker['protocol_sha256'] == ph
            assert worker['worker_wall_seconds'] <= (180 if kind == 'gate' else 1500)
            assert worker['peak_cpu_rss_kib'] <= 8 * 1024**2
        for kind, folder in [('gate', 'source-gates'), ('warmup', 'warmup')]:
            for mode in ['LP96', 'BP_geometry_LP96']:
                path = directory / folder / f'gpu{gpu}-{mode}'
                r = json.loads((path / 'result.json').read_text())
                r.update(kind=kind, directory=str(path.relative_to(root)))
                assert r['source_gate'] and r['model_inputs'] == (864 if kind == 'gate' else 96)
                assert all(row['source_parity']['fresh_diagnostics']['status'] == 'passed' for row in r['rows'])
                cohorts.append(r)
    for pair in range(4):
        for mode in ['LP96', 'BP_geometry_LP96']:
            path = directory / 'runs' / f'pair{pair}-{mode}'
            r = json.loads((path / 'result.json').read_text())
            r.update(kind='timing', pair=pair, directory=str(path.relative_to(root)))
            assert not r['source_gate'] and r['model_inputs'] == 12288
            assert r['retained_couplings'] == (pair == 0)
            cohorts.append(r)
    return cohorts


def main(root, directory, workers):
    started = time.perf_counter()
    ph = sha(directory / 'protocol.json')
    p = json.loads((directory / 'protocol.json').read_text())
    seal = json.loads((directory / 'sealed-replays.json').read_text())
    assert seal['status'] == 'complete' and seal['protocol_sha256'] == ph
    assert seal['labels_or_shortestness_outcomes_joined'] is False
    for name, digest in {**seal['cohort_and_worker_receipt_sha256'], **seal['fresh_input_receipt_sha256']}.items():
        assert sha(directory / name) == digest
    for name, digest in {**p['source_sha256'], **p['source_evidence_sha256']}.items():
        assert sha(root / name) == digest, name
    inputs = json.loads((directory / 'independent-input-audit.json').read_text())
    datasets = json.loads((directory / 'datasets.json').read_text())
    assert inputs['status'] == 'passed' and inputs['protocol_sha256'] == datasets['protocol_sha256'] == ph
    assert inputs['datasets'] == datasets['datasets'] and inputs['inputs_checked'] == inputs['unique'] == 4096
    assert inputs['unique_wall_masks'] == inputs['endpoint_records_checked'] == inputs['seeded_original_maze_replays'] == 4096
    assert all(r['overlap'] == r['wall_overlap'] == 0 for r in inputs['prior_manifest_checks'])
    assert workers == p['budgets']['independent_audit_processes'] == 16
    assert p['budgets']['independent_final_audit_wall_seconds'] == 600
    cohorts = load_cohorts(root, directory, ph)
    cost = verify_costs(root, directory, ph, cohorts)
    partials = directory / 'audit-partials'
    partials.mkdir(exist_ok=False)
    direct, replay, source = [], [], []
    source_p = {**p, 'datasets': p['source_datasets']}
    source_cohorts = [r for r in cohorts if r['kind'] in ['gate', 'warmup']]
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {}
        for job in p['jobs']:
            futures[pool.submit(job_check, (root, directory, job, p))] = ('direct', job['id'])
        for shard in range(16):
            futures[pool.submit(one_shard, (root, directory, p, datasets['datasets'], shard, cohorts))] = ('replay', str(shard))
        for shard in [0, 15]:
            futures[pool.submit(source_shard, (root, directory, source_p, shard, source_cohorts))] = ('source', str(shard))
        for future in as_completed(futures):
            kind, key = futures[future]
            result = future.result()
            {'direct': direct, 'replay': replay, 'source': source}[kind].append(result)
            (partials / f'{kind}-{key}.json').write_text(json.dumps(result) + '\n')
            print(json.dumps({'event': 'fresh_audit_progress', 'kind': kind, 'key': key,
                              'direct_jobs': len(direct), 'replay_shards': len(replay),
                              'source_shards': len(source)}), flush=True)
    except BaseException:
        for child in list((getattr(pool, '_processes', None) or {}).values()):
            if child.is_alive():
                child.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    assert len(direct) == 48 and len(replay) == 16 and len(source) == 2
    rows = sorted([row for shard in replay for row in shard['rows']],
                  key=lambda r: (r['input_index'], r['model_seed'], ['BP_AdamW', 'LP96', 'cascade'].index(r['arm'])))
    assert len(rows) == 36864
    bycase = {(r['input_index'], r['model_seed'], r['arm']): r for r in rows}
    assert len(bycase) == 36864
    direct_jobs = {r['job']['id']: r for r in direct}
    for row in rows:
        if row['arm'] == 'cascade':
            continue
        i, seed = row['input_index'], row['model_seed']
        independently_checked = direct_jobs[f'shard{i // 256:02d}-s{seed}']['success'][0 if row['arm'] == 'BP_AdamW' else 7][i % 256]
        assert row['success'] == independently_checked
    arms = ['BP_AdamW', 'LP96', 'cascade']
    joint = {arm: [all(bycase[i, seed, arm]['success'] for seed in range(3)) for i in range(4096)] for arm in arms}
    models = {arm: sum(r['success'] for r in rows if r['arm'] == arm) for arm in arms}
    gains = sum(a and not b for a, b in zip(joint['cascade'], joint['LP96']))
    losses = sum(b and not a for a, b in zip(joint['cascade'], joint['LP96']))
    absolute_result = absolute(int(sum(joint['cascade'])))
    ni = noninferiority(int(gains), int(losses))
    accounting = {}
    for arm in arms:
        selected = [r for r in rows if r['arm'] == arm]
        accounting[arm] = {'model_inputs': 12288, 'successes': models[arm],
            'hidden_failures': sum(not r['hidden_good'] for r in selected),
            'nonfinite_outputs': sum(not r['finite'] for r in selected),
            'not_stopped': sum(not r['stopped'] for r in selected),
            'native_caps': sum(r['native_cap'] for r in selected),
            'block_caps': sum(r['block_cap'] for r in selected),
            'numerical_rejections': sum(not r['numerically_accepted'] for r in selected),
            'wrong_numerically_accepted': sum(r['numerically_accepted'] and not r['graph_valid_shortest'] for r in selected)}
    candidate = [r for r in rows if r['arm'] == 'cascade']
    valid_longer = sum(r['BP_geometry_valid_longer'] for r in candidate)
    assert all(not r['routed'] for r in candidate if r['BP_geometry_valid_longer'])
    cases = [{'input_index': i, **{arm: int(joint[arm][i]) for arm in arms}} for i in range(4096)]
    for name, entries in [('per-model-input.csv', rows), ('per-input.csv', cases)]:
        with (directory / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0])); writer.writeheader(); writer.writerows(entries)
    elapsed = time.perf_counter() - started
    assert elapsed <= 600
    result = {'status': 'passed', 'hypothesis': 'H2bq', 'protocol_sha256': ph,
        'sealed_replays_sha256': sha(directory / 'sealed-replays.json'),
        'primary_endpoints': {'absolute_joint_success_999': absolute_result,
                              'paired_LP96_noninferiority': ni},
        'cost_endpoint': cost, 'joint_success': {arm: int(sum(joint[arm])) for arm in arms},
        'model_success': models, 'failure_accounting': accounting,
        'actual_routed_model_inputs_per_replay': sum(r['routed'] for r in candidate),
        'geometry_valid_longer_BP_outputs_retained': valid_longer,
        'source_gate_and_warmup_model_copies_checked': sum(r['model_pipeline_copies_checked'] for r in source),
        'fresh_final_internal_states_directly_checked': sum(r['all_final_states'] for r in direct),
        'direct_candidate_original_factor_checks': sum(r['candidate_literal_checks'] for r in direct),
        'direct_accepted_moves': sum(r['accepted_moves'] for r in direct),
        'fresh_measured_replay_model_pipeline_copies': sum(r['measured_model_pipeline_copies'] for r in replay),
        'replay_candidate_score_comparisons': sum(r['candidate_score_comparisons'] for r in replay),
        'replay_active_trace_states': sum(r['active_trace_states'] for r in replay),
        'maximum_direct_margin_error': max(r['max_final_margin_error'] for r in direct),
        'maximum_direct_energy_error': max(r['max_final_energy_error'] for r in direct),
        'maximum_direct_candidate_error': max(r['max_literal_error'] for r in direct),
        'maximum_replay_candidate_error': max(r['maximum_candidate_score_error'] for r in replay),
        'per_model_rows_sha256': sha(directory / 'per-model-input.csv'), 'per_input_rows_sha256': sha(directory / 'per-input.csv'),
        'fresh_independent_inputs': 4096, 'planned_jobs': 48, 'primary_model_maze_arm_evaluations': 36864,
        'accuracy_pair': 0, 'workers': workers, 'seconds': elapsed, 'source_sha256': sha(Path(__file__)),
        'scope': 'Complete fresh cohort, three fixed models, predetermined pair0 accuracy and exact repeated-source integrity. Separate absolute, paired noninferiority and measured complete-cost endpoints; no simultaneous95% claim or dropped numerical/valid-longer failure.'}
    (directory / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    progress = args.directory / 'audit-worker.json'
    if progress.exists() or (args.directory / 'audit.json').exists():
        raise FileExistsError('Refusing an existing fresh audit attempt')
    started = time.perf_counter()
    state = {'status': 'running', 'pid': os.getpid(), 'wall_budget_seconds': 600, 'workers': args.workers}
    progress.write_text(json.dumps(state, indent=2) + '\n')
    def alarm(signum, frame):
        raise TimeoutError('Frozen fresh independent audit wall budget exhausted')
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 600)
    try:
        main(args.root, args.directory, args.workers)
    except BaseException as error:
        state.update(status='incomplete', error=repr(error), traceback=traceback.format_exc(),
                     wall_seconds=time.perf_counter() - started, partial_receipts_retained=True)
        progress.write_text(json.dumps(state, indent=2) + '\n')
        raise
    else:
        state.update(status='complete', wall_seconds=time.perf_counter() - started)
        progress.write_text(json.dumps(state, indent=2) + '\n')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
