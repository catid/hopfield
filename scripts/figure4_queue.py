#!/usr/bin/env python3
"""Prepare and run the 100 Figure 4 curves after each Table 7 lane finishes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

from objective_queue import ROOT, COMMIT, digest, now, write, verify, gpu_snapshot, progress


def points(updates):
    out = set(range(1, min(64, updates) + 1)) | {updates}
    k = 1.
    while k < updates:
        out.add(round(k)); k *= 2 ** (1 / 8)
    return sorted(out)


def build(args):
    manifest_path = ROOT / 'configs/reproduction_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    base = json.loads((ROOT / 'configs/objective_jobs.json').read_text())
    jobs = [j for j in manifest['jobs'] if j['family'] == 'figure4']
    recipes = [x for x in manifest['frozen_recipe_materializations']
               if any('plan-dense-sudoku-' in p for p in x['to'])]
    assert len(recipes) == 5 and len(jobs) == 100
    final_lanes = {(j['method'], j['seed']): j['lane'] for j in base['jobs']
                   if j['family'] == 'table7_objectives' and j['task'] == 'sudoku'}
    assigned = {lane: 0 for lane in base['lanes']}
    for job in jobs:
        cmd = job['commands'][0]
        job['method'] = method = cmd[cmd.index('--method') + 1]
        job['seed'] = seed = int(cmd[cmd.index('--seed') + 1])
        dense = seed < 3 and method != 'bptt'
        if dense:
            lane = final_lanes[(method, seed)]
        else:
            lane = min(assigned, key=assigned.get)
        assigned[lane] += 1
        job['lane'], job['task'], job['dense'] = lane, 'sudoku', dense
        job['run_relative'] = (cmd[cmd.index('--out') + 1] if dense else job['commands'][1][-1])
        job['frozen_choice'] = manifest['frozen_training_objective_choices'][f'sudoku/{method}']
        job['updates_resolved'] = 60000 if method == 'bptt' else 32 if method == 'pl' else 2000
        job['fit_cap_s'] = 3600 if dense else 310
        job['expected_probe_updates_if_schedule_completed'] = points(job['updates_resolved'])
        job['expected_probe_n'] = 512
        job['clock_source'] = f'runs/ALT/final/sudoku-{method}-s{seed}/log.jsonl' if dense else job['run_relative'] + '/log.jsonl'
        job['status'] = 'queued_after_table7'
        assert ('--long' in cmd) == (method == 'bptt')
    assert {(j['method'], j['seed']) for j in jobs} == {(m, s) for m in ('pl', 'mpf', 'cd', 'bptt1', 'bptt') for s in range(20)}
    data = json.loads((ROOT / 'artifacts/deployment/data_manifest.json').read_text())
    needed = {f'frm/split/train-s{s}.npz' for s in range(20)} | {'frm/dev10k/dev.npz'}
    config = {**base, 'manifest_sha256': digest(manifest_path), 'created_utc': now(), 'jobs': jobs, 'external_jobs': [],
              'data_manifest': [x for x in data if x['path'] in needed], 'recipe_materializations': recipes,
              'protocol': {'seeds': list(range(20)), 'methods': ['pl', 'mpf', 'cd', 'bptt1', 'bptt'],
                  'dev': 'dev10k rows 0:512; no test-set scoring or tuning',
                  'probe': 'every update through 64; eight points per doubling thereafter; final update',
                  'dense': '12 runs: four non-full-BPTT methods, seeds 0-2; use matched final-run fit clock',
                  'deferred': '88 runs: remaining seeds plus full BPTT all20; use own fit clock',
                  'bptt': '60,000-update cosine at lr60, knob10, l1 0.1; distinct from Table7 lr2 long runs',
                  'time_axis': 'raw fit_s retained; startup = fit[1] - median(diff(fit[1:22])); paper adds ready_s from final seeds0-2',
                  'reuse': 'Reuse final-run clocks for dense seeds0-2. Sparse Table7 checkpoints cannot supply missing dense probes.',
                  'timing': 'Blackwell hardware; retain per-host/GPU curves and never claim H200 timing equivalence'}}
    assert len(config['data_manifest']) == 21
    write(ROOT / 'configs/figure4_jobs.json', config)
    for lane in base['lanes']:
        lines = []
        for job in jobs:
            if job['lane'] == lane:
                lines += [f'# {job["id"]}'] + [shlex.join([base['python'], '-u'] + cmd[1:]) for cmd in job['commands']]
        path = ROOT / 'configs/figure4_queues' / (lane.replace(':', '-gpu') + '.sh')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Auditable command list; use figure4_queue.py worker for provenance, lock, and resume.\n' + '\n'.join(lines) + '\n')
    print(json.dumps({'jobs': len(jobs), 'per_lane': assigned, 'dense': sum(j['dense'] for j in jobs),
                      'deferred': sum(not j['dense'] for j in jobs),
                      'scheduled_probe_count': sum(len(j['expected_probe_updates_if_schedule_completed']) for j in jobs)}))


def validate_curve(run, job):
    config, final = [json.loads((run / name).read_text()) for name in ('config.json', 'final.json')]
    assert (config['method'], config['seed'], config['updates_resolved']) == (job['method'], job['seed'], job['updates_resolved'])
    assert (config['probe_per_octave'], config['probe_every_until']) == (8, 64)
    for key, value in job['frozen_choice'].items():
        assert config.get(key) == value, (key, config.get(key), value)
    logs = [json.loads(line) for line in (run / 'log.jsonl').read_text().splitlines()]
    logs = [r for r in logs if 'update' in r]
    end = logs[-1]['update']
    expected = {p for p in job['expected_probe_updates_if_schedule_completed'] if p <= end} | {end}
    saved = {int(p.stem[7:]) for p in run.glob('model-u*.pt')}
    if job['dense']:
        probes = {r['update']: r['probe'] for r in logs if 'probe' in r}
    else:
        probes = {int(k): v for k, v in json.loads((run / 'probes.json').read_text()).items()}
    assert set(probes) == saved == expected, (len(probes), len(saved), len(expected))
    assert all(r['n'] == 512 and 0 <= r['solved'] <= 512 for r in probes.values())
    return {'n_probes': len(probes), 'probe_n': 512, 'last_update': end, 'stop': final['stop'],
            'schedule_completed': end == job['updates_resolved'], 'probe_updates': sorted(probes),
            'model_sha256': {p.name: digest(p) for p in sorted(run.glob('model-u*.pt'))}}


def validate_dense_clock(run, job, config):
    final = Path(config['source']) / 'experiments/runs/ALT/final' / f'sudoku-{job["method"]}-s{job["seed"]}'
    a, b = [json.loads((p / 'config.json').read_text()) for p in (run, final)]
    fields = ('task', 'method', 'seed', 'lr', 'knob', 'l1', 'reg', 'updates_resolved', 'n_train', 'aug', 'pool')
    assert all(a.get(k) == b.get(k) for k in fields), 'Dense and final recipes differ'
    common = sorted({p.name for p in run.glob('model-u*.pt')} & {p.name for p in final.glob('model-u*.pt')})
    assert common, 'No shared checkpoints validate the final-run clock'
    hashes = {name: {'dense': digest(run / name), 'final': digest(final / name)} for name in common}
    unequal = [name for name, h in hashes.items() if h['dense'] != h['final']]
    tensor_equal = {}
    if unequal:
        code = ('import json,sys,torch;from pathlib import Path;'
                'a,b=map(Path,sys.argv[1:3]);out={};'
                '\nfor n in sys.argv[3:]:\n x=torch.load(a/n,map_location="cpu",weights_only=True);y=torch.load(b/n,map_location="cpu",weights_only=True);out[n]=x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x)'
                '\nprint(json.dumps(out))')
        tensor_equal = json.loads(subprocess.check_output([config['python'], '-c', code, str(run), str(final), *unequal], text=True))
    return {'final_run': str(final), 'compared_checkpoints': hashes, 'tensor_comparison_for_different_archives': tensor_equal,
            'matched': all(h['dense'] == h['final'] or tensor_equal.get(n, False) for n, h in hashes.items()),
            'final_log_sha256': digest(final / 'log.jsonl'), 'dense_log_sha256': digest(run / 'log.jsonl')}


def claim_job(config_path, original, lane, base, config):
    """Atomically consult live ownership and claim an untouched job locally."""
    lock_path = ROOT / 'artifacts/figure4/dispatch.lock'
    with lock_path.open('a') as dispatch:
        fcntl.flock(dispatch, fcntl.LOCK_EX)
        live = json.loads(Path(config_path).read_text())
        job = next(j for j in live['jobs'] if j['id'] == original['id'])
        assert {k: v for k, v in job.items() if k != 'lane'} == {k: v for k, v in original.items() if k != 'lane'}, 'Live recipe changed'
        if job['lane'] != lane:
            return None
        for key in ('source', 'source_sha256', 'data_manifest', 'fresh_hbar_prerequisite_sha256'):
            assert live[key] == config[key], f'Live configuration changed {key}'
        source = Path(config['source']); run = source / 'experiments' / job['run_relative']
        identity = hashlib.sha256(json.dumps({'job': job, 'source': config['source_sha256'],
                    'data': config['data_manifest'], 'recipes': config['recipe_materializations'],
                    'hbar': config['fresh_hbar_prerequisite_sha256']}, sort_keys=True).encode()).hexdigest()
        owner = run / 'figure4_provenance.json'
        if owner.exists():
            assert json.loads(owner.read_text())['identity'] == identity
        elif run.exists() and any(run.iterdir()):
            raise RuntimeError(f'Refusing unprovenanced output: {run}')
        else:
            write(owner, {**base, 'identity': identity, 'job': job, 'dispatch_config_sha256': digest(config_path)})
        return job, run, identity


def worker(args):
    config_path = Path(args.config); config = json.loads(config_path.read_text())
    host, index = args.lane.split(':'); gpu = int(index)
    assert host == socket.gethostname().split('.')[0].lower() and args.lane in config['lanes']
    artifacts = ROOT / 'artifacts/figure4'; artifacts.mkdir(parents=True, exist_ok=True)
    state_path = artifacts / f'{host}-gpu{gpu}-state.json'
    objective_state = ROOT / 'artifacts/objectives' / f'{host}-gpu{gpu}-state.json'
    gate = config.get('wait_for_table7_by_lane', {}).get(args.lane, True)
    while gate and (not objective_state.exists() or json.loads(objective_state.read_text())['status'] != 'queue_complete'):
        write(state_path, {'status': 'queued_after_table7', 'lane': args.lane, 'worker_pid': os.getpid(), 'utc': now()})
        time.sleep(30)
    # The same physical-GPU lock as the objective worker, acquired only after
    # the entire existing queue has declared completion.
    lock_path = Path(config.get('physical_gpu_locks', {}).get(args.lane,
                     str(ROOT / 'artifacts/objectives' / f'gpu-{gpu}.lock')))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    validation = verify(config)
    source = Path(config['source'])
    for recipe in config['recipe_materializations']:
        for rel in recipe['to']:
            path = source / rel
            if path.exists() and path.read_text() != recipe['text']:
                raise RuntimeError(f'Frozen recipe differs: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
            temp.write_text(recipe['text']); os.replace(temp, path)
    base = {'lane': args.lane, 'worker_pid': os.getpid(), 'source_commit': COMMIT,
            'config_sha256': digest(config_path), 'validation': validation, 'started_utc': now(),
            'dynamic_dispatch_v1': True}
    base['runtime'] = json.loads(subprocess.check_output([config['python'], '-c',
        'import json,platform,torch,triton; print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__)))'], text=True))
    write(artifacts / f'{host}-gpu{gpu}-worker.json', base)
    for original in config['jobs']:
        claimed = claim_job(config_path, original, args.lane, base, config)
        if claimed is None:
            continue
        job, run, identity = claimed
        record_path = artifacts / 'jobs' / job['id'] / 'execution.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else {**base, 'identity': identity, 'job': job, 'steps': []}
        if record.get('status') == 'complete':
            assert all(digest(run / name) == sha for name, sha in record['outputs'].items())
            continue
        while gpu_snapshot(gpu)['owners']:
            write(state_path, {**base, 'job': job['id'], 'status': 'waiting_for_gpu', 'utc': now()}); time.sleep(15)
        env_changes = {'CUDA_VISIBLE_DEVICES': str(gpu), 'CRAM_DATA': config['data'], 'V1_DATA': config['data'],
                       'FRM_DATA': str(Path(config['data']) / 'frm/split'), 'PYTHONPATH': str(source),
                       'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'WANDB_MODE': 'disabled'}
        record.update(status='running', env=env_changes, gpu=gpu_snapshot(gpu)); write(record_path, record)
        for stage, raw in enumerate(job['commands']):
            # Probe files are resumable and may be partial; never skip the probe
            # command merely because probes.json exists.
            if stage == 0 and (run / 'final.json').exists():
                continue
            argv = [config['python'], '-u'] + raw[1:]
            if stage == 1 and args.lane in config.get('mps_probe_lanes', []):
                argv = ['/usr/bin/python3', str(ROOT / 'scripts/run_mps_probes.py'), '--source', str(source),
                        '--python', config['python'], '--run', job['run_relative'], '--gpu', str(gpu),
                        '--artifacts', str(record_path.parent / 'mps_probes')]
            start = time.monotonic(); step = {'stage': stage, 'argv': argv, 'cwd': str(source / 'experiments'), 'started_utc': now()}
            with (record_path.parent / ('train.log' if stage == 0 else 'probe.log')).open('ab', buffering=0) as log:
                child = subprocess.Popen(argv, cwd=source / 'experiments', env={**os.environ, **env_changes},
                                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                while child.poll() is None:
                    state = {**base, 'status': 'running', 'job': job['id'], 'stage': stage, 'child_pid': child.pid,
                             'gpu': gpu_snapshot(gpu), 'progress': progress(run), 'elapsed_s': time.monotonic()-start, 'utc': now()}
                    write(state_path, state)
                    with (record_path.parent / 'monitor.jsonl').open('a') as f:
                        f.write(json.dumps(state) + '\n')
                    try: child.wait(timeout=15)
                    except subprocess.TimeoutExpired: pass
            step.update(exit_code=child.returncode, wall_seconds=time.monotonic()-start, finished_utc=now())
            record['steps'].append(step); write(record_path, record)
            if child.returncode != 0:
                record.update(status='failed'); write(record_path, record)
                raise RuntimeError(f'Figure 4 child failed: {job["id"]}')
        validation = validate_curve(run, job)
        if job['dense']:
            validation['clock_match'] = validate_dense_clock(run, job, config)
        record.update(status='complete', validation=validation, finished_utc=now(),
                      outputs={name: digest(run / name) for name in ('config.json', 'final.json', 'log.jsonl') + (() if job['dense'] else ('probes.json',))})
        write(record_path, record)
        write(state_path, {**base, 'status': 'complete', 'job': job['id'], 'utc': now()})
    write(state_path, {**base, 'status': 'queue_complete', 'utc': now()})


def collect(args):
    config = json.loads(Path(args.config).read_text())
    root = ROOT / 'artifacts/figure4'
    def fetch(host):
        dest = root / host; dest.mkdir(parents=True, exist_ok=True)
        base = ['rsync', '-a', '--partial', '-e', 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes',
                '--include=*/', '--include=*.json', '--include=*.jsonl', '--include=*.log', '--exclude=*']
        for remote, local in [('artifacts/figure4/', str(dest) + '/'),
                              ('objective_source/experiments/runs/', str(dest / 'runs') + '/')]:
            here = host == socket.gethostname().split('.')[0].lower()
            exclusions = [f'--exclude=/{h}/' for h in ('foureyes', 'ripper', 'loki', 'work')] if here and remote.startswith('artifacts') else []
            src = f'{ROOT}/{remote}' if here else f'{host}:{ROOT}/{remote}'
            subprocess.run(base[:2] + exclusions + base[2:] + [src, local], check=True, capture_output=True, text=True)
        return host
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch, host) for host in sorted({x.split(':')[0] for x in config['lanes']})]
        for future in futures:
            future.result()
    states = {}
    for lane in config['lanes']:
        host, gpu = lane.split(':')
        path = root / host / f'{host}-gpu{gpu}-state.json'
        if path.exists():
            states[lane] = json.loads(path.read_text())
    records = []
    for job in config['jobs']:
        host = job['lane'].split(':')[0]
        path = root / host / 'jobs' / job['id'] / 'execution.json'
        record = {k: job[k] for k in ('id', 'lane', 'method', 'seed', 'dense', 'run_relative', 'fit_cap_s', 'updates_resolved')}
        record.update(status='queued_after_table7' if config.get('wait_for_table7_by_lane', {}).get(job['lane'], True)
                      else 'pending', expected_probes=len(job['expected_probe_updates_if_schedule_completed']))
        if states.get(job['lane'], {}).get('status') == 'queued_after_d2':
            record['status'] = 'queued_after_d2'
        if path.exists():
            execution = json.loads(path.read_text())
            record['status'] = execution['status']
            record['validation'] = execution.get('validation')
            if record['status'] == 'complete':
                run = root / host / job['run_relative']
                record['output_hashes_match'] = all((run / name).exists() and digest(run / name) == sha
                                                   for name, sha in execution['outputs'].items())
                if not record['output_hashes_match']:
                    record['status'] = 'collection_incomplete'
        records.append(record)
    counts = {s: sum(r['status'] == s for r in records) for s in sorted({r['status'] for r in records})}
    write(root / 'summary.json', {'created_utc': now(), 'source_commit': COMMIT, 'config_sha256': digest(args.config),
          'protocol': config['protocol'], 'status_counts': counts, 'lanes': states, 'jobs': records,
          'expected_probes_full_schedules': sum(r['expected_probes'] for r in records),
          'completed_curve_runs': sum(r['status'] == 'complete' for r in records)})
    print(json.dumps({'summary': str(root / 'summary.json'), 'status_counts': counts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build')
    p = sub.add_parser('worker'); p.add_argument('--lane', required=True)
    p.add_argument('--config', default=str(ROOT / 'configs/figure4_jobs.json'))
    p = sub.add_parser('collect'); p.add_argument('--config', default=str(ROOT / 'configs/figure4_jobs.json'))
    args = parser.parse_args(); globals()[args.command](args)
