#!/usr/bin/env python3
"""Persistent per-GPU Table 7 queues with frozen jobs and auditable resumes.

Build config on the coordinating host; copy it and this script to each worker.
Workers run only their explicitly assigned lane and never use imported results.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path('/home/catid/hopfield')
COMMIT = '62bbe830c7d6f87f2ba18f30a28039258b986454'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def build(args):
    manifest_path = ROOT / 'configs/reproduction_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    all_jobs = [j for j in manifest['jobs'] if j['family'] in ('table7_objectives', 'table7_long_schedules')]
    external = [j for j in all_jobs if '3840000' in j['id']]
    jobs = [j for j in all_jobs if j not in external]
    lanes = ['loki:1', 'loki:0', 'work:0', 'work:1']
    initial = ['objective-maze-pl-s0', 'objective-maze-bptt-s0', 'objective-maze-pl-s1', 'objective-maze-pl-s2']
    by_id = {j['id']: j for j in jobs}
    # Short early measurements, followed by all frozen methods and the long
    # supplementary schedules. Exact jobs are never dropped to shorten runtime.
    method_order = ['pl', 'mpl', 'mpf', 'dsm', 'bptt1', 'bptt5', 'bptt', 'chl', 'imp', 'eqp', 'cd', 'pcd']
    rest = [j for j in jobs if j['id'] not in initial]
    def order(j):
        cmd = j['commands'][0]
        method = cmd[cmd.index('--method') + 1]
        task = cmd[cmd.index('--task') + 1]
        return (j['family'] == 'table7_long_schedules', task == 'sudoku', method_order.index(method), j['id'])
    ordered = [by_id[k] for k in initial] + sorted(rest, key=order)
    for idx, job in enumerate(ordered):
        job['lane'] = lanes[idx % len(lanes)]
        cmd = job['commands'][0]
        job['run_relative'] = cmd[cmd.index('--out') + 1]
        job['task'] = cmd[cmd.index('--task') + 1]
        job['method'] = cmd[cmd.index('--method') + 1]
        job['seed'] = int(cmd[cmd.index('--seed') + 1])
    data = json.loads((ROOT / 'artifacts/deployment/data_manifest.json').read_text())
    data = [x for x in data if x['path'] in {
        'frm/split/train-s0.npz', 'frm/split/train-s1.npz', 'frm/split/train-s2.npz',
        'frm/dev10k/dev.npz', 'full-test/full-all.npz',
        *[f'eqr/maze-30x30-unique-1k/{s}/all__{a}.npy' for s in ('train', 'test') for a in ('inputs', 'labels')]}]
    sources = {}
    for subdir in ('cram', 'experiments'):
        for path in sorted((ROOT / 'upstream' / subdir).glob('*.py')):
            sources[str(path.relative_to(ROOT / 'upstream'))] = digest(path)
    prerequisites = {}
    for seed in range(3):
        path = Path(f'experiments/runs/CRAM-S/full/full-aug-s{seed}/model.pt')
        prerequisites[str(path)] = digest(ROOT / 'upstream' / path)
    config = {'schema_version': 1, 'created_utc': now(), 'paper': 'arXiv:2611.17309v1',
              'source_commit': COMMIT, 'manifest_sha256': digest(manifest_path),
              'workspace': str(ROOT), 'source': str(ROOT / 'objective_source'),
              'python': str(ROOT / 'upstream/.venv/bin/python'), 'data': str(ROOT / 'upstream/data'),
              'source_sha256': sources, 'data_manifest': data, 'fresh_hbar_prerequisite_sha256': prerequisites,
              'lanes': lanes, 'jobs': ordered,
              'external_jobs': [{'id': j['id'], 'owner': 'root', 'reason': 'Already running; do not duplicate',
                                 'commands': j['commands']} for j in external],
              'protocol': {'methods': method_order, 'seeds': [0, 1, 2], 'sudoku_test_n': 50000,
                           'maze_test_n': 1000, 'sudoku_probe_n': 512, 'maze_train_n': 900,
                           'maze_dev_n': 100, 'temperature_selection': 'Sudoku dev10k rows512:2560 only',
                           'selection': 'Highest development probe accuracy, earliest update on ties',
                           'timing': 'Cold fits on Blackwell; no direct H200 timing equivalence',
                           'control': 'Paper BPTT uses SGD; this is not a BP AdamW control experiment'}}
    write(ROOT / 'configs/objective_jobs.json', config)
    print(json.dumps({'managed': len(ordered), 'external': len(external), 'lanes': lanes}))


def gpu_snapshot(gpu):
    raw = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
        '--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,power.draw',
        '--format=csv,noheader,nounits'], text=True).strip()
    values = [s.strip() for s in raw.split(',')]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                        '--format=csv,noheader,nounits'], text=True).strip()
    owners = [line for line in processes.splitlines() if values[1] in line]
    return {'raw': raw, 'uuid': values[1], 'owners': owners}


def progress(run):
    result = {}
    log = run / 'log.jsonl'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 8192))
            lines = f.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            try:
                result['last_update'] = json.loads(line)
                break
            except json.JSONDecodeError:
                pass
    for name in ('ckpt.pt', 'final.json', 'test.json'):
        p = run / name
        if p.exists():
            result[name] = {'bytes': p.stat().st_size, 'mtime': p.stat().st_mtime}
    return result


def verify(config):
    source, data = Path(config['source']), Path(config['data'])
    for rel, expected in config['source_sha256'].items():
        if digest(source / rel) != expected:
            raise RuntimeError(f'Source hash mismatch: {rel}')
    for item in config['data_manifest']:
        if digest(data / item['path']) != item['sha256']:
            raise RuntimeError(f'Data hash mismatch: {item["path"]}')
    for rel, expected in config['fresh_hbar_prerequisite_sha256'].items():
        if digest(source / rel) != expected:
            raise RuntimeError(f'Fresh hbar prerequisite missing or changed: {rel}')
    return {'source_checked_files': len(config['source_sha256']),
            'data_checked_files': len(config['data_manifest']), 'hbar_models_checked': 3, 'utc': now()}


def frozen_choice(job):
    argv = job['commands'][0]
    return {key: float(argv[argv.index('--' + key) + 1])
            for key in ('lr', 'knob', 'l1', 'reg') if '--' + key in argv}


def common_t99(finals, startup, ready):
    """Paper T99: first shared probe update whose three-run mean is >=99%."""
    by_seed = [{r['update']: r for r in f['curve']} for f in finals]
    shared = sorted(set.intersection(*(set(r) for r in by_seed)))
    curve = []
    for update in shared:
        records = [r[update] for r in by_seed]
        mean = statistics.fmean(r['solved'] / r['n'] for r in records)
        fit = [r['fit_s'] for r in records]
        adjusted = [f - s + r for f, s, r in zip(fit, startup, ready)]
        curve.append({'update': update, 'mean_solved_percent': 100 * mean,
                      'fit_s_per_seed': fit, 'paper_adjusted_fit_s_per_seed': adjusted})
    crossing = next((r for r in curve if r['mean_solved_percent'] >= 99.), None)
    return {'update': crossing['update'] if crossing else None,
            'fit_s_per_seed': crossing['fit_s_per_seed'] if crossing else [None] * len(finals),
            'paper_adjusted_fit_s_per_seed': crossing['paper_adjusted_fit_s_per_seed'] if crossing else [None] * len(finals),
            'mean_solved_percent': crossing['mean_solved_percent'] if crossing else None, 'common_curve': curve}


def count_updates(config, base_record, gpu):
    """Run paper arithmetic counts while the worker still holds its GPU lock.

    These are frozen published choices, not a new tuning sweep. In particular,
    choice.json deliberately has no invented grid rows or development scores.
    """
    source = Path(config['source'])
    artifacts = ROOT / 'artifacts/objectives'
    state_path = artifacts / f'{base_record["host"]}-gpu{gpu}-state.json'
    for job in config['jobs']:
        if job['family'] != 'table7_objectives' or job['seed'] != 0:
            continue
        rel = Path(job['run_relative'])
        path = source / 'experiments' / rel.parent.parent / 'tune' / f'{job["task"]}-{job["method"]}' / 'choice.json'
        value = {'choice': frozen_choice(job), 'provenance': {
            'kind': 'frozen_published_choice', 'configuration_source': job['configuration_source'],
            'manifest_sha256': config['manifest_sha256'], 'source_commit': COMMIT,
            'locally_retuned': False, 'job_id': job['id']}}
        if path.exists() and json.loads(path.read_text()) != value:
            raise RuntimeError(f'Refusing to replace existing choice: {path}')
        write(path, value)
    for task in ('maze', 'sudoku'):
        result_path = artifacts / f'operation-counts-{task}.json'
        record_path = artifacts / f'operation-counts-{task}-execution.json'
        if record_path.exists():
            previous = json.loads(record_path.read_text())
            if previous.get('status') == 'complete' and result_path.exists() and digest(result_path) == previous['output_sha256']:
                continue
        while gpu_snapshot(gpu)['owners']:
            write(state_path, {**base_record, 'status': 'waiting_for_gpu_counts', 'task': task, 'utc': now()})
            time.sleep(15)
        argv = [config['python'], '-u', 'training_methods.py', 'count', '--tasks', task,
                '--full', '--out', str(result_path)]
        env_changes = {'CUDA_VISIBLE_DEVICES': str(gpu), 'CRAM_DATA': config['data'], 'V1_DATA': config['data'],
                       'PYTHONPATH': str(source), 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
                       'ALT_MAZE_KERNEL': 'free' if task == 'maze' else 'separable',
                       'ALT_MAZE_OPT': 'sgd-cosine' if task == 'maze' else 'adam'}
        record = {**base_record, 'status': 'running', 'task': task, 'argv': argv, 'env': env_changes,
                  'cwd': str(source / 'experiments'), 'started_utc': now(),
                  'counter_sha256': digest(source / 'baselines/flops/fullcount.py'),
                  'choices': {j['method']: frozen_choice(j) for j in config['jobs']
                              if j['family'] == 'table7_objectives' and j['seed'] == 0 and j['task'] == task}}
        write(record_path, record)
        start = time.monotonic()
        with (artifacts / f'operation-counts-{task}.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(argv, cwd=source / 'experiments', env={**os.environ, **env_changes},
                                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            while child.poll() is None:
                write(state_path, {**base_record, 'status': 'counting_operations', 'task': task,
                      'child_pid': child.pid, 'gpu': gpu_snapshot(gpu), 'elapsed_s': time.monotonic() - start, 'utc': now()})
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
        if child.returncode != 0 or not result_path.exists():
            record.update(status='failed', exit_code=child.returncode, finished_utc=now())
            write(record_path, record)
            raise RuntimeError(f'Operation counting failed: {task}; see {artifacts}')
        result = json.loads(result_path.read_text())
        if set(result['rows']) != {f'{task}/{m}' for m in config['protocol']['methods']}:
            raise RuntimeError('Operation counter omitted a method')
        record.update(status='complete', exit_code=0, output_sha256=digest(result_path),
                      wall_seconds=time.monotonic() - start, finished_utc=now())
        write(record_path, record)


def worker(args):
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    host, gpu = args.lane.split(':'); gpu = int(gpu)
    if socket.gethostname().split('.')[0].lower() != host:
        raise RuntimeError('Lane belongs to another host')
    if args.lane not in config['lanes']:
        raise RuntimeError('Lane is not authorized in this config')
    source = Path(config['source']); artifacts = ROOT / 'artifacts/objectives'
    artifacts.mkdir(parents=True, exist_ok=True)
    lock = (artifacts / f'gpu-{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    validation = verify(config)
    runtime = subprocess.check_output([config['python'], '-c',
        'import json,torch,platform; print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda)))'], text=True)
    base_record = {'lane': args.lane, 'worker_pid': os.getpid(), 'host': host, 'source': str(source),
                   'source_commit': COMMIT, 'configuration_sha256': digest(config_path),
                   'validation': validation, 'runtime': json.loads(runtime), 'started_utc': now()}
    write(artifacts / f'{host}-gpu{gpu}-worker.json', base_record)
    state_path = artifacts / f'{host}-gpu{gpu}-state.json'
    if args.count_first:
        count_updates(config, base_record, gpu)
    for job in config['jobs']:
        if job['lane'] != args.lane:
            continue
        run = source / 'experiments' / job['run_relative']
        job_artifacts = artifacts / 'jobs' / job['id']; job_artifacts.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(json.dumps({'job': job, 'source': config['source_sha256'],
                    'data': config['data_manifest'], 'hbar': config['fresh_hbar_prerequisite_sha256']}, sort_keys=True).encode()).hexdigest()
        owner = run / 'objective_provenance.json'
        if owner.exists():
            if json.loads(owner.read_text())['identity'] != identity:
                raise RuntimeError(f'Existing output identity mismatch: {run}')
        elif run.exists() and any(run.iterdir()):
            # Exactly one measured bootstrap fit predates this queue. Its
            # recorded command and source checksum authorize adoption.
            bootstrap_path = artifacts / 'bootstrap_maze_pl_s0.json'
            if job['id'] != 'objective-maze-pl-s0' or not bootstrap_path.exists():
                raise RuntimeError(f'Refusing unprovenanced existing outputs: {run}')
            bootstrap = json.loads(bootstrap_path.read_text())
            if bootstrap['source'] != str(source) or bootstrap['training_methods_sha256'] != config['source_sha256']['experiments/training_methods.py']:
                raise RuntimeError('Bootstrap provenance mismatch')
            expected = [config['python'], '-u'] + job['commands'][0][1:]
            if bootstrap['argv'] != expected:
                raise RuntimeError('Bootstrap command mismatch')
            write(owner, {'identity': identity, 'bootstrap': bootstrap, 'job': job, **base_record})
        else:
            write(owner, {'identity': identity, 'job': job, **base_record})
        execution_path = job_artifacts / 'execution.json'
        existing = json.loads(execution_path.read_text()) if execution_path.exists() else {}
        if existing.get('status') == 'complete':
            if digest(run / 'test.json') != existing['outputs']['test.json']:
                raise RuntimeError('Completed test output changed')
            continue
        # Always check NVML ownership; a different process is never stopped.
        while True:
            snapshot = gpu_snapshot(gpu)
            if not snapshot['owners']:
                break
            write(state_path, {**base_record, 'status': 'waiting_for_gpu', 'job': job['id'], 'gpu': snapshot, 'utc': now()})
            time.sleep(15)
        env_changes = {'CUDA_VISIBLE_DEVICES': str(gpu), 'CRAM_DATA': config['data'], 'V1_DATA': config['data'],
                       'PYTHONPATH': str(source), 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '8',
                       'MKL_NUM_THREADS': '8', 'WANDB_MODE': 'disabled', **job.get('env', {})}
        env = {**os.environ, **env_changes}
        record = {**base_record, 'identity': identity, 'job': job, 'env': env_changes,
                  'gpu': snapshot, 'status': 'running', 'utc': now(), 'steps': existing.get('steps', [])}
        write(execution_path, record)
        failed = False
        for stage, raw in enumerate(job['commands']):
            artifact = run / ('final.json' if stage == 0 else 'test.json')
            if artifact.exists():
                record['steps'].append({'stage': stage, 'status': 'verified_existing', 'sha256': digest(artifact), 'utc': now()})
                continue
            argv = [config['python'], '-u'] + raw[1:]
            log_path = job_artifacts / ('train.log' if stage == 0 else 'test.log')
            start = time.monotonic()
            step_record = {'stage': stage, 'argv': argv, 'cwd': str(source / 'experiments'), 'started_utc': now()}
            with log_path.open('ab', buffering=0) as log:
                child = subprocess.Popen(argv, cwd=source / 'experiments', env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                step_record['pid'] = child.pid
                def stop_child(signum, _frame):
                    os.killpg(child.pid, signum)
                    raise SystemExit(128 + signum)
                signal.signal(signal.SIGTERM, stop_child)
                signal.signal(signal.SIGINT, stop_child)
                while child.poll() is None:
                    sample = {**base_record, 'status': 'running', 'job': job['id'], 'stage': stage,
                              'child_pid': child.pid, 'elapsed_s': time.monotonic() - start,
                              'utc': now(), 'gpu': gpu_snapshot(gpu), 'progress': progress(run)}
                    write(state_path, sample)
                    with (job_artifacts / 'monitor.jsonl').open('a') as f:
                        f.write(json.dumps(sample) + '\n')
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                code = child.returncode
            step_record.update(exit_code=code, finished_utc=now(), wall_seconds=time.monotonic()-start)
            record['steps'].append(step_record)
            if code != 0 or not artifact.exists():
                record.update(status='failed', reason=f'Stage {stage} exit={code}, output_exists={artifact.exists()}')
                failed = True
                break
            write(execution_path, record)
        if not failed:
            result = json.loads((run / 'test.json').read_text())
            if result['test']['n'] != (50000 if job['task'] == 'sudoku' else 1000):
                raise RuntimeError('Unexpected test sample count')
            record.update(status='complete', finished_utc=now(), test=result['test'],
                          outputs={name: digest(run / name) for name in ('config.json', 'final.json', 'test.json')})
        write(execution_path, record)
        write(state_path, {**base_record, 'status': record['status'], 'job': job['id'], 'utc': now()})
        print(json.dumps({'job': job['id'], 'status': record['status'], 'test': record.get('test')}), flush=True)
    write(state_path, {**base_record, 'status': 'queue_complete', 'utc': now()})


def collect(args):
    config = json.loads(Path(args.config).read_text())
    root = ROOT / 'artifacts/objectives'
    hosts = sorted({lane.split(':')[0] for lane in config['lanes']})
    def one_host(host):
        dest = root / host
        dest.mkdir(parents=True, exist_ok=True)
        base = ['rsync', '-a', '--partial', '-e', 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes',
                '--include=*/', '--include=*.json', '--include=*.jsonl', '--include=*.log', '--exclude=*']
        for remote, local in [('artifacts/objectives/', str(dest) + '/'),
                              ('objective_source/experiments/runs/', str(dest / 'runs') + '/')]:
            subprocess.run(base + [f'{host}:{ROOT}/{remote}', local], check=True, capture_output=True, text=True)
        return {'host': host, 'utc': now(), 'status': 'collected'}
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        futures = {host: pool.submit(one_host, host) for host in hosts}
        results = []
        for host, future in futures.items():
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({'host': host, 'status': 'failed', 'error': str(exc), 'utc': now()})
    write(root / 'collection.json', {'results': results, 'utc': now()})
    from collect_external_bptt import collect as collect_external
    collect_external()
    summarize(args)
    if any(x['status'] == 'failed' for x in results):
        raise RuntimeError('A host collection failed; see collection.json')


def summarize(args):
    config = json.loads(Path(args.config).read_text())
    root = ROOT / 'artifacts/objectives'
    records, groups, lanes, counts = [], {}, {}, {}
    for lane in config['lanes']:
        host, gpu = lane.split(':')
        state_path = root / host / f'{host}-gpu{gpu}-state.json'
        if state_path.exists():
            lanes[lane] = json.loads(state_path.read_text())
    for task in ('maze', 'sudoku'):
        for host in sorted({lane.split(':')[0] for lane in config['lanes']}):
            path = root / host / f'operation-counts-{task}.json'
            provenance = root / host / f'operation-counts-{task}-execution.json'
            if path.exists() and provenance.exists():
                execution = json.loads(provenance.read_text())
                if execution.get('status') == 'complete' and digest(path) == execution['output_sha256']:
                    counts.update(json.loads(path.read_text())['rows'])
    for job in config['jobs']:
        host, _ = job['lane'].split(':')
        run = root / host / job['run_relative']
        execution_path = root / host / 'jobs' / job['id'] / 'execution.json'
        record = {key: job[key] for key in ('id', 'family', 'lane', 'task', 'method', 'seed', 'run_relative')}
        record.update(status='pending', frozen_choice=frozen_choice(job))
        if execution_path.exists():
            execution = json.loads(execution_path.read_text())
            record['status'] = execution['status']
            record['execution_path'] = str(execution_path.relative_to(ROOT))
            if execution.get('reason'):
                record['reason'] = execution['reason']
            if execution['status'] == 'complete':
                missing = [name for name, expected in execution['outputs'].items()
                           if not (run / name).exists() or digest(run / name) != expected]
                if missing:
                    record.update(status='collection_incomplete', missing_or_mismatched=missing)
                else:
                    record.update(final=json.loads((run / 'final.json').read_text()),
                                  result=json.loads((run / 'test.json').read_text()))
                    early = {}
                    with (run / 'log.jsonl').open() as log:
                        for line in log:
                            entry = json.loads(line)
                            if 'update' in entry:
                                early[entry['update']] = entry['fit_s']
                            if len(early) >= 21:
                                break
                    n = min(20, max(early) - 1)
                    startup = early[1] - statistics.median(early[k + 1] - early[k] for k in range(1, n + 1))
                    record['startup_s'] = startup
                    record['startup_estimate_later_updates'] = n
        state = lanes.get(job['lane'], {})
        if state.get('job') == job['id'] and record['status'] != 'complete':
            record['worker_status'] = state['status']
            record['progress'] = state.get('progress')
        records.append(record)
        # Group by the complete recipe, so long and intermediate schedules can
        # never replace or mingle with a short published-choice row.
        argv = job['commands'][0]
        schedule = argv[argv.index('--updates') + 1] if '--updates' in argv else 'standard'
        key = f'{job["task"]}/{job["method"]}/{schedule}'
        groups.setdefault(key, []).append(record)
    external_path = ROOT / 'artifacts/long_bptt/collected/summary.json'
    collected_external = json.loads(external_path.read_text())['jobs'] if external_path.exists() else []
    external = []
    for job in config['external_jobs']:
        cmd = job['commands'][0]; seed = int(cmd[cmd.index('--seed') + 1])
        got = next((j for j in collected_external if j['seed'] == seed), {})
        rec = {'id': job['id'], 'family': 'table7_external_long_bptt', 'task': 'sudoku', 'method': 'bptt',
               'seed': seed, 'frozen_choice': frozen_choice(job), 'status': 'externally_managed', **got}
        rec['collector_id'] = rec['id']; rec['id'] = job['id']
        external.append(rec)
    groups['sudoku/bptt/3840000'] = external
    rows = {}
    for key, seeds in groups.items():
        seeds = sorted(seeds, key=lambda x: x['seed'])
        done = [s for s in seeds if s['status'] == 'complete']
        row = {'status': 'complete' if len(done) == 3 else 'incomplete', 'job_ids': [s['id'] for s in seeds],
               'completed_seeds': [s['seed'] for s in done], 'expected_seeds': [0, 1, 2],
               'frozen_choice': seeds[0]['frozen_choice'], 'locally_retuned': False}
        if len(done) == 3:
            fins, tests = [s['final'] for s in done], [s['result'] for s in done]
            rates = [100 * t['test']['solved'] / t['test']['n'] for t in tests]
            def median_censored(values):
                value = statistics.median(math.inf if v is None else v for v in values)
                return None if value == math.inf else value
            row.update(solved_percent_mean=statistics.fmean(rates), solved_percent_sample_sd=statistics.stdev(rates),
                       solved_percent_per_seed=rates, n_test_per_seed=[t['test']['n'] for t in tests],
                       wrong_stops=sum(t['test']['wrong_stops'] for t in tests),
                       budget_failures=sum(t['test']['budget_failures'] for t in tests),
                       T99_s_per_seed=[f['T99_s'] for f in fins],
                       T99_s_median=median_censored([f['T99_s'] for f in fins]),
                       T99_update_per_seed=[f['T99_update'] for f in fins],
                       T99_update_median=median_censored([f['T99_update'] for f in fins]),
                       full_schedule_fit_s_per_seed=[f['fit_s'] for f in fins],
                       selected_fit_plus_ready_s_per_seed=[t['fit_s'] for t in tests],
                       mean_legacy_inference_flop=statistics.fmean(t['test']['mean_flop'] for t in tests))
            adjusted_t99 = [None if s['final']['T99_s'] is None else
                            s['final']['T99_s'] - s['startup_s'] + s['result']['ready_s'] for s in done]
            row.update(startup_s_per_seed=[s['startup_s'] for s in done],
                       ready_s_per_seed=[s['result']['ready_s'] for s in done],
                       paper_adjusted_T99_s_per_seed=adjusted_t99,
                       paper_adjusted_T99_s_median=median_censored(adjusted_t99),
                       paper_adjusted_selected_fit_s_per_seed=[s['result']['fit_s'] - s['startup_s'] for s in done])
            row['individual_crossings'] = {k: row[k] for k in (
                'T99_s_per_seed', 'T99_s_median', 'T99_update_per_seed', 'T99_update_median',
                'paper_adjusted_T99_s_per_seed', 'paper_adjusted_T99_s_median')}
            crossing = common_t99(fins, row['startup_s_per_seed'], row['ready_s_per_seed'])
            row.update(T99_definition='First common probe update where the mean solve rate of all three seeds reaches99percent; median fit time at that update.',
                       T99_update_common_mean=crossing['update'], T99_update_median=crossing['update'],
                       T99_update_per_seed=[crossing['update']] * 3,
                       T99_s_per_seed=crossing['fit_s_per_seed'],
                       T99_s_median=median_censored(crossing['fit_s_per_seed']),
                       paper_adjusted_T99_s_per_seed=crossing['paper_adjusted_fit_s_per_seed'],
                       paper_adjusted_T99_s_median=median_censored(crossing['paper_adjusted_fit_s_per_seed']),
                       T99_mean_probe_solved_percent=crossing['mean_solved_percent'],
                       common_probe_curve=crossing['common_curve'])
            count = counts.get('/'.join(key.split('/')[:2]))
            # These counts depend on task, method and knob, not the LR or update
            # budget. The complete frozen choice remains visible for auditing.
            if count and count['choice'].get('knob') == row['frozen_choice'].get('knob'):
                per = count['flop_per_update']
                row.update(fullcount_flop_per_update=per,
                           flopcountermode_per_update=count['flopcountermode_next_update'],
                           T99_fullcount_flop=None if row['T99_update_median'] is None else per * row['T99_update_median'])
        rows[key] = row
    status_counts = {status: sum(r['status'] == status for r in records) for status in sorted({r['status'] for r in records})}
    budgets = {}
    for lane in config['lanes']:
        remaining = [j for j, r in zip(config['jobs'], records) if j['lane'] == lane and r['status'] != 'complete']
        caps = []
        for j in remaining:
            cmd = j['commands'][0]
            caps.append(float(cmd[cmd.index('--cap-s') + 1]) if '--cap-s' in cmd else 3600.)
        budgets[lane] = {'remaining_jobs': len(remaining), 'declared_remaining_fit_cap_hours': sum(caps) / 3600,
                         'note': 'Upper bounds on fit clocks, not estimated wall time; probes and tests are additional.'}
    summary = {'schema_version': 1, 'generated_utc': now(), 'source_commit': COMMIT,
               'config_sha256': digest(args.config), 'protocol': config['protocol'],
               'timing_note': 'Fresh cold fits on Blackwell; probe and final evaluation wall time is separate from fit_s.',
               'inference_count_note': 'mean_legacy_inference_flop uses upstream search accounting, not paper FC2.',
               'tuning_note': 'Published frozen choices reused. No local tuning sweep or invented development scores.',
               'status_counts': status_counts, 'lanes': lanes, 'jobs': records, 'rows': rows,
               'remaining_fit_cap_budget_by_lane': budgets,
               'operation_counts': counts,
               'external_jobs': external,
               'external_status_counts': {s: sum(j['status'] == s for j in external) for s in sorted({j['status'] for j in external})},
               'scope_complete': all(j['status'] == 'complete' for j in records + external) and len(counts) == 24}
    write(root / 'summary.json', summary)
    print(json.dumps({'summary': str(root / 'summary.json'), 'managed_status': status_counts,
                      'complete_three_seed_rows': sum(r['status'] == 'complete' for r in rows.values()),
                      'operation_count_rows': len(counts), 'external': len(config['external_jobs'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build')
    p = sub.add_parser('worker'); p.add_argument('--lane', required=True); p.add_argument('--count-first', action='store_true')
    p.add_argument('--config', default=str(ROOT / 'configs/objective_jobs.json'))
    for command in ('collect', 'summarize'):
        p = sub.add_parser(command)
        p.add_argument('--config', default=str(ROOT / 'configs/objective_jobs.json'))
    arguments = parser.parse_args()
    globals()[arguments.command](arguments)
