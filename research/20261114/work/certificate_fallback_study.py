"""H2bj fixed source routing and actual BP subset replay; no oracle routing."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import study as S
import selective_fallback as F
import larger_relaxation as L
import fast_npz as Z
from certificate_fallback_policy import accepted, chosen_model
from confirmation96_study import load_models, protocol as source_protocol

PP = S.HERE/'certificate-fallback-protocol.json'
DEST = S.OUT/'certificate_fallback'
SOURCE = S.OUT/'loop80_confirmation'


def protocol():
    p = json.loads(PP.read_text())
    for field in ['source_sha256', 'source_evidence_sha256']:
        for path, h in p[field].items():
            assert S.sha(S.ROOT/path) == h, path
    return p


def freeze():
    assert not PP.exists()
    proposal = S.HERE/'certificate-fallback-proposal.json'
    p = copy.deepcopy(json.loads(proposal.read_text()))
    source = json.loads((SOURCE/'protocol.json').read_text())
    a = json.loads((SOURCE/'audit.json').read_text())
    assert a['status'] == 'passed' and a['all_final_states'] == 98304
    synthetic = DEST/'synthetic-gate.json'
    sg = json.loads(synthetic.read_text())
    assert sg['status'] == 'passed' and sg['synthetic_certificate_cases'] == 288
    for name, h in sg['source_sha256'].items():
        assert S.sha(S.HERE/name) == h
    datasets = json.loads((SOURCE/'datasets.json').read_text())['datasets']
    jobs = []
    for job in source['jobs']:
        run = SOURCE/'runs'/job['id']
        receipt = json.loads((run/'result.json').read_text())
        assert receipt['status'] == 'complete'
        assert receipt['protocol_sha256'] == S.sha(SOURCE/'protocol.json')
        assert S.sha(run/'outputs.npz') == receipt['outputs_sha256']
        jobs.append({**job, 'outputs_sha256': receipt['outputs_sha256'],
                     'receipt_sha256': S.sha(run/'result.json'),
                     'batch_sha256': {f'batch{i:02d}.npz': S.sha(run/f'batch{i:02d}.npz')
                                      for i in range(8)}})
    names = ['certificate_fallback_policy.py', 'preflight_certificate_fallback.py',
             'certificate_fallback_study.py', 'audit_certificate_fallback.py']
    p.update(status='frozen_before_routing_and_gpu_replays', frozen_utc=S.utc(),
             source_jobs=jobs, datasets=datasets, models=source['models'],
             expected_routes=[{'input_index': 4078, 'model_seed': seed} for seed in range(3)])
    p['source_sha256'] = {**source['source_sha256'],
                         **{str((S.HERE/n).relative_to(S.ROOT)): S.sha(S.HERE/n) for n in names}}
    evidence = [proposal, synthetic, SOURCE/'protocol.json', SOURCE/'audit.json',
                SOURCE/'datasets.json', SOURCE/'model-extraction.json']
    evidence += [SOURCE/f'model-seed{s}-{arm}.npz' for s in range(3)
                 for arm in ['pl', 'old_binary']]
    p['source_evidence_sha256'] = {str(f.relative_to(S.ROOT)): S.sha(f) for f in evidence}
    S.save(PP, p)
    S.save(DEST/'protocol.json', p)
    S.emit('certificate_fallback_frozen', sha256=S.sha(PP))


def route():
    started = time.perf_counter()
    p = protocol()
    rows = []
    jobs = []
    counts = {str(s): {'model_inputs': 0, 'accepted': 0, 'routed': 0} for s in range(3)}
    for job in p['source_jobs']:
        source = SOURCE/'runs'/job['id']/'outputs.npz'
        assert S.sha(source) == job['outputs_sha256']
        # The routing pass cannot load prediction, labels, distances or audit outcomes.
        with np.load(source) as z:
            fields = dict(hidden_good=z['hidden_good'], finite=z['finite'][7],
                          stopped=z['stopped'][7], native_cap=z['block_native_cap'][5],
                          block_cap=z['block_cap'][5], margin=z['minimum_flip_margin64'][7])
        assert all(x.shape == (256,) for x in fields.values())
        accept = accepted(**fields)
        selected = chosen_model(accept)
        path = DEST/'routing'/f'{job["id"]}.npz'
        path.parent.mkdir(exist_ok=True)
        Z.save(path, **fields, accepted=accept, chosen_model=selected)
        record = {'job': job, 'file': str(path.relative_to(DEST)),
                  'sha256': S.sha(path), 'accepted': int(accept.sum()),
                  'routed': int((~accept).sum())}
        jobs.append(record)
        for i in np.flatnonzero(~accept):
            rows.append({'job_id': job['id'], 'shard': job['shard'],
                         'input_index': job['shard']*256+int(i),
                         'local_index': int(i), 'batch': int(i)//32,
                         'batch_index': int(i)%32, 'model_seed': job['seed'],
                         'gpu': 0 if job['seed'] in [0, 2] else 1})
        counter = counts[str(job['seed'])]
        counter['model_inputs'] += 256
        counter['accepted'] += int(accept.sum())
        counter['routed'] += int((~accept).sum())
        if time.perf_counter()-started > p['budgets']['routing_cpu_wall_seconds']:
            S.save(DEST/'routing-partial.json', {'status': 'budget_exhausted', 'jobs': jobs,
                                               'rows': rows, 'counts': counts})
            raise TimeoutError('Frozen routing CPU budget; no extension')
    actual = sorted((r['input_index'], r['model_seed']) for r in rows)
    expected = [(r['input_index'], r['model_seed']) for r in p['expected_routes']]
    status = 'passed' if actual == expected and all(c['model_inputs'] == 4096 for c in counts.values()) else 'cohort_mismatch'
    S.save(DEST/'routing.json', {'status': status, 'protocol_sha256': S.sha(PP),
                               'rows': rows, 'jobs': jobs, 'counts_by_seed': counts,
                               'model_inputs': sum(c['model_inputs'] for c in counts.values()),
                               'seconds': time.perf_counter()-started,
                               'scope': 'Every source case routed using numerical certificate fields only.'})
    assert status == 'passed', 'Unexpected or empty route cohort; no GPU execution authorized'


def cpu_output(z):
    return {'prediction': z[0].bool().cpu().numpy(), 'iterations': z[1].cpu().numpy(),
            'stopped': z[2].cpu().numpy(), 'finite': z[3].cpu().numpy(),
            'final_logits': z[4].cpu().numpy()}


def parity(cpu, old, indices):
    names = {'prediction': 'predictions', 'iterations': 'iterations', 'stopped': 'stopped',
             'finite': 'finite', 'final_logits': 'native_final_logits'}
    return {k: bool(np.array_equal(v, old[names[k]][0, indices])) for k, v in cpu.items()}


def evaluate(gpu):
    started = time.perf_counter()
    lock, inventory = F.acquire(gpu)
    torch, _, _ = S.imports()
    p = protocol()
    routes = json.loads((DEST/'routing.json').read_text())
    assert routes['status'] == 'passed' and routes['protocol_sha256'] == S.sha(PP)
    assert routes['model_inputs'] == 12288 and len(routes['rows']) == 3
    assigned = [r for r in routes['rows'] if r['gpu'] == gpu]
    assert len(assigned) == (2 if gpu == 0 else 1)
    state = {'status': 'running', 'gpu': gpu, 'inventory': inventory, 'pid': os.getpid(),
             'started_utc': S.utc(), 'completed_gates': [], 'completed_fallbacks': [],
             'watchdog_checks': 0}
    def watch():
        state['watchdog_checks'] += 1
        state.update(worker_seconds=time.perf_counter()-started, updated_utc=S.utc())
        if state['worker_seconds'] > p['budgets']['gpu_gate_and_fallback_worker_wall_seconds']:
            state['status'] = 'budget_exhausted'
            S.save(DEST/f'worker{gpu}.json', state)
            raise TimeoutError('Frozen GPU worker budget; no extension')
        if (DEST/f'failure-evaluate-{1-gpu}.json').exists():
            raise RuntimeError('Peer worker failed; stopping further source execution')
        S.save(DEST/f'worker{gpu}.json', state)
    watch()
    sp = source_protocol()
    cache = []
    # All gates assigned to this worker pass before any subset execution.
    for row in assigned:
        seed = row['model_seed']
        models, doubles = load_models(sp, seed)
        job = next(j for j in p['source_jobs'] if j['id'] == row['job_id'])
        ds = p['datasets'][row['shard']]
        assert S.sha(S.ROOT/ds['path']) == ds['sha256']
        with np.load(S.ROOT/ds['path']) as z:
            tokens = z['tokens'][row['batch']*32:(row['batch']+1)*32]
        batch = SOURCE/'runs'/row['job_id']/f'batch{row["batch"]:02d}.npz'
        assert S.sha(batch) == job['batch_sha256'][batch.name]
        with np.load(batch) as z:
            old = {k: z[k] for k in ['couplings', 'hidden_good', 'predictions', 'iterations',
                                     'stopped', 'finite', 'native_final_logits']}
        torch.cuda.synchronize(); before = time.perf_counter()
        t = torch.tensor(tokens.astype(np.int64), device='cuda')
        a = torch.tensor(old['couplings'], device='cuda')
        torch.cuda.synchronize(); setup_seconds = time.perf_counter()-before
        with torch.no_grad():
            torch.cuda.synchronize(); before = time.perf_counter()
            full = L.solve(models[0], t, a, torch.full_like(a[:, 0], 3.), 8192, 0.)
            torch.cuda.synchronize(); solve_seconds = time.perf_counter()-before
            before = time.perf_counter(); cpu = cpu_output(full)
            torch.cuda.synchronize(); return_seconds = time.perf_counter()-before
        checks = parity(cpu, old, np.arange(32))
        path = DEST/'gates'/f'seed{seed}.npz'; path.parent.mkdir(exist_ok=True)
        Z.save(path, **cpu)
        receipt = {'status': 'passed' if all(checks.values()) else 'parity_failed',
                   'route': row, 'checks': checks, 'source_batch_sha256': S.sha(batch),
                   'output_sha256': S.sha(path), 'input_setup_seconds': setup_seconds,
                   'solve_seconds': solve_seconds, 'cpu_return_seconds': return_seconds}
        S.save(path.with_suffix('.json'), receipt)
        if not all(checks.values()):
            raise AssertionError('Source gate failed: cross-certificate endpoint incomplete')
        state['completed_gates'].append(seed); watch()
        cache.append((row, models, doubles, tokens, old, t, a))
    S.save(DEST/f'gate-worker{gpu}.json', {'status': 'passed', 'gpu': gpu,
                                        'protocol_sha256': S.sha(PP), 'seeds': state['completed_gates']})
    # A failure on either GPU forbids subset execution on both.
    while not all((DEST/f'gate-worker{g}.json').exists() for g in [0, 1]):
        time.sleep(.1)
        watch()
    for g in [0, 1]:
        gate = json.loads((DEST/f'gate-worker{g}.json').read_text())
        assert gate['status'] == 'passed' and gate['protocol_sha256'] == S.sha(PP)
    for row, models, doubles, tokens, old, t, a in cache:
        seed = row['model_seed']; index = row['batch_index']
        with torch.no_grad():
            torch.cuda.synchronize(); before = time.perf_counter()
            ix = torch.tensor([index], device='cuda'); tt = t[ix]; aa = a[ix]
            torch.cuda.synchronize(); gather_seconds = time.perf_counter()-before
            before = time.perf_counter()
            out = L.solve(models[0], tt, aa, torch.full_like(aa[:, 0], 3.), 8192, 0.)
            torch.cuda.synchronize(); solve_seconds = time.perf_counter()-before
            before = time.perf_counter(); cpu = cpu_output(out)
            torch.cuda.synchronize(); return_seconds = time.perf_counter()-before
            checks = parity(cpu, old, [index])
            path = DEST/'fallback'/f'i{row["input_index"]}-s{seed}.npz'; path.parent.mkdir(exist_ok=True)
            # Preserve replay immediately, before numerical diagnosis or any label access.
            Z.save(path, **cpu)
            preliminary = {'status': 'passed' if all(checks.values()) else 'parity_failed',
                           'route': row, 'protocol_sha256': S.sha(PP), 'checks': checks,
                           'gather_seconds': gather_seconds, 'solve_seconds': solve_seconds,
                           'cpu_return_seconds': return_seconds, 'output_sha256': S.sha(path)}
            S.save(path.with_suffix('.json'), preliminary)
            if not all(checks.values()):
                raise AssertionError('Subset parity failed: cross-certificate endpoint incomplete')
            before = time.perf_counter(); margins = []; energies = []
            for model in doubles:
                board = out[0].double()
                margins.append(model.flip_delta(tt, board, aa.double()).masked_fill(tt != 1, float('inf')).flatten(1).amin(1).cpu().numpy())
                energies.append(model.energy(tt, board, aa.double()).cpu().numpy())
            torch.cuda.synchronize(); certificate_seconds = time.perf_counter()-before
        evidence = DEST/'diagnostics'/path.name; evidence.parent.mkdir(exist_ok=True)
        Z.save(evidence, tokens=tokens[index:index+1], couplings=old['couplings'][index:index+1],
               hidden_good=old['hidden_good'][index:index+1], margins=np.stack(margins),
               energies=np.stack(energies))
        preliminary.update(certificate_seconds=certificate_seconds,
                           diagnostics_sha256=S.sha(evidence),
                           bp_checkpoint_sha256=p['models'][str(seed)]['pl']['sha256'],
                           lp_checkpoint_sha256=p['models'][str(seed)]['old_binary']['sha256'])
        S.save(path.with_suffix('.json'), preliminary)
        state['completed_fallbacks'].append(seed); watch()
    state.update(status='complete', worker_seconds=time.perf_counter()-started, updated_utc=S.utc())
    S.save(DEST/f'worker{gpu}.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'route', 'evaluate'])
    parser.add_argument('--gpu', type=int, choices=[0, 1])
    args = parser.parse_args()
    try:
        evaluate(args.gpu) if args.command == 'evaluate' else globals()[args.command]()
    except BaseException as error:
        import traceback
        S.save(DEST/f'failure-{args.command}-{args.gpu}.json', {
            'status': 'incomplete', 'error': repr(error), 'traceback': traceback.format_exc(),
            'scope': 'Infrastructure, parity or budget failure; do not infer algorithmic certificate failure.'})
        raise
