"""H3c: fixed-trajectory training checkpoints, without test-driven selection."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'research/20261114/loki/h3c'
OUT = ROOT / 'artifacts/research/20261114/loki/h3c'
OLD = ROOT / 'artifacts/research/20261114/loki'
UP = ROOT / 'upstream'
sys.path[:0] = [str(SRC.parent), str(UP), str(UP / 'experiments')]
import study as S


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value): S.save(path, value)
def emit(event, **values): S.emit(event, **values)
def protocol(): return read(SRC / 'protocol.json')


def prepare():
    assert not (SRC / 'protocol.json').exists(), 'Protocol is immutable'
    parent = read(SRC.parent / 'h3b/protocol.json')
    caches = {
        'adam-nodecay': OLD / 'h3b/n10/fits/adam-nodecay/model.pt',
        'sgd-pre': OLD / 'n10/fits/sgd-pre/model.pt',
        'adamw-decay': OLD / 'n10/fits/adamw-lr0.3/model.pt',
    }
    sources = {key: value for key, value in parent['source_sha256'].items()
               if key.startswith('upstream/') or key.endswith('/loki/study.py')}
    sources[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    data = dict(parent['data_sha256'])
    data.update(parent['cohorts']['10']['data_sha256'])
    data[str((OLD / 'cohorts-n10.npz').relative_to(ROOT))] = sha(OLD / 'cohorts-n10.npz')
    p = dict(id='H3c-training-prefix', frozen_utc=now(),
        hypothesis='No-decay Adam at the fixed 1000-update prefix is within 0.1 percentage point of preconditioned SGD at 4000 updates in N20 accuracy, using at most half the median complete in-memory fitting time.',
        antecedent='H3b showed that removing weight decay at LR 0.3 closed the full-test gap. H3c asks whether the same fixed trajectory reaches adequate accuracy earlier.',
        primary=dict(arm='adam-nodecay', updates=1000, control='sgd-pre', control_updates=4000,
                     accuracy_margin_pp=0.1, maximum_median_fit_time_ratio=0.5),
        checkpoints=[250, 500, 1000, 2000, 4000],
        arms={'adam-nodecay': dict(optimizer='adamw', lr=.3, weight_decay=0., precond=False),
              'sgd-pre': dict(optimizer='sgd', lr=30., weight_decay=0., precond=True)},
        fit=dict(parent['fit'], rows=97554, schedule_denominator=4000,
                 prefix_schedule='All checkpoints are prefixes of the unchanged 4000-step cosine schedule, not separately retuned schedules.'),
        search=dict(unique_inputs=6513, chains=20, cap=20000, seeds=[0, 1, 2],
                    temperature=1., stop='local', margin=.001,
                    input_file='artifacts/research/20261114/loki/h3b/test-inputs.npz'),
        decision=dict(confidence='One-sided 95%, paired unique-input bootstrap 2000 draws, seed 20261114; all three search seeds stay within each input.',
            safeguard='Exact one-sided 95% Clopper-Pearson upper bound on input fraction with any paired accuracy regression <0.001.',
            rule='Supported if accuracy safeguard passes and median fit-time ratio <=0.5. Falsified if accuracy bootstrap upper difference < -0.1 pp or observed median ratio >0.5; otherwise inconclusive. Timing decision is descriptive of this fixed machine and timing protocol, not a population inference.',
            secondary='All other checkpoint outcomes, coverage, N1, wrong stops, capped chains and move cost are secondary. No checkpoint, LR, temperature, budget or stop is selected from these outcomes.'),
        timing=dict(device='Loki GPU0 for both compared arms, exclusive GPU lock, no other campaign work until timing completes',
            repeats=3, warmups_each=1, order=[['adam-nodecay', 'sgd-pre'], ['sgd-pre', 'adam-nodecay'], ['adam-nodecay', 'sgd-pre']],
            scope='Complete in-memory fitting: preloaded host arrays and fitting indices, then GPU transfer, task construction, all updates, training-derived unit/tau/delta, and CPU inference-ready model. Excludes dataset disk I/O, Python import, CUDA context, checkpoint disk writes. Eager PyTorch has no torch.compile phase. No intermediate snapshots in timed fits.',
            checks='Each warm and measured fitted model must exactly equal its corresponding frozen trajectory checkpoint; every timing sample retained.'),
        cache_models={arm: dict(path=str(path.relative_to(ROOT)), sha256=sha(path)) for arm, path in caches.items()},
        cache_results='H3b primary final outputs are reused at 4000 updates only after tensorwise and scalarwise exact model parity. AdamW-decay 4000 is a labeled cached reference, not a new trajectory.',
        validation='CPU algebraic positive-scale invariance check with margin scaled too; GPU 4000-step exact cached-model parity before search; CPU saved-board exhaustive-solution/givens validation after search.',
        source_sha256=sources, data_sha256=data, parent_protocol_sha256=sha(SRC.parent / 'h3b/protocol.json'),
        limits=['Official test inputs were already used in H3b; this is an adaptive research follow-up, not untouched external confirmation.',
                'The 1000-update hypothesis is frozen before any H3c checkpoint accuracy is seen; all checkpoints are reported.',
                'One deterministic full-batch fitted trajectory per optimizer; search seeds are not independent training seeds.',
                'Geometry-derived preconditioning is retained in the SGD control; the optimizers have fixed unequal historical hyperparameter selection.',
                'Timing is within Loki Blackwell hardware, not an H200 comparison or end-to-end inference speedup.'])
    save(SRC / 'protocol.json', p)
    emit('protocol_frozen', sha256=sha(SRC / 'protocol.json'))


def validate_cpu():
    import numpy as np
    import torch
    sys.path.insert(0, str(SRC.parent / 'h3b'))
    from cram import cram_nqueens as Q
    model = torch.load(OLD / 'n10/fits/adamw-lr0.3/model.pt', weights_only=False)
    alpha = 2.  # Exact positive power-of-two scaling isolates the algebraic invariant.
    scaled = {key: value * alpha if torch.is_tensor(value) or isinstance(value, float) else value for key, value in model.items()}
    inputs = np.load(OLD / 'h3b/test-inputs.npz')['inputs'][:13] == 2
    a = Q.solve(model, inputs, chains=2, cap=37, seed=883, stop='local', temperature=1., margin=.001)
    b = Q.solve(scaled, inputs, chains=2, cap=37, seed=883, stop='local', temperature=1., margin=.001 * alpha)
    checks = [np.array_equal(a[i], b[i]) for i in [0, 1, 3, 4]]
    checks.append(np.array_equal(a[2] * alpha, b[2]))
    assert all(checks)
    save(OUT / 'validation.json', dict(status='passed', utc=now(), positive_scale_invariance_checks=checks,
        explanation='For alpha>0: fields, energies, unit, delta and margin all scale by alpha, so Gibbs logits, unhappy sets, ordering and stopping inequalities are unchanged in exact arithmetic. Keeping an absolute margin fixed only changes the relative stopping tolerance. This exact-power-of-two CPU fixture confirms the implementation invariant without measuring research accuracy.',
        protocol_sha256=sha(SRC / 'protocol.json'), source_sha256=sha(__file__)))


def same_model(a, b):
    import torch
    keys = ['A', 'b', 'W', 'B', 'S']
    equal = {key: torch.equal(a[key], b[key]) for key in keys}
    equal.update({key: a[key] == b[key] for key in ['tau', 'unit', 'delta', 'arm']})
    assert all(equal.values()), equal
    return equal


def loaded_training():
    import numpy as np
    import nqueens_offset as O
    x, y = O.Q.load_tokens(10, 'train')
    rows = np.load(OLD / 'cohorts-n10.npz')['fit_rows']
    return x, y, rows


def fit(arm, updates, data, progress, snapshots=False):
    import torch
    import nqueens_offset as O
    import nqueens_free as N
    import training_methods as AT
    frozen = protocol(); cfg = frozen['arms'][arm]
    checkpoints = set(frozen['checkpoints'])
    x, y, rows = data
    torch.cuda.synchronize(); started = time.perf_counter()
    xi = torch.as_tensor(x[rows], device='cuda') == 2
    target = (torch.as_tensor(y[rows], device='cuda') == 2).long()
    task = O.OffsetTask(xi, target, 10, 'ising', dense=False)
    gen = torch.Generator(device='cuda').manual_seed(0)
    bufs = [torch.zeros_like(t) for t in task.params()]
    counts = torch.bincount(O.offsets(10, 'cuda')[0]).float(); pre = counts / counts.mean()
    opt = torch.optim.AdamW(task.params(), lr=cfg['lr'], weight_decay=0., foreach=False) if arm == 'adam-nodecay' else None
    def ready():
        with torch.no_grad():
            A, b = task.A.detach(), task.b.detach()
            W, B = O.pair_tables(A, b, 'ising', 10)
            energy, tau, unit, delta = N.ready(W, B, target)
            return dict(W=W.cpu(), B=B.cpu(), S=energy.cpu(), tau=tau, unit=unit, delta=delta,
                        A=A.cpu(), b=b.cpu(), arm='ising')
    for k in range(updates):
        task.begin(); loss = AT.loss_fn('pl', None, task, task.ctx, gen, {})
        grads = torch.autograd.grad(loss, task.params())
        lr = cfg['lr'] * .5 * (1 + math.cos(math.pi * k / 4000))
        if opt is not None:
            opt.param_groups[0]['lr'] = lr
            for par, grad in zip(task.params(), grads): par.grad = grad
            opt.step()
        else:
            grads = (grads[0] / pre, grads[1])
            with torch.no_grad():
                for par, grad, buf in zip(task.params(), grads, bufs):
                    buf.mul_(.5).add_(grad); par.sub_(lr * buf)
        if (k + 1) % 250 == 0:
            progress(stage='fit', arm=arm, update=k + 1, limit=updates, loss=float(loss.detach()))
        if snapshots and k + 1 in checkpoints:
            model = ready(); dest = OUT / f'fits/{arm}/u{k+1}'
            dest.mkdir(parents=True, exist_ok=True); torch.save(model, dest / 'model.pt')
            save(dest / 'fit.json', dict(status='complete', arm=arm, updates=k+1, schedule_denominator=4000,
                rows=len(rows), final_loss=float(loss.detach()), unit=model['unit'], delta=model['delta'], tau=model['tau'],
                model_sha256=sha(dest / 'model.pt'), protocol_sha256=sha(SRC / 'protocol.json'), source_sha256=sha(__file__),
                timing_scope='Untimed trajectory with intermediate snapshots; primary timing uses separate actual prefix fits.'))
    model = ready()
    torch.cuda.synchronize(); seconds = time.perf_counter() - started
    return model, seconds


def time_fits(data, progress):
    import torch
    p = protocol(); records = []
    targets = {'adam-nodecay': 1000, 'sgd-pre': 4000}
    for arm in targets:
        model, seconds = fit(arm, targets[arm], data, progress)
        expected = torch.load(OUT / f'fits/{arm}/u{targets[arm]}/model.pt', weights_only=False)
        records.append(dict(arm=arm, updates=targets[arm], warmup=True, seconds=seconds, parity=same_model(model, expected)))
    for repeat, order in enumerate(p['timing']['order']):
        for arm in order:
            model, seconds = fit(arm, targets[arm], data, progress)
            expected = torch.load(OUT / f'fits/{arm}/u{targets[arm]}/model.pt', weights_only=False)
            records.append(dict(arm=arm, updates=targets[arm], warmup=False, repeat=repeat, seconds=seconds,
                                parity=same_model(model, expected)))
    save(OUT / 'timing.json', dict(status='complete', utc=now(), records=records, gpu=0,
        protocol_sha256=sha(SRC / 'protocol.json'), source_sha256=sha(__file__), scope=p['timing']['scope']))


def search(arm, update, seed, progress):
    import numpy as np
    import torch
    from cram import cram_nqueens as Q
    from cram import nqueens_data as D
    modelpath = OUT / f'fits/{arm}/u{update}/model.pt'
    model = torch.load(modelpath, weights_only=False)
    inputs = np.load(OLD / 'h3b/test-inputs.npz')['inputs'] == 2
    progress(stage='search', arm=arm, update=update, seed=seed, inputs=len(inputs))
    emit('search_start', arm=arm, update=update, seed=seed)
    start = time.perf_counter()
    boards, stopped, energy, moves, flop = Q.solve(model, inputs, chains=20, temperature=1., cap=20000,
        seed=seed, device='cuda', stop='local', margin=.001)
    seconds = time.perf_counter() - start
    ok = S.independent_ok(boards, inputs, 10)
    solutions = np.asarray(D.completions(np.zeros(100, bool), 10), bool)
    counts = np.array([solutions[:, row].all(1).sum() for row in inputs]); denom = np.minimum(counts, 20)
    found = np.array([min(len({b.tobytes() for b in samples[valid]}), cap)
                      for samples, valid, cap in zip(boards, ok, denom)])
    n20 = ok[np.arange(len(inputs)), energy.argmin(1)]
    dest = OUT / f'search/{arm}/u{update}/s{seed}'; dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / 'records.npz', inputs=inputs, boards=np.packbits(boards, -1), stopped=stopped,
        energy=energy, moves=moves, flop=flop, valid=ok, n20=n20, n1=ok[:, 0], found=found, denom=denom, solution_counts=counts)
    result = dict(status='complete', utc=now(), arm=arm, updates=update, seed=seed, unique_inputs=len(inputs),
        n20_accuracy_pct=100*float(n20.mean()), coverage_pct=100*float(found.sum()/denom.sum()),
        wrong_stops=int((stopped & ~ok).sum()), budget_failures=int((~stopped).sum()),
        mean_moves=float(moves.mean()), search_seconds=seconds, model_path=str(modelpath.relative_to(ROOT)),
        model_sha256=sha(modelpath), records_sha256=sha(dest / 'records.npz'),
        protocol_sha256=sha(SRC / 'protocol.json'), source_sha256=sha(__file__))
    save(dest / 'result.json', result)
    emit('search_complete', **{key: result[key] for key in ['arm', 'updates', 'seed', 'n20_accuracy_pct', 'coverage_pct', 'search_seconds']})


def worker(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', CRAM_DATA=str(UP / 'data'))
    lock = (ROOT / f'artifacts/gpu{gpu}.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    query = lambda *args: subprocess.check_output(args, text=True, timeout=20).strip()
    assert query('hostname') == 'loki'
    device = query('nvidia-smi', f'--id={gpu}', '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader')
    assert device.split(',')[1].strip() not in query('nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader')
    import torch
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    torch.empty(1, device='cuda'); torch.cuda.synchronize()
    p = protocol()
    assert read(OUT / 'validation.json')['status'] == 'passed'
    for group in ('source_sha256', 'data_sha256'):
        for path, digest in p[group].items(): assert sha(ROOT / path) == digest, path
    base = dict(status='running', started_utc=now(), host='loki', gpu=gpu, device=device,
                pid=os.getpid(), protocol_sha256=sha(SRC / 'protocol.json'))
    receipt = OUT / f'worker-g{gpu}.json'
    def progress(**values): save(receipt, dict(base, updated_utc=now(), progress=values))
    progress(stage='starting')
    try:
        arm = ['adam-nodecay', 'sgd-pre'][gpu]
        data = loaded_training()
        if not (OUT / f'parity-{arm}.json').exists():
            model, _ = fit(arm, 4000, data, progress, snapshots=True)
            cache = p['cache_models'][arm]; assert sha(ROOT / cache['path']) == cache['sha256']
            equality = same_model(model, torch.load(ROOT / cache['path'], weights_only=False))
            save(OUT / f'parity-{arm}.json', dict(status='passed', utc=now(), equality=equality,
                cache_model_sha256=cache['sha256'], snapshot_model_sha256=sha(OUT / f'fits/{arm}/u4000/model.pt')))
        while not all((OUT / f'parity-{a}.json').exists() for a in p['arms']): time.sleep(1)
        if gpu == 0 and not (OUT / 'timing.json').exists(): time_fits(data, progress)
        while not (OUT / 'timing.json').exists():
            progress(stage='waiting_for_uncontended_timing'); time.sleep(1)
        del data
        # The primary checkpoint is first; all remaining checkpoints are frozen secondary work.
        jobs = [(a, u, seed) for u in [1000, 250, 500, 2000] for seed in [0, 1, 2] for a in p['arms']]
        while True:
            pending = worked = False
            for a, u, seed in jobs:
                dest = OUT / f'search/{a}/u{u}/s{seed}'; dest.mkdir(parents=True, exist_ok=True)
                if (dest / 'result.json').exists(): continue
                pending = True; claim = (dest / 'job.lock').open('a')
                try: fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: claim.close(); continue
                try:
                    if not (dest / 'result.json').exists(): search(a, u, seed, progress); worked = True
                finally: claim.close()
            if not pending: break
            if not worked: time.sleep(1)
        save(receipt, dict(base, status='complete', completed_utc=now()))
    except BaseException as exc:
        save(receipt, dict(base, status='failed', completed_utc=now(), error=repr(exc))); raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='mode', required=True)
    sp.add_parser('prepare'); sp.add_parser('validate')
    worker_parser = sp.add_parser('worker'); worker_parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare': prepare()
    elif args.mode == 'validate': validate_cpu()
    else: worker(args.gpu)
