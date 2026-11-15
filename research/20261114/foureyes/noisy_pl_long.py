"""H4s: matched noisy-source likelihood fits, followed by frozen graph selection.

The training producer has no Sudoku geometry. Truth enters the separate auditor.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
UP = ROOT / 'upstream'
OUT = ROOT / 'artifacts/research/20261114/foureyes/noisy_pl_long'
PROTOCOL = HERE / 'noisy_pl_long.json'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def dependencies(gpu=None):
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    sys.path[:0] = [str(UP), str(UP / 'experiments'), str(HERE.parent / 'ripper')]
    import numpy as np
    import torch
    import sudoku as CF
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    return np, torch, CF


def fit_adam(torch, CF, sol, free, draws, config):
    W = torch.zeros(3240, 9, 9, device=sol.device, requires_grad=True)
    B = torch.zeros(81, 9, device=sol.device, requires_grad=True)
    optimizer = torch.optim.AdamW([W, B], lr=config['lr'], weight_decay=config['weight_decay'])
    pair = CF.pair_index(sol.device)
    history = []
    start = time.monotonic()
    for k, draw in enumerate(draws):
        optimizer.param_groups[0]['lr'] = config['lr'] * .5 * (1 + math.cos(math.pi * k / len(draws)))
        relabeled = draw.to(sol.device).argsort(1).gather(1, sol)
        optimizer.zero_grad(set_to_none=True)
        loss = CF.pl_loss(CF.full_S(W, pair), B, relabeled, free)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    if sol.is_cuda:
        torch.cuda.synchronize()
    return W.detach(), B.detach(), time.monotonic() - start, history


def transform(W, torch):
    # Equivalent digit-orbit canonical coefficient, evaluated directly in FP64.
    # This avoids the redundant FP32 repeat-and-average in the earlier wrapper.
    eye = torch.eye(9, dtype=torch.bool, device=W.device)
    J = (W.double()[:, eye].mean(1) - W.double()[:, ~eye].mean(1)).clamp_min(0)
    positive = J[J > 0].sort().values
    if len(positive) < 2 or not bool(torch.isfinite(J).all()):
        return None, J, dict(eligible=False, reason='insufficient finite positive coefficients')
    gaps = positive.diff()
    ix = int(gaps.argmax())
    if not float(gaps[ix]) > 0:
        return None, J, dict(eligible=False, reason='no positive coefficient gap')
    threshold = float((positive[ix] + positive[ix + 1]) / 2)
    support = J > threshold
    return support, J, dict(eligible=True, gap_index=ix, gap=float(gaps[ix]), threshold=threshold,
                            retained=int(support.sum()), minimum_threshold_distance=float((J-threshold).abs().min()))


def prepare(p):
    np, torch, CF = dependencies()
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    cache = ROOT / 'artifacts/research/sudoku_source_cache' / p['source_sha256_value']
    cm = json.loads((cache / 'manifest.json').read_text())
    assert sha(ROOT / p['source']) == p['source_sha256_value'] == cm['source_sha256']
    for name, digest in cm['files'].items():
        assert sha(cache / name) == digest
    clean = np.load(cache / 'solutions.npy', mmap_mode='r')
    given = np.load(cache / 'given.npy', mmap_mode='r')
    excluded = np.zeros(len(clean), bool)
    for spec in p['exclusions']:
        manifest = ROOT / spec['path']
        assert sha(manifest) == spec['sha256']
        for entry in json.loads(manifest.read_text())['inputs']:
            f = ROOT / entry['completion_path']
            assert sha(f) == entry['completion_sha256']
            excluded[np.load(f)['source_row']] = True
    eligible = np.flatnonzero(~excluded)
    entries = []
    for bank in range(p['banks']):
        rows = np.random.default_rng(p['bank_rng_base'] + bank).choice(eligible, p['n'], replace=False)
        s = np.asarray(clean[rows]); free = ~given[rows]
        rng = np.random.default_rng(p['noise_rng_base'] + bank)
        cells = np.stack([rng.choice(np.flatnonzero(f), 9, replace=False) for f in free])
        values = (np.take_along_axis(s, cells, 1) + rng.integers(1, 9, cells.shape)) % 9
        noisy = s.copy(); np.put_along_axis(noisy, cells, values, 1)
        file = OUT / 'inputs' / f'b{bank:02d}.npz'; file.parent.mkdir(exist_ok=True)
        np.savez_compressed(file, source_row=rows, solutions=s, free=free, noisy=noisy, cells=cells, values=values)
        entries.append(dict(bank=bank, path=str(file.relative_to(ROOT)), sha256=sha(file)))
    write(OUT / 'inputs.json', dict(protocol_sha256=sha(PROTOCOL), source_sha256=sha(__file__),
                                  banks=entries, excluded_rows=int(excluded.sum()), eligible_rows=len(eligible)))
    write(OUT / 'fit_queue.json', [dict(bank=b, status='pending') for b in range(p['banks'])])


def claim(gpu, bank=None, status=None):
    with (OUT / 'fit_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = json.loads((OUT / 'fit_queue.json').read_text())
        if bank is None:
            job = next((j for j in jobs if j['status'] == 'pending'), None)
            if job is not None:
                job.update(status='running', gpu=gpu, pid=os.getpid(), start=time.time())
        else:
            job = next(j for j in jobs if j['bank'] == bank)
            job.update(status=status, finish=time.time())
        write(OUT / 'fit_queue.json', jobs)
        return job


def worker(p, gpu):
    assert platform.node() == 'foureyes'
    lock = (ROOT / f'artifacts/gpu{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    assert uuid not in owners
    np, torch, CF = dependencies(gpu)
    from noise_chernoff import fit_counts
    assert json.loads((OUT / 'gate.json').read_text())['status'] == 'passed'
    manifest = json.loads((OUT / 'inputs.json').read_text())
    assert manifest['protocol_sha256'] == sha(PROTOCOL) and manifest['source_sha256'] == sha(__file__)
    while (job := claim(gpu)) is not None:
        bank = job['bank']; start = time.monotonic()
        try:
            entry = manifest['banks'][bank]; path = ROOT / entry['path']; assert sha(path) == entry['sha256']
            z = np.load(path); s = torch.as_tensor(z['noisy'].astype(np.int64), device='cuda')
            free = torch.as_tensor(z['free'], device='cuda')
            draws = torch.rand((p['updates'], p['n'], 9), generator=torch.Generator().manual_seed(p['augmentation_rng_base'] + bank))
            dest = OUT / 'fits' / f'b{bank:02d}'; dest.mkdir(exist_ok=False, parents=True)
            results = {}; arrays = {}; records = {}
            for arm in ['adamw', 'paper_pl', 'long_pl']:
                if arm == 'adamw':
                    W, B, seconds, history = fit_adam(torch, CF, s, free, draws[:32], p['adamw'])
                else:
                    W, B, seconds = CF.train(CF.FullConfig(**p['long_pl' if arm=='long_pl' else 'paper_pl']), s, free, draws=draws)
                    history = []
                assert torch.isfinite(W).all() and torch.isfinite(B).all()
                support, coefficients, meta = transform(W, torch)
                torch.save(dict(W=W.cpu(), B=B.cpu()), dest / f'{arm}.pt')
                arrays[f'{arm}_coefficients'] = coefficients.cpu().numpy()
                if support is not None:
                    arrays[f'{arm}_support'] = support.cpu().numpy()
                records[arm] = dict(**meta, loop_s=seconds, loss_history=history)
            ii, jj = np.triu_indices(81, 1)
            counts = (z['noisy'][:, ii] == z['noisy'][:, jj]).sum(0)
            distinct = np.unique(counts); choices = np.flatnonzero(distinct[1:] <= np.median(counts))
            ix = int(choices[np.argmax(np.diff(distinct)[choices])]) if len(choices) else None
            threshold = int((distinct[ix] + distinct[ix + 1]) // 2) if ix is not None else 0
            arrays.update(counts=counts, restricted_support=counts <= threshold)
            support, threshold9, mean = fit_counts(counts, z['free'], 9, .01)
            arrays.update(chernoff9_support=support, chernoff9_thresholds=threshold9)
            records.update(restricted=dict(threshold=threshold, eligible=True), chernoff9=dict(eligible=True))
            np.savez_compressed(dest / 'models.npz', **arrays)
            write(dest / 'receipt.json', dict(bank=bank, gpu=gpu, pid=os.getpid(), protocol_sha256=sha(PROTOCOL),
                  input_sha256=entry['sha256'], augmentation_seed=p['augmentation_rng_base'] + bank,
                  arms=records, wall_s=time.monotonic()-start, torch=torch.__version__, cuda=torch.version.cuda,
                  output_sha256={f.name:sha(f) for f in dest.iterdir() if f.is_file()}))
            assert time.monotonic()-start < p['max_fit_bank_s']
            claim(gpu, bank, 'complete')
            print(json.dumps(dict(bank=bank, status='complete', wall_s=time.monotonic()-start)), flush=True)
        except BaseException:
            claim(gpu, bank, 'failed')
            raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--gpu', type=int)
    args = parser.parse_args(); p = json.loads(PROTOCOL.read_text())
    for path, digest in p['source_sha256'].items():
        assert sha(ROOT / path) == digest, path
    if args.prepare:
        prepare(p)
    else:
        worker(p, args.gpu)


if __name__ == '__main__':
    main()
