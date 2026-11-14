#!/usr/bin/env python3
"""Bounded, isolated continuation of copied DE checkpoints on an idle GPU.

Uses unchanged frozen-source preparation, GeneralSearch and step_check. Never
reads or writes a production output directory after the snapshot was copied.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path('/home/catid/hopfield')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def gpu(index):
    raw = subprocess.check_output(['nvidia-smi', f'--id={index}',
        '--query-gpu=index,uuid,name,power.limit,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used',
        '--format=csv,noheader'], text=True).strip()
    uuid = raw.split(',')[1].strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    return {'raw': raw, 'uuid': uuid, 'owners': [line for line in apps.splitlines() if uuid in line]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lane', required=True); p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--checks', type=int, default=16); p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--seeds', type=int, nargs='+', default=[4, 5])
    p.add_argument('--source', type=Path, default=ROOT / 'dataeff_source')
    args = p.parse_args(); host, index = args.lane.split(':'); index = int(index)
    assert host == socket.gethostname().split('.')[0].lower()
    assert 1 <= args.checks <= 16 and 1 <= args.repeats <= 3
    assert args.seeds and len(set(args.seeds)) == len(args.seeds) and set(args.seeds) <= {4, 5}
    bundle = args.bundle.resolve()
    assert bundle.is_relative_to(ROOT / 'artifacts/checkpoint_resume_benchmark')
    out = bundle / 'runs' / args.lane.replace(':', '-'); out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'result.json').exists(), 'Do not overwrite a measured benchmark'
    locks = []
    for lockpath in (ROOT / f'artifacts/gpu{index}.lock', ROOT / f'artifacts/objectives/gpu-{index}.lock'):
        lockpath.parent.mkdir(parents=True, exist_ok=True)
        lock = lockpath.open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(lock)
    before = gpu(index); assert not before['owners'], before
    manifest = json.loads((bundle / 'inputs.json').read_text())
    for rel, expected in manifest['source_sha256'].items(): assert sha(args.source / rel) == expected, rel
    for item in manifest['files']:
        if item['seed'] not in args.seeds:
            continue
        snapshot = bundle / 'inputs' / f'seed{item["seed"]}' / Path(item['snapshot']).name
        assert sha(snapshot) == item['sha256'], snapshot
    os.environ.update(CUDA_VISIBLE_DEVICES=str(index), CRAM_DATA=manifest['data'], V1_DATA=manifest['data'],
                      DE_VARIANTS='99', CRAM_E_ROOT='DE2-v99-recreated42', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    sys.path[:0] = [str(args.source), str(args.source / 'experiments')]
    import numpy as np
    import torch
    import triton
    import data_efficiency as DE
    CF, GS, SS = DE.CF, DE.GS, DE.E1.SS
    torch.set_num_threads(4)
    def tensor_digest(t):
        a = t.detach().cpu().contiguous()
        return {'dtype': str(a.dtype), 'shape': list(a.shape), 'sha256': hashlib.sha256(a.numpy().tobytes()).hexdigest()}
    def equal(a, b):
        assert set(a) == set(b)
        return {k: (bool(torch.equal(a[k], b[k])) and tensor_digest(a[k]) == tensor_digest(b[k])
                    if isinstance(a[k], torch.Tensor) else a[k] == b[k])
                for k in a if k != 'elapsed'}
    sol, free, idx = SS.load_puzzles('full-all', 10000, 10000)
    data_path = Path(manifest['data']) / 'full-test/full-all.npz'
    assert sha(data_path) == manifest['data_sha256'], 'Frozen test dataset changed'
    result = {'status': 'running', 'lane': args.lane, 'pid': os.getpid(), 'script_sha256': sha(__file__),
              'source_commit': manifest['source_commit'], 'source_sha256': manifest['source_sha256'],
              'bundle_manifest_sha256': sha(bundle / 'inputs.json'), 'data_sha256': sha(data_path),
              'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'triton': triton.__version__, 'gpu_before': before, 'checks': args.checks, 'repeats': args.repeats,
              'puzzle_rows': [int(idx[0]), int(idx[-1])], 'n': len(idx), 'seeds': [],
              'production_output_written': False, 'fit_or_hyperparameter_change': False}
    write(out / 'result.json', result)
    for seed in args.seeds:
        directory = bundle / 'inputs' / f'seed{seed}'
        fit = json.loads((directory / 'fit.json').read_text())
        model = torch.load(directory / 'model.pt', map_location='cuda', weights_only=True)
        train = np.load(directory / 'train.npz')
        m = CF.prepare(model['W'], model['B'], torch.as_tensor(train['solutions'].astype(np.int64), device='cuda'))
        assert m['tau_train'] == fit['tau'] and m['hbar'] == fit['hbar']
        hi, lo, theta = DE.E1.REL; unit = m['hbar']; period = CF.FROZEN['anneal'][2]
        ss = GS.GeneralSearch(m['S'], model['B'], sol, free, temp=hi*unit, theta=theta*unit,
            focused=True, C=CF.FROZEN['C'], tau=m['tau_train'], tol=CF.FROZEN['tol'], ebar=m['ebar'],
            cap=CF.FROZEN['cap'], check=CF.FROZEN['check'], seed=seed, anneal=(hi*unit, lo*unit, period))
        signature = {'k': 1, 'seed': seed, 'model_sha256': fit['model_sha256'], 'data': 'test10k',
            'n': len(idx), 'g': m['g'], 'ebar': m['ebar'], 'hbar': unit, 'tau': m['tau_train'],
            'D': ss.D, 'fl_move': ss.fl_move, 'rel': list(DE.E1.REL),
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in CF.FROZEN.items() if k not in ('anneal', 'theta')}}
        checkpoint = torch.load(directory / 'ckpt.pt', map_location='cpu', weights_only=False)
        assert checkpoint['signature'] == signature, 'Native resume signature mismatch'
        initial = checkpoint['state']; ss.load_state_dict(initial)
        assert all(equal(initial, ss.state_dict()).values()), 'Initial load is not exact'
        # Compile every shape required by an actual check, then restore the same
        # checkpoint before timing. Compilation and restoration are not timed.
        torch.cuda.synchronize(); warm_start = time.perf_counter()
        with torch.no_grad(): ss.step_check()
        torch.cuda.synchronize(); warm_s = time.perf_counter() - warm_start
        row = {'seed': seed, 'signature': signature, 'initial_t': initial['t'],
               'initial_active': int(initial['act'].numel()), 'compile_warm_check_s': warm_s,
               'checkpoint_sha256': sha(directory / 'ckpt.pt'), 'trials': [],
               'static_tensors': {k: tensor_digest(getattr(ss, k)) for k in ('Sg', 'Bc', 'nbr', 'R')},
               'static_scalars': {k: getattr(ss, k) for k in ('D', 'thr', 'temp', 'theta', 'tau', 'tol',
                    'ebar', 'fl_move', 'T_cap', 'C', 'check', 'seed', 'anneal', 'restart', 'growth')}}
        reference = None
        for repeat in range(args.repeats):
            ss.load_state_dict(initial); torch.cuda.synchronize(); times = []; start = time.perf_counter()
            with torch.no_grad():
                for step in range(args.checks):
                    assert not ss.done()
                    t = time.perf_counter(); ss.step_check(); torch.cuda.synchronize()
                    times.append(time.perf_counter() - t)
            total = time.perf_counter() - start
            ss.elapsed = initial['elapsed'] + total
            state = ss.state_dict()
            agreement = equal(reference, state) if reference is not None else None
            if agreement is not None: assert all(agreement.values()), agreement
            else:
                reference = state
                torch.save({'signature': signature, 'state': state}, out / f'seed{seed}-after.pt')
            row['trials'].append({'repeat': repeat, 'wall_s': total, 'check_seconds': times,
                'moves_per_second': args.checks*ss.check/total, 'final_t': ss.t, 'final_active': len(ss.act),
                'same_host_repeat_exact': agreement, 'gpu': gpu(index)})
        row['median_seconds_per_check'] = statistics.median(t['wall_s']/args.checks for t in row['trials'])
        row['final_state_tensors'] = {k: tensor_digest(v) for k, v in reference.items() if isinstance(v, torch.Tensor)}
        row['final_state_scalars_except_elapsed'] = {k: v for k, v in reference.items() if not isinstance(v, torch.Tensor) and k != 'elapsed'}
        result['seeds'].append(row); write(out / 'result.json', result)
        del reference, checkpoint, ss, model, m
        torch.cuda.empty_cache()
    result.update(status='complete', gpu_after=gpu(index))
    write(out / 'result.json', result)
    print(json.dumps({'status': 'complete', 'lane': args.lane,
                      'seconds_per_check': [r['median_seconds_per_check'] for r in result['seeds']]}), flush=True)


if __name__ == '__main__': main()
