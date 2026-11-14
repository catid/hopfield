"""Bounded exact-state and throughput checks for optional GPU move sharding.

Reads an isolated snapshot, never a production output directory. Original
source, search checks, refreshes, compaction and budgets remain unchanged.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'dataeff_source'), str(ROOT / 'dataeff_source/experiments')]
import numpy as np
import torch
import triton
import sudoku as CF
from cram import cram_sudoku_general as GS
from cram import cram_sudoku_search as SS
from sharded_sudoku_moves import ShardedMoves


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def state_digests(search):
    out = {}
    for key, value in search.state_dict().items():
        if key == 'elapsed':
            continue
        if isinstance(value, torch.Tensor):
            out[key] = {'dtype': str(value.dtype), 'shape': list(value.shape),
                        'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}
        else:
            out[key] = value
    return out


def compare(expected, actual, label):
    keys = set(expected) | set(actual)
    bad = [key for key in sorted(keys) if expected.get(key) != actual.get(key)]
    assert not bad, f'{label}: bitwise state mismatch in {bad}'
    return {'label': label, 'equal': True, 'state_fields': len(keys)}


def build(snapshot, device):
    fit = json.loads((snapshot / 'fit.json').read_text())
    saved = torch.load(snapshot / 'ckpt.pt', map_location='cpu', weights_only=False)
    assert sha(snapshot / 'model.pt') == fit['model_sha256'] == saved['signature']['model_sha256']
    sd = torch.load(snapshot / 'model.pt', map_location=device, weights_only=True)
    tr = np.load(snapshot / 'train.npz')
    prepared = CF.prepare(sd['W'], sd['B'], torch.as_tensor(tr['solutions'].astype(np.int64), device=device))
    assert prepared['tau_train'] == fit['tau'] and prepared['hbar'] == fit['hbar']
    sol, free, idx = SS.load_puzzles('full-all', 10000, 10000)
    unit = prepared['hbar']
    search = GS.GeneralSearch(prepared['S'], sd['B'], sol, free, temp=.4 * unit, theta=.45 * unit,
        focused=True, C=CF.FROZEN['C'], tau=prepared['tau_train'], tol=CF.FROZEN['tol'],
        ebar=prepared['ebar'], cap=CF.FROZEN['cap'], check=CF.FROZEN['check'],
        seed=fit['seed'], anneal=(.4 * unit, .31 * unit, CF.FROZEN['anneal'][2]), device=device)
    signature = {'k': fit['k'], 'seed': fit['seed'], 'model_sha256': fit['model_sha256'], 'data': 'test10k',
        'n': len(idx), 'g': prepared['g'], 'ebar': prepared['ebar'], 'hbar': unit, 'tau': prepared['tau_train'],
        'D': search.D, 'fl_move': search.fl_move, 'rel': [.4, .31, .45],
        **{k: list(v) if isinstance(v, tuple) else v for k, v in CF.FROZEN.items() if k not in ('anneal', 'theta')}}
    assert signature == saved['signature'], 'Snapshot protocol differs from original search'
    search.load_state_dict(saved['state'])
    return search, saved


@contextlib.contextmanager
def reserve(devices):
    locks = []
    try:
        assert not os.environ.get('CUDA_VISIBLE_DEVICES'), 'Use physical device indices without remapping'
        for device in devices:
            for path in [ROOT / f'artifacts/gpu{device}.lock', ROOT / f'artifacts/objectives/gpu-{device}.lock']:
                path.parent.mkdir(parents=True, exist_ok=True)
                lock = path.open('a')
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locks.append(lock)
            uuid = subprocess.check_output(['nvidia-smi', f'--id={device}', '--query-gpu=uuid',
                                            '--format=csv,noheader'], text=True).strip()
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                            '--format=csv,noheader'], text=True)
            assert not any(uuid in line for line in apps.splitlines()), f'GPU {device} has an owner'
        yield
    finally:
        for lock in reversed(locks):
            lock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--devices', default='0,1,2,3')
    ap.add_argument('--checks', type=int, default=8)
    ap.add_argument('--timing-checks', type=int, default=32)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert args.snapshot.resolve().is_relative_to((ROOT / 'artifacts').resolve())
    assert args.checks >= 2 and args.timing_checks >= 4
    devices = [int(x) for x in args.devices.split(',')]
    assert len(set(devices)) == len(devices)
    assert not args.output.exists(), 'Preserve prior benchmark evidence'
    manifest_path = args.snapshot.parent.parent / 'inputs.json'
    manifest = json.loads(manifest_path.read_text())
    for rel, expected in manifest['source_sha256'].items():
        assert sha(ROOT / 'dataeff_source' / rel) == expected, rel
    for name in ('model.pt', 'fit.json', 'train.npz', 'ckpt.pt'):
        expected = next(row['sha256'] for row in manifest['files']
                        if Path(row['snapshot']).resolve() == (args.snapshot / name).resolve())
        assert sha(args.snapshot / name) == expected, name
    os.environ['CRAM_DATA'] = str(ROOT / 'upstream/data')
    torch.set_num_threads(4)
    out = {'status': 'running', 'host': socket.gethostname(), 'devices': devices,
           'manifest_sha256': sha(manifest_path), 'validated_source_files': len(manifest['source_sha256']),
           'versions': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'triton': triton.__version__},
           'matmul_flags': {'tf32': torch.backends.cuda.matmul.allow_tf32,
                            'float32_precision': torch.get_float32_matmul_precision()},
           'data_path': os.environ['CRAM_DATA'],
           'test_data_sha256': sha(ROOT / 'upstream/data/full-test/full-all.npz'),
           'snapshot_sha256': {name: sha(args.snapshot / name) for name in ('model.pt', 'fit.json', 'train.npz', 'ckpt.pt')},
           'script_sha256': sha(__file__), 'sharding_sha256': sha(ROOT / 'scripts/sharded_sudoku_moves.py'),
           'checks': args.checks, 'timing_checks': args.timing_checks, 'comparisons': [],
           'original_source_sha256': {name: sha(ROOT / 'dataeff_source/cram' / name)
             for name in ('cram_sudoku_general.py', 'cram_sudoku_search.py')}}
    assert out['test_data_sha256'] == manifest['data_sha256']
    out['driver_and_hardware'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,power.limit',
                                                         '--format=csv,noheader'], text=True)
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    previous_hook = sys.excepthook
    def record_failure(kind, error, tb):
        out.update(status='failed', error=''.join(traceback.format_exception(kind, error, tb)),
                   elapsed_s=time.monotonic() - started)
        args.output.write_text(json.dumps(out, indent=2) + '\n')
        previous_hook(kind, error, tb)
    sys.excepthook = record_failure
    with reserve(devices), torch.no_grad():
        search, saved = build(args.snapshot, f'cuda:{devices[0]}')
        original_moves = search._moves
        sharded = ShardedMoves([f'cuda:{d}' for d in devices])
        original_state = saved['state']
        out['starting_move'] = search.t
        out['hardware'] = [torch.cuda.get_device_name(d) for d in devices]
        out['peer_access'] = [[torch.cuda.can_device_access_peer(a, b) if a != b else True for b in devices] for a in devices]

        for fixture in ('original_checkpoint', 'forced_compaction'):
            search._moves = original_moves
            search.load_state_dict(original_state)
            if fixture == 'forced_compaction':
                # Bench-only synthetic stopping pattern exercises reordered
                # global chain indices and uneven shard ranges after compact.
                search.stopped[search.act[::3]] = True
                search._compact()
            initial = search.state_dict()
            original_moves(search.check)
            pre_refresh = state_digests(search)
            search.load_state_dict(initial)
            sharded(search, search.check)
            out['comparisons'].append(compare(pre_refresh, state_digests(search), fixture + '/pre_refresh'))
            search.load_state_dict(initial)
            search._moves = original_moves
            expected = []
            for i in range(args.checks):
                search.step_check()
                expected.append(state_digests(search))
            search.load_state_dict(initial)
            sharded.install(search)
            for i in range(args.checks):
                search.step_check()
                out['comparisons'].append(compare(expected[i], state_digests(search), f'{fixture}/check{i+1}'))
            print(json.dumps({'fixture': fixture, 'status': 'exact', 'checks': args.checks}), flush=True)

        # Cold compilation is excluded by the parity checks above. Alternate
        # order across repeats; replay precisely the same starting state.
        timings = {'original': [], 'sharded': []}
        final = {}
        for order in (('original', 'sharded'), ('sharded', 'original'), ('original', 'sharded')):
            for mode in order:
                search.load_state_dict(original_state)
                if mode == 'original':
                    search._moves = original_moves
                else:
                    sharded.install(search)
                torch.cuda.synchronize(devices[0])
                t0 = time.perf_counter()
                for _ in range(args.timing_checks):
                    search.step_check()
                torch.cuda.synchronize(devices[0])
                timings[mode].append(time.perf_counter() - t0)
                digest = state_digests(search)
                if mode in final:
                    out['comparisons'].append(compare(final[mode], digest, f'{mode}/repeat'))
                final[mode] = digest
                print(json.dumps({'mode': mode, 'seconds': timings[mode][-1]}), flush=True)
        out['comparisons'].append(compare(final['original'], final['sharded'], 'timing_final'))
        out.update(status='validated', exact_state=True, timings_s=timings,
                   speedup=float(np.median(timings['original']) / np.median(timings['sharded'])),
                   final_state_digests=final['sharded'], elapsed_s=time.monotonic()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('status', 'exact_state', 'speedup', 'timings_s', 'elapsed_s')}))


if __name__ == '__main__':
    main()
