"""Independent end-to-end Maze-Unique fit and complete test evaluation on CPU.

Only hopfield_cram implements the model, loss and search. Pinned dataset arrays
are decoded directly; the released checkpoint is used only for a post-fit
parameter comparison. Exact NumPy minibatch order follows the paper recipe.
CPU Gibbs RNG and batching are explicitly distinct from the CUDA trajectories.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hopfield_cram import MazeEnergy, pseudolikelihood, focused_gibbs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(1)
    out = ROOT / 'artifacts/reference_maze_unique' / f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise FileExistsError('Final result already exists; no test rerun.')
    dataset = ROOT / 'upstream/data/eqr/maze-30x30-unique-1k'
    expected = {str(Path(r['path']).relative_to('eqr/maze-30x30-unique-1k')): r['sha256']
                for r in json.loads((ROOT / 'artifacts/deployment/data_manifest.json').read_text())
                if r['path'].startswith('eqr/maze-30x30-unique-1k/')}
    def load(split):
        arrays = []
        for name in ('inputs', 'labels'):
            path = dataset / split / f'all__{name}.npy'
            assert sha(path) == expected[f'{split}/all__{name}.npy']
            arrays.append(np.load(path).astype(np.int64))
        x, y = arrays
        return torch.from_numpy(x - 1), torch.from_numpy(((y == 5) | (x == 3) | (x == 4)).astype(np.int64))
    tokens, labels = load('train')
    assert tokens.shape == labels.shape == (1000, 900)
    free = tokens == 1
    rng = np.random.default_rng(args.seed)
    schedule = []
    while len(schedule) < 20:
        order = rng.permutation(len(tokens))
        schedule.extend(order[lo:lo + 64] for lo in range(0, len(tokens) - 63, 64))
    schedule = np.stack(schedule[:20])
    np.save(out / 'training_indices.npy', schedule)
    model = MazeEnergy()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.5)
    losses = []
    begin = time.perf_counter()
    for update, idx in enumerate(schedule):
        optimizer.zero_grad(set_to_none=True)
        loss = pseudolikelihood(model, labels[idx], free[idx], free[idx])
        assert bool(torch.isfinite(loss))
        loss.backward()
        optimizer.param_groups[0]['lr'] = 0.5 * (1 + math.cos(math.pi * update / 20))
        optimizer.step()
        losses.append(float(loss.detach()))
    train_s = time.perf_counter() - begin
    torch.save(model.state_dict(), out / 'model.pt')
    comparison_path = ROOT / f'artifacts/fresh_maze/maze_unique/seed{args.seed}/model.pt'
    previous = torch.load(comparison_path, map_location='cpu', weights_only=True)
    differences = {'tables': float((model.tables.detach() - previous['T']).abs().max()),
                   'unary_cost': float((model.unary_cost.detach() + previous['b']).abs().max())}
    tokens, labels = load('test')
    assert tokens.shape == labels.shape == (1000, 900)
    predictions, moves, stopped = [], [], []
    begin = time.perf_counter()
    for lo in range(0, len(tokens), args.batch_size):
        tok = tokens[lo:lo + args.batch_size]
        result = focused_gibbs(model, (tok > 0).long(), tok == 1, tok == 1,
                               max_moves=20000, temperature=0.0, stop='fixed', seed=args.seed + lo)
        predictions.append(result.boards.numpy())
        moves.append(result.moves.numpy())
        stopped.append(result.stopped.numpy())
        print(json.dumps({'seed': args.seed, 'completed': lo + len(tok), 'total': len(tokens),
                          'elapsed_s': time.perf_counter() - begin}), flush=True)
    search_s = time.perf_counter() - begin
    predictions, moves, stopped = map(np.concatenate, (predictions, moves, stopped))
    exact = (predictions == labels.numpy()).all(1)
    np.savez_compressed(out / 'predictions.npz', predictions=predictions, moves=moves, stopped=stopped)
    record = {'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'seed': args.seed,
              'n': 1000, 'solved': int(exact.sum()), 'wrong_stops': int((stopped & ~exact).sum()),
              'budget_failures': int((~stopped).sum()), 'wrong_indices': np.flatnonzero(~exact).tolist(),
              'train_loop_s': train_s, 'search_s': search_s, 'losses': losses,
              'max_weight_difference_from_fresh_optimized_fit': differences,
              'training_schedule': '20 NumPy default_rng(seed) permutation minibatches,64rows,drop-last; zero start,SGD momentum0.5,cosineLR1,noL1',
              'search': {'device': 'cpu', 'threads': 1, 'dtype': 'float32', 'batch_size': args.batch_size,
                         'seed_rule': 'seed + batch_start; CPU RNG differs from optimized CUDA search',
                         'temperature': 0, 'max_moves': 20000, 'stop': 'fixed', 'initial': 'all open cells on'},
              'model_sha256': sha(out / 'model.pt'), 'prediction_sha256': sha(out / 'predictions.npz'),
              'comparison_checkpoint_sha256': sha(comparison_path), 'data_sha256': expected,
              'source_sha256': {str(f.relative_to(ROOT)): sha(f) for f in (ROOT / 'src/hopfield_cram').glob('*.py')},
              'script_sha256': sha(__file__), 'torch': torch.__version__, 'python': sys.version}
    (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
