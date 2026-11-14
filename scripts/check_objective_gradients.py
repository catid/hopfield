#!/usr/bin/env python3
"""CPU sanity check of D.2 objective gradients on declared training batches.

This is a new local diagnostic. The release does not specify the model/batch
behind its reported 0.99999 cosine or its beta=0.5 divergence observation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CRAM_DATA'] = str(args.data)
    os.environ['FRM_DATA'] = str(args.data / 'frm/split')
    sys.path.insert(0, str(args.source)); sys.path.insert(0, str(args.source / 'experiments'))
    import training_methods as A
    torch.set_num_threads(8)
    start = time.monotonic()
    task = A.Sudoku(0, torch.device('cpu'), n_train=32, aug=False)
    task.setup(); ctx = task.batch(torch.Generator().manual_seed(0))
    rows = []
    for state in ('zero', 'seeded_small_weights'):
        with torch.no_grad():
            gen = torch.Generator().manual_seed(20261113)
            for p in task.params():
                p.zero_() if state == 'zero' else p.copy_(1e-3 * torch.randn(p.shape, generator=gen))
        gradients = {}
        for method, steps in (('bptt', 10), ('bptt', 30), ('imp', 30)):
            task.begin()
            loss = A.loss_fn(method, steps, task, ctx, torch.Generator().manual_seed(0), {'_reg': 0.})
            gradients[f'{method}{steps}'] = torch.cat([x.flatten() for x in torch.autograd.grad(loss, task.params())])
        for beta in (.01, .5):
            task.begin()
            zero = torch.zeros((*ctx['y'].shape, task.C))
            with torch.no_grad():
                u0 = A.relax(task, ctx, zero, 30)
                up = A.relax(task, ctx, u0, 30, nudge=beta)
                um = A.relax(task, ctx, u0, 30, nudge=-beta)
                vp = A.clamp(up.softmax(-1), ctx, task.C)
                vm = A.clamp(um.softmax(-1), ctx, task.C)
            loss = (task.energy(vp, ctx) - task.energy(vm, ctx)).sum() / (2 * beta * ctx['free'].sum())
            eqp = torch.cat([x.flatten() for x in torch.autograd.grad(loss, task.params())])
            row = {'state': state, 'beta': beta, 'eqp_steps_free_and_each_nudge': 30,
                   'eqp_gradient_finite': bool(eqp.isfinite().all()), 'eqp_gradient_norm': float(eqp.norm()),
                   'positive_logit_absmax': float(up.abs().max()), 'negative_logit_absmax': float(um.abs().max()),
                   'comparisons': {}}
            for name, reference in gradients.items():
                row['comparisons'][name] = {
                    'cosine': float(torch.nn.functional.cosine_similarity(eqp[None].double(), reference[None].double())),
                    'reference_norm': float(reference.norm()),
                    'relative_norm_error': float((eqp - reference).norm() / reference.norm()),
                    'reference_gradient_finite': bool(reference.isfinite().all())}
            rows.append(row)
    def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = {'status': 'complete', 'kind': 'new_local_sanity_check_not_exact_published_diagnostic',
              'device': 'cpu', 'python': platform.python_version(), 'torch': torch.__version__, 'threads': 8,
              'training_split': 'train-s0', 'n_train': 32, 'subset_rng': 'numpy.default_rng(1000), first32 permutation indices',
              'augmentation': False, 'initialization_states': ['zero', 'Gaussian std0.001 seed20261113'],
              'source_sha256': sha(args.source / 'experiments/training_methods.py'),
              'data_sha256': sha(args.data / 'frm/split/train-s0.npz'), 'rows': rows,
              'wall_seconds': time.monotonic() - start,
              'limitation': 'A single gradient cannot reproduce a training-divergence claim. The published diagnostic model, batch, seed and iteration setting are unspecified.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'wall_seconds': result['wall_seconds'], 'rows': rows}))


if __name__ == '__main__':
    main()
