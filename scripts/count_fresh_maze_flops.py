"""Rebuild paper-convention FLOPs from single-pass saved test step counts.

Measure operator costs afresh on training mazes, using the official FullCounter.
No inference or fitting is repeated on the test split.
"""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / 'upstream'
OUT = ROOT / 'artifacts/fresh_maze'
sys.path[:0] = [str(UP), str(UP / 'experiments'), str(UP / 'baselines/flops')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CRAM_DATA'] = str(UP / 'data')
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import cram_kernels_extra as C
import maze_unique as U
import maze_hard_latent as H
from cram import maze_unique as M
from cram import maze_h3 as M3


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, d):
    p.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')


def unique(seed):
    d = OUT / 'maze_unique' / f'seed{seed}'
    model = U.make_model(U.Train(**json.loads((d / 'config.json').read_text())))
    model.load_state_dict(torch.load(d / 'model.pt', map_location='cpu', weights_only=True))
    model.eval()
    data = M.load('train')
    rows = []
    for i in range(6):
        x, y, fr = M.to_device(data, [i], torch.device('cpu'))
        h = {}
        with torch.no_grad():
            count = C._count(lambda: h.setdefault('r', C.gibbs_one(model, x, y, fr)))
        rows.append(dict(maze=i, m=int(h['r']['moves'][0]), count=count))
    fit = C.linear(rows)
    assert fit['total']['max_abs_residual'] == 0
    saved = json.loads((d / 'test.json').read_text())
    steps = np.asarray(saved['per_maze']['steps'])
    old = M.flop_per_step(model) + steps * (4 * 8 + 2 * 900)
    assert abs(old.mean() - saved['mean_flop']) < 1e-8
    per = fit['total']['setup'] + fit['total']['per'] * steps
    return dict(task='maze_unique', seed=seed, n=len(per), legacy_flops_mean=float(old.mean()),
                fc2_flops_mean=float(per.mean()), fc2_flops_p99=float(np.quantile(per, .99)),
                fit=fit, training_count_rows=rows, source_test_sha256=digest(d / 'test.json'),
                source_model_sha256=digest(d / 'model.pt'), formula='setup + per_move * saved_test_moves')


def hard(seed):
    d = OUT / 'maze_hard' / f'seed{seed}'
    model = H.build(H.Recipe(**json.loads((d / 'config.json').read_text())))
    model.load_state_dict(torch.load(d / 'model.pt', map_location='cpu', weights_only=True))
    model.eval()
    tok, _ = M3.load('train')
    zrows, drows, att = [], [], []
    for i in range(8):
        T = torch.as_tensor(tok[i:i + 1])
        h = {}
        with torch.no_grad():
            cz = C._count(lambda: h.setdefault('z', model.settle(T)))
            psi, ps = h['z']
            ca = C._count(lambda: h.setdefault('a', torch.softmax(model.att_scores(T, psi), 1)))
            cd = C._count(lambda: h.setdefault('d', M3.descend(model, T, h['a'], steps=2048, eta=.2)))
        zrows.append(dict(maze=i, m=int(ps[0]), count=cz))
        drows.append(dict(maze=i, m=int(h['d'][1][0]), count=cd))
        att.append(ca)
    fz, fd = C.linear(zrows), C.linear(drows)
    assert all(f['total']['max_abs_residual'] == 0 for f in (fz, fd))
    assert len({a['total'] for a in att}) == 1
    saved = json.loads((d / 'test.json').read_text())
    fl = np.asarray(saved['per_maze']['flop'], dtype=np.float64)
    Fp, Fa, Ff = M3.flop_psi_step(model), M3.flop_attention(model), M3.flop_field(model)
    ps, used = C._decode_maze_hard(fl, Fp, Fa, Ff).T
    actual_used = np.load(d / 'test_predictions.npz')['steps']
    assert np.array_equal(used, actual_used)
    per = fz['total']['setup'] + fz['total']['per'] * ps + att[0]['total'] + fd['total']['setup'] + fd['total']['per'] * used
    return dict(task='maze_hard', seed=seed, n=len(per), legacy_flops_mean=float(fl.mean()),
                fc2_flops_mean=float(per.mean()), fc2_flops_p99=float(np.quantile(per, .99)),
                latent_fit=fz, descent_fit=fd, attention=att[0],
                training_latent_count_rows=zrows, training_descent_count_rows=drows,
                decoded_test_descent_steps_match_saved_predictions=True,
                source_test_sha256=digest(d / 'test.json'),source_model_sha256=digest(d / 'model.pt'),
                formula='latent_setup + latent_per * saved_psi_steps + attention + descent_setup + descent_per * saved_descent_steps')


if __name__ == '__main__':
    begin=time.monotonic()
    records=[]
    for task, fn in [('maze_unique', unique), ('maze_hard', hard)]:
        for seed in range(3):
            p=OUT / task / f'seed{seed}' / 'fc2_flops.json'
            if p.exists():
                rec=json.loads(p.read_text())
            else:
                rec=fn(seed)
                rec.update(convention=C.CONVENTION, counted_on='first training mazes, CPU, batch1; extrapolated from test steps already saved',
                           torch=torch.__version__, hostname=platform.node(), script_sha256=digest(__file__),
                           counter_sha256={str(UP / 'baselines/flops' / n):digest(UP / 'baselines/flops' / n) for n in ['fullcount.py','cram_kernels_extra.py','cram_kernels.py']})
                write(p,rec)
            records.append({k:rec[k] for k in ['task','seed','n','legacy_flops_mean','fc2_flops_mean','fc2_flops_p99']})
            print(json.dumps(records[-1]),flush=True)
    write(OUT/'fc2_summary.json',dict(records=records,elapsed_s=time.monotonic()-begin))
