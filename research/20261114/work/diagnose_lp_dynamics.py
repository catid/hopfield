"""Post-outcome H2y fixed-state probes; never extend or replace primary runs."""
import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import study as S
import selective_fallback as F
from lp_transfer import PP, DEST, ARMS, protocol

D = DEST / 'dynamics'


def load_job(job):
    run = DEST / 'runs' / job['id']
    receipt = json.loads((run / 'result.json').read_text())
    assert S.sha(run / 'outputs.npz') == receipt['outputs_sha256']
    with np.load(run / 'outputs.npz') as x:
        output = {k: x[k] for k in ['final_logits', 'couplings', 'predictions']}
    with np.load(DEST / f"shard{job['shard']:02d}.npz") as x:
        tokens, indices = x['tokens'], x['indices']
    return tokens, indices, output, receipt


def run(gpu):
    lock, inventory = F.acquire(gpu)
    torch, _, _ = S.imports()
    p = protocol()
    started = time.perf_counter()
    D.mkdir(exist_ok=True)
    target = D / f'gpu{gpu}.json'
    if target.exists():
        raise FileExistsError(target)
    models = {}
    for seed in range(3):
        models[seed] = []
        for arm in ARMS:
            model = S.model_for(seed, 120, 'original', 'cuda')
            model.load_state_dict(torch.load(S.ROOT / p['models'][str(seed)][arm]['path'], map_location='cuda', weights_only=True))
            models[seed].append(model)
    jobs = [j for j in p['jobs'] if j['gpu'] == gpu]
    rows, evidence, representatives = [], [], []
    load_wait = probe_seconds = 0.
    # One archive prefetch overlaps decompression with useful device probes.
    with ThreadPoolExecutor(max_workers=1) as pool, torch.no_grad():
        future = pool.submit(load_job, jobs[0])
        for ji, job in enumerate(jobs):
            before = time.perf_counter()
            tokens, indices, out, receipt = future.result()
            load_wait += time.perf_counter() - before
            if ji + 1 < len(jobs):
                future = pool.submit(load_job, jobs[ji + 1])
            for lo in range(0, 256, 64):
                torch.cuda.synchronize()
                before = time.perf_counter()
                t = torch.tensor(tokens[lo:lo+64].astype(np.int64), device='cuda')
                a = torch.tensor(out['couplings'][lo:lo+64], device='cuda')
                free = t == 1
                def maxabs(x):
                    return x.abs().masked_fill(~free, 0).flatten(1).amax(1)
                for v, (arm, model) in enumerate(zip(ARMS, models[job['seed']])):
                    u0 = torch.tensor(out['final_logits'][v, lo:lo+64], device='cuda')
                    assert np.array_equal((u0 > 0).cpu().numpy(), out['predictions'][v, lo:lo+64])
                    p0 = torch.sigmoid(u0)
                    f0 = model.field(t, p0, a)
                    u1 = u0 + .5 * (f0 - u0)
                    p1 = torch.sigmoid(u1)
                    f1 = model.field(t, p1, a)
                    u2 = u1 + .5 * (f1 - u1)
                    p2 = torch.sigmoid(u2)
                    f2 = model.field(t, p2, a)
                    binary = (u0 > 0).to(a.dtype)
                    fb = model.field(t, binary, a)
                    metric = {
                        'residual_u0_inf': maxabs(f0-u0),
                        'residual_u1_inf': maxabs(f1-u1),
                        'residual_u2_inf': maxabs(f2-u2),
                        'one_step_displacement_inf': maxabs(u1-u0),
                        'two_step_return_error_inf': maxabs(u2-u0),
                        'second_step_displacement_inf': maxabs(u2-u1),
                        'rounded_field_residual_inf': maxabs(fb-u0),
                        'rounding_field_change_inf': maxabs(fb-f0),
                        'one_step_changed_rounded_free': (((u1>0)!=(u0>0)) & free).sum((1,2)),
                        'two_step_changed_rounded_free': (((u2>0)!=(u0>0)) & free).sum((1,2)),
                        'fractional_05_95_free': ((p0>.05)&(p0<.95)&free).sum((1,2)),
                        'near_half_45_55_free': ((p0>.45)&(p0<.55)&free).sum((1,2)),
                        'free_cells': free.sum((1,2)),
                        'selected_free_cells': ((u0>0)&free).sum((1,2)),
                    }
                    for k, u in enumerate([u0, u1, u2]):
                        delta = model.flip_delta(t, (u>0).to(a.dtype), a)
                        metric[f'minimum_binary_flip_u{k}'] = delta.masked_fill(~free, float('inf')).flatten(1).amin(1)
                        metric[f'negative_binary_flips_u{k}'] = ((delta < -1e-6)&free).sum((1,2))
                    values = {k:x.cpu().numpy() for k,x in metric.items()}
                    assert all(np.isfinite(x).all() for x in values.values())
                    for i, index in enumerate(indices[lo:lo+64]):
                        rows.append({'input_index':int(index), 'model_seed':job['seed'], 'arm':arm,
                                     **{k:x[i].item() for k,x in values.items()}})
                    # Deterministic first input, first shard, all models and arms.
                    if job['shard'] == 0 and lo == 0:
                        name = f"representative-s{job['seed']}-{arm}.npz"
                        arrays = {k:x[:1].cpu().numpy() for k,x in {'tokens':t,'couplings':a,'u0':u0,'u1':u1,'u2':u2,'p0':p0,'p1':p1,'p2':p2,'field0':f0,'field1':f1,'field2':f2,'binary_field0':fb}.items()}
                        arrays.update(theta=np.concatenate([model.T.cpu().numpy().ravel(),model.b.cpu().numpy(),model.kappa.cpu().numpy()]), mu_offsets=np.array(model.mu_offsets), aoffs=np.array(model.aoffs))
                        np.savez_compressed(D/name, **arrays)
                        representatives.append({'file':name,'sha256':S.sha(D/name),'input_index':0,'model_seed':job['seed'],'arm':arm})
                torch.cuda.synchronize()
                probe_seconds += time.perf_counter() - before
            evidence.append({'job':job['id'],'source_outputs_sha256':receipt['outputs_sha256']})
            S.emit('lp_saved_dynamics_job',gpu=gpu,job=job['id'],model_input_states=len(rows),seconds=time.perf_counter()-started)
    csvpath = D/f'per-state-gpu{gpu}.csv'
    with csvpath.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    S.save(target,{'status':'complete','utc':S.utc(),'gpu':gpu,'inventory':inventory,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'source_states':len(rows),'map_probes':2*len(rows),'new_primary_trajectories':0,'seconds':time.perf_counter()-started,'archive_wait_seconds':load_wait,'probe_and_transfer_seconds':probe_seconds,'per_state_csv_sha256':S.sha(csvpath),'representatives':representatives,'source_output_evidence':evidence,'scope':'Post-outcome descriptive native one/two-map probes at retained final logits; no stop-mask. No continuation credit, primary changes, cap extension, rescue, or new independent inputs.'})


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[0,1],required=True)
    run(parser.parse_args().gpu)
