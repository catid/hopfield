"""Independent H2bj routing, source parity and direct-factor certificate audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy, flip, LD
from verify_hard_negative_feasibility import valid_shortest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def audit(directory, root):
    start = time.perf_counter()
    p = json.loads((directory/'protocol.json').read_text())
    protocol_hash = sha(directory/'protocol.json')
    source = root/'artifacts/research/20261114/work/loop80_confirmation'
    assert sha(root/'research/20261114/work/certificate-fallback-protocol.json') == protocol_hash
    for field in ['source_sha256', 'source_evidence_sha256']:
        for name, h in p[field].items():
            assert sha(root/name) == h, name
    routed = json.loads((directory/'routing.json').read_text())
    assert routed['status'] == 'passed' and routed['protocol_sha256'] == protocol_hash
    assert routed['model_inputs'] == 12288 and len(routed['rows']) == 3
    old_audit = json.loads((source/'audit.json').read_text())
    source_outcomes = {r['job']['id']: np.asarray(r['success'], bool) for r in old_audit['rows']}
    all_accept = np.zeros((3, 4096), bool)
    bp = np.zeros((3, 4096), bool)
    lp = np.zeros((3, 4096), bool)
    route_table = []
    for row in routed['jobs']:
        job = row['job']; seed = job['seed']; sl = slice(job['shard']*256, (job['shard']+1)*256)
        expected_job = next(j for j in p['source_jobs'] if j['id'] == job['id'])
        assert expected_job == job
        assert sha(source/'runs'/job['id']/'outputs.npz') == job['outputs_sha256']
        path = directory/row['file']; assert sha(path) == row['sha256']
        with np.load(path) as z:
            fields = {k: z[k] for k in z.files}
        with np.load(source/'runs'/job['id']/'outputs.npz') as z:
            original = dict(hidden_good=z['hidden_good'], finite=z['finite'][7],
                            stopped=z['stopped'][7], native_cap=z['block_native_cap'][5],
                            block_cap=z['block_cap'][5], margin=z['minimum_flip_margin64'][7])
        assert all(np.array_equal(fields[k], v, equal_nan=True) for k, v in original.items())
        accept = np.asarray([
            bool(fields['hidden_good'][i]) and bool(fields['finite'][i]) and bool(fields['stopped'][i])
            and not bool(fields['native_cap'][i]) and not bool(fields['block_cap'][i])
            and bool(np.isfinite(fields['margin'][i])) and float(fields['margin'][i]) >= -1e-6
            for i in range(256)
        ])
        assert np.array_equal(accept, fields['accepted'])
        assert fields['chosen_model'].tolist() == [0 if x else 1 for x in accept]
        assert int(accept.sum()) == row['accepted'] and int((~accept).sum()) == row['routed']
        all_accept[seed, sl] = accept
        bp[seed, sl] = source_outcomes[job['id']][0]
        lp[seed, sl] = source_outcomes[job['id']][7]
        for i, ok in enumerate(accept):
            route_table.append({'input_index': job['shard']*256+i, 'model_seed': seed,
                                'lp96_accepted': int(ok), 'routed_to_bp': int(not ok),
                                'output_model': 'LP' if ok else 'BP',
                                'output_checkpoint_sha256': p['models'][str(seed)]['old_binary' if ok else 'pl']['sha256'],
                                'source_outputs_sha256': job['outputs_sha256']})
        if time.perf_counter()-start > p['budgets']['independent_cpu_audit_wall_seconds']:
            raise TimeoutError('Frozen independent audit cap; no extension')
    actual_routes = sorted((int(i), int(s)) for s, i in np.argwhere(~all_accept))
    assert actual_routes == [(r['input_index'], r['model_seed']) for r in p['expected_routes']]
    assert len(route_table) == 12288 and int(all_accept.sum()) == 12285
    # Accepted outputs are exact source references. Every original success is kept.
    assert not np.any(lp & ~all_accept)
    primary = lp.copy(); strict = lp.copy(); rows = []; maximum_margin_error = maximum_energy_error = 0.
    for row in routed['rows']:
        seed = row['model_seed']; index = row['input_index']; local = row['batch_index']
        name = f'i{index}-s{seed}'
        path = directory/'fallback'/f'{name}.npz'
        receipt = json.loads(path.with_suffix('.json').read_text())
        assert receipt['status'] == 'passed' and receipt['route'] == row
        assert receipt['protocol_sha256'] == protocol_hash and receipt['output_sha256'] == sha(path)
        assert all(receipt['checks'].values())
        assert receipt['bp_checkpoint_sha256'] == p['models'][str(seed)]['pl']['sha256']
        assert receipt['lp_checkpoint_sha256'] == p['models'][str(seed)]['old_binary']['sha256']
        diagnostics = directory/'diagnostics'/f'{name}.npz'
        assert sha(diagnostics) == receipt['diagnostics_sha256']
        with np.load(path) as z: replay = {k: z[k] for k in z.files}
        with np.load(diagnostics) as z: diag = {k: z[k] for k in z.files}
        job = next(j for j in p['source_jobs'] if j['id'] == row['job_id'])
        batch = source/'runs'/row['job_id']/f'batch{row["batch"]:02d}.npz'
        assert sha(batch) == job['batch_sha256'][batch.name]
        with np.load(batch) as z: old = {k: z[k] for k in ['couplings','hidden_good','predictions','iterations','stopped','finite','native_final_logits']}
        mapping = {'prediction':'predictions','iterations':'iterations','stopped':'stopped','finite':'finite','final_logits':'native_final_logits'}
        assert all(np.array_equal(replay[k], old[v][0, local:local+1]) for k, v in mapping.items())
        gate = directory/'gates'/f'seed{seed}.npz'; gr = json.loads(gate.with_suffix('.json').read_text())
        assert gr['status'] == 'passed' and gr['source_batch_sha256'] == sha(batch)
        assert gr['output_sha256'] == sha(gate) and all(gr['checks'].values())
        with np.load(gate) as z: assert all(np.array_equal(z[k], old[v][0]) for k, v in mapping.items())
        ds = p['datasets'][row['shard']]; assert sha(root/ds['path']) == ds['sha256']
        with np.load(root/ds['path']) as z:
            tokens = z['tokens'][row['local_index']:row['local_index']+1]
            labels = z['labels'][row['local_index']:row['local_index']+1]
        assert np.array_equal(diag['tokens'], tokens)
        assert np.array_equal(diag['couplings'], old['couplings'][local:local+1])
        assert np.array_equal(diag['hidden_good'], old['hidden_good'][local:local+1])
        board = replay['prediction']; assert np.array_equal(board[tokens != 1], (tokens >= 2)[tokens != 1])
        margins = []; energies = []; locations = []
        for v, arm in enumerate(['pl', 'old_binary']):
            with np.load(source/f'model-seed{seed}-{arm}.npz') as z:
                theta = z['theta'].astype(LD); mu = z['mu_offsets']; ao = z['aoffs']
            delta = np.where(tokens == 1, flip(tokens, board, diag['couplings'].astype(LD), theta, mu, ao), np.inf)
            margin = float(delta.min()); margins.append(margin)
            location = np.unravel_index(int(delta.argmin()), delta.shape); locations.append(list(map(int, location[1:])))
            entropy = LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(tokens == 1).sum((1,2))
            e = float((energy(tokens, board, diag['couplings'].astype(LD), theta, mu, ao)+entropy)[0]); energies.append(e)
            margin_error = abs(margin-float(diag['margins'][v,0])); energy_error = abs(e-float(diag['energies'][v,0]))
            assert margin_error < 1e-9 and energy_error < 1e-6
            maximum_margin_error = max(maximum_margin_error, margin_error)
            maximum_energy_error = max(maximum_energy_error, energy_error)
        own = bool(diag['hidden_good'][0] and replay['finite'][0] and replay['stopped'][0]
                   and np.isfinite(margins[0]) and margins[0] >= -1e-6)
        dual = bool(own and np.isfinite(margins[1]) and margins[1] >= -1e-6)
        # Graph outcomes are computed only after sealed replay files and receipts pass.
        assert valid_shortest(tokens[0], labels[0])
        graph = bool(valid_shortest(tokens[0], board[0]))
        primary[seed,index] = own and graph
        strict[seed,index] = dual and graph
        rows.append({'input_index':index, 'model_seed':seed, 'bp_model_certificate':own,
                     'original_lp_certificate':dual, 'graph_valid_shortest':graph,
                     'bp_original_margin':margins[0], 'lp_original_margin':margins[1],
                     'minimum_flip_locations_bp_lp':locations, 'original_energies_bp_lp':energies,
                     'subset_parity':'passed', 'source_batch_parity_inputs':32})
    byroute = {(r['input_index'],r['model_seed']):r for r in rows}
    for row in route_table:
        i,s = row['input_index'],row['model_seed']
        row.update(bp_success=int(bp[s,i]), lp96_success=int(lp[s,i]),
                   fallback_success=int(primary[s,i]), strict_success=int(strict[s,i]),
                   bp_fallback_certificate=int(byroute[i,s]['bp_model_certificate']) if (i,s) in byroute else '',
                   cross_lp_certificate=int(byroute[i,s]['original_lp_certificate']) if (i,s) in byroute else '')
    route_table.sort(key=lambda r:(r['input_index'],r['model_seed']))
    with (directory/'per-model-input.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(route_table[0]));w.writeheader();w.writerows(route_table)
    arms={'bp_adamw_native':bp,'lp96':lp,'lp96_bp_fallback':primary,'lp96_bp_dual_certificate':strict}
    joint={arm:x.all(0) for arm,x in arms.items()}
    perinput=[{'input_index':i,**{arm:int(x[i]) for arm,x in joint.items()},
               'routed_models':int((~all_accept[:,i]).sum())} for i in range(4096)]
    with (directory/'per-input.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(perinput[0]));w.writeheader();w.writerows(perinput)
    workers=[]
    for gpu in [0,1]:
        r=json.loads((directory/f'worker{gpu}.json').read_text())
        assert r['status']=='complete' and r['worker_seconds']<=120
        assert r['completed_gates']==r['completed_fallbacks']==([0,2] if gpu==0 else [1])
        workers.append(r)
    gains=int((joint['lp96_bp_fallback']&~joint['lp96']).sum())
    losses=int((~joint['lp96_bp_fallback']&joint['lp96']).sum())
    result={'status':'passed','hypothesis':'H2bj','protocol_sha256':protocol_hash,
            'model_inputs_routing_checked':len(route_table),'accepted_lp_model_inputs':int(all_accept.sum()),
            'routed_model_inputs':len(rows),'all_routed_cases':rows,
            'joint_success':{arm:int(x.sum()) for arm,x in joint.items()},
            'model_success':{arm:int(x.sum()) for arm,x in arms.items()},
            'source_joint_gains':gains,'source_joint_losses':losses,
            'execution_decision':'supported' if joint['lp96_bp_fallback'].all() and losses==0 else 'falsified',
            'cross_certificate_decision':'supported' if all(r['original_lp_certificate'] for r in rows) else 'falsified',
            'maximum_margin_error':maximum_margin_error,'maximum_energy_error':maximum_energy_error,
            'worker_receipts':workers,'seconds':time.perf_counter()-start,
            'source_sha256':sha(Path(__file__)),
            'scope':'Complete archived H2bh cohort; known BP rescue execution verification plus finite cross-model local-certificate diagnostic. Zero new scientific totals. No fresh accuracy, shortestness from certificates, global optimality or H2bh confidence upgrade.'}
    assert result['seconds']<=p['budgets']['independent_cpu_audit_wall_seconds']
    save(directory/'audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('root',type=Path)
    a=parser.parse_args();print(json.dumps(audit(a.directory,a.root),indent=2))
