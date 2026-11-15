"""Independent H2bl direct-factor and graph joins after the complete margin seal."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import flip,LD
from verify_hard_negative_feasibility import valid_shortest
from audit_connectivity_screen import path_valid


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def scalar(value):return float(value) if np.isfinite(value) else None


def one(args):
    root,directory,p,shard,seal=args;started=time.perf_counter();source=root/'artifacts/research/20261114/work/loop80_confirmation';ds=p['datasets'][shard];assert sha(root/ds['path'])==ds['sha256']
    # This function runs only after main() verifies the complete 48-job seal.
    with np.load(root/ds['path']) as z:tokens=z['tokens'];labels=z['labels'];indices=z['indices']
    assert np.array_equal(indices,np.arange(shard*256,(shard+1)*256))
    assert all(valid_shortest(t,b) for t,b in zip(tokens,labels));shortest_cells=labels.sum((1,2));rows=[];maximum_bp_error=maximum_lp_error=maximum_witness_error=0.;decisions=0;graph_cache={}
    for job in [j for j in p['jobs'] if j['shard']==shard]:
        seed=job['seed'];sourcepath=source/'runs'/job['id']/'outputs.npz';assert sha(sourcepath)==job['outputs_sha256']
        with np.load(sourcepath) as z:board=z['predictions'][0];couplings=z['couplings'].astype(LD);hidden=z['hidden_good'];finite=z['finite'][0];stopped=z['stopped'][0];old_margin=z['minimum_flip_margin64'][0]
        run=directory/'runs'/job['id'];bound=next(r for r in seal['rows'] if r['job']['id']==job['id']);assert sha(run/'margins.npz')==bound['margin_sha256'] and sha(run/'result.json')==bound['receipt_sha256']
        receipt=json.loads((run/'result.json').read_text());assert not receipt['labels_distances_outcomes_loaded'] and receipt['n']==256 and receipt['source_outputs_sha256']==job['outputs_sha256']
        with np.load(run/'margins.npz') as z:actual={k:z[k] for k in z.files}
        assert all(v.shape==(256,) for v in actual.values());clamps=((tokens==1)|(board==(tokens>=2))).all((1,2))
        for k,v in [('hidden_good',hidden),('finite',finite),('stopped',stopped),('clamps_ok',clamps),('bp_margin',old_margin)]:assert np.array_equal(actual[k],v,equal_nan=True)
        computed=[]
        for arm in ['pl','old_binary']:
            with np.load(source/f'model-seed{seed}-{arm}.npz') as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
            parts=[]
            for lo in range(0,256,32):
                delta=np.where(tokens[lo:lo+32]==1,flip(tokens[lo:lo+32],board[lo:lo+32],couplings[lo:lo+32],theta,mu,ao),np.inf).reshape(32,-1);margins=delta.min(1);parts.append(margins)
                if arm=='old_binary':
                    witness=actual['witness_flat'][lo:lo+32];assert np.all((witness>=0)&(witness<180*180))
                    assert (tokens[lo:lo+32].reshape(32,-1)[np.arange(32),witness]==1).all();values=delta[np.arange(32),witness];mask=np.isfinite(values)&np.isfinite(margins)
                    if mask.any():err=float(abs(values[mask]-margins[mask]).max());assert err<1e-9;maximum_witness_error=max(maximum_witness_error,err)
            computed.append(np.concatenate(parts))
        for v,key in enumerate(['bp_margin','lp_margin']):
            assert np.array_equal(np.isfinite(computed[v]),np.isfinite(actual[key]));mask=np.isfinite(computed[v])
            if mask.any():err=float(abs(computed[v][mask]-actual[key][mask]).max());assert err<1e-9
            else:err=0.
            if v==0:maximum_bp_error=max(maximum_bp_error,err)
            else:maximum_lp_error=max(maximum_lp_error,err)
        bp_accept=hidden&finite&stopped&clamps&np.isfinite(computed[0])&(computed[0]>=LD(-1e-6));lp_accept=hidden&clamps&np.isfinite(computed[1])&(computed[1]>=LD(-1e-6));dual=bp_accept&lp_accept
        for expected,key in [(bp_accept,'bp_own_accept'),(lp_accept,'lp_cross_accept'),(dual,'dual_accept')]:assert np.array_equal(expected,actual[key]);decisions+=len(expected)
        for i in range(256):
            key=(i,np.packbits(board[i]).tobytes())
            if key not in graph_cache:graph_cache[key]=path_valid(tokens[i],board[i])
            graph=bool(graph_cache[key]);shortest=graph and int(board[i].sum())==int(shortest_cells[i])
            rows.append({'input_index':shard*256+i,'model_seed':seed,'source_job':job['id'],'graph_valid_path':int(graph),'graph_valid_shortest':int(shortest),
                'hidden_good':int(hidden[i]),'BP_finite':int(finite[i]),'BP_stopped':int(stopped[i]),'clamps_ok':int(clamps[i]),
                'bp_original_margin':scalar(actual['bp_margin'][i]),'lp_cross_margin':scalar(actual['lp_margin'][i]),'lp_witness_flat':int(actual['witness_flat'][i]),
                'bp_own_accept':int(bp_accept[i]),'lp_cross_accept':int(lp_accept[i]),'dual_accept':int(dual[i])})
    result={'shard':shard,'status':'passed','rows':rows,'shortest_labels_checked':256,'decisions_checked':decisions,
        'maximum_bp_margin_error':maximum_bp_error,'maximum_lp_margin_error':maximum_lp_error,'maximum_witness_error':maximum_witness_error,'seconds':time.perf_counter()-started}
    save(directory/'partial-rows'/f'shard{shard:02d}.json',result);return result


def matrix(rows,key):
    good=[r for r in rows if r['graph_valid_shortest']];bad=[r for r in rows if not r['graph_valid_shortest']]
    return {'graph_successes':len(good),'accepted_graph_successes':sum(r[key] for r in good),'rejected_graph_successes':sum(not r[key] for r in good),
            'graph_failures':len(bad),'accepted_graph_failures':sum(r[key] for r in bad),'rejected_graph_failures':sum(not r[key] for r in bad)}


def main(directory,root,workers):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());ph=sha(directory/'protocol.json');sealed=directory/'sealed-margins.json';seal=json.loads(sealed.read_text())
    assert seal['status']=='complete' and seal['protocol_sha256']==ph and seal['model_inputs']==12288 and len(seal['rows'])==48 and not seal['labels_joined']
    assert sorted(r['job']['id'] for r in seal['rows'])==sorted(j['id'] for j in p['jobs'])
    for field in ['source_sha256','source_evidence_sha256']:
        for path,h in p[field].items():assert sha(root/path)==h,path
    for r in seal['rows']:
        run=directory/'runs'/r['job']['id'];assert sha(run/'margins.npz')==r['margin_sha256'] and sha(run/'result.json')==r['receipt_sha256']
    assert all(w['status']=='complete' and w['seconds']<=120 and w['peak_rss_kib']<=8*1024**2 for w in seal['worker_receipts'])
    (directory/'partial-rows').mkdir(exist_ok=True);results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,(root,directory,p,s,seal)) for s in range(16)]
        for f in as_completed(futures):
            r=f.result();results.append(r);print(json.dumps({'event':'cross_certificate_shard_audited','shard':r['shard'],'completed_shards':len(results)}),flush=True)
    rows=sorted([r for shard in results for r in shard['rows']],key=lambda r:(r['input_index'],r['model_seed']));assert len(rows)==12288 and len({(r['input_index'],r['model_seed']) for r in rows})==12288
    names=['bp_own_accept','lp_cross_accept','dual_accept'];tables={k:matrix(rows,k) for k in names};primary=tables['dual_accept'];assert primary['graph_failures']==510 and primary['graph_successes']==11778
    rejected=primary['rejected_graph_failures'];retained=primary['accepted_graph_successes']
    endpoints={'error_rejection':{'rejected':rejected,'denominator':510,'observed_percent':100*rejected/510,'minimum_percent':90.,'decision':'supported' if 100*rejected>=90*510 else 'falsified'},
        'valid_retention':{'retained':retained,'denominator':11778,'observed_percent':100*retained/11778,'minimum_percent':99.,'decision':'supported' if 100*retained>=99*11778 else 'falsified'}}
    with (directory/'per-model-input.csv').open('w',newline='') as stream:w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'status':'passed','hypothesis':'H2bl','protocol_sha256':ph,'sealed_margins_sha256':sha(sealed),'source_model_inputs':12288,
        'shortest_labels_checked':sum(r['shortest_labels_checked'] for r in results),'control_confusion_matrices':tables,'primary_endpoints':endpoints,
        'per_seed_controls':{str(s):{k:matrix([r for r in rows if r['model_seed']==s],k) for k in names} for s in range(3)},
        'bp_numerical_rejections':sum(not r['bp_own_accept'] for r in rows),'hidden_failures':sum(not r['hidden_good'] for r in rows),
        'bp_nonfinite':sum(not r['BP_finite'] for r in rows),'bp_not_stopped':sum(not r['BP_stopped'] for r in rows),
        'lp_nonfinite_margins':sum(r['lp_cross_margin'] is None for r in rows),'clamp_failures':sum(not r['clamps_ok'] for r in rows),
        'maximum_bp_margin_error':max(r['maximum_bp_margin_error'] for r in results),'maximum_lp_margin_error':max(r['maximum_lp_margin_error'] for r in results),
        'maximum_witness_error':max(r['maximum_witness_error'] for r in results),'acceptance_decisions_checked':sum(r['decisions_checked'] for r in results),
        'rows_sha256':sha(directory/'per-model-input.csv'),'worker_receipts':seal['worker_receipts'],'cpu_shard_receipts':[{k:v for k,v in r.items() if k!='rows'} for r in sorted(results,key=lambda r:r['shard'])],
        'seconds':time.perf_counter()-started,'processes':workers,'source_sha256':sha(Path(__file__)),
        'scope':'All archived H2bh BP boards. All margins and acceptance flags sealed before graph labels were joined. Finite source targets only; no fresh trajectory, population claim, shortestness/global proof from a local certificate or executed BP-first cascade.'}
    assert result['seconds']<=300;save(directory/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['worker_receipts','cpu_shard_receipts']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('root',type=Path);parser.add_argument('--workers',type=int,default=16);a=parser.parse_args();main(a.directory,a.root,a.workers)
