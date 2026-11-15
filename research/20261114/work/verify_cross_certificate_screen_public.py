"""Independent full H2bl scalar decisions and direct-factor negative/control fixtures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import flip,LD
from verify_hard_negative_feasibility import valid_shortest
from audit_connectivity_screen import path_valid


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());a=json.loads((directory/'audit.json').read_text());seal=json.loads((directory/'sealed-margins.json').read_text());d=directory/'public-fixtures';m=json.loads((d/'manifest.json').read_text())
    ph=sha(directory/'protocol.json');assert ph==a['protocol_sha256']==m['protocol_sha256']==seal['protocol_sha256'];assert sha(directory/'audit.json')==m['audit_sha256']
    assert sha(directory/'sealed-margins.json')==a['sealed_margins_sha256'] and len(seal['rows'])==48 and seal['model_inputs']==12288 and not seal['labels_joined']
    source=json.loads((d/'source-audit.json').read_text());assert sha(d/'source-audit.json')==m['source_audit_sha256']==p['source_evidence_sha256']['artifacts/research/20261114/work/loop80_confirmation/audit.json'];source_rows={r['job']['id']:r for r in source['rows']}
    with (directory/'per-model-input.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==12288 and sha(directory/'per-model-input.csv')==a['rows_sha256'];bycase={(int(r['input_index']),int(r['model_seed'])):r for r in rows};assert len(bycase)==12288
    matrices={k:{'graph_successes':0,'accepted_graph_successes':0,'rejected_graph_successes':0,'graph_failures':0,'accepted_graph_failures':0,'rejected_graph_failures':0} for k in ['bp_own_accept','lp_cross_accept','dual_accept']};values={}
    for entry in m['scalar_archives']:
        path=d/entry['file'];job=entry['job'];assert sha(path)==entry['sha256'];bound=next(x for x in seal['rows'] if x['job']['id']==job['id']);assert bound['job']==job and bound['margin_sha256']==entry['sha256']
        with np.load(path) as z:fields={k:z[k] for k in z.files}
        values[entry['file']]=fields;assert all(x.shape==(256,) for x in fields.values())
        own=fields['hidden_good']&fields['finite']&fields['stopped']&fields['clamps_ok']&np.isfinite(fields['bp_margin'])&(fields['bp_margin']>=-1e-6)
        cross=fields['hidden_good']&fields['clamps_ok']&np.isfinite(fields['lp_margin'])&(fields['lp_margin']>=-1e-6)
        for key,expected in [('bp_own_accept',own),('lp_cross_accept',cross),('dual_accept',own&cross)]:assert np.array_equal(fields[key],expected)
        for i in range(256):
            row=bycase[job['shard']*256+i,job['seed']];good=int(row['graph_valid_shortest']);assert row['source_job']==job['id']
            # The original audited BP cohort has no numerical failures, so its
            # success labels equal graph-shortest labels for these fixed boards.
            assert own[i] and good==source_rows[job['id']]['success'][0][i]
            for field,column in [('bp_margin','bp_original_margin'),('lp_margin','lp_cross_margin')]:assert fields[field][i]==float(row[column])
            assert int(fields['witness_flat'][i])==int(row['lp_witness_flat'])
            for key,table in matrices.items():
                accepted=int(fields[key][i]);assert accepted==int(row[key]);suffix='graph_successes' if good else 'graph_failures';table[suffix]+=1;table[('accepted_' if accepted else 'rejected_')+suffix]+=1
    assert matrices==a['control_confusion_matrices'];primary=matrices['dual_accept'];assert primary['graph_failures']==510 and primary['graph_successes']==11778
    rejection=primary['rejected_graph_failures'];retention=primary['accepted_graph_successes']
    assert a['primary_endpoints']['error_rejection']['decision']==('supported' if 100*rejection>=90*510 else 'falsified')
    assert a['primary_endpoints']['valid_retention']['decision']==('supported' if 100*retention>=99*11778 else 'falsified')
    fixed={(r['input_index'],r['model_seed']) for r in m['fixed_cases']};assert fixed=={(shard*256,s) for shard in range(16) for s in range(3)}
    supplement={(r['input_index'],r['model_seed']) for r in m['negative_or_retention_loss_cases']};expected={(int(r['input_index']),int(r['model_seed'])) for r in rows if r['graph_valid_shortest']=='0' or r['dual_accept']=='0'};assert supplement==expected
    max_bp=max_lp=max_witness=0.;checked=0;cache={};labels=set()
    for record in m['fixed_cases']+m['negative_or_retention_loss_cases']:
        path=d/record['file'];assert sha(path)==record['sha256'];row=bycase[record['input_index'],record['model_seed']]
        with np.load(path) as z:f={k:z[k] for k in z.files}
        scalar=values[record['scalar_file']];i=record['local_index'];assert all(np.array_equal(f[k],v[i:i+1],equal_nan=True) for k,v in scalar.items())
        t=f['tokens'];b=f['board'];aa=f['couplings'].astype(LD);theta=f['theta'].astype(LD);mu=f['mu_offsets'];ao=f['aoffs'];th=hashlib.sha256(t.tobytes()).digest()
        if th not in labels:assert valid_shortest(t[0],f['labels'][0]);labels.add(th)
        for v,key in enumerate(['bp_margin','lp_margin']):
            delta=np.where(t==1,flip(t,b,aa,theta[v],mu,ao),np.inf).reshape(1,-1);margin=float(delta.min());error=abs(margin-float(f[key][0]));assert error<1e-9
            if v==0:max_bp=max(max_bp,error)
            else:
                max_lp=max(max_lp,error);w=int(f['witness_flat'][0]);assert t.reshape(1,-1)[0,w]==1;error=abs(float(delta[0,w])-margin);assert error<1e-9;max_witness=max(max_witness,error)
        bk=(th,np.packbits(b).tobytes())
        if bk not in cache:cache[bk]=(path_valid(t[0],b[0]),valid_shortest(t[0],b[0]))
        valid,shortest=cache[bk];assert int(valid)==int(row['graph_valid_path']) and int(shortest)==int(row['graph_valid_shortest']);checked+=1
    return {'status':'passed','model_input_scalar_rows':12288,'control_confusion_matrices':matrices,'primary_endpoints':a['primary_endpoints'],
        'fixed_state_fixtures':len(m['fixed_cases']),'complete_negative_or_retention_loss_fixtures':len(m['negative_or_retention_loss_cases']),
        'source_boards_checked_in_fixtures':checked,'original_objective_margin_checks':2*checked,'maximum_bp_margin_error':max_bp,'maximum_lp_margin_error':max_lp,'maximum_witness_error':max_witness,
        'manifest_sha256':sha(d/'manifest.json'),'source_sha256':sha(Path(__file__)),'seconds':time.perf_counter()-started,
        'scope':'All sealed scalar decisions and finite targets; direct BP/LP factors and independent graphs on fixed controls and every negative/retention-loss case. Remaining accepted-positive source states were checked in the full independent Work audit and bound to the original H2bh audit. No fresh samples or global/shortestness proof from local certificates.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);print(json.dumps(verify(parser.parse_args().directory),indent=2))
