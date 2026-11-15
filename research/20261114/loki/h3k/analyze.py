"""Separate source-bank inference from prespecified descriptive target searches."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3k';OUT=ROOT/'artifacts/research/20261114/loki/h3k';PUBLIC=ROOT/'results/research/loki/h3k'
spec=importlib.util.spec_from_file_location('n13_audit',SRC.parent/'h3j/analyze.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)
J.SRC=SRC;J.OUT=OUT
HASHES={}


def sha(path):
    path=Path(path)
    with path.open('rb') as f:v=hashlib.file_digest(f,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=v;return v
def read(path):sha(path);return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    started=time.perf_counter();torch.set_num_threads(1);p=read(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    preflight=read(OUT/'independent-preflight.json');assert preflight['status']=='passed' and preflight['all_identities_exact']
    for path,digest in preflight['input_sha256'].items():assert sha(ROOT/path)==digest,path
    validations=[read(OUT/f'backend-validation-{device}.json') for device in ['cpu','cuda']]
    for receipt in validations:
        assert receipt['status']=='passed' and receipt['backend_sha256']==sha(SRC.parent/'h3i/backend.py')
        assert receipt['source_sha256']==sha(SRC/'driver.py') and receipt['protocol_sha256']==sha(SRC/'protocol.json')
    plan=read(OUT/'identity-plan.json');mapping=read(OUT/'identity-mapping.json');groups=read(OUT/'energy-groups.json')
    assert mapping['status']=='complete' and mapping['plan_sha256']==sha(OUT/'identity-plan.json')
    assert plan['identity_receipt_sha256']==sha(OUT/'independent-preflight.json') and plan['source_sha256']==sha(SRC/'reuse_identities.py')
    assert plan['groups']==groups['groups'] and plan['actual_unique_search_jobs']==groups['distinct_energies']*3
    for item in mapping['mappings']:
        source=read(ROOT/item['source_result']);derived=read(ROOT/item['derived_result'])
        assert sha(ROOT/item['source_result'])==item['source_result_sha256'] and sha(ROOT/item['derived_result'])==item['derived_result_sha256']
        assert source['status']=='complete' and derived['status']=='derived_by_exact_identity' and not derived['independent_search']
        assert derived['source_result_sha256']==item['source_result_sha256'] and derived['identity_plan_sha256']==sha(OUT/'identity-plan.json')
        assert source['records_sha256']==derived['records_sha256']==item['records_sha256']
        assert derived['search_seconds']==0 and derived['shared_source_search_seconds']==source['search_seconds']
    d=dict(np.load(OUT/'data.npz'));keys={row.tobytes() for row in d['all_solutions']};n=len(d['test_inputs'])
    PUBLIC.mkdir(parents=True,exist_ok=True);canonical={arm:g['representative'] for g in groups['groups'] for arm in g['members']}
    models={};model_audits=[]
    for cfg in p['arms']:
        folder=OUT/f"models/{cfg['arm']}";rec=read(folder/'model.json')
        assert rec['model_sha256']==sha(folder/'model.pt') and rec['construction_sha256']==sha(folder/'construction.npz')
        models[cfg['arm']]=torch.load(folder/'model.pt',weights_only=False);model_audits.append(rec)
    for group in groups['groups']:
        base=models[group['representative']]
        for arm in group['members']:
            model=models[arm];assert all(torch.equal(base[k],model[k]) for k in ['S','B']) and all(base[k]==model[k] for k in ['q','delta','tau','unit'])
    audited={};capacity_by_group={};unique_diagnostics=[]
    for group in groups['groups']:
        arm=group['representative'];model=models[arm];counts,cap=J.H.feasible_certificates(model,d);capacity_by_group[arm]=cap
        for seed in [0,1,2]:
            z,row,diag=J.search_audit(arm,seed,d,keys,set(),model)
            assert not (z['stopped']&z['valid'])[counts==0].any()
            audited[arm,seed]=(z,row,diag);unique_diagnostics.append(diag)
    rows=[];perseed=[];logical_evidence={};capacity=[]
    for cfg in p['arms']:
        arm=cfg['arm'];representative=canonical[arm];current=[];zs=[]
        audit=next(x for x in preflight['target_models'] if x['arm']==arm)
        for seed in [0,1,2]:
            z,base_row,_=audited[representative,seed];folder=OUT/f'search/{arm}/s{seed}';rec=read(folder/'result.json')
            assert rec['records_sha256']==sha(folder/'records.npz')==base_row['records_sha256']
            assert rec['model_sha256']==sha(ROOT/rec['model_path'])
            row=dict(base_row,arm=arm,model_sha256=rec['model_sha256'],result_sha256=sha(folder/'result.json'),search_seconds=rec['search_seconds'],
                source_draw=cfg['draw'],source_patterns=cfg['source_patterns'],independent_search=arm==representative,derived_from='' if arm==representative else representative)
            zs.append(z);current.append(row);perseed.append(row)
        arr={k:np.stack([z[k] for z in zs]) for k in ['n20','n20_certificate','n1','found']};logical_evidence[arm]=arr
        rows.append(dict(arm=arm,draw=cfg['draw'],source_patterns=cfg['source_patterns'],exact_four_line_recovery=audit['exact_four_line_recovery'],
            false_forbidden=audit['false_forbidden'],missed_forbidden=audit['missed_forbidden'],unique_inputs=n,chains=20,search_seeds=[0,1,2],
            original_accuracy_pct=100*float(arr['n20'].mean()),certificate_first_accuracy_pct=100*float(arr['n20_certificate'].mean()),
            n1_accuracy_pct=100*float(arr['n1'].mean()),coverage_pct=100*float(arr['found'].sum(1).mean()/zs[0]['denom'].sum()),
            wrong_stops=sum(x['wrong_stops'] for x in current),capped_chains=sum(x['capped_chains'] for x in current),
            mean_moves=float(np.mean([x['mean_moves'] for x in current])),search_seconds_total=sum(x['search_seconds'] for x in current),
            energy_representative=representative,independent_search=arm==representative))
        capacity.append(dict(arm=arm,representative=representative,**capacity_by_group[representative]))
    behavior=[]
    for size in [1,8]:
        subset=[r for r in rows if r['source_patterns']==size]
        behavior.append(dict(source_patterns=size,prespecified_source_draws=6,search_seeds_per_model=3,
            original_accuracy_pct=float(np.mean([r['original_accuracy_pct'] for r in subset])),
            certificate_first_accuracy_pct=float(np.mean([r['certificate_first_accuracy_pct'] for r in subset])),
            coverage_pct=float(np.mean([r['coverage_pct'] for r in subset])),exact_line_models=sum(r['exact_four_line_recovery'] for r in subset),
            interpretation='Descriptive equal-weight mean across the first6 prespecified source draws, retaining identical logical models; not an estimate from all512 source draws.'))
    workers=[read(OUT/f'worker-g{g}.json') for g in [0,1]];assert all(w['status']=='complete' for w in workers)
    for w in workers:
        assert dt.datetime.fromisoformat(preflight['utc'])<dt.datetime.fromisoformat(w['started_utc'])
        assert dt.datetime.fromisoformat(plan['frozen_utc'])<dt.datetime.fromisoformat(w['started_utc'])
        assert all(dt.datetime.fromisoformat(v['utc'])<dt.datetime.fromisoformat(w['started_utc']) for v in validations)
    gpu_seconds=sum((dt.datetime.fromisoformat(w['completed_utc'])-dt.datetime.fromisoformat(w['started_utc'])).total_seconds() for w in workers)
    assert gpu_seconds<1200
    for path in [Path(__file__),SRC.parent/'h3j/analyze.py',SRC.parent/'h3h/analyze.py',SRC.parent/'h3g/analyze.py',SRC.parent/'analyze.py']:sha(path)
    HASHES.update(J.HASHES)
    source_draws=[dict(r,source_pattern_indices=';'.join(map(str,d['source_draw_indices'][r['draw'],:r['source_patterns']]))) for r in preflight['source_outcomes']]
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3k',decision=preflight['primary']['decision'],
        primary=preflight['primary'],source_recovery=preflight['source_recovery'],rows=rows,prespecified_behavior_summary=behavior,
        dataset=p['dataset'],calibration=p['calibration'],target_model_audits=preflight['target_models'],output_diagnostics=unique_diagnostics,
        posthoc_valid_certificate_capacity=capacity,bp_adamw='Not run; this study measures source-line identification and zero-shot target behavior under a fixed supplied line-family prior.',
        independent_saved_boards=groups['actual_unique_search_jobs']*n*20,derived_record_boards=groups['avoided_identical_jobs']*n*20,
        independent_source_draws=512,total_source_fits=2560,source_fit_artifact_sha256=sha(OUT/'source-fits.npz'),
        identity_plan=plan,identity_mapping=mapping,backend_validations=validations,worker_receipts=workers,
        gpu_worker_seconds=gpu_seconds,input_sha256=HASHES,limits=p['limits'],audit_seconds=time.perf_counter()-started)
    save(OUT/'summary.json',summary);save(PUBLIC/'summary.json',summary)
    for name,records in [('per_seed.csv',perseed),('source_draws.csv',source_draws),('source_recovery.csv',preflight['source_recovery'])]:
        with (PUBLIC/name).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(dict(status='validated',primary=summary['primary'],behavior=behavior,rows=rows,distinct_energies=groups['distinct_energies'],
        independent_saved_boards=summary['independent_saved_boards'],gpu_worker_seconds=gpu_seconds,audit_seconds=summary['audit_seconds'])))


if __name__=='__main__':main()
