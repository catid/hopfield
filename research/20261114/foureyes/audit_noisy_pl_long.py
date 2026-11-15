"""Independent source, model and saved-board audit for H4s."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import csv
import hashlib
import json
import os
import sys
import time
import numpy as np
import torch
from audit_restart import validity
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/20261114/foureyes/noisy_pl_long';PUBLIC=ROOT/'results/research/foureyes/noisy_pl_long'
sys.path.insert(0,str(ROOT/'research/20261114/ripper'))
from bootstrap_two_way import resample_means

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def audit_job(job):
    torch.set_num_threads(1);start=time.monotonic();dest=OUT/job['job_id'];rec=json.loads((dest/'receipt.json').read_text());evidence={}
    assert rec['status']=='complete' and rec['preflight_checks']==8 and rec['seed']==0 and rec['chains']==4
    evidence[str((dest/'receipt.json').relative_to(ROOT))]=sha(dest/'receipt.json')
    for name,expected in rec['source_sha256'].items():assert sha(ROOT/name)==expected;evidence[name]=expected
    path=ROOT/job['model_path'];assert sha(path)==job['model_sha256'];evidence[job['model_path']]=sha(path);m=np.load(path)
    support=m['support']
    assert hashlib.sha256(support.tobytes()).hexdigest()==job['support_sha256']
    assert len(rec['checkpoints'])==1;point=rec['checkpoints'][0];assert point['cap']==262144
    directory=dest/'moves-262144'
    for name,expected in point['output_sha256'].items():assert sha(directory/name)==expected;evidence[str((directory/name).relative_to(ROOT))]=expected
    b=np.load(directory/'boards.npz');target=np.load(OUT/'audit-target.npz')
    assert np.array_equal(b['solutions'],target['solutions']) and np.array_equal(b['free'],target['free']) and np.array_equal(b['puzzle'],target['puzzle'])
    good=validity(b['boards'],target['solutions'],target['free']);assert np.array_equal(good,b['valid']) and np.array_equal(good,b['exact']) and np.array_equal(good,(b['boards']==target['solutions']).all(1))
    assert int(good.sum())==point['solved'] and int((b['stopped']&~good).sum())==point['wrong_stops']
    assert (b['moves']<=262144).all() and (b['moves'][~b['stopped']]==262144).all()
    state=torch.load(directory/'state.pt',weights_only=False,map_location='cpu');capture=torch.load(directory/'capture.pt',weights_only=True,map_location='cpu')
    returned=torch.where(state['stopped'][:,None],capture['cert_boards'],capture['best_boards']).numpy()
    assert np.array_equal(returned,b['boards']) and np.array_equal(state['stopped'],b['stopped'])
    assert np.array_equal(torch.where(state['stopped'],state['ok'],state['bestok']).numpy(),good)
    ii,jj=np.triu_indices(81,1);adjacency=np.zeros((81,81),dtype=np.int64);adjacency[ii[support],jj[support]]=1;adjacency+=adjacency.T.copy()
    y=state['y'].numpy();fields=np.einsum('ij,hjv->hiv',adjacency,np.eye(9,dtype=np.int64)[y])
    energy=(y[:,ii[support]]==y[:,jj[support]]).sum(1)
    assert np.array_equal(fields,state['G'].numpy()) and np.array_equal(energy,state['P'].numpy())
    returned_energy=(returned[:,ii[support]]==returned[:,jj[support]]).sum(1);assert (returned_energy[b['stopped']]==0).all()
    row,col=np.arange(81)//9,np.arange(81)%9;truth=(row[ii]==row[jj])|(col[ii]==col[jj])|((row[ii]//3==row[jj]//3)&(col[ii]//3==col[jj]//3))
    if (support[truth]).all():assert good[b['stopped']].all()
    print(json.dumps(dict(job=job['job_id'],boards=len(good),wall_s=time.monotonic()-start)),flush=True)
    return dict(job_id=job['job_id'],good=good,stopped=b['stopped'],moves=b['moves'],support_sha256=job['support_sha256'],input_sha256=evidence,search_seconds=point['cumulative_search_seconds'])

def main():
    start=time.monotonic();p=json.loads((HERE/'noisy_pl_long.json').read_text());prepared=json.loads((OUT/'prepared.json').read_text());jobs=json.loads((OUT/'queue.json').read_text())
    assert all(j['status']=='complete' for j in jobs);assert prepared['protocol_sha256']==sha(HERE/'noisy_pl_long.json') and prepared['producer_sha256']==sha(HERE/'map_noisy_pl_long.py')
    evidence={}
    for name,expected in (p['source_sha256']|p['frozen_inputs']).items():assert sha(ROOT/name)==expected;evidence[name]=expected
    data=np.load(ROOT/'upstream/data/full-test/full-all.npz');lo,hi=p['rows_half_open']
    np.savez_compressed(OUT/'audit-target.npz',solutions=data['solutions'][lo:hi].astype(np.int8)-1,free=~data['clue_mask'][lo:hi].astype(bool),puzzle=np.arange(lo,hi))
    from audit_noisy_pl_long_source import audit_sources
    cache=OUT/'source-audit.json'
    source_audit=json.loads(cache.read_text()) if cache.exists() else audit_sources()
    assert source_audit['protocol_sha256']==sha(HERE/'noisy_pl_long.json') and source_audit['auditor_sha256']==sha(HERE/'audit_noisy_pl_long_source.py')
    for name,digest in source_audit['input_sha256'].items():assert sha(ROOT/name)==digest
    public_models={(r['bank'],r['arm']):r['support_sha256'] for r in source_audit['models']}
    with ProcessPoolExecutor(max_workers=6) as pool:results=list(pool.map(audit_job,jobs))
    unique={r['job_id']:r for r in results};rows=[];arrays={}
    for mapping in prepared['mappings']:
        bank,arm=mapping['bank'],mapping['arm'];r=unique[mapping['actual']]
        assert r['support_sha256']==mapping['support_sha256']==public_models[bank,arm] and mapping['seed']==p['search_seed']==0
        arrays[bank,arm]=r['good'].astype(np.int8)
        rows.append(dict(bank=bank,arm=arm,actual_job=mapping['actual'],solved=int(r['good'].sum()),accuracy_pct=float(r['good'].mean()*100),wrong_stops=int((r['stopped']&~r['good']).sum()),mean_moves=float(r['moves'].mean())))
    assert len(rows)==p['banks']*len(p['arms']) and not prepared['ineligible']
    grouped=[]
    for arm in p['arms']:
        rs=[r for r in rows if r['arm']==arm];grouped.append(dict(arm=arm,mean_accuracy_pct=float(np.mean([r['accuracy_pct'] for r in rs])),logical_wrong_stops=sum(r['wrong_stops'] for r in rs),mean_moves=float(np.mean([r['mean_moves'] for r in rs]))))
    delta=np.stack([arrays[b,p['primary_arm']]-arrays[b,p['control_arm']] for b in range(p['banks'])])
    boot=resample_means(delta,20261115444,draws=5000);lower,upper=map(float,np.quantile(boot,[.025,.975]));wrong=next(r['logical_wrong_stops'] for r in grouped if r['arm']==p['primary_arm'])
    primary=dict(gain_pp=float(delta.mean()*100),paired_two_way95_interval_pp=[lower,upper],target_pp=-2.,logical_wrong_stops=wrong,decision='falsified' if wrong or upper<-2 else 'supported' if lower>=-2 else 'inconclusive')
    for r in results:evidence.update(r['input_sha256'])
    for file in [Path(__file__),HERE/'audit_restart.py',OUT/'prepared.json',OUT/'queue.json',OUT/'launch-result.json',OUT/'search-launch.json',HERE/'noisy_pl_long_search.py',HERE/'audit_noisy_pl_long_source.py',OUT/'source-audit.json',OUT/'gate.json',OUT/'audit-target.npz']:evidence[str(file.relative_to(ROOT))]=sha(file)
    result=dict(status='validated',audited_utc=datetime.now(timezone.utc).isoformat(),primary=primary,grouped=grouped,unique_searches=len(jobs),logical_cells=len(rows),unique_saved_boards_audited=len(jobs)*10000,unique_puzzles=10000,source_banks=p['banks'],source_audit=source_audit,common_search_seed=0,
        all_integer_fields_and_energies_exact=True,model_mappings=prepared['mappings'],logical_bank_outcomes=rows,input_sha256=evidence,limits=p['limits'],audit_wall_s=time.monotonic()-start,launch=json.loads((OUT/'launch-result.json').read_text()))
    PUBLIC.mkdir(exist_ok=True,parents=True);write(PUBLIC/'summary.json',result);write(OUT/'audit.json',result)
    with (PUBLIC/'per_bank.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'audited-outcomes.npz',**{arm:np.stack([arrays[b,arm] for b in range(p['banks'])]) for arm in p['arms']})
    print(json.dumps(dict(primary=primary,grouped=grouped,audit_wall_s=result['audit_wall_s'])),flush=True)

if __name__=='__main__':main()
