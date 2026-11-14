"""Independent audit of noisy-label counts, all model mappings and saved states."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import torch
from audit_restart import validity

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/20261114/foureyes/noise'
PUBLIC=ROOT/'results/research/foureyes/noise'


def main():
    evidence={}
    def verify(path,expected=None):
        key=str(path.relative_to(ROOT))
        if key not in evidence:
            with path.open('rb') as f:evidence[key]=hashlib.file_digest(f,'sha256').hexdigest()
        if expected is not None:assert evidence[key]==expected,path
        return evidence[key]
    p=json.loads((HERE/'noise.json').read_text())
    prepared=json.loads((OUT/'prepared.json').read_text())
    verify(OUT/'prepared.json')
    verify(HERE/'noise.json',prepared['protocol_sha256'])
    verify(HERE/'prepare_noise.py',prepared['source_sha256'])
    verify(Path(__file__));verify(HERE/'audit_restart.py')
    for name,digest in (p['frozen_inputs']|p['source_sha256']).items():verify(ROOT/name,digest)
    jobs=json.loads((OUT/'queue.json').read_text())
    assert all(j['status']=='complete' for j in jobs)
    ii,jj=np.triu_indices(81,1)
    row,col=np.arange(81)//9,np.arange(81)%9
    truth=(row[ii]==row[jj])|(col[ii]==col[jj])|((row[ii]//3==row[jj]//3)&(col[ii]//3==col[jj]//3))
    models={};graphs=[]
    for item in prepared['models']:
        verify(ROOT/item['model_path'],item['model_sha256'])
        z=np.load(ROOT/item['model_path'])
        seed,arm=item['seed'],item['arm']
        source=np.load(ROOT/f'artifacts/research/20261114/ripper/raw/inputs/s{seed}/train.npz')
        clean=source['solutions'][:300].astype(np.int64);free=source['free'][:300].astype(bool)
        assert np.array_equal(z['clean'],clean) and np.array_equal(z['free'],free)
        assert validity(clean,clean,free).all()
        noisy=clean.copy();rng=np.random.default_rng(p['noise_rng_base']+seed)
        for r in rng.choice(300,size=3,replace=False):
            cell=int(rng.choice(np.flatnonzero(free[r])))
            noisy[r,cell]=(clean[r,cell]+rng.integers(1,9))%9
        expected=clean if arm=='clean' else noisy
        assert np.array_equal(z['train'],expected)
        counts=(expected[:,ii]==expected[:,jj]).sum(0)
        support=counts<=p['thresholds'][arm]
        assert np.array_equal(z['counts'],counts) and np.array_equal(z['support'],support)
        assert hashlib.sha256(support.tobytes()).hexdigest()==item['support_sha256']
        models[item['job_id']]=support
        graphs.append(dict(seed=seed,arm=arm,edges=int(support.sum()),missing_true_edges=int((truth&~support).sum()),
            extra_edges=int((~truth&support).sum()),exact_peer_graph=bool(np.array_equal(support,truth)),
            corrupted_cells=int((expected!=clean).sum())))
    datafile=ROOT/'upstream/data/full-test/full-all.npz'
    prior=json.loads((ROOT/'results/research/ripper/raw_compare.json').read_text())
    verify(datafile,prior['input_sha256'][str(datafile.relative_to(ROOT))])
    z=np.load(datafile);lo,hi=p['rows_half_open']
    target=z['solutions'][lo:hi].astype(np.int8)-1;free=~z['clue_mask'][lo:hi].astype(bool)
    unique={};checked=0
    for job in jobs:
        dest=OUT/job['job_id'];receipt=json.loads((dest/'receipt.json').read_text())
        verify(dest/'receipt.json')
        assert receipt['status']=='complete' and receipt['preflight_checks']==8
        for name,digest in (receipt['input_sha256']|receipt['source_sha256']).items():verify(ROOT/name,digest)
        assert len(receipt['checkpoints'])==1
        point=receipt['checkpoints'][0];assert point['cap']==262144
        directory=dest/'moves-262144'
        for name,digest in point['output_sha256'].items():verify(directory/name,digest)
        b=np.load(directory/'boards.npz')
        assert np.array_equal(b['solutions'],target) and np.array_equal(b['free'],free) and np.array_equal(b['puzzle'],np.arange(lo,hi))
        good=validity(b['boards'],target,free)
        assert np.array_equal(good,(b['boards']==target).all(1)) and np.array_equal(good,b['valid']) and np.array_equal(good,b['exact'])
        assert int(good.sum())==point['solved'] and int((b['stopped']&~good).sum())==point['wrong_stops']
        assert (b['moves']<=262144).all() and (b['moves'][~b['stopped']]==262144).all()
        state=torch.load(directory/'state.pt',weights_only=False,map_location='cpu')
        capture=torch.load(directory/'capture.pt',weights_only=True,map_location='cpu')
        returned=torch.where(state['stopped'][:,None],capture['cert_boards'],capture['best_boards']).numpy()
        assert np.array_equal(returned,b['boards']) and np.array_equal(state['stopped'],b['stopped'])
        assert np.array_equal(torch.where(state['stopped'],state['ok'],state['bestok']).numpy(),good)
        support=models[job['job_id']]
        adjacency=np.zeros((81,81),dtype=np.int64);adjacency[ii[support],jj[support]]=1;adjacency+=adjacency.T.copy()
        y=state['y'].numpy()
        fields=np.einsum('ij,hjv->hiv',adjacency,np.eye(9,dtype=np.int64)[y])
        energy=(y[:,ii[support]]==y[:,jj[support]]).sum(1)
        assert np.array_equal(fields,state['G'].numpy()) and np.array_equal(energy,state['P'].numpy())
        returned_energy=(returned[:,ii[support]]==returned[:,jj[support]]).sum(1)
        assert (returned_energy[b['stopped']]==0).all()
        unique[job['job_id']]=dict(good=good,stopped=b['stopped'],moves=b['moves'])
        checked+=len(good)
    arrays={};rows=[]
    for mapping in prepared['mappings']:
        assert np.array_equal(models[mapping['requested']],models[mapping['actual']])
        result=unique[mapping['actual']];good=result['good']
        arrays[mapping['seed'],mapping['arm']]=good.astype(float)
        rows.append(dict(seed=mapping['seed'],arm=mapping['arm'],actual_job=mapping['actual'],n=len(good),
            solved=int(good.sum()),accuracy_pct=float(good.mean()*100),wrong_stops=int((result['stopped']&~good).sum()),
            mean_moves=float(result['moves'].mean())))
    grouped=[]
    for arm in p['arms']:
        rs=[r for r in rows if r['arm']==arm]
        grouped.append(dict(arm=arm,mean_accuracy_pct=float(np.mean([r['accuracy_pct'] for r in rs])),
            per_seed_solved=[r['solved'] for r in rs],wrong_stops=sum(r['wrong_stops'] for r in rs)))
    delta=np.stack([arrays[s,p['primary_arm']]-arrays[s,p['control_arm']] for s in range(6)])
    rng=np.random.default_rng(20261114)
    boot=np.array([100*delta[rng.integers(6,size=6)][:,rng.integers(10000,size=10000)].mean() for _ in range(5000)])
    lower,upper=map(float,np.quantile(boot,[.025,.975]))
    wrong=next(r['wrong_stops'] for r in grouped if r['arm']==p['primary_arm'])
    decision='falsified' if wrong or upper<5 else 'supported' if lower>=5 else 'inconclusive'
    primary=dict(gain_pp=float(delta.mean()*100),paired95_interval_pp=[lower,upper],threshold_pp=5,
        wrong_stops=wrong,decision=decision)
    summary=dict(status='validated',utc=datetime.now(timezone.utc).isoformat(),primary=primary,grouped=grouped,
        rows=rows,graph_diagnostics=graphs,saved_boards_verified=checked,requested_cells=len(rows),unique_searches=len(jobs),
        model_mappings=prepared['mappings'],corruption_rng_replayed=True,integer_fields_and_energy_exact=True,
        input_sha256=evidence,limitations=p['limits'])
    PUBLIC.mkdir(parents=True,exist_ok=True)
    for path in (OUT/'audit.json',PUBLIC/'summary.json'):path.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for name,rs in [('per_seed.csv',rows),('graph_diagnostics.csv',graphs)]:
        with (PUBLIC/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');writer.writeheader();writer.writerows(rs)
    print(json.dumps(dict(primary=primary,grouped=grouped,saved_boards_verified=checked)))


if __name__=='__main__':main()
