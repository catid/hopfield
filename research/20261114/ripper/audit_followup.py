"""Saved-output audit for the frozen confirmation and low-data H1 studies."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/research/20261114/ripper'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('study',choices=['confirmation','lowdata']);a=ap.parse_args()
    ppath=HERE/f'protocol-{a.study}.json';p=json.loads(ppath.read_text());out=BASE/a.study
    n=p['rows_half_open'][1]-p['rows_half_open'][0];rows=[];arrays={};hashes={}
    for k in p['primary_k']:
      for seed in p['training_seeds']:
       for arm in p['arms']:
        d=out/f'jobs/k{k}-s{seed}-{arm}';r=json.loads((d/'receipt.json').read_text());assert r['status']=='complete'
        assert r['protocol_sha256']==sha(ppath) and r['wrapper_sha256']==sha(HERE/'followup.py')
        assert r['development_rows_half_open']==p['rows_half_open'] and r['test_rows']==n
        for name,h in r['output_sha256'].items():assert sha(d/name)==h;hashes[str((d/name).relative_to(ROOT))]=h
        hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
        entry=next(x for x in p['frozen_inputs'] if x['k']==k and x['seed']==seed)
        for name,h in entry['sha256'].items():
            q=ROOT/entry['input_dir']/name;assert sha(q)==h;hashes[str(q.relative_to(ROOT))]=h
        for name,h in r['source_sha256'].items():assert sha(ROOT/name)==h;hashes[name]=h
        z=np.load(d/'boards.npz');b=z['boards'];y=z['solutions'];f=z['free'];st=z['stopped'];moves=z['moves'];assert b.shape==(n,81)
        assert np.array_equal(z['puzzle'],np.arange(*p['rows_half_open']))
        exact=(b==y).all(1);valid=[]
        for board,target,free in zip(b,y,f):
            q=board.reshape(9,9);want=set(range(9));good=all(set(q[i].tolist())==want and set(q[:,i].tolist())==want for i in range(9))
            good=good and all(set(q[i:i+3,j:j+3].reshape(-1).tolist())==want for i in [0,3,6] for j in [0,3,6]) and bool(np.all((board==target)|free));valid.append(good)
        assert np.array_equal(exact,valid) and np.array_equal(exact,z['exact']) and np.array_equal(valid,z['valid'])
        assert r['solved']==int(exact.sum()) and r['wrong_stops']==int((st&~exact).sum()) and r['budget_failures']==int((~st).sum())
        assert np.all((moves>0)&(moves<=p['move_cap_per_chain'])) and np.all(moves[~st]==r['terminal_move'])
        assert r['terminal_move']==p['move_cap_per_chain'] or np.all(st)
        rr=json.loads((d/'result.json').read_text());assert np.array_equal(rr['per_puzzle']['solved'],exact)
        assert np.array_equal(rr['per_puzzle']['stopped'],st) and np.allclose(rr['per_puzzle']['flops'],moves*r['fl_move']*4,rtol=0,atol=0)
        arrays[k,seed,arm]=exact.astype(float)
        rows.append({key:r[key] for key in ['job_id','k','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s']})
    diff=np.array([[arrays[k,s,'projected']-arrays[k,s,'original'] for k in p['primary_k']] for s in p['training_seeds']])
    rng=np.random.default_rng(20261114);boot=[]
    for _ in range(5000):
        s=rng.integers(0,len(p['training_seeds']),len(p['training_seeds']));i=rng.integers(0,n,n);boot.append(float(100*diff[s][:,:,i].mean()))
    ci=np.quantile(boot,[.025,.975]).tolist();effect=float(100*diff.mean())
    decision='supported_for_frozen_protocol' if ci[0]>=5 else 'falsified_for_frozen_protocol' if ci[1]<5 else 'inconclusive'
    grouped=[]
    for k in p['primary_k']:
      for arm in p['arms']:
        rs=[r for r in rows if r['k']==k and r['arm']==arm];grouped.append({'k':k,'arm':arm,'solved_per_seed':[r['solved'] for r in rs],'mean_accuracy_pct':float(100*np.mean([r['solved']/n for r in rs])),'wrong_stops':sum(r['wrong_stops'] for r in rs)})
    for q in [ppath,HERE/'followup.py',HERE/'PROTOCOL_ERRATA.json',Path(__file__),BASE/'followup-preflight.json']:
        hashes[str(q.relative_to(ROOT))]=sha(q)
    rec={'status':'validated','study':a.study,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':len(rows),'n_per_run':n,'saved_boards_independently_verified':len(rows)*n,'primary_effect_pp':effect,'two_way_bootstrap_95pct_interval_pp':ci,'decision':decision,'primary_per_seed_pp':(100*diff.mean((1,2))).tolist(),'new_seeds3to5_effect_pp':float(100*diff[3:].mean()) if len(diff)>3 else None,'grouped':grouped,'rows':rows,'input_sha256':hashes,'limitations':['Approximate paired bootstrap with only'+str(len(p['training_seeds']))+'independent training-seed clusters.','Development data screen/confirmation, not untouched external-test generalization.','Only262144movesperchain; not paper-budget accuracy.','Frozen protocol has documented inherited-prose errata; controlling numeric fields and hypothesis were executed unchanged.'],'new_inference':False}
    (out/'audit.json').write_text(json.dumps(rec,indent=2)+'\n')
    lines=[f'# H1 {a.study}: validated results','',f'All {len(rows)} jobs and {len(rows)*n:,} saved boards validated. Primary projected-minus-original effect **{effect:.4f}pp**, paired two-way bootstrap95% interval **[{ci[0]:.4f}, {ci[1]:.4f}]pp**; preregistered5pp hypothesis: **{decision}**.','',f"Development rows{p['rows_half_open']}, {n} puzzles per run, seeds{p['training_seeds']}, fixed262144moves/chain and4chains. The frozen protocol bytes are retained; [editorial errata](PROTOCOL_ERRATA.json) explain inherited stale prose. The numerical cohort/arms were unchanged.",'','| k | Arm | Solved per seed | Mean accuracy (%) | Wrong stops |','|---|---|---|---:|---:|','| — | BP AdamW: outside post-fit protocol | Not run | — | — |']
    for r in grouped:lines.append(f"| {r['k']} | {r['arm']} | {' / '.join(map(str,r['solved_per_seed']))} | {r['mean_accuracy_pct']:.4f} | {r['wrong_stops']} |")
    if rec['new_seeds3to5_effect_pp'] is not None:lines+=['',f"Newly evaluated model seeds3–5 alone show **{rec['new_seeds3to5_effect_pp']:.4f}pp** mean improvement. Seeds0–2 reuse the discovery models on disjoint rows; all6 are included as preregistered."]
    lines+=['','Intervals are exploratory with few model seeds; these results do not establish performance on an untouched external distribution. Search RNG streams can diverge after model-specific compaction. The saved boards permit full independent validity checks.','']
    (HERE/f'REPORT-{a.study}.md').write_text('\n'.join(lines))
    print(json.dumps({key:rec[key] for key in ['status','study','jobs','saved_boards_independently_verified','primary_effect_pp','two_way_bootstrap_95pct_interval_pp','new_seeds3to5_effect_pp','decision']}))

if __name__=='__main__':main()
