"""Independent saved-board audit and frozen held-out inference, CPU only."""
import csv
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_mechanism import interval

OUT=ROOT/'artifacts/research/20261114/ripper/heldout'
PUBLIC=ROOT/'results/research/ripper'


def main():
    p=json.loads((HERE/'protocol-heldout.json').read_text());arrays={};rows=[];hashes={}
    q=json.loads((OUT/'queue.json').read_text());assert len(q)==25 and all(j['status']=='complete' for j in q)
    for j in q:
        d=OUT/'jobs'/j['job_id'];r=json.loads((d/'receipt.json').read_text())
        assert r['status']=='complete' and r['protocol_sha256']==sha(HERE/'protocol-heldout.json')
        assert r['wrapper_sha256']==sha(HERE/'heldout.py') and r['backend_sha256']==sha(HERE/'sparse_backend.py') and r['capture_sha256']==sha(HERE/'run.py')
        assert r['test_rows']==10000 and r['test_rows_half_open']==[20000,30000]
        for name,h in r['output_sha256'].items():assert sha(d/name)==h
        z=np.load(d/'boards.npz');b=z['boards'];sol=z['solutions'];free=z['free'];st=z['stopped'];moves=z['moves']
        assert b.shape==(10000,81) and np.array_equal(z['puzzle'],np.arange(20000,30000))
        # Independent count-based checker, avoiding the inference runner's sorting checker.
        boards=b.reshape(-1,9,9);boxes=boards.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
        valid=((b==sol)|free).all(1)
        for digit in range(9):
            valid &= ((boards==digit).sum(1)==1).all(1)
            valid &= ((boards==digit).sum(2)==1).all(1)
            valid &= ((boxes==digit).sum(2)==1).all(1)
        exact=(b==sol).all(1)
        assert np.array_equal(valid,exact) and np.array_equal(z['valid'],valid) and np.array_equal(z['exact'],exact)
        assert int(exact.sum())==r['solved'] and int((st&~exact).sum())==r['wrong_stops']
        assert int((~st).sum())==r['budget_failures'] and np.all((moves>0)&(moves<=262144))
        result=json.loads((d/'result.json').read_text())['per_puzzle'];assert np.array_equal(result['solved'],exact) and np.array_equal(result['stopped'],st)
        assert np.allclose(result['flops'],moves*r['fl_move']*4,rtol=0,atol=0)
        arrays[r['k'],r['seed'],r['arm']]=exact.astype(float)
        rows.append({key:r[key] for key in ['job_id','k','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s']})
        hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
    d1=OUT/'jobs/k17-s0-canonical_repulsive_general';d2=OUT/'jobs/k17-s0-canonical_repulsive_sparse'
    a,b=np.load(d1/'boards.npz'),np.load(d2/'boards.npz')
    assert a.files==b.files and all(np.array_equal(a[n],b[n]) for n in a.files)
    u=torch.load(d1/'final_state.pt',weights_only=True,map_location='cpu');v=torch.load(d2/'final_state.pt',weights_only=True,map_location='cpu')
    assert u.keys()==v.keys()
    fields=[]
    for key in u:
        if key=='elapsed':continue
        assert torch.equal(u[key],v[key]) if torch.is_tensor(u[key]) else u[key]==v[key],key
        fields.append(key)
    diff=np.array([[arrays[k,s,'canonical_repulsive_sparse']-arrays[k,s,'original_general'] for k in p['k']] for s in p['training_seeds']])
    ci=interval(diff);effect=float(100*diff.mean());decision='supported_for_frozen_protocol' if ci[0]>=5 else 'falsified_for_frozen_protocol' if ci[1]<5 else 'inconclusive'
    grouped=[]
    for k in p['k']:
        for arm in p['arms']:
            rs=[r for r in rows if r['k']==k and r['arm']==arm]
            grouped.append({'k':k,'arm':arm,'solved_per_seed':[r['solved'] for r in rs],'mean_accuracy_pct':float(np.mean([r['solved']/100 for r in rs])),'wrong_stops':sum(r['wrong_stops'] for r in rs),'n_per_seed':10000,'seeds':6})
    wrong=sum(r['wrong_stops'] for r in rows if r['arm']=='canonical_repulsive_sparse')
    rec={'status':'validated','utc':now(),'jobs':25,'saved_boards_verified':250000,'primary_effect_pp':effect,'primary_ci95_pp':ci,'primary_decision':decision,'primary_per_seed_pp':(100*diff.mean((1,2))).tolist(),'repulsive_wrong_stops':wrong,'zero_wrong_stop_component':'passed_observed_cohort' if wrong==0 else 'falsified','integration_exact_boards_stops_moves':True,'integration_exact_state_fields_excluding_elapsed':fields,'integration_state_field_count':len(fields),'grouped':grouped,'rows':rows,'input_sha256':hashes,'protocol_sha256':sha(HERE/'protocol-heldout.json'),'new_inference':False,'limitations':['Six independent training-seed clusters; approximate paired bootstrap.','Fixed10000-puzzle research held-out cohort, same dataset family as development; no external-distribution claim.','Fixed short262144move/chain budget,not full paper-budget accuracy.','Original and canonical use different backends; full-cohort k17seed0 plus18model development parity gates validate equivalence within measured scope.','Reported job walltimes were not paired exclusiveGPU benchmarks and do not support cross-arm speed claims.']}
    save(OUT/'audit.json',rec)
    # Public export is a separate explicit flag, allowing another agent to freeze a shared-cohort study first.
    import sys
    if '--export' not in sys.argv:
        print('Audited250000 savedboards and fullcohort24statefield parity; results held locally.');return
    export(rec)


def export(rec):
    rows=rec['rows'];grouped=rec['grouped'];effect=rec['primary_effect_pp'];ci=rec['primary_ci95_pp']
    wrong=rec['repulsive_wrong_stops'];decision=rec['primary_decision']
    save(PUBLIC/'heldout.json',rec)
    with (PUBLIC/'heldout_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=json.loads((PUBLIC/'summary.json').read_text());summary['heldout']=rec;save(PUBLIC/'summary.json',summary)
    lines=['# H1e: fixed held-out canonical repulsion test','',f'The fixed canonical-repulsive method improves mean solve accuracy by **{effect:.4f} percentage points**, approximate paired95% interval **[{ci[0]:.4f}, {ci[1]:.4f}]**. The frozen≥5point hypothesis is **{decision}**. Repulsive wrongstops: **{wrong}** across120000model-puzzle evaluations.','',
           'All250000saved boards were independently checked by digit counts in rows, columns and boxes plus preservation of clues. The prespecified10000-puzzle k17seed0 sparse/general integration pair has identical full returned boards, stop times, and24finalstate fields excluding elapsed walltime.','',
           '| k | Method | Mean accuracy (%) | Wrong stops |','|---:|---|---:|---:|','| — | BP AdamW: outside post-training intervention protocol; not run | — | — |']
    for r in grouped:lines.append(f'| {r["k"]} | {r["arm"]} | {r["mean_accuracy_pct"]:.4f} | {r["wrong_stops"]} |')
    lines+=['','This uses fixed full-all rows[20000,30000), k17/21, six model/search seeds0–5,262144moves perchain,fourchains. Models, transformations, temperatures and stopping parameters were frozen before cohort results. This cohort is held out from this research development sequence, not claimed to be a pristine external distribution. No test tuning or modelselection occurred.','',
            '[Frozen protocol](protocol-heldout.json) · [Runner](heldout.py) · [Audit](audit_heldout.py) · [Public results](../../../results/research/ripper/heldout.json)','']
    (HERE/'REPORT-heldout.md').write_text('\n'.join(lines))
    with (PUBLIC/'README.md').open('a') as f:
        f.write(f'\nThe sparse backend audit now reports **2.824×** geometric-mean warm inference speedup with exact measured state/board parity; see `sparse.json` and `sparse_timings.csv`. The fixed held-out study reports **+{effect:.2f}pp**,95% interval **{ci[0]:.2f}–{ci[1]:.2f}pp**, with **{wrong}** repulsive wrong stops. See `heldout.json` and `heldout_per_seed.csv`.\n')
    print('Public heldout exports written.')


if __name__=='__main__':
    import sys
    if '--export-existing' in sys.argv:
        export(json.loads((OUT/'audit.json').read_text()))
    else:main()
