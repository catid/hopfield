"""H1j no-optimizer count-control inference audit."""
import csv
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_mechanism import interval

OUT=ROOT/'artifacts/research/20261114/ripper/count_search'
PUBLIC=ROOT/'results/research/ripper'


def main():
    p=json.loads((HERE/'protocol-count-search.json').read_text());arrays={};rows=[];hashes={}
    q=json.loads((OUT/'queue.json').read_text());assert len(q)==24 and all(j['status']=='complete' for j in q)
    source=ROOT/'upstream/data/full-test/full-all.npz'
    data=np.load(source);refsol=data['solutions'][40000:50000].astype(np.int8)-1;reffree=~data['clue_mask'][40000:50000].astype(bool)
    hashes[str(source.relative_to(ROOT))]=sha(source)
    continuations=[]
    for j in q:
        d=OUT/'jobs'/j['job_id'];r=json.loads((d/'receipt.json').read_text())
        assert r['status']=='complete' and r['protocol_sha256']==sha(HERE/'protocol-count-search.json')
        assert r['wrapper_sha256']==sha(HERE/'count_search.py') and r['backend_sha256']==sha(HERE/'sparse_backend.py') and r['capture_sha256']==sha(HERE/'run.py')
        assert r['test_rows']==10000 and r['test_rows_half_open']==[40000,50000]
        for name,h in r['output_sha256'].items():assert sha(d/name)==h
        z=np.load(d/'boards.npz');b=z['boards'];sol=z['solutions'];free=z['free'];st=z['stopped'];moves=z['moves']
        assert b.shape==(10000,81) and np.array_equal(z['puzzle'],np.arange(40000,50000))
        # Independent count-based checker, avoiding the inference runner's sorting checker.
        boards=b.reshape(-1,9,9);boxes=boards.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
        valid=((b==sol)|free).all(1)
        for digit in range(9):
            valid &= ((boards==digit).sum(1)==1).all(1)
            valid &= ((boards==digit).sum(2)==1).all(1)
            valid &= ((boxes==digit).sum(2)==1).all(1)
        assert np.array_equal(sol,refsol) and np.array_equal(free,reffree)
        exact=(b==sol).all(1)
        assert np.array_equal(valid,exact) and np.array_equal(z['valid'],valid) and np.array_equal(z['exact'],exact)
        assert int(exact.sum())==r['solved'] and int((st&~exact).sum())==r['wrong_stops']
        assert int((~st).sum())==r['budget_failures'] and np.all((moves>0)&(moves<=262144))
        result=json.loads((d/'result.json').read_text())['per_puzzle'];assert np.array_equal(result['solved'],exact) and np.array_equal(result['stopped'],st)
        assert np.allclose(result['flops'],moves*r['fl_move']*4,rtol=0,atol=0)
        arrays[r['k'],r['seed'],r['arm']]=exact.astype(float)
        rows.append({key:r[key] for key in ['job_id','k','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s']})
        hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
        if r['arm']=='count_sparse':
            state=torch.load(d/'final_state.pt',weights_only=True,map_location='cpu');capture=torch.load(d/'capture_state.pt',weights_only=True,map_location='cpu')
            assert state['t']==262144 and len(state)==25
            selected=torch.where(state['stopped'][:,None],capture['cert_boards'],capture['best_boards']).numpy()
            assert np.array_equal(selected,b) and np.array_equal(state['stopped'].numpy(),st)
            continuations.append({'k':r['k'],'seed':r['seed'],'state':str((d/'final_state.pt').relative_to(ROOT)),'state_sha256':sha(d/'final_state.pt'),'capture':str((d/'capture_state.pt').relative_to(ROOT)),'capture_sha256':sha(d/'capture_state.pt'),'active_puzzles':len(state['act'])})
    # Deduplication is justified by exact input support equality, not approximate search outcomes.
    base=ROOT/'artifacts/research/20261114/ripper';counts=[np.load(base/f'count/k{k}-s{s}/counts.npz')['support'] for k in p['k'] for s in p['training_seeds']]
    assert all(np.array_equal(counts[0],x) for x in counts)
    for seed in p['training_seeds']:
        rr=json.loads((OUT/f'jobs/k10-s{seed}-count_sparse/receipt.json').read_text())
        assert rr['used_hbar']==1 and rr['statistics']['tau_train']==0 and rr['statistics']['ebar']==0
    diff=np.array([[arrays[10,s,'count_sparse']-arrays[k,s,'normalized_sparse'] for k in p['k']] for s in p['training_seeds']])
    ci=interval(diff);effect=float(100*diff.mean());decision='noninferior_within_2pp' if ci[0]>=-2 else 'inferior_beyond_2pp' if ci[1]<-2 else 'inconclusive'
    grouped=[]
    for k in p['k']:
        for arm in p['arms']:
            rs=[r for r in rows if r['k']==(10 if arm=='count_sparse' else k) and r['arm']==arm]
            grouped.append({'k':k,'arm':arm,'solved_per_seed':[r['solved'] for r in rs],'mean_accuracy_pct':float(np.mean([r['solved']/100 for r in rs])),'wrong_stops':sum(r['wrong_stops'] for r in rs),'n_per_seed':10000,'seeds':6,'reused_identical_count_runs':arm=='count_sparse' and k!=10})
    countwrong=sum(r['wrong_stops'] for r in rows if r['arm']=='count_sparse')
    rec={'status':'validated','utc':now(),'jobs':24,'saved_boards_verified':240000,'primary_effect_pp':effect,'primary_ci95_pp':ci,'decision':decision,'primary_per_seed_pp':(100*diff.mean((1,2))).tolist(),'count_wrong_stops':countwrong,'grouped':grouped,'rows':rows,'count_checkpoints':continuations,'deduplicated_count_graphs':18,'executed_count_searches':6,'input_sha256':hashes,'protocol_sha256':sha(HERE/'protocol-count-search.json'),'new_inference':False,'interpretation':'Counting training equality events recovers the same graph without optimizer training; unit1 inference is compared directly, not assumed equal from real-arithmetic scale equivalence.','limitations':['Count trajectories are reused across k only because graph,unit,statistics,seed,puzzles andsearcharguments are identical.','18source datasets do not provide18independent recoveredstructures.','Alltrainingdatasets include99storedsymmetryvariants/base,includingSudokuspatialaugmentations.','Fixed new10000-puzzle officialcorpuscohort,not externaldistribution.','No isolatedhardware speedcomparison betweentrainingandcounting is made.']}
    save(OUT/'audit.json',rec);save(PUBLIC/'count_search.json',rec)
    with (PUBLIC/'count_search_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=json.loads((PUBLIC/'summary.json').read_text());summary['count_graph']=json.loads((PUBLIC/'count.json').read_text());summary['count_inference']=rec;save(PUBLIC/'summary.json',summary)
    countacc=next(r['mean_accuracy_pct'] for r in grouped if r['arm']=='count_sparse')
    lines=['# H1i/H1j: a no-optimizer equality counter recovers the same graph','',
        f'Connecting cells that never share a training digit recovers the same 810-edge graph as the learned transformations on all 18 augmented training datasets. **No optimizer is needed for this structural recovery.** On the new fixed 10,000-puzzle cohort, the unit-weight count graph solves **{countacc:.4f}%**, with **{countwrong} wrong stops** across 60,000 unique search evaluations.','',
        f'Its mean accuracy difference from the learned normalized models is **{effect:.4f} percentage points**, approximate paired 95% interval **[{ci[0]:.4f}, {ci[1]:.4f}]**. The frozen two-point noninferiority result is **{decision}**. All 240,000 saved boards were independently verified against digit-count/clue constraints and the source dataset slice.','',
        '| Base training puzzles | Method | Mean accuracy (%) | Wrong stops |','|---:|---|---:|---:|','| — | BP AdamW: outside this intervention/no-fit control protocol; not run | — | — |']
    for r in grouped:lines.append(f'| {r["k"]} | {r["arm"]} | {r["mean_accuracy_pct"]:.4f} | {r["wrong_stops"]} |')
    lines+=['','The count graph is constructed from equality events in training solutions only. The Sudoku peer graph is consulted only afterward for auditing. Each unique base puzzle has 99 stored symmetry variants, including valid Sudoku spatial transformations. Therefore this finding does not show that 10 unaugmented puzzles recover the graph; augmentation supplies task structure. Digit permutations preserve equality events, so digit-only augmentation cannot change a never-equal graph.','',
        'All 18 count graphs are identical. For each search seed, graph, unit, tau/ebar, puzzle cohort and all search arguments are byte-identical across training sizes; only six count searches were executed and explicitly reused across k. Learned normalized weights retain different floating coefficients, so all 18 of those searches were measured. Neither duplicate labels nor repeated source fits are counted as independent structures.','',
        'Inference protocol: full-all rows [40,000, 50,000), six seeds, four chains, 262,144 moves per chain. This is a short-budget comparison within the official corpus.','',
        '[Count protocol](protocol-count.json) · [Count implementation](count_graph.py) · [Inference protocol](protocol-count-search.json) · [Inference runner](count_search.py) · [Audit](audit_count_search.py) · [Tracked support evidence](../../../results/research/ripper/count.json) · [Tracked inference results](../../../results/research/ripper/count_search.json)','']
    (HERE/'REPORT-count.md').write_text('\n'.join(lines));print(json.dumps({'effect_pp':effect,'ci95_pp':ci,'decision':decision,'count_accuracy_pct':countacc,'wrong_stops':countwrong}))


if __name__=='__main__':main()
