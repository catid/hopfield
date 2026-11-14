"""H1h independent board/data audit and fixed confirmation contrasts."""
import csv
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_mechanism import interval

OUT=ROOT/'artifacts/research/20261114/ripper/finaltest'
PUBLIC=ROOT/'results/research/ripper'


def main():
    p=json.loads((HERE/'protocol-finaltest.json').read_text());arrays={};rows=[];hashes={}
    q=json.loads((OUT/'queue.json').read_text());assert len(q)==54 and all(j['status']=='complete' for j in q)
    source=ROOT/'upstream/data/full-test/full-all.npz'
    data=np.load(source);refsol=data['solutions'][30000:40000].astype(np.int8)-1;reffree=~data['clue_mask'][30000:40000].astype(bool)
    hashes[str(source.relative_to(ROOT))]=sha(source)
    continuations=[]
    for j in q:
        d=OUT/'jobs'/j['job_id'];r=json.loads((d/'receipt.json').read_text())
        assert r['status']=='complete' and r['protocol_sha256']==sha(HERE/'protocol-finaltest.json')
        assert r['wrapper_sha256']==sha(HERE/'finaltest.py') and r['backend_sha256']==sha(HERE/'sparse_backend.py') and r['capture_sha256']==sha(HERE/'run.py')
        assert r['test_rows']==10000 and r['test_rows_half_open']==[30000,40000]
        for name,h in r['output_sha256'].items():assert sha(d/name)==h
        z=np.load(d/'boards.npz');b=z['boards'];sol=z['solutions'];free=z['free'];st=z['stopped'];moves=z['moves']
        assert b.shape==(10000,81) and np.array_equal(z['puzzle'],np.arange(30000,40000))
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
        if r['arm']=='normalized_sparse':
            state=torch.load(d/'final_state.pt',weights_only=True,map_location='cpu');capture=torch.load(d/'capture_state.pt',weights_only=True,map_location='cpu')
            assert state['t']==262144 and len(state)==25
            selected=torch.where(state['stopped'][:,None],capture['cert_boards'],capture['best_boards']).numpy()
            assert np.array_equal(selected,b) and np.array_equal(state['stopped'].numpy(),st)
            continuations.append({'k':r['k'],'seed':r['seed'],'state':str((d/'final_state.pt').relative_to(ROOT)),'state_sha256':sha(d/'final_state.pt'),'capture':str((d/'capture_state.pt').relative_to(ROOT)),'capture_sha256':sha(d/'capture_state.pt'),'active_puzzles':len(state['act'])})
    grouped=[];effects=[]
    for k in p['k']:
        for arm in p['arms']:
            rs=[r for r in rows if r['k']==k and r['arm']==arm]
            grouped.append({'k':k,'arm':arm,'solved_per_seed':[r['solved'] for r in rs],'mean_accuracy_pct':float(np.mean([r['solved']/100 for r in rs])),'wrong_stops':sum(r['wrong_stops'] for r in rs),'n_per_seed':10000,'seeds':6})
        for control in ['canonical_repulsive_sparse','original_general']:
            diff=np.array([[arrays[k,s,'normalized_sparse']-arrays[k,s,control]] for s in p['training_seeds']])
            ci=interval(diff);effect=float(100*diff.mean());threshold=30 if k==10 else 5
            decision=('supported_for_frozen_protocol' if ci[0]>=threshold else 'falsified_for_frozen_protocol' if ci[1]<threshold else 'inconclusive') if control=='canonical_repulsive_sparse' else 'descriptive_control_comparison'
            effects.append({'k':k,'contrast':['normalized_sparse',control],'effect_pp':effect,'ci95_pp':ci,'decision':decision,'per_seed_pp':(100*diff.mean((1,2))).tolist()})
    primary=effects[0];wrong=sum(r['wrong_stops'] for r in rows if r['arm']=='normalized_sparse')
    rec={'status':'validated','utc':now(),'jobs':54,'saved_boards_verified':540000,'primary_effect_pp':primary['effect_pp'],'primary_ci95_pp':primary['ci95_pp'],'primary_decision':primary['decision'],'normalized_wrong_stops':wrong,'zero_wrong_stop_component':'passed_observed_cohort' if wrong==0 else 'falsified','effects':effects,'grouped':grouped,'rows':rows,'continuation_checkpoints':continuations,'input_sha256':hashes,'protocol_sha256':sha(HERE/'protocol-finaltest.json'),'new_inference':False,'limitations':['All normalized graphs are identical up to global weight scale; repeated source fits do not create independent structural evidence.','Variation among normalized runs is largely search seed and floating-point arithmetic.','Approximate paired bootstrap over seed clusters and shared puzzle indices.','Fixed short262144move/chain budget; full paper-budget accuracy is not tested here.','New fixed10000-puzzle cohort from same official corpus, not external-distribution validation.','Job walltimes were not paired isolated benchmarks; no cross-arm speed claim.']}
    save(OUT/'audit.json',rec);save(PUBLIC/'finaltest.json',rec)
    with (PUBLIC/'finaltest_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=json.loads((PUBLIC/'summary.json').read_text());summary['normalized_confirmation']=rec;save(PUBLIC/'summary.json',summary)
    lines=['# H1h: independent confirmation of the fixed normalized method','',f'On the new fixed test cohort, normalized k10 improves accuracy by **{primary["effect_pp"]:.4f} percentage points** over canonical repulsion, approximate paired 95% interval **[{primary["ci95_pp"][0]:.4f}, {primary["ci95_pp"][1]:.4f}]**. The frozen ≥30-point hypothesis is **{primary["decision"]}**. There were **{wrong} wrong stops** across 180,000 normalized evaluations.','',
           'All 540,000 saved boards were independently checked by counting digits in rows, columns and boxes and preserving clues. Saved solutions and clue masks were compared with the specified source dataset slice. All 18 normalized trajectory checkpoints and their capture arrays were verified for later continuation.','',
           '| Training puzzles | Method | Mean accuracy (%) | Wrong stops |','|---:|---|---:|---:|','| — | BP AdamW: outside post-training intervention protocol; not run | — | — |']
    for r in grouped:lines.append(f'| {r["k"]} | {r["arm"]} | {r["mean_accuracy_pct"]:.4f} | {r["wrong_stops"]} |')
    lines+=['','The method uses digit-label averaging, nonnegative canonical equal-digit couplings, largest-gap support selection, uniform retained weights, and temperature/focus normalized to that common weight. It uses only learned parameters and training statistics; no Sudoku peer graph is supplied. Every transformation and parameter was fixed before this cohort was evaluated.','',
            'All recovered normalized graphs are identical up to a global coupling scale. Repeated source fits do not create independent structural evidence; transformed-model variation is largely search-seed and numerical variation.','',
            'Protocol: full-all rows [30,000, 40,000), k10/17/21, six seeds, four chains, 262,144 moves per chain. This is a fixed short-budget test within the same official corpus, not a new external distribution.','',
            '[Frozen protocol](protocol-finaltest.json) · [Runner](finaltest.py) · [Audit](audit_finaltest.py) · [Tracked results](../../../results/research/ripper/finaltest.json)','']
    (HERE/'REPORT-finaltest.md').write_text('\n'.join(lines))
    print(json.dumps(effects))


if __name__=='__main__':main()
