"""Saved-output H1f audit; geometry is used only after the frozen transforms."""
import csv
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_mechanism import interval

BASE=ROOT/'artifacts/research/20261114/ripper'
OUT=BASE/'gap'
PUBLIC=ROOT/'results/research/ripper'


def main():
    p=json.loads((HERE/'protocol-gap.json').read_text());arrays={};rows=[];hashes={};graph=[]
    q=json.loads((OUT/'queue.json').read_text());assert len(q)==54 and all(j['status']=='complete' for j in q)
    pair=torch.triu_indices(81,81,1);r=torch.arange(81)//9;c=torch.arange(81)%9
    peers=(r[:,None]==r[None])|(c[:,None]==c[None])|((r[:,None]//3==r[None]//3)&(c[:,None]//3==c[None]//3));truth=peers[pair[0],pair[1]];eye=torch.eye(9,dtype=torch.bool)
    for j in q:
        d=OUT/'jobs'/j['job_id'];r=json.loads((d/'receipt.json').read_text());k,s,arm=r['k'],r['seed'],r['arm']
        assert r['status']=='complete' and r['protocol_sha256']==sha(HERE/'protocol-gap.json') and r['wrapper_sha256']==sha(HERE/'gap.py')
        assert r['backend_sha256']==sha(HERE/'sparse_backend.py') and r['preflight_checks']==5 and r['preflight_state_fields']==25
        assert r['rows_half_open']==[6656,8704] and r['test_rows']==2048
        for name,h in r['output_sha256'].items():assert sha(d/name)==h
        z=np.load(d/'boards.npz');b=z['boards'];target=z['solutions'];free=z['free'];st=z['stopped'];moves=z['moves']
        assert b.shape==(2048,81) and np.array_equal(z['puzzle'],np.arange(6656,8704))
        boards=b.reshape(-1,9,9);boxes=boards.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9);valid=((b==target)|free).all(1)
        for digit in range(9):
            valid &= ((boards==digit).sum(1)==1).all(1)
            valid &= ((boards==digit).sum(2)==1).all(1)
            valid &= ((boxes==digit).sum(2)==1).all(1)
        exact=(b==target).all(1);assert np.array_equal(valid,exact) and np.array_equal(z['valid'],valid) and np.array_equal(z['exact'],exact)
        assert int(exact.sum())==r['solved'] and int((st&~exact).sum())==r['wrong_stops']
        assert np.all((moves>0)&(moves<=262144)) and int((~st).sum())==r['budget_failures']
        result=json.loads((d/'result.json').read_text())['per_puzzle'];assert np.array_equal(result['solved'],exact) and np.array_equal(result['stopped'],st)
        assert np.allclose(result['flops'],moves*r['fl_move']*4,rtol=0,atol=0)
        src=BASE/f'mechanism/jobs/k{k}-s{s}-canonical_repulsive';assert sha(src/'searched_model.pt')==r['source_model_sha256']
        before=torch.load(src/'searched_model.pt',weights_only=True,map_location='cpu')['W'][:,0,0]
        positive=before[before>0].sort().values;ix=int((positive[1:]-positive[:-1]).argmax());threshold=float((positive[ix].double()+positive[ix+1].double())/2);keep=before>threshold
        assert threshold==r['transform']['threshold'] and int(keep.sum())==r['transform']['retained']
        expected=before.clone()
        if arm=='gap_pruned':expected[~keep]=0
        elif arm=='gap_uniform':expected=torch.where(keep,torch.full_like(before,r['transform']['uniform_level']),torch.zeros_like(before))
        else:assert arm=='repulsive'
        model=torch.load(d/'searched_model.pt',weights_only=True,map_location='cpu');assert torch.equal(model['W'],expected[:,None,None]*eye) and not torch.any(model['B'])
        if arm=='gap_uniform':assert abs(float(before[keep].double().mean())-r['transform']['uniform_level'])<2e-7
        prior=json.loads((src/'receipt.json').read_text());assert prior['used_hbar']==r['used_hbar']
        graph.append({'k':k,'seed':s,'arm':arm,'retained_edges':int((expected>0).sum()),'false_positive_edges':int(((expected>0)&~truth).sum()),'missing_true_edges':int(((expected==0)&truth).sum()),'exact_peer_graph':bool(torch.equal(expected>0,truth)),'threshold':threshold,'gap':r['transform']['gap'],'minimum_true_weight':float(expected[truth].min()),'tau_plus_tolerance':r['statistics']['tau_train']+.001})
        arrays[k,s,arm]=exact.astype(float);rows.append({key:r[key] for key in ['job_id','k','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s']});hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
    diff=np.array([[arrays[10,s,'gap_pruned']-arrays[10,s,'repulsive']] for s in range(6)]);ci=interval(diff);effect=float(100*diff.mean())
    decision='supported_for_frozen_protocol' if ci[0]>=20 else 'falsified_for_frozen_protocol' if ci[1]<20 else 'inconclusive'
    grouped=[];secondary=[]
    for k in p['k']:
        for arm in p['arms']:
            rs=[r for r in rows if r['k']==k and r['arm']==arm];grouped.append({'k':k,'arm':arm,'solved_per_seed':[r['solved'] for r in rs],'mean_accuracy_pct':float(np.mean([r['solved']/20.48 for r in rs])),'wrong_stops':sum(r['wrong_stops'] for r in rs),'n_per_seed':2048,'seeds':6})
        for a,b in [('gap_pruned','repulsive'),('gap_uniform','gap_pruned')]:
            x=np.array([[arrays[k,s,a]-arrays[k,s,b]] for s in range(6)])
            secondary.append({'k':k,'contrast':[a,b],'effect_pp':float(100*x.mean()),'ci95_pp':interval(x)})
    rec={'status':'validated','utc':now(),'jobs':54,'saved_boards_verified':110592,'primary_effect_pp':effect,'primary_ci95_pp':ci,'primary_decision':decision,'primary_per_seed_pp':(100*diff.mean((1,2))).tolist(),'secondary':secondary,'grouped':grouped,'rows':rows,'graph_diagnostics':graph,'pruned_models_exact_peer_graph':sum(g['exact_peer_graph'] for g in graph if g['arm']=='gap_pruned'),'uniform_models_exact_peer_graph':sum(g['exact_peer_graph'] for g in graph if g['arm']=='gap_uniform'),'input_sha256':hashes,'new_inference':False,'limitations':['Gap rule was motivated by training-weight inspection, including observed810edges; graph recovery is exploratory post-selection diagnostic.','No geometry supplied to fitting,transform orsearch; groundtruth graph used only in this audit.','Six model seeds and shared development puzzles; bootstrap remains approximate.','Fixed short search budget; no full paper-budget accuracy claim.','All transformedmodels use original raw-projection thermal scale; effects cannot be attributed to rescaled temperatures.','Job walltimes are not isolated paired benchmarks.']}
    save(OUT/'audit.json',rec);save(PUBLIC/'gap.json',rec)
    with (PUBLIC/'gap_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=json.loads((PUBLIC/'summary.json').read_text());summary['gap_pruning']=rec;save(PUBLIC/'summary.json',summary)
    lines=['# H1f: prune weak surplus couplings using the learned coefficient gap','',f'At k10, largest-gap pruning changes solve accuracy by **{effect:.4f} percentage points**, approximate paired95% interval **[{ci[0]:.4f}, {ci[1]:.4f}]**. The frozen≥20point hypothesis is **{decision}**. All110592saved boards were independently checked.','',
           '| k | Method | Mean accuracy (%) | Wrong stops |','|---:|---|---:|---:|','| — | BP AdamW: outside post-training intervention protocol; not run | — | — |']
    for r in grouped:lines.append(f'| {r["k"]} | {r["arm"]} | {r["mean_accuracy_pct"]:.4f} | {r["wrong_stops"]} |')
    lines+=['','The gap rule uses sorted positive learned coefficients only, retains values above the midpoint of the first largest gap, and never queries a Sudoku peer graph. The separate uniform arm replaces retained coefficients with their mean. Both retain the same raw-projection temperature/focus scale and recompute only training-derived stopping/focus statistics.','',
            f'Post-selection graph audit: all{rec["pruned_models_exact_peer_graph"]}/18 pruned models recover the exact810-edge Sudoku peer graph, including allsix k10fits. Training-weight inspection motivated the pruning rule before new puzzle evaluation; this is an exploratory graph-recovery finding, not independently preregistered discovery.','',
            'Fresh development rows[6656,8704), allsix model/search seeds0–5, k10/17/21,262144moves perchain,fourchains. New transformedmodels additionally passed initial+4check exact25field/capturedboard parity gates against GeneralSearch.','',
            '[Frozen protocol](protocol-gap.json) · [Runner](gap.py) · [Audit](audit_gap.py) · [Tracked evidence](../../../results/research/ripper/gap.json)','']
    (HERE/'REPORT-gap.md').write_text('\n'.join(lines))
    print(json.dumps({key:rec[key] for key in ['primary_effect_pp','primary_ci95_pp','primary_decision','pruned_models_exact_peer_graph','secondary']}))


if __name__=='__main__':main()
