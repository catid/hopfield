"""CPU-only H1g audit: identical model weights, different search units."""
import csv
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_mechanism import interval

BASE=ROOT/'artifacts/research/20261114/ripper'
OUT=BASE/'scale'
PUBLIC=ROOT/'results/research/ripper'


def main():
    p=json.loads((HERE/'protocol-scale.json').read_text());q=json.loads((OUT/'queue.json').read_text())
    assert len(q)==36 and all(j['status']=='complete' for j in q)
    rows=[];arrays={};hashes={};units=[]
    for j in q:
        d=OUT/'jobs'/j['job_id'];r=json.loads((d/'receipt.json').read_text());k,s,arm=r['k'],r['seed'],r['arm']
        assert r['status']=='complete' and r['protocol_sha256']==sha(HERE/'protocol-scale.json') and r['wrapper_sha256']==sha(HERE/'scale.py')
        assert r['backend_sha256']==sha(HERE/'sparse_backend.py') and r['preflight_checks']==5 and r['preflight_state_fields']==25
        assert r['rows_half_open']==[8704,10000] and r['test_rows']==1296
        for name,h in r['output_sha256'].items():assert sha(d/name)==h
        z=np.load(d/'boards.npz');b=z['boards'];target=z['solutions'];free=z['free'];st=z['stopped'];moves=z['moves']
        assert b.shape==(1296,81) and np.array_equal(z['puzzle'],np.arange(8704,10000))
        boards=b.reshape(-1,9,9);boxes=boards.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9);valid=((b==target)|free).all(1)
        for digit in range(9):
            valid &= ((boards==digit).sum(1)==1).all(1)
            valid &= ((boards==digit).sum(2)==1).all(1)
            valid &= ((boxes==digit).sum(2)==1).all(1)
        exact=(b==target).all(1);assert np.array_equal(valid,exact) and np.array_equal(z['valid'],valid) and np.array_equal(z['exact'],exact)
        assert int(exact.sum())==r['solved'] and int((st&~exact).sum())==r['wrong_stops'] and int((~st).sum())==r['budget_failures']
        assert np.all((moves>0)&(moves<=262144))
        result=json.loads((d/'result.json').read_text())['per_puzzle'];assert np.array_equal(result['solved'],exact) and np.array_equal(result['stopped'],st)
        assert np.allclose(result['flops'],moves*r['fl_move']*4,rtol=0,atol=0)
        src=BASE/f'gap/jobs/k{k}-s{s}-gap_uniform';assert sha(src/'searched_model.pt')==r['source_model_sha256']
        before=torch.load(src/'searched_model.pt',weights_only=True,map_location='cpu');after=torch.load(d/'searched_model.pt',weights_only=True,map_location='cpu')
        assert before.keys()==after.keys() and all(torch.equal(before[key],after[key]) for key in before)
        J=before['W'][:,0,0];level=float(J[J>0][0]);assert bool((J[J>0]==level).all())
        prior=json.loads((src/'receipt.json').read_text());expected=prior['used_hbar'] if arm=='inherited_unit' else level
        assert expected==r['used_hbar'] and r['statistics']['tau_train']==prior['statistics']['tau_train'] and r['statistics']['ebar']==prior['statistics']['ebar']
        units.append({'k':k,'seed':s,'arm':arm,'unit':r['used_hbar'],'level':level,'unit_over_level':r['used_hbar']/level})
        arrays[k,s,arm]=exact.astype(float);rows.append({key:r[key] for key in ['job_id','k','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s']});hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
    effects=[];grouped=[]
    for k in p['k']:
        diff=np.array([[arrays[k,s,'normalized_unit']-arrays[k,s,'inherited_unit']] for s in range(6)]);ci=interval(diff);effect=float(100*diff.mean())
        if k==10:decision='supported_for_frozen_protocol' if ci[0]>=20 else 'falsified_for_frozen_protocol' if ci[1]<20 else 'inconclusive'
        else:decision='equivalent_within_5pp' if ci[0]>=-5 and ci[1]<=5 else 'non_equivalent' if ci[0]>5 or ci[1]<-5 else 'inconclusive'
        effects.append({'k':k,'effect_pp':effect,'ci95_pp':ci,'decision':decision,'per_seed_pp':(100*diff.mean((1,2))).tolist()})
        for arm in p['arms']:
            rs=[r for r in rows if r['k']==k and r['arm']==arm];grouped.append({'k':k,'arm':arm,'solved_per_seed':[r['solved'] for r in rs],'mean_accuracy_pct':float(np.mean([r['solved']/12.96 for r in rs])),'wrong_stops':sum(r['wrong_stops'] for r in rs),'n_per_seed':1296,'seeds':6})
    primary=effects[0];rec={'status':'validated','utc':now(),'jobs':36,'saved_boards_verified':46656,'primary_effect_pp':primary['effect_pp'],'primary_ci95_pp':primary['ci95_pp'],'primary_decision':primary['decision'],'effects':effects,'grouped':grouped,'rows':rows,'units':units,'weights_exactly_equal_to_frozen_input':True,'input_sha256':hashes,'new_inference':False,'limitations':['Fixed short-budget development study, no external-distribution claim.','Last1296 development rows were unused before this study; all10000 development rows are now consumed across research waves.','Six model-seed clusters; paired two-way bootstrap approximate.','Only thermal/focus unit changes; weights,graph,trainingtau,ebar andsearchbudget are exactly preserved.','Observed invariance/equivalence is statistical, not exact floating-point scale invariance.']}
    save(OUT/'audit.json',rec);save(PUBLIC/'scale.json',rec)
    with (PUBLIC/'scale_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=json.loads((PUBLIC/'summary.json').read_text());summary['scale_normalization']=rec;save(PUBLIC/'summary.json',summary)
    lines=['# H1g: normalize temperature and focus to the retained coupling level','',f'At k10, scale normalization changes accuracy by **{primary["effect_pp"]:.4f} percentage points**, approximate paired95% interval **[{primary["ci95_pp"][0]:.4f}, {primary["ci95_pp"][1]:.4f}]**. The frozen≥20point hypothesis is **{primary["decision"]}**. All46656saved boards were independently checked, and every model weight exactly matches its frozen uniformized input.','',
           '| k | Thermal/focus unit | Mean accuracy (%) | Wrong stops |','|---:|---|---:|---:|','| — | BP AdamW: outside post-training intervention protocol; not run | — | — |']
    for r in grouped:lines.append(f'| {r["k"]} | {r["arm"]} | {r["mean_accuracy_pct"]:.4f} | {r["wrong_stops"]} |')
    lines+=['','The normalized arm uses the common retained coupling as its temperature/focus unit. The inherited arm retains the earlier raw-projection hbar. Weights, graph, training-only tau/ebar, random seeds and move budget are unchanged. No Sudoku geometry is supplied.','']
    for e in effects[1:]:lines.append(f'At k{e["k"]}, the effect is {e["effect_pp"]:.4f} points,95% interval[{e["ci95_pp"][0]:.4f}, {e["ci95_pp"][1]:.4f}], decision: **{e["decision"]}** under the frozen±5point equivalence margin.')
    lines+=['','The fresh development interval was[8704,10000),1296puzzles permodel. All development rows have now been used across the research sequence.','',
            '[Frozen protocol](protocol-scale.json) · [Runner](scale.py) · [Audit](audit_scale.py) · [Tracked results](../../../results/research/ripper/scale.json)','']
    (HERE/'REPORT-scale.md').write_text('\n'.join(lines));print(json.dumps(effects))


if __name__=='__main__':main()
