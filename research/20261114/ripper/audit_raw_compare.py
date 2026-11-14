"""Independent raw-board count/PL audit with frozen four-contrast family."""
import csv
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_mechanism import interval

BASE=ROOT/'artifacts/research/20261114/ripper'
OUT=BASE/'raw_compare'
PUBLIC=ROOT/'results/research/ripper'


def main():
    p=json.loads((HERE/'protocol-raw-compare.json').read_text());q=json.loads((OUT/'queue.json').read_text())
    assert len(q)==48 and all(j['status']=='complete' for j in q)
    wrappers={sha(HERE/'raw_compare.py'),sha(HERE/'raw_compare_v2.py')};err=json.loads((HERE/'ERRATUM-raw-compare.json').read_text())
    assert wrappers=={err['old_wrapper_sha256'],err['corrected_wrapper_sha256']} and err['protocol_unchanged']==sha(HERE/'protocol-raw-compare.json')
    datafile=ROOT/'upstream/data/full-test/full-all.npz';data=np.load(datafile);refsol=data['solutions'][60000:70000].astype(np.int8)-1;reffree=~data['clue_mask'][60000:70000].astype(bool)
    hashes={str(datafile.relative_to(ROOT)):sha(datafile)};rows=[];arrays={};oracle=[];overlaps=[];pair=np.triu_indices(81,1)
    testrows=np.ascontiguousarray(refsol).view(np.dtype((np.void,81))).ravel()
    for j in q:
        d=OUT/'jobs'/j['job_id'];r=json.loads((d/'receipt.json').read_text());n,s,arm=r['k'],r['seed'],r['arm']
        assert r['status']=='complete' and r['protocol_sha256']==sha(HERE/'protocol-raw-compare.json') and r['wrapper_sha256'] in wrappers
        assert r['backend_sha256']==sha(HERE/'sparse_backend.py') and r['rows_half_open']==[60000,70000] and r['test_rows']==10000 and r['training_rows']==n
        for name,h in r['output_sha256'].items():assert sha(d/name)==h
        entry=next(e for e in p['training_inputs'] if e['seed']==s);trainpath=ROOT/entry['path'];assert sha(trainpath)==entry['sha256']==r['training_sha256'];tr=np.load(trainpath)
        train=tr['solutions'][:n];fr=tr['free'][:n];fc=fr.sum(1)
        assert int(fc.min())==r['free_cells_per_training_board']['min'] and int(fc.max())==r['free_cells_per_training_board']['max']
        assert abs(float(fc.mean())-r['free_cells_per_training_board']['mean'])<1e-5
        overlaps.append({'n':n,'seed':s,'arm':arm,'training_complete_boards_in_test':int(np.isin(np.ascontiguousarray(train).view(np.dtype((np.void,81))).ravel(),testrows).sum())})
        model=torch.load(d/'searched_model.pt',weights_only=True,map_location='cpu');fit=torch.load(d/'fitted_model.pt',weights_only=True,map_location='cpu');assert all(torch.equal(model[k],fit[k]) for k in model)
        if arm=='count':
            supportfile=BASE/f'raw/counts/n{n}-s{s}/support.npz';support=np.load(supportfile)['support']
            assert r['transform']['support_sha256']==sha(supportfile)
            assert torch.equal(model['W'],torch.from_numpy(support.astype(np.float32))[:,None,None]*torch.eye(9)) and not torch.any(model['B'])
            assert not np.any(train[:,pair[0][support]]==train[:,pair[1][support]])
            assert r['used_hbar']==1 and r['statistics']['tau_train']==0 and r['statistics']['ebar']==0
            assert abs(r['statistics']['hbar']-1)<=1e-7
            energy=(refsol[:,pair[0][support]]==refsol[:,pair[1][support]]).sum(1)
            assert np.array_equal(energy,np.load(d/'oracle_target_energy.npy')) and int((energy==0).sum())==r['oracle_zero_energy_target_count']
            hashes[str((d/'oracle_target_energy.npy').relative_to(ROOT))]=sha(d/'oracle_target_energy.npy')
        else:
            expected={'aug':arm=='pl_digit','lam':3e-3/9,'lr':540.0,'momentum':.5,'updates':32,'split':s,'seed':s}
            assert r['fit_config']==expected and r['used_hbar']==r['statistics']['hbar']
        z=np.load(d/'boards.npz');b=z['boards'];target=z['solutions'];free=z['free'];st=z['stopped'];moves=z['moves'];assert b.shape==(10000,81)
        assert np.array_equal(target,refsol) and np.array_equal(free,reffree) and np.array_equal(z['puzzle'],np.arange(60000,70000))
        boards=b.reshape(-1,9,9);boxes=boards.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9);valid=((b==target)|free).all(1)
        for digit in range(9):
            valid &= ((boards==digit).sum(1)==1).all(1)
            valid &= ((boards==digit).sum(2)==1).all(1)
            valid &= ((boxes==digit).sum(2)==1).all(1)
        exact=(b==target).all(1);assert np.array_equal(valid,exact) and np.array_equal(z['valid'],valid) and np.array_equal(z['exact'],exact)
        assert int(exact.sum())==r['solved'] and int((st&~exact).sum())==r['wrong_stops'] and int((~st).sum())==r['budget_failures']
        assert np.all((moves>0)&(moves<=262144));result=json.loads((d/'result.json').read_text())['per_puzzle']
        assert np.array_equal(result['solved'],exact) and np.array_equal(result['stopped'],st) and np.allclose(result['flops'],moves*r['fl_move']*4,rtol=0,atol=0)
        if arm=='count':
            assert r['wrong_stops']==0 and np.all(~st|(energy==0))
            oracle.append({'n':n,'seed':s,'zero_energy_target_count':int((energy==0).sum()),'n_test':10000,'correct_certified_stops':int((st&exact).sum()),'correct_uncertified_returns':int((~st&exact).sum()),'total_correct_returns':int(exact.sum()),'mean_target_energy':float(energy.mean())})
        arrays[n,s,arm]=exact.astype(float);rows.append({key:r[key] for key in ['job_id','k','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s']});hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
    for s in range(6):
        a=np.load(BASE/f'raw/counts/n100-s{s}/support.npz')['support'];b=np.load(BASE/f'raw/counts/n300-s{s}/support.npz')['support'];assert np.array_equal(a,b)
        arrays[300,s,'count']=arrays[100,s,'count']
    spec=[(n,a) for n in (100,300) for a in ('pl_raw','pl_digit')]
    diff=np.array([[arrays[n,s,'count']-arrays[n,s,a] for n,a in spec] for s in range(6)])
    rng=np.random.default_rng(20261114);boot=np.empty((5000,4))
    for draw in range(5000):
        ix=rng.integers(0,6,6);puzzles=rng.integers(0,10000,10000);boot[draw]=100*diff[ix][:,:,puzzles].mean((0,2))
    ci=np.quantile(boot,[.00625,.99375],axis=0);effects=[]
    for i,(n,a) in enumerate(spec):
        lo,hi=float(ci[0,i]),float(ci[1,i]);decision='noninferior_within_5pp' if lo>=-5 else 'inferior_beyond_5pp' if hi<-5 else 'inconclusive'
        effects.append({'n':n,'contrast':['count',a],'effect_pp':float(100*diff[:,i].mean()),'family_ci98_75_pp':[lo,hi],'decision':decision})
    joint='supported_joint_noninferiority' if all(e['decision']=='noninferior_within_5pp' for e in effects) else 'falsified_joint_noninferiority' if any(e['decision']=='inferior_beyond_5pp' for e in effects) else 'inconclusive'
    secondary=[]
    for a in ('pl_raw','pl_digit'):
        x=np.array([[arrays[30,s,'count']-arrays[30,s,a]] for s in range(6)]);secondary.append({'n':30,'contrast':['count',a],'effect_pp':float(100*x.mean()),'descriptive_ci95_pp':interval(x)})
    grouped=[]
    for n in (30,100,300):
        for arm in p['arms']:
            rs=[r for r in rows if r['k']==(100 if n==300 and arm=='count' else n) and r['arm']==arm]
            grouped.append({'n':n,'arm':arm,'solved_per_seed':[r['solved'] for r in rs],'mean_accuracy_pct':float(np.mean([r['solved']/100 for r in rs])),'wrong_stops':sum(r['wrong_stops'] for r in rs),'n_per_seed':10000,'seeds':6,'reused_count_n100':n==300 and arm=='count'})
    rec={'status':'validated','utc':now(),'jobs':48,'saved_boards_verified':480000,'primary_decision':joint,'primary_contrasts':effects,'secondary_n30':secondary,'grouped':grouped,'rows':rows,'oracle_diagnostic':oracle,'training_test_completeboard_overlap':overlaps,'input_sha256':hashes,'new_inference':False,'erratum_sha256':sha(HERE/'ERRATUM-raw-compare.json'),'limitations':['Sizes selected after exploratory training-only structure curve; test comparisons frozen before outcomes.','Four primary contrasts use approximate paired two-way bootstrap andBonferroni98.75percent intervals.','Same official associatedquestion masks retained; no all-free substitution.','PLdigit usesper-updatedigitrelabeling only; no spatialaugmentation inanyarm.','CountN300 scores reusedfrombyte-identicalN100model/settings,notindependentduplicate runs.','Targetzero-energyfraction upperbounds correctcertified stops only; correctunstoppedbestboards mayhavepositiveenergy.','Job walltimes not isolatedpaired speedbenchmarks.','This tests fixedoriginalPLrecipes,not alloptimizedPLmethods.']}
    save(OUT/'audit.json',rec);save(PUBLIC/'raw_compare.json',rec)
    for name,rs in [('raw_compare_per_seed.csv',rows),('raw_compare_oracle.csv',oracle)]:
        with (PUBLIC/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    summary=json.loads((PUBLIC/'summary.json').read_text());summary['raw_count_PL_comparison']=rec;save(PUBLIC/'summary.json',summary)
    lines=['# H1o: direct counting versus fixed pseudolikelihood recipes on raw boards','',f'The four frozen count-versus-PL contrasts at 100 and 300 raw complete boards give **{joint}** under the five-point margin. Each contrast uses a 98.75% paired bootstrap interval, with Bonferroni correction for the four-contrast family. All 480,000 returned boards were independently checked.','',
           '| Raw training boards | Method | Mean accuracy (%) | Wrong stops |','|---:|---|---:|---:|','| — | BP AdamW: outside this fixed-PL-recipe comparison; not run | — | — |']
    for r in grouped:lines.append(f'| {r["n"]} | {r["arm"]} | {r["mean_accuracy_pct"]:.4f} | {r["wrong_stops"]} |')
    lines+=['','Both PL arms use the original 32-update proximal pseudolikelihood recipe and unchanged fitted weights. `pl_raw` has no augmentation; `pl_digit` uses the original per-update digit relabeling. All arms preserve the official puzzle masks associated with the sampled complete boards. No spatial augmentation or largest-gap transformation is used.','',
            'Count graphs at 100 and 300 boards are byte-identical, with the same unit, stopping/focus statistics, search seed and test inputs. The six count searches are explicitly reused across these sizes. The six 30-board count graphs remain distinct.','',
            'The oracle column measures which fixed test solutions have zero count energy. It is a ceiling for correct certified stops; overall returned-board accuracy can exceed it because an unstopped best board may be correct at positive energy. The diagnostic never feeds back to graph construction or search.','',
            'Evaluation uses full-all rows [60,000,70,000), six seeds, four chains and 262,144 moves per chain. A documented pre-inference assertion correction enforces the already frozen literal count unit of one when a floating mean reduction returns 0.99999994. Completed earlier runs were unchanged.','',
            '[Frozen protocol](protocol-raw-compare.json) · [Original runner](raw_compare.py) · [Corrected runner](raw_compare_v2.py) · [Erratum](ERRATUM-raw-compare.json) · [Audit](audit_raw_compare.py) · [Tracked results](../../../results/research/ripper/raw_compare.json)','']
    (HERE/'REPORT-raw-compare.md').write_text('\n'.join(lines));print(json.dumps({'primary_decision':joint,'effects':effects,'grouped':grouped,'secondary_n30':secondary}))


if __name__=='__main__':main()
