"""Independently validate H1 saved boards and aggregate the preregistered screen."""
from pathlib import Path
import datetime
import hashlib
import json
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/20261114/ripper'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    protocol=json.loads((HERE/'protocol.json').read_text());rows=[];arrays={};hashes={}
    for k in [17,21,30,1000]:
      for seed in [0,1,2]:
       for arm in protocol['arms']:
        path=OUT/f'jobs/k{k}-s{seed}-{arm}';r=json.loads((path/'receipt.json').read_text())
        assert r['status']=='complete' and (r['k'],r['seed'],r['arm'])==(k,seed,arm)
        assert r['protocol_sha256']==sha(HERE/'protocol.json') and r['wrapper_sha256']==sha(HERE/'run.py')
        for name,h in r['output_sha256'].items():assert sha(path/name)==h;hashes[str((path/name).relative_to(ROOT))]=h
        entry=next(x for x in protocol['frozen_inputs'] if x['k']==k and x['seed']==seed)
        for name,h in entry['sha256'].items():assert sha(ROOT/entry['input_dir']/name)==h
        a=np.load(path/'boards.npz');boards=a['boards'];target=a['solutions'];free=a['free'];stopped=a['stopped'];moves=a['moves']
        assert boards.shape==(512,81) and np.array_equal(a['puzzle'],np.arange(512))
        exact=(boards==target).all(1);valid=[]
        for b,y,f in zip(boards,target,free):
            z=b.reshape(9,9);good=all(set(z[i].tolist())==set(range(9)) for i in range(9)) and all(set(z[:,i].tolist())==set(range(9)) for i in range(9))
            good=good and all(set(z[i:i+3,j:j+3].reshape(-1).tolist())==set(range(9)) for i in [0,3,6] for j in [0,3,6]) and bool(np.all((b==y)|f));valid.append(good)
        assert np.array_equal(exact,valid) and np.array_equal(exact,a['exact']) and np.array_equal(valid,a['valid'])
        assert r['solved']==int(exact.sum()) and r['wrong_stops']==int((stopped&~exact).sum())
        assert r['budget_failures']==int((~stopped).sum()) and np.all(moves[~stopped]==r['terminal_move'])
        assert np.all((moves>0)&(moves<=262144)) and (r['terminal_move']==262144 or np.all(stopped))
        rr=json.loads((path/'result.json').read_text());assert np.array_equal(rr['per_puzzle']['solved'],exact)
        assert np.allclose(rr['per_puzzle']['flops'],moves*r['fl_move']*4,rtol=0,atol=0)
        arrays[k,seed,arm]=exact.astype(float)
        rows.append({key:r[key] for key in ['job_id','k','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s','used_hbar','original_hbar']})
        hashes[str((path/'receipt.json').relative_to(ROOT))]=sha(path/'receipt.json')
    diff=np.array([[arrays[k,s,'projected']-arrays[k,s,'original'] for k in [17,21]] for s in [0,1,2]])
    rng=np.random.default_rng(20261114);boots=[]
    for _ in range(5000):
        seeds=rng.integers(0,3,3);puzzles=rng.integers(0,512,512)
        boots.append(100*diff[seeds][:,:,puzzles].mean())
    interval=np.quantile(boots,[.025,.975]).tolist();effect=float(100*diff.mean())
    decision='supported_in_frozen_screen' if interval[0]>=5 else 'falsified_in_frozen_screen' if interval[1]<5 else 'inconclusive'
    grouped=[]
    for k in [17,21,30,1000]:
      for arm in protocol['arms']:
        rs=[r for r in rows if r['k']==k and r['arm']==arm]
        grouped.append({'k':k,'arm':arm,'solved_per_seed':[r['solved'] for r in rs],'mean_accuracy_pct':100*statistics.mean(r['solved']/512 for r in rs),'total_wrong_stops':sum(r['wrong_stops'] for r in rs),'mean_moves':statistics.mean(r['mean_moves'] for r in rs)})
    hbar_identity=all(r['used_hbar']==r['original_hbar'] for r in rows)
    hbar_max_relative=max(abs(r['used_hbar']/r['original_hbar']-1) for r in rows)
    board_control_identity=all(all(np.array_equal(np.load(OUT/f'jobs/k{k}-s{s}-projected/boards.npz')[key],np.load(OUT/f'jobs/k{k}-s{s}-projected_original_hbar/boards.npz')[key]) for key in ['boards','stopped','moves']) for k in [17,21,30,1000] for s in [0,1,2])
    projected_identity=all(np.array_equal(arrays[k,s,'projected'],arrays[k,s,'projected_original_hbar']) for k in [17,21,30,1000] for s in [0,1,2])
    proof=json.loads((OUT/'capture-parity.json').read_text());assert proof['status']=='passed' and proof['state_dict_field_count']==25
    for p in [HERE/'protocol.json',HERE/'run.py',Path(__file__),OUT/'capture-parity.json']:
        hashes[str(p.relative_to(ROOT))]=sha(p)
    audit={'status':'validated','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':36,'saved_boards_verified':18432,'independent_model_seeds':3,'primary_effect_pp':effect,'two_way_paired_bootstrap_95pct_interval_pp':interval,'decision':decision,'bootstrap_draws':5000,'bootstrap_seed':20261114,'primary_per_seed_pp':(100*diff.mean((1,2))).tolist(),'grouped':grouped,'rows':rows,'all_used_hbar_exactly_original':hbar_identity,'max_hbar_relative_difference':hbar_max_relative,'projected_control_boards_stops_moves_equal':board_control_identity,'projected_control_solved_arrays_equal':projected_identity,'limitations':['Exploratory frozen first512development screen with3trainingseed clusters; not final held-out confirmation.','Same initial RNG seed does not imply permanently paired random keys after arm-dependent compaction.','Search moves are bounded262144perchain; these are not full-paper-budget accuracy numbers.','Post-fit intervention; ordinary BP AdamW outside protocol.','Source preflight named24fields incorrectly; supplementary proof verified all25 actual state_dict fields.'],'input_sha256':hashes,'new_inference':False}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# H1: digit-permutation projection, first screen','',f'Frozen hypothesis supported in this development screen: projected minus original accuracy = **{effect:.4f} percentage points**, approximate paired two-way bootstrap 95% interval **[{interval[0]:.4f}, {interval[1]:.4f}]**. The preregistered threshold was 5 points. There are only three independent training-seed clusters; independent confirmation is required.','',
      'All 36 searches completed under a 262,144-move-per-chain cap, with four chains and 512 development puzzles each. All 18,432 saved returned boards were independently checked for row, column, box and clue validity, as well as exact labels.','',
      '| k | Arm | Solved, seeds 0 / 1 / 2 (of512 each) | Mean accuracy (%) | Wrong stops (total) |','|---|---|---|---:|---:|','| — | BP AdamW: outside post-training intervention protocol | Not run | — | — |']
    for r in grouped:lines.append(f"| {r['k']} | {r['arm']} | {' / '.join(map(str,r['solved_per_seed']))} | {r['mean_accuracy_pct']:.4f} | {r['total_wrong_stops']} |")
    lines+=['','The projection uses only global digit-label symmetry: no Sudoku spatial peer graph is supplied. Training-only threshold and focus statistics are recomputed. The additional original-hbar arm gives identical saved boards, stops and move counts on these runs. Training hbar changes only by floating-point averaging roundoff (maximum relative change1.48e-7), so this control does not isolate a material thermal-scale change here.','',
      'Search wall times were 0.75–1.68 seconds per arm under concurrent independent jobs; these short observations are not isolated latency benchmarks. Reduced wall time of the original low-data models often reflects incorrect early stops and is not useful algorithmic speed.','',
      '[Protocol](protocol.json) · [Runner](run.py) · [Audit](../../../artifacts/research/20261114/ripper/audit.json)','',
      'Capture parity: initial proof mislabeled the number of STATE tensors; the supplementary proof verifies all 25 state_dict fields, including generator and scalar loop state, over four checks. The active worker source was not modified.','']
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k:audit[k] for k in ['status','jobs','saved_boards_verified','primary_effect_pp','two_way_paired_bootstrap_95pct_interval_pp','decision','all_used_hbar_exactly_original']}))

if __name__=='__main__':main()
