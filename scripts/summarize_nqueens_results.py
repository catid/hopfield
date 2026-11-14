"""Recompute N-Queens Table 3 metrics from saved chains; never run inference."""
import json,pathlib,statistics,sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
repo = ROOT / 'upstream'
out=(ROOT / 'artifacts/multi_solution')
sys.path[:0]=[str(repo),str(repo/'experiments')]
from cram.nqueens_data import load_split,valid
published=json.loads((repo/'results/multi-solution-table.json').read_text())
results=[]
for n in (8,10):
 inputs,labels,stored=load_split(n,'test')
 counts=dict(np.load(out/f'nq{n}-solution-counts.npz'))
 inverse=counts['inverse'];S=counts['counts'];cap=np.minimum(S,20)
 assert np.array_equal(S[inverse],stored.reshape(-1))
 for kind in ('released','fresh','free','hand'):
  method={'hand':'CRAM (hand-set)','free':'CRAM (free pair tables)'}.get(kind,'CRAM (offset-shared pair tables)')
  expected=next(row for row in published['rows'] if row['method']==method)['cells'][f'nq{n}']
  per_seed=[]
  for seed in (0,1,2):
   if kind=='released':folder=out/'released'/f'nq{n}'/f's{seed}'
   elif kind=='fresh':folder=repo/'experiments/runs/NQ/nq11'/f'n{n}'/('shared' if n==8 else 'final')/f'pl-ising-s{seed}'
   elif kind=='free':folder=repo/'experiments/runs/NQ/alt'/f'n{n}'/'final'/f"{'mpl' if n==8 else 'cd'}-s{seed}"
   else:folder=repo/'experiments/runs/NQ/alt'/f'n{n}'/'hand'/f'test-s{seed}'
   path=folder/'records.npz'
   if kind=='free' and not (folder/'final.json').exists():continue
   if not path.exists():continue
   z=dict(np.load(path));boards=np.unpackbits(z['boards'],axis=-1)[...,:n*n].astype(bool)
   ok=valid(boards,n)&(boards>=inputs[:,None,:]).all(axis=-1)
   selected=np.argmin(z['energy'],axis=1)
   found=np.array([len({board.tobytes() for board in boards[i,ok[i]]}) for i in range(len(inputs))])
   per_row_found=np.minimum(found,cap[inverse])
   found_per_puzzle=np.bincount(inverse,weights=per_row_found,minlength=len(S))/np.bincount(inverse,minlength=len(S))
   per_seed.append({'seed':seed,'rows':len(inputs),'distinct_inputs':len(S),'acc_n1':100*float(ok[:,0].mean()),'acc_n20':100*float(ok[np.arange(len(inputs)),selected].mean()),'cov20_per_puzzle':100*float(found_per_puzzle.sum()/cap.sum()),'cov20_per_row':100*float(per_row_found.sum()/cap[inverse].sum()),'wrong_stops':int((z['stopped']&~ok).sum()),'budget_failures':int((~z['stopped']).sum()),'records':str(path)})
  if not per_seed:continue
  metrics={k:{'mean':statistics.mean([r[k] for r in per_seed]),'sd':statistics.stdev([r[k] for r in per_seed]) if len(per_seed)>1 else None,'per_seed':[r[k] for r in per_seed]} for k in ['acc_n1','acc_n20','cov20_per_puzzle','cov20_per_row']}
  difference={k:metrics[k]['mean']-expected[k]['mean'] for k in metrics}
  results.append({'task':f'nqueens-{n}','kind':kind,'method':method,'split':'test','complete_seeds':len(per_seed)==3,'seeds':[r['seed'] for r in per_seed],'chains':20,'metrics':metrics,'paper_reference':{k:expected[k] for k in metrics},'mean_difference_percentage_points':difference,'per_seed':per_seed,'metric_rule':'N1 first chain; N20 lowest-energy chain; coverage any valid completion capped at min(S,20), averaged over duplicate rows within each input and pooled over distinct inputs.'})
(out/'nqueens-paper-results.json').write_text(json.dumps(results,indent=2)+'\n')
for r in results:print(json.dumps({'task':r['task'],'kind':r['kind'],'complete_seeds':r['complete_seeds'],'means':{k:v['mean'] for k,v in r['metrics'].items()},'difference':r['mean_difference_percentage_points']}),flush=True)
