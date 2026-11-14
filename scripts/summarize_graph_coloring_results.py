"""Recompute Table 3 graph-coloring metrics from locally generated raw chains."""
import json,pathlib,statistics,sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
repo = ROOT / 'upstream'
sys.path[:0]=[str(repo),str(repo/'experiments')]
from cram import graph_coloring_data as G
published=json.loads((repo/'results/multi-solution-table.json').read_text())
rows=[]
for kind,method in [('learned','CRAM (free pair tables)'),('hand','CRAM (hand-set)')]:
 reference=next(row for row in published['rows'] if row['method']==method)
 for n,tag,T in [(8,'n8-full-u1024-lr10-lam0.0001',2),(10,'n10-full-u256-lr10-lam0.001',3)]:
  x,y=G.load_split(n,'test');edge,_=G.convert(x,y,n);A=G.adjacency(edge,n)
  per_seed=[]
  for seed in range(3):
   if kind=='learned':folder=repo/'experiments/runs/CRAM-GC/final'/tag/f'test-terms-T{T}-S20-cap10000-s{seed}'
   else:folder=repo/'experiments/runs/CRAM-GC/hand'/f'n{n}'/f'test-s{seed}'
   record=json.loads((folder/'final.json').read_text());raw=dict(np.load(folder/'outputs.npz'))
   conf=G.conflicts(A,raw['boards']);selected=np.argmin(raw['energy'],axis=1)
   per_seed.append({'seed':seed,'conflicts_n1':int(conf[:,0].sum()),'conflicts_n20':int(conf[np.arange(len(x)),selected].sum()),'cov20':100*record['coverage_20'],'wrong_stops_all':record['wrong_stops_all']})
  metrics={k:{'mean':statistics.mean([r[k] for r in per_seed]),'sd':statistics.stdev([r[k] for r in per_seed]),'per_seed':[r[k] for r in per_seed]} for k in ['conflicts_n1','conflicts_n20','cov20']}
  ref=reference['cells'][f'gc{n}']
  matches={k:metrics[k]==ref[k] for k in metrics}
  rows.append({'task':f'graph-coloring-{n}','kind':kind,'method':method,'split':'test','graphs':len(x),'seeds':[0,1,2],'chains':20,'metrics':metrics,'matches_paper_export_each_metric':matches,'all_paper_metrics_exact':all(matches.values()),'per_seed':per_seed,'reference_origin':'Imported published results/multi-solution-table.json; not a locally rerun baseline.'})
out=(ROOT / 'artifacts/multi_solution/graph-coloring-paper-results.json')
out.write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(json.dumps({'task':r['task'],'kind':r['kind'],'metrics':r['metrics'],'all_paper_metrics_exact':r['all_paper_metrics_exact']}),flush=True)
