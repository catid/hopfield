"""Run each released N-Queens checkpoint once; retain samples for paper metrics."""
import json
import pathlib
import sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
repo = ROOT / 'upstream'
sys.path[:0]=[str(repo), str(repo/'experiments')]
from checkpoints import evaluate as E
import nqueens_free as N

n=int(sys.argv[1])
out=(ROOT / 'artifacts/multi_solution/released')/f'nq{n}'
out.mkdir(parents=True,exist_ok=True)
torch.backends.cuda.matmul.allow_tf32=False
original=N.score
rows=[]
for seed in (0,1,2):
 dest=out/f's{seed}'
 dest.mkdir(exist_ok=True)
 if (dest/'comparison.json').exists():
  assert (dest/'records.npz').exists()
  row=json.loads((dest/'comparison.json').read_text())
  rows.append(row)
  print(json.dumps(dict(row,reused_saved_result=True)),flush=True)
  continue
 def save_score(*args, **kwargs):
  kwargs['records']=dest/'records.npz'
  result=original(*args,**kwargs)
  (dest/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
  return result
 N.score=save_score
 try:
  got,want,same=E.nqueens(n,seed)
 finally:
  N.score=original
 row={'n':n,'seed':seed,'matches_upstream_diagnostic':same,'got':got,'expected':want,'records':str(dest/'records.npz')}
 rows.append(row)
 (dest/'comparison.json').write_text(json.dumps(row,indent=2)+'\n')
 print(json.dumps(row),flush=True)
(out/'comparisons.json').write_text(json.dumps(rows,indent=2)+'\n')
raise SystemExit(0 if all(x['matches_upstream_diagnostic'] for x in rows) else 1)
