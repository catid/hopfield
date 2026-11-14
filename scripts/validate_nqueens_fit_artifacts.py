"""Compare fresh N-Queens models, references, and indexed/dense fit timing."""
import json,pathlib,hashlib,sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
repo = ROOT / 'upstream'
out=(ROOT / 'artifacts/multi_solution')
sys.path.insert(0,str(repo))
from checkpoints.load import path

def tensor_digest(model):
 h=hashlib.sha256()
 for key in ['A','b','W','B','S']:
  value=model[key].contiguous().cpu().numpy()
  h.update(json.dumps({'name':key,'shape':list(value.shape),'dtype':str(value.dtype)},sort_keys=True).encode())
  h.update(value.tobytes())
 return h.hexdigest()

all_rows=[]
for n in (8,10):
 rows=[]
 for seed in (0,1,2):
  folder=repo/'experiments/runs/NQ/nq11'/f'n{n}'/('shared' if n==8 else 'final')/f'pl-ising-s{seed}'
  if not all((folder/file).exists() for file in ['fit.json','fit-dense.json','model.pt','final.json']):continue
  info=json.loads((folder/'fit.json').read_text());dense=json.loads((folder/'fit-dense.json').read_text())
  model=torch.load(folder/'model.pt',weights_only=False);ref=torch.load(path(f'nqueens-{n}',seed),weights_only=False)
  diffs={k:{'max_abs':float((model[k].double()-ref[k].double()).abs().max()),'reference_max_abs':float(ref[k].double().abs().max()),'exact_equal':bool(torch.equal(model[k],ref[k]))} for k in ['A','b','W','B','S']}
  row={'n':n,'seed':seed,'rows':info['rows'],'config':{k:info[k] for k in ['method','arm','lr','knob','lam','precond','updates']},'params_learned':info['params_learned'],'params_nonzero':info['params_nonzero'],'cold_index_seconds':info['cold_seconds'],'warm_index_seconds':info['warm_seconds'],'cold_dense_seconds':dense['cold_seconds'],'warm_dense_seconds':dense['warm_seconds'],'dense_vs_index_speedup':info['warm_seconds']/dense['warm_seconds'],'dense_vs_scored':dense['vs_scored_model'],'released_checkpoint_comparison':diffs,'model_sha256':hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest(),'model_tensors_sha256':tensor_digest(model),'reference_tensors_sha256':tensor_digest(ref)}
  rows.append(row);all_rows.append(row)
 (out/f'nq{n}-fit-validation.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'nqueens-fit-validation.json').write_text(json.dumps(all_rows,indent=2)+'\n')
print(json.dumps({'fits_validated':len(all_rows),'source_models_unchanged':True,'timing_comparison_scope':'Same GPU and recipe; indexed scored fit versus dense mapping retime, TF32 disabled. Unrelated jobs may use other GPUs/CPU cores.'}))
