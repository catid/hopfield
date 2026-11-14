"""Post-hoc sufficient energy-certificate check; geometry never enters training."""
from pathlib import Path
import datetime
import hashlib
import json
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/research/20261114/ripper/mechanism'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    pair=torch.triu_indices(81,81,1);r=torch.arange(81)//9;c=torch.arange(81)%9
    peers=(r[:,None]==r[None])|(c[:,None]==c[None])|((r[:,None]//3==r[None]//3)&(c[:,None]//3==c[None]//3))
    truth=peers[pair[0],pair[1]];assert int(truth.sum())==810
    eye=torch.eye(9,dtype=torch.bool);rows=[]
    for k in [10,17,21]:
      for seed in range(6):
        d=OUT/f'jobs/k{k}-s{seed}-canonical_repulsive';m=torch.load(d/'searched_model.pt',map_location='cpu',weights_only=True);rec=json.loads((d/'receipt.json').read_text())
        W=m['W'];B=m['B'];j=W[:,0,0]
        assert not torch.any(B) and torch.equal(W,j[:,None,None]*eye) and not torch.any(j<0)
        floor=float(j[truth].min());ceiling=rec['statistics']['tau_train']+.001
        valid_graph=bool((j[truth]>0).all());cert=valid_graph and floor>ceiling
        rows.append({'k':k,'seed':seed,'all_810_true_peer_weights_positive':valid_graph,'nonnegative_extra_edges':int(((j>0)&~truth).sum()),'minimum_true_peer_weight':floor,'runtime_tau_plus_tolerance':ceiling,'exact_energy_gap':floor-ceiling,'sufficient_real_arithmetic_certificate':cert,'model_sha256':sha(d/'searched_model.pt'),'receipt_sha256':sha(d/'receipt.json')})
    result={'status':'completed_posthoc_weight_audit','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'proof':'With nonnegative J and B=0, any invalid Sudoku board has at least one equal-digit true peer pair, hence E>=min_true_peer_J. If tau+tol<that minimum, exact E<=tau+tol implies Sudoku validity. Extra nonnegative edges may reject valid boards but cannot invalidate this implication.','geometry_scope':'Ground-truth peer graph is used only for this post-hoc sufficient-bound audit, never in parameter transforms, fit or inference.','numerical_limit':'This proves an exact-energy implication for the saved coefficients, not a worst-case floating-point error bound for incremental Triton energies. The runtime certificate can differ under numerical drift; saved returned boards are independently validated separately.','source_sha256':sha(__file__),'rows':rows,'certified_models':sum(r['sufficient_real_arithmetic_certificate'] for r in rows),'total_models':len(rows)}
    (OUT/'repulsive-certificate-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
