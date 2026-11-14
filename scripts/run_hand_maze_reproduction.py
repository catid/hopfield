"""Hand-set Maze-Unique development gates and one full test pass per inference seed."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'upstream'
OUT=ROOT/'artifacts/fresh_maze/hand_maze_unique'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['CRAM_DATA']=str(UP/'data')
sys.path[:0]=[str(UP),str(UP/'experiments'),str(UP/'baselines/flops')]
import numpy as np
import torch
from run_maze_reproduction import path_audit,patched,save,sha256,utc

torch.set_num_threads(1);torch.set_num_interop_threads(1)
import hand_set as H
from cram import maze_unique as M
import cram_kernels_extra as C


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    H.OUT=OUT
    H.main_check()
    # Independent exhaustive local-neighborhood cost check (all 2^8 patterns).
    model=H.maze_hand();w=model.kernel().detach().numpy()[1,1]
    offsets=[(i,j) for i in range(3) for j in range(3) if (i,j)!=(1,1)]
    maxerr=0.
    for bits in range(256):
        p=np.zeros((3,3),float)
        for k,ij in enumerate(offsets):p[ij]=(bits>>k)&1
        cost=float(-model.b[1])+float((p*w).sum())
        sides=p[0,1]+p[1,0]+p[1,2]+p[2,1]
        diags=p[0,0]+p[0,2]+p[2,0]+p[2,2]
        expected=1.5-sides-.1*diags
        maxerr=max(maxerr,abs(cost-expected))
        assert abs(cost-expected)<1e-6
        assert (cost<0)==(sides>=2)
    save(OUT/'source_consistency.json',dict(stored_values=sum(p.numel() for p in model.parameters()),
         learned_parameters=0,paper_effective_nonzero_values=[-1.0,-.1,1.5],
         effective_coefficient_count=3,local_neighbor_patterns_checked=256,max_abs_field_error=maxerr,
         all_patterns_dead_end_rule_exact=True,mapping='No model conversion: official factorized 16-value state used directly; effective current coefficients checked independently.'))
    save(OUT/'provenance.json',dict(started_utc=utc(),hostname=platform.node(),torch=torch.__version__,python=sys.version,
         device='cpu',threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
         upstream_commit='62bbe830c7d6f87f2ba18f30a28039258b986454',
         script_sha256=sha256(__file__),source_sha256={str(p.relative_to(UP)):sha256(p) for p in [UP/'experiments/hand_set.py',UP/'experiments/maze_unique.py',UP/'cram/maze_unique.py',UP/'cram/cram_maze.py']},
         dataset_sha256={str(p.relative_to(UP/'data')):sha256(p) for p in sorted((UP/'data/eqr/maze-30x30-unique-1k').glob('*/*.npy'))}))
    summaries=[]
    for seed in range(3):
        final=OUT/f'summary-s{seed}.json'
        if final.exists():
            summaries.append(json.loads(final.read_text()));continue
        H.main_maze('dev100',seed,'cpu')
        captured=[];orig=M.gibbs
        def capture(*a,**kw):
            r=orig(*a,**kw)
            captured.append(dict(inputs=a[1].cpu().numpy(),labels=a[2].cpu().numpy(),predictions=r['pred'].cpu().numpy(),stopped=r['stopped'].cpu().numpy()))
            return r
        with patched(M,'gibbs',capture):H.main_maze('test',seed,'cpu')
        arrays={k:np.concatenate([a[k] for a in captured]) for k in captured[0]}
        audit=path_audit(arrays['inputs'],arrays['predictions'],arrays['labels'])
        np.savez_compressed(OUT/f'test_predictions-s{seed}.npz',**arrays)
        save(OUT/f'independent_test_audit-s{seed}.json',audit)
        result=json.loads(H.maze_out('test',seed).read_text())
        assert audit['exact_solved']==result['solved'] and audit['n']==1000
        rec=dict(seed=seed,**{k:result[k] for k in ['n','solved','wrong_stops','budget_failures','mean_flop']},
                 independent_shortest_paths=audit['shortest_paths'],completed_utc=utc())
        save(final,rec);summaries.append(rec)
    # Count once on training inputs; hand-set weights are identical across seeds.
    rows=[];data=M.load('train')
    for i in range(6):
        x,y,fr=M.to_device(data,[i],torch.device('cpu'));h={}
        with torch.no_grad():count=C._count(lambda:h.setdefault('r',C.gibbs_one(model,x,y,fr)))
        rows.append(dict(maze=i,m=int(h['r']['moves'][0]),count=count))
    fit=C.linear(rows)
    assert fit['total']['max_abs_residual']==0
    for rec in summaries:
        result=json.loads(H.maze_out('test',rec['seed']).read_text());steps=np.asarray(result['per_maze']['steps'])
        fl=fit['total']['setup']+fit['total']['per']*steps
        rec['fc2_flops_mean']=float(fl.mean())
        save(OUT/f"summary-s{rec['seed']}.json",rec)
    save(OUT/'fc2_flops.json',dict(convention=C.CONVENTION,fit=fit,training_rows=rows,formula='setup + per * saved_test_moves',
         source_counter_sha256=sha256(UP/'baselines/flops/fullcount.py'),summaries=summaries))
    save(OUT/'summary.json',dict(seeds=summaries))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
