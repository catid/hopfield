"""Posthoc decomposition of H3u false stopped states; no fit or search."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3u';OUT=ROOT/'artifacts/research/20261114/loki/h3u';PUBLIC=ROOT/'results/research/loki/h3u'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):return dict(min=float(a.min()),max=float(a.max()),mean=float(a.mean()))
def main():
    torch.set_num_threads(1);m=torch.load(OUT/'models/bp-adamw-grouped/model.pt',weights_only=False);design=np.load(OUT/'source-design.npz');J=m['S'][:,:,1,1].numpy().astype(np.float64);h=m['B'][:,1].numpy().astype(np.float64)
    i,j=np.triu_indices(169,1);dr=j//13-i//13;dc=j%13-i%13;source_lookup={tuple(o):int(c) for o,c in zip(design['offsets'],design['counts'])}
    source_seen=np.array([(int(r),int(c)) in source_lookup for r,c in zip(dr,dc)]);line=(dr==0)|(dc==0)|(abs(dr)==abs(dc));camel=((dr==1)&(abs(dc)==3))|((dr==3)&(abs(dc)==1));extras=np.array([source_lookup.get((int(r),int(c)),1)==0 for r,c in zip(dr,dc)])&~line&~camel
    masks=dict(inferred_line=line,camel=camel,source_global_extra=extras,other_source_finite=source_seen&~line&~camel&~extras,new_offline=~source_seen&~line)
    assert sum(v.astype(int) for v in masks.values()).min()==1 and sum(v.astype(int) for v in masks.values()).max()==1
    states=[];givens=[];paths=[SRC/'mechanism.py',SRC/'protocol.json',OUT/'models/bp-adamw-grouped/model.pt',OUT/'source-design.npz',PUBLIC/'summary.json'];positions=[]
    for seed in [0,1,2]:
        path=OUT/f'search/bp-adamw-grouped/s{seed}/records.npz';paths.append(path);z=np.load(path);boards=np.unpackbits(z['boards'],axis=-1)[...,:169].astype(bool);wrong=z['stopped']&~z['valid'];a,b=np.where(wrong)
        states.append(boards[a,b]);givens.append(z['inputs'][a]);positions += [dict(seed=seed,input_index=int(x),chain_index=int(y)) for x,y in zip(a,b)]
    x=np.concatenate(states).astype(np.float64);g=np.concatenate(givens);assert len(x)==9965 and (x.sum(1)==13).all();board=x.reshape(-1,13,13);assert (board.sum(1)==1).all() and (board.sum(2)==1).all()
    occupied=x[:,i]*x[:,j];components={k:occupied[:,v]@J[i[v],j[v]] for k,v in masks.items()};components['unary']=x@h;energy=sum(components.values());assert np.max(abs(energy-(.5*x*(x@J)).sum(1)-x@h))<1e-10
    field=x@J+h;cost=(1-2*x)*field;cost[g]=np.inf;minimum=cost.min(1);assert (minimum>=m['delta']-.001).all();conflict=(occupied[:,line|camel]).sum(1);assert (conflict>0).all()
    sample=0;cells=np.flatnonzero(x[sample]);edges=[]
    for a,b in zip(i[line|camel],j[line|camel]):
        if x[sample,a] and x[sample,b]:edges.append(dict(i=int(a),j=int(b),dr=int(b//13-a//13),dc=int(b%13-a%13),coupling=float(J[a,b]),i_given=bool(g[sample,a]),j_given=bool(g[sample,b]),i_remove_cost=float(-(field[sample,a])),j_remove_cost=float(-(field[sample,b]))))
    report=dict(status='validated_posthoc',wrong_stopped_states=len(x),all_q13=True,all_one_queen_per_row_and_column=True,
       conflict_edges=dict(diagonal=int(occupied[:,(dr!=0)&(dc!=0)&(abs(dr)==abs(dc))].sum()),camel=int(occupied[:,camel].sum())),
       minimum_free_margin=stats(minimum),frozen_required_margin=m['delta']-.001,energy_components={k:stats(v) for k,v in components.items()},new_offline_occupied_pairs=stats(occupied[:,masks['new_offline']].sum(1)),
       new_offline_interpretation='Exactly two newly representable off-line occupied pairs appear in every false stop. Each carries the shared default−1coefficient. Their total−2energy is constant within these states, while incident local margins still include those terms. No observed false conflict uses a new endpoint line offset.',
       example=dict(**positions[sample],given_cells=np.flatnonzero(g[sample]).tolist(),occupied_cells=cells.tolist(),true_conflict_edges=edges,energy=float(energy[sample]),minimum_free_margin=float(minimum[sample]),energy_components={k:float(v[sample]) for k,v in components.items()}),
       interpretation='Positive diagonal/camel coefficients do not imply constraint enforcement: negative couplings to other occupied cells can leave every free-cell removal cost above the source-calibrated threshold even when true conflicts remain. This is posthoc mechanism evidence, not a tuned repair.',
       input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    (PUBLIC/'false-stop-mechanism.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
