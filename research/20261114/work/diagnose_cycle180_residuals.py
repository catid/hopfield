"""Post-H2av candidate-size attribution, without any solver trajectories."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
import time
import numpy as np
import study as S
from verify_training_feasibility import energy,LD
from verify_graph_block_source import connected
from verify_cycle_cut_geometry_checked import geometry,cycle_rank
from cycle_transfer_180 import PP,DEST


def chains(tokens,board):
    h,w=board.shape;degree=np.zeros_like(tokens,np.int8);degree[1:]+=board[:-1];degree[:-1]+=board[1:];degree[:,1:]+=board[:,:-1];degree[:,:-1]+=board[:,1:];remaining=set(map(int,np.flatnonzero(board&(tokens==1)&(degree>=1)&(degree<=2))));parts=[]
    while remaining:
        u=min(remaining);remaining.remove(u);stack=[u];part={u}
        while stack:
            u=stack.pop();r,c=divmod(u,w)
            for v in ([u-w] if r else [])+([u+w] if r+1<h else [])+([u-1] if c else [])+([u+1] if c+1<w else []):
                if v in remaining:remaining.remove(v);part.add(v);stack.append(v)
        parts.append(tuple(sorted(part)))
    return sorted(parts),degree


def one(args):
    row,manifest=args;job=row['job'];targets={v:np.flatnonzero(~np.array(row['success'][v],bool)) for v in [3,4]};allbad=sorted(set(i for ids in targets.values() for i in ids));out=[]
    if not allbad:return out
    with np.load(S.ROOT/manifest['datasets'][job['shard']]['path']) as z:tokens=z['tokens'][allbad]
    with np.load(DEST/'runs'/job['id']/'outputs.npz') as z:pred=z['predictions'][:,allbad];readout=z['couplings'][allbad].astype(LD)
    with np.load(DEST/f'model-seed{job["seed"]}-old_binary.npz') as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
    for v,ids in targets.items():
        for original in ids:
            i=allbad.index(original);tok=tokens[i];board=pred[v,i];parts,degree=chains(tok,board);old,_=geometry(tok,board);assert not old;rank=cycle_rank(board);candidates=[]
            for cells in parts:
                changed=board.copy();changed.flat[list(cells)]=False;after=cycle_rank(changed);reach=connected(tok,changed)
                if not reach or after>=rank:continue
                delta=float((energy(tok[None],changed[None],readout[i:i+1],theta,mu,ao)-energy(tok[None],board[None],readout[i:i+1],theta,mu,ao))[0]);candidates.append({'size':len(cells),'cycle_rank_after':after,'original_energy_delta':delta,'eligible_at256':2<=len(cells)<=256,'eligible_at512':2<=len(cells)<=512,'cells':list(cells)})
            out.append({'job':job['id'],'seed':job['seed'],'input_index':job['shard']*256+int(original),'arm':['energy_cycle_cut','lexical_cycle_cut'][v-3],'selected_cells':int(board.sum()),'cycle_rank':rank,'degree_counts':dict(Counter(map(int,degree[board]))),'all_chain_sizes':[len(x) for x in parts],'admitted_cycle_reducing_components_without_size_filter':candidates,'eligible_256':sum(x['eligible_at256'] for x in candidates),'eligible_512':sum(x['eligible_at512'] for x in candidates)})
    return out


def main():
    started=time.perf_counter();a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed';manifest=json.loads((DEST/'datasets.json').read_text())
    with ProcessPoolExecutor(max_workers=16) as pool:rows=[x for group in pool.map(one,[(r,manifest) for r in a['rows']]) for x in group]
    summary={}
    for arm in ['energy_cycle_cut','lexical_cycle_cut']:
        group=[r for r in rows if r['arm']==arm];summary[arm]={'failed_model_inputs':len(group),'distinct_inputs':len({r['input_index'] for r in group}),'cases_with_eligible512_component':sum(r['eligible_512']>0 for r in group),'eligible_unbounded_size_histogram':dict(Counter(x['size'] for r in group for x in r['admitted_cycle_reducing_components_without_size_filter']))}
    r={'status':'complete','protocol_sha256':S.sha(PP),'audit_sha256':S.sha(DEST/'audit.json'),'source_sha256':S.sha(__file__),'summary':summary,'rows':rows,'seconds':time.perf_counter()-started,'new_searches':0,'scope':'Posthoc independent geometry and literal cost attribution on allfailedH2avcycleoutputs; no solver/native trajectories or newinputs. Counts potential newlyadmittedmoves, not claimedrescues.'};S.save(DEST/'residual-size-diagnosis.json',r);print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
