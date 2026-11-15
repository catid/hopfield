"""H2bf finite exhaustive and fixed-boundary equivalence gate; no speed endpoint."""
import itertools
import json
import time
import numpy as np
import study as S
import connected_block_moves as OLD
import bridge_veto_geometry as NEW
from verify_large_cycle_geometry import raw_chains
from verify_graph_block_source import connected

def independent(t,b,limit):
    raw,stats=raw_chains(t,b,limit);kept=[]
    for cells in raw:
        changed=b.copy();changed.flat[list(cells)]=False
        if connected(t,changed):kept.append(cells)
    return kept,{**stats,"unfiltered_candidate_count":len(raw),"connectivity_vetoed_components":len(raw)-len(kept),"reachability_checks":len(raw)}

DEST=S.OUT/'veto_speed_preflight'


def fixtures():
    for shape,pairs in [((3,3),[(0,8)]),((2,3),itertools.permutations(range(6),2))]:
        for start,goal in pairs:
            free=[i for i in range(np.prod(shape)) if i not in [start,goal]]
            for values in itertools.product(range(3),repeat=len(free)):
                t=np.ones(shape,np.int8);b=np.zeros(shape,bool);t.flat[start]=2;t.flat[goal]=3;b.flat[start]=b.flat[goal]=True
                for i,value in zip(free,values):t.flat[i]=0 if value==2 else 1;b.flat[i]=value==1
                yield 'exhaustive',t,b
    for si,shape in enumerate([(5,7),(7,5),(11,13)]):
        for i in range(32):
            rng=np.random.default_rng(np.random.SeedSequence([20261115,2,64,si,i]));t=np.where(rng.random(shape)<.2,0,1).astype(np.int8);b=(t==1)&(rng.random(shape)<.65);t[0,0]=2;t[-1,-1]=3;b[0,0]=b[-1,-1]=True;yield 'rectangular',t,b
    for length in [255,256,257,511,512,513]:
        t=np.ones((1,length+2),np.int8);t[0,0]=2;t[0,-1]=3;b=np.ones_like(t,bool);yield 'boundary',t,b
        t=np.ones((3,length+2),np.int8);t[1,1:-1]=0;t[0,0]=2;t[0,-1]=3;b=t!=0;yield 'boundary',t,b


def main():
    started=time.perf_counter();counts={};candidates=0
    for family,t,b in fixtures():
        old=OLD.candidates(t,b);new=NEW.candidates(t,b);check=independent(t,b,256);assert old==new==check,(family,t.tolist(),b.tolist(),old,new,check);counts[family]=counts.get(family,0)+1;candidates+=len(old[0]);assert time.perf_counter()-started<120
    assert counts=={'exhaustive':4617,'rectangular':96,'boundary':12};DEST.mkdir(exist_ok=True);S.save(DEST/'synthetic-gate.json',{'status':'passed','counts':counts,'candidate_tuples':candidates,'seconds':time.perf_counter()-started,'proposal_sha256':S.sha(S.HERE/'veto-speed-proposal.json'),'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in ['bridge_veto_geometry.py','preflight_veto_geometry.py','graph_block_moves.py','lexical_cycle_cut_moves.py','verify_large_cycle_geometry.py']},'scope':'Legacy, bridge traversal and independent NumPy geometry identical on every fixed fixture. Gate wall time only; no benchmark speed endpoint.'})


if __name__=='__main__':main()
