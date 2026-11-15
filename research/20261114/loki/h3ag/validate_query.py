"""Synthetic cache-index/query gates; no experimental model or target data."""
import numpy as np
import ast
import time
from common import OUT,SRC,save,sha,now
from query import solve

def main():
    boards=np.zeros((3,225),bool)
    for k in range(3):boards[k,15*np.arange(15)+(np.arange(15)+k)%15]=True
    packed=np.packbits(boards.T,axis=1,bitorder='little');index=[int.from_bytes(row.tobytes(),'little')for row in packed];inputs=np.zeros((4,225),bool);inputs[0,0]=True;inputs[1,1]=True;inputs[2,[0,17]]=True
    for bank in[0,1]:
        for seed in[0,1,2]:
            z=solve(boards,index,inputs,bank,seed);expected=np.array([boards[:,g].all(1).sum()for g in inputs]);assert np.array_equal(z['available'],expected)and expected.tolist()==[1,1,0,3];assert np.array_equal(z['abstained'],expected==0)
            for t in range(4):
                if expected[t]:assert(z['boards'][t].sum(-1)==15).all()and(z['boards'][t]>=inputs[t]).all()and len(np.unique(z['indices'][t]))==min(20,expected[t])
                else:assert not z['boards'][t].any()and not z['stopped'][t].any()and(z['indices'][t]==-1).all()
            assert np.array_equal(z['indices'],solve(boards,index,inputs,bank,seed)['indices'])
    old=ast.parse((SRC.parent/'h3ae/compiler.py').read_text());new=ast.parse((SRC/'compiler.py').read_text())
    for name in['forward','reverse_literal']:
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name==name);assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    save(OUT/'query-validation.json',dict(status='passed',utc=now(),synthetic_boards=3,synthetic_queries=4,family_seed_combinations=6,literal_match_counts=True,abstention_and_unique_then_padding_verified=True,reproducible_sample_stream=True,enumerator_ASTs_identical_to_frozen_H3ae=True,no_experimental_model_or_target_data=True,query_sha256=sha(SRC/'query.py'),validation_sha256=sha(__file__)));print('QUERY_GATE_PASSED')
if __name__=='__main__':main()
