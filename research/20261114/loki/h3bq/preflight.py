"""Toy gates and seed collision audit before any prospective study outcome."""
import itertools
import json
import math
import tempfile
import time
from pathlib import Path
import numpy as np
from common import ROOT,SRC,read,save,sha,now
from compiler import Budget,compile_tree
import synthetic_gate
import firstlex_gate
from weights import integer_weights


def main():
    start=time.perf_counter();assert not(SRC/'protocol.json').exists();budget=Budget()
    synthetic_gate.OUT=SRC;firstlex_gate.OUT=SRC
    w=synthetic_gate.run(budget);f=firstlex_gate.run(budget)
    (SRC/'synthetic-gate.json').rename(SRC/'preflight-weighted-gate.json');(SRC/'firstlex-synthetic-gate.json').rename(SRC/'preflight-firstlex-gate.json')
    fixtures=[]
    for M,k in [(7,0),(8,1),(15,2),(16,0),(20,2)]:
        count=sum(1 for _ in itertools.combinations(range(M),8));prior=sum(1 for _ in itertools.combinations(range(55),k));score=math.comb(55,k)*math.comb(M,8) if M>=8 else None
        assert score==(prior*count if count else None);fixtures.append(dict(population=M,orbit_count=k,eight_subsets=count,prior_subsets=prior,score=str(score) if score is not None else None))
    toy=np.arange(20,dtype=np.uint8)[:,None];indices=np.random.default_rng(813).choice(20,8,replace=False);assert len(set(indices.tolist()))==8 and np.array_equal(toy[indices],indices[:,None])
    large=str(10**5000);assert len(large)==5001 and int(json.loads(json.dumps(large)))==10**5000
    old={};paths={}
    def walk(x,path,context=''):
        if isinstance(x,dict):
            for k,v in x.items():walk(v,path,context+'/'+k)
        elif isinstance(x,list):
            for v in x:walk(v,path,context)
        elif isinstance(x,int) and 'seed' in context.lower():old.setdefault(x,[]).append(str(path.relative_to(ROOT))+':'+context)
    for path in sorted(SRC.parent.glob('*/protocol.json')):paths[str(path.relative_to(ROOT))]=sha(path);walk(read(path),path)
    truths=read(SRC.parent/'h3bf/protocol.json')['truths'];seeds=[2026600000+64*r['candidate']+b for r in truths for b in range(64)];collisions={str(s):old[s] for s in seeds if s in old};assert len(seeds)==len(set(seeds))==1408 and not collisions
    save(SRC/'seed-audit.json',dict(utc=now(),new_seed_formula='2026600000+64*candidateID+bank',new_seed_values=seeds,historical_explicit_seed_values=sorted(old),historical_protocol_sha256=paths,collisions=collisions,scope='Explicit integer fields under seed-named keys in prior Loki protocols; no claim about undocumented RNG streams.'))
    result=dict(utc=now(),all_toy_gates_passed=True,weighted_gate_sha256=sha(SRC/'preflight-weighted-gate.json'),firstlex_gate_sha256=sha(SRC/'preflight-firstlex-gate.json'),eight_likelihood_fixtures=fixtures,eight_direct_draw_fixture=dict(seed=813,population=20,indices=indices.tolist()),large_integer_decimal_serialization_digits=5001,seed_audit_sha256=sha(SRC/'seed-audit.json'),new_source_draws=0,new_target_calls=0,elapsed_seconds=time.perf_counter()-start)
    save(SRC/'preflight.json',result);print(result)


if __name__=='__main__':main()
