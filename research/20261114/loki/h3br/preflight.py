"""Exact full-ticket prior audit and seed gates before any prospective outcomes."""
import itertools
import json
import math
import time
import numpy as np
from common import ROOT,SRC,read,save,sha,now
from compiler import Budget
import synthetic_gate
import firstlex_gate
from prior import prior_rows,truth_from_ticket


def main():
    start=time.perf_counter();assert not(SRC/'protocol.json').exists();budget=Budget();synthetic_gate.OUT=SRC;firstlex_gate.OUT=SRC;synthetic_gate.run(budget);firstlex_gate.run(budget)
    (SRC/'synthetic-gate.json').rename(SRC/'preflight-weighted-gate.json');(SRC/'firstlex-synthetic-gate.json').rename(SRC/'preflight-firstlex-gate.json')
    normalizers=read(SRC.parent/'h3bf/protocol.json')['normalizers'];selection=read(ROOT/normalizers['selection_path']);packed=np.load(ROOT/normalizers['bits_path'])['bits'];counts=np.unpackbits(packed,axis=1,count=14200,bitorder='little').sum(1);assert counts.tolist()==[r['complete_source_cardinality'] for r in selection['candidates']]
    rows=prior_rows(counts)
    assert all(row['candidate']==old['candidate'] and row['orbits']==old['orbits'] and row['source_population']==old['complete_source_cardinality'] for row,old in zip(rows,selection['candidates']))
    mass=sum(r['prior_integer_weight'] for r in rows);mapped=[truth_from_ticket(rows,ticket) for ticket in range(mass)];multiplicities=np.bincount(mapped,minlength=1541).tolist();assert mass==4436 and multiplicities==[r['prior_integer_weight'] for r in rows] and sum(r['eligible'] for r in rows)==1522
    for row in rows:assert row['prior_integer_weight']==(1485//math.comb(55,len(row['orbits'])) if row['source_population']>=8 else 0)
    save(SRC/'prior-gate.json',dict(utc=now(),conditional_total_mass=mass,eligible_models=1522,ineligible_models=19,eligible_counts_by_orbit_count={str(k):sum(r['eligible'] and len(r['orbits'])==k for r in rows) for k in [0,1,2]},integer_weights_by_orbit_count={'0':1485,'1':27,'2':1},all4436_tickets_enumerated=True,ticket_to_candidate=mapped,model_ticket_multiplicities=multiplicities,prior_rows=rows,source_cardinalities_independently_recounted_from_packed_masks=True,source_selection_sha256=sha(ROOT/normalizers['selection_path']),source_bits_sha256=sha(ROOT/normalizers['bits_path'])))
    fixtures=[]
    for M,k in [(7,0),(8,1),(15,2),(16,0),(20,2)]:
        count=sum(1 for _ in itertools.combinations(range(M),8));prior=math.comb(55,k);score=math.comb(55,k)*math.comb(M,8) if M>=8 else None;assert score==(prior*count if count else None);fixtures.append(dict(population=M,orbit_count=k,eight_subsets=count,prior_subsets=prior,score=str(score) if score is not None else None))
    large=str(10**5000);assert len(large)==5001 and int(json.loads(json.dumps(large)))==10**5000
    old={};paths={}
    def walk(x,path,context=''):
        if isinstance(x,dict):
            for k,v in x.items():walk(v,path,context+'/'+k)
        elif isinstance(x,list):
            for v in x:walk(v,path,context)
        elif isinstance(x,int) and 'seed' in context.lower():old.setdefault(x,[]).append(str(path.relative_to(ROOT))+':'+context)
    for path in sorted(SRC.parent.glob('*/protocol.json')):paths[str(path.relative_to(ROOT))]=sha(path);walk(read(path),path)
    seeds=[seed for case in range(4096) for seed in [2026700000+case,2026800000+case]];collisions={str(s):old[s] for s in seeds if s in old};assert len(seeds)==len(set(seeds))==8192 and not collisions
    save(SRC/'seed-audit.json',dict(utc=now(),new_seed_formula=dict(truth='2026700000+case',source='2026800000+case'),new_seed_values=seeds,historical_explicit_seed_values=sorted(old),historical_protocol_sha256=paths,collisions=collisions,scope='Explicit integer fields under seed-named keys in prior Loki protocols; no claim about undocumented RNG streams.'))
    result=dict(utc=now(),all_toy_gates_passed=True,weighted_gate_sha256=sha(SRC/'preflight-weighted-gate.json'),firstlex_gate_sha256=sha(SRC/'preflight-firstlex-gate.json'),prior_gate_sha256=sha(SRC/'prior-gate.json'),eight_likelihood_fixtures=fixtures,large_integer_decimal_serialization_digits=5001,seed_audit_sha256=sha(SRC/'seed-audit.json'),new_truth_draws=0,new_source_draws=0,new_target_calls=0,elapsed_seconds=time.perf_counter()-start);save(SRC/'preflight.json',result);print(result)


if __name__=='__main__':main()
