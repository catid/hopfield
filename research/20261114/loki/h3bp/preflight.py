"""Before-freeze toy gates and validation of already-public feasible witnesses."""
from fractions import Fraction
import importlib.util
import itertools
import json
import math
import time
from common import ROOT,SRC,OUT,read,save,sha,now
from compiler import Budget,ResourceCap,compile_tree
from weights import integer_weights
from weight_mode import uniform_weights
from synthetic_gate import run as synthetic_gate
from proof_gate import require_ready_proof


def main():
    start=time.perf_counter();assert not(SRC/'protocol.json').exists();OUT.mkdir(parents=True,exist_ok=True);budget=Budget();toy=synthetic_gate(budget);scores=[str(math.comb(55,k)*math.comb(M,8)) for M,k in [(8,1),(15,2),(16,0)]];weights,receipt=integer_weights([0,1,2],scores);assert all(Fraction(w,weights[0])==Fraction(int(scores[0]),int(d)) for w,d in zip(weights,scores));large=str(10**5000);assert len(large)==5001 and int(json.loads(json.dumps(dict(value=large)))['value'])==10**5000
    uniform,uniform_receipt=uniform_weights([7,8,9]);uniform_proof=compile_tree([7,8,9],uniform,[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate([0,1,2,4,7])],2,budget);assert uniform==[1,1,1] and uniform_proof['optimal_cost']=='5' and uniform_proof['states'][uniform_proof['root_state']]['chosen_split']['board_index']==1
    save(SRC/'preflight-synthetic-gate.json',toy);folder=OUT/'preflight-proofs';folder.mkdir(exist_ok=True);proof=compile_tree([0,1],[1,1],[dict(signature=1,lexfirst_board_index=0,board_count=1)],4,budget);save(folder/'valid.json',proof);assert require_ready_proof(folder/'valid.json',sha(folder/'valid.json'),[0,1])==proof;save(folder/'invalid.json',dict(candidates=[0,1],depth_limit=4,feasible=False));rejected={}
    for name,path,digest in [('missing',folder/'absent.json','0'*64),('hash_mismatch',folder/'valid.json','0'*64),('infeasible_or_capped',folder/'invalid.json',sha(folder/'invalid.json'))]:
        try:require_ready_proof(path,digest,[0,1])
        except RuntimeError:rejected[name]=True
        else:raise AssertionError('Proof gate failed '+name)
    dependency=ROOT/'research/20261114/loki/h3bi/verify_public.py';spec=importlib.util.spec_from_file_location('h3bp_old_witness_gate',dependency);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);M=ROOT/'results/research/loki/h3bm';cases=read(M/'case-results.json');groups=read(M/'groups.json');ids=sorted({r['arms']['8']['tree_group'] for r in cases if r['arms']['8']['compiler_feasible']});checks=[]
    def check(name,value):assert value,name;checks.append(name)
    for group in ids:
        proof=read(M/f'group{group}/proof.json');catalog=read(M/f'group{group}/catalog.json')['signatures'];helper.certificate(proof,catalog,check,f'group{group}');assert proof['feasible'] and proof['depth_limit']==4
    assert len(ids)==439 and sum(r['arms']['8']['compiler_feasible'] is True for r in cases)==1067 and sum(r['arms']['8']['compiler_feasible'] is False for r in cases)==54 and sum(r['arms']['8']['tree_group'] is None for r in cases)==287
    result=dict(utc=now(),weighted_synthetic_gates_passed=True,weighted_synthetic_gate_sha256=sha(SRC/'preflight-synthetic-gate.json'),C_M_8_inherited_MAP_weight_fixture=receipt,uniform_weight_fixture=uniform_receipt,uniform_three_model_optimal_cost='5',uniform_three_model_lexfirst_board=1,large_integer_decimal_serialization_digits=5001,missing_mismatched_infeasible_proofs_rejected=rejected,all439_prior_feasible_depth4_witnesses_validated=True,prior_witness_checks=len(checks),parent_case_results_sha256=sha(M/'case-results.json'),parent_groups_sha256=sha(M/'groups.json'),independent_witness_verifier_sha256=sha(dependency),elapsed_seconds=time.perf_counter()-start);save(SRC/'preflight.json',result);print(dict(preflight='passed',checks=len(checks),elapsed_seconds=result['elapsed_seconds']))


if __name__=='__main__':main()
