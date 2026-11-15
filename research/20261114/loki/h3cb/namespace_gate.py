"""Fixed common-selector fixtures with different tree weights and paid labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from common import SRC, save, read, sha, load
from compiler import Budget
from ledger import Ledger, canonical
from oracle import PaidOracle
from arm import ArmRunner
from audit_io import AuditIO
from audit_queries import PaidQueries
from audit_arm import verify as verify_arm
from endpoint import summarize


def run(folder,budget=None):
    folder.mkdir(parents=True,exist_ok=False)
    predictions=np.eye(20,dtype=bool);predictions[19]=predictions[18]
    bits=np.packbits(predictions,axis=1,bitorder='little')
    boards=np.arange(20,dtype=np.uint8)[:,None]
    sources=[dict(case=i,candidate=truth,initial_candidates=candidates,
                  source_record_sha256=hashlib.sha256(canonical(dict(case=i,fixture='fixed_common_selector'))).hexdigest())
             for i,(truth,candidates) in enumerate([(0,list(range(20))),(19,list(range(20))),(0,[0,19]),(0,[0,19]),(0,[0]),(19,[18,19]),(0,list(range(6))),(5,list(range(6)))])]
    scores={'H':[str(1<<i) for i in range(20)],'T':[str(1<<(19-i)) for i in range(20)]}
    ledger=Ledger(folder/'query-ledger.jsonl')
    oracle=PaidOracle(ledger,boards,bits)
    budget=Budget(200000,1000000) if budget is None else budget
    before_states,before_work=budget.states,budget.splits
    statuses=[dict(case=i,H='pending',T='pending') for i in range(len(sources))]
    runners={}
    for arm in ['H','T']:
        target=folder/arm;target.mkdir()
        np.save(target/'predictions-bits.npy',bits);save(target/'selector-scores.json',scores['H']);save(target/'tree-scores.json',scores[arm])
        runner=ArmRunner(arm,target,predictions,scores['H'],scores[arm],budget,100000,len(boards))
        runners[arm]=runner;runner.run(sources,ledger,oracle.ask,statuses)
    ledger.close()
    assert runners['H'].compiler is not runners['T'].compiler
    assert runners['H'].compiler.cache is not runners['T'].compiler.cache
    assert runners['H'].cost_model.cache is not runners['T'].cost_model.cache
    from audit_namespace_gate import verify as replay_fixture
    checks=0
    def check(name,value):
        nonlocal checks
        assert value,name
        checks+=1
    io=AuditIO(folder,SRC,check)
    replayed=replay_fixture(folder,io,check,budget.states-before_states,budget.splits-before_work)
    from pairing import pair
    paired=[pair(h,t,scores['T'],[runners['H'].handoff_proof(h),runners['T'].handoff_proof(t)])
            for h,t in zip(runners['H'].results,runners['T'].results)]
    assert summarize(paired)==replayed['common_routing_fixture']
    assert [r['matched_expected_tree_costs'] for r in paired]==replayed['matched_expected_costs']
    for fixture,expected in [([dict(H_calls=100,T_calls=98,H_exact=True,T_exact=True)],'supported'),
                            ([dict(H_calls=100,T_calls=98,H_exact=False,T_exact=False)],'supported'),
                            ([dict(H_calls=100,T_calls=99,H_exact=True,T_exact=True)],'falsified'),
                            ([dict(H_calls=0,T_calls=0,H_exact=True,T_exact=True)],'inconclusive_zero_comparator')]:
        assert summarize(fixture)['decision']==expected
    for h,t in [(True,False),(False,True)]:
        try:summarize([dict(H_calls=100,T_calls=98,H_exact=h,T_exact=t)])
        except AssertionError:pass
        else:raise AssertionError('Unequal exact flags accepted')
    storage=io.finish()
    from role_gate import run as role_gate
    roles=role_gate(folder,budget.states-before_states,budget.splits-before_work)
    result=dict(status='passed_common_selector_tree_weight_namespace_and_payment_gate',checks=checks,
                actual_oracle_calls=0,fresh_truth_draws=0,fresh_source_draws=0,
                synthetic_memo_states=budget.states-before_states,synthetic_work_steps=budget.splits-before_work,
                **replayed,storage=storage,score_role_mutations=roles,separate_caches_and_selectors_verified=True,
                all_synthetic_queries_and_proofs_replayed=True,
                source_sha256={name:sha(SRC/name) for name in ['arm.py','oracle.py','endpoint.py','audit_arm.py',
                    'audit_queries.py','namespace_gate.py','audit_namespace_gate.py','pairing.py','audit_pairing.py','policy.py','counterfactual.py','audit_trees.py','audit_costs.py','role_gate.py']})
    save(folder/'summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    print(run(parser.parse_args().folder))
