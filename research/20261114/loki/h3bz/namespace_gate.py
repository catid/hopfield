"""Fixed fixtures exercise score isolation, disagreement, and separately paid calls."""
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
    predictions=np.eye(20,dtype=bool)
    bits=np.packbits(predictions,axis=1,bitorder='little')
    boards=np.arange(20,dtype=np.uint8)[:,None]
    sources=[dict(case=i,candidate=truth,initial_candidates=candidates,
                  source_record_sha256=hashlib.sha256(canonical(dict(case=i,fixture='fixed_one_hot'))).hexdigest())
             for i,(truth,candidates) in enumerate([(0,list(range(20))),(19,list(range(20))),(0,[0,19]),(0,[0,19])])]
    scores={'H':list(map(str,range(1,21))),'U':list(map(str,range(20,0,-1)))}
    ledger=Ledger(folder/'query-ledger.jsonl')
    oracle=PaidOracle(ledger,boards,bits)
    budget=Budget(200000,1000000) if budget is None else budget
    before_states,before_work=budget.states,budget.splits
    statuses=[dict(case=i,H='pending',U='pending') for i in range(4)]
    runners={}
    for arm in ['H','U']:
        target=folder/arm;target.mkdir()
        np.save(target/'predictions-bits.npy',bits);save(target/'score-values8.json',scores[arm])
        runner=ArmRunner(arm,target,predictions,scores[arm],budget,100000,len(boards))
        runners[arm]=runner;runner.run(sources,ledger,oracle.ask,statuses)
    ledger.close()
    assert runners['H'].compiler is not runners['U'].compiler
    assert runners['H'].compiler.cache is not runners['U'].compiler.cache
    assert runners['H'].cost_model.cache is not runners['U'].cost_model.cache
    from audit_namespace_gate import verify as replay_fixture
    checks=0
    def check(name,value):
        nonlocal checks
        assert value,name
        checks+=1
    io=AuditIO(folder,SRC,check)
    replayed=replay_fixture(folder,io,check,budget.states-before_states,budget.splits-before_work)
    assert summarize([dict(H_calls=h['membership_calls'],U_calls=u['membership_calls'],
                     H_exact=h['exact_true_singleton'],U_exact=u['exact_true_singleton'])
                     for h,u in zip(runners['H'].results,runners['U'].results)])==replayed['exact_disagreement_fixture']
    for fixture,expected in [([dict(H_calls=100,U_calls=98,H_exact=True,U_exact=True)],'supported'),
                            ([dict(H_calls=100,U_calls=98,H_exact=False,U_exact=True)],'supported'),
                            ([dict(H_calls=100,U_calls=98,H_exact=True,U_exact=False)],'falsified'),
                            ([dict(H_calls=100,U_calls=99,H_exact=True,U_exact=True)],'falsified'),
                            ([dict(H_calls=0,U_calls=0,H_exact=True,U_exact=True)],'inconclusive_zero_comparator')]:
        assert summarize(fixture)['decision']==expected
    storage=io.finish()
    result=dict(status='passed_two_arm_namespace_disagreement_and_payment_gate',checks=checks,
                actual_oracle_calls=0,fresh_truth_draws=0,fresh_source_draws=0,
                synthetic_memo_states=budget.states-before_states,synthetic_work_steps=budget.splits-before_work,
                **replayed,storage=storage,separate_caches_and_selectors_verified=True,
                all_synthetic_queries_and_proofs_replayed=True,
                source_sha256={name:sha(SRC/name) for name in ['arm.py','oracle.py','endpoint.py','audit_arm.py',
                    'audit_queries.py','namespace_gate.py','audit_namespace_gate.py']})
    save(folder/'summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    print(run(parser.parse_args().folder))
