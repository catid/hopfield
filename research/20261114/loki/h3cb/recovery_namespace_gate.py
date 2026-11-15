"""Real exact-policy arms plus paid fallback on fixed synthetic namespaces."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from common import SRC,save,sha,load
from compiler import Budget
from ledger import Ledger,canonical
from oracle import PaidOracle
from arm import ArmRunner
from audit_io import AuditIO
from audit_queries import PaidQueries
from audit_arm import verify as verify_arm
from fallback import ScanBudget
from recovery_pairing import pair
from recovery_endpoint import summarize
from audit_recovery import pair as independent_pair,endpoint as independent_endpoint


def run(folder,budget=None,scan=None):
    folder.mkdir(parents=True,exist_ok=False);budget=Budget() if budget is None else budget;scan=ScanBudget() if scan is None else scan
    before_states,before_work,before_cells,before_requests=budget.states,budget.splits,scan.cells,scan.requests
    pred=np.eye(24,dtype=bool);pred[23]=pred[22];bits=np.packbits(pred,axis=1,bitorder='little');boards=np.arange(24,dtype=np.uint8)[:,None]
    source_specs=[(0,list(range(24))),(4,list(range(24))),(7,list(range(24))),(21,list(range(24))),(23,list(range(24))),
                  (0,[0,1,2]),(0,[0]),(23,[22,23]),(4,list(range(24)))]
    sources=[dict(case=i,candidate=c,initial_candidates=active,source_record_sha256=hashlib.sha256(canonical(dict(case=i,fixture='fixed_budget_recovery'))).hexdigest()) for i,(c,active) in enumerate(source_specs)]
    selector=['1']*24;trees=['1']*24;statuses=[dict(case=i,B='pending',E='pending') for i in range(len(sources))]
    ledger=Ledger(folder/'query-ledger.jsonl');oracle=PaidOracle(ledger,boards,bits,arms=('B','E'));runners={}
    for arm in ['B','E']:
        target=folder/arm;target.mkdir();np.save(target/'predictions-bits.npy',bits);save(target/'selector-scores.json',selector);save(target/'tree-scores.json',trees)
        runner=ArmRunner(arm,target,pred,selector,trees,budget,100000,len(boards),scan_budget=scan)
        runners[arm]=runner;runner.run(sources,ledger,oracle.ask,statuses)
    paired=[pair(b,e) for b,e in zip(runners['B'].results,runners['E'].results)]
    for record in paired:ledger.append('paired_case_result',record)
    outcome=summarize(paired);ledger.append('study_complete',outcome);ledger.close()
    checks=0
    def check(name,value):
        nonlocal checks
        assert value,name;checks+=1
    chain=load(SRC.parent/'h3bj/verify_public.py','h3cb_recovery_fixture_chain').chain
    replay=PaidQueries(chain((folder/'query-ledger.jsonl').read_bytes()),pred,boards,check,arms=('B','E'));io=AuditIO(folder,SRC,check)
    audits={}
    for arm in ['B','E']:audits[arm]=verify_arm(arm,folder/arm,SRC,sources,runners[arm].results,pred,boards,selector,trees,replay,io,check)
    rebuilt=[]
    for b,e,stored in zip(runners['B'].results,runners['E'].results,paired):
        row=independent_pair(b,e,check);check('full_pair_and_ledger',row==stored==replay.take('paired_case_result')['data']);rebuilt.append(row)
    check('full_joint_endpoint_and_ledger',independent_endpoint(rebuilt)==outcome==replay.take('study_complete')['data'] and replay.at==len(replay.events))
    check('all_actual_calls_separately_paid',oracle.arm_calls==replay.calls and sum(replay.calls.values())==outcome['actual_membership_calls'])
    check('both_rescued_and_unresolved_are_retained',outcome['exact_transitions']['E_only_exact']>0 and outcome['exact_transitions']['neither_exact']>0)
    check('shared_prefix_labels_independently_reacquired',sum(r['independently_paid_cross_arm_reacquisitions'] for r in paired)>0)
    check('separate_exact_caches',runners['B'].compiler.cache is not runners['E'].compiler.cache and runners['B'].cost_model.cache is not runners['E'].cost_model.cache)
    check('complete_charged_fallback_scans',scan.cells-before_cells==sum(r['fallback']['candidate_board_cells'] for a in runners.values() for r in a.results) and
          scan.requests-before_requests==sum(r['fallback']['selection_requests'] for a in runners.values() for r in a.results))
    # Exact boundary arithmetic with enough baseline failures and no omitted calls.
    def artificial(F,R,B,E):
        result=[dict(B_exact=False,E_exact=i<R,B_calls=4,E_calls=4) for i in range(F)]
        result.append(dict(B_exact=True,E_exact=True,B_calls=B-4*F,E_calls=E-4*F));return result
    endpoint_cases=[(8,6,100,103,'supported'),(8,5,100,103,'falsified'),(8,6,100,104,'falsified'),
                    (8,8,100,100,'supported'),(7,7,100,100,'inconclusive_fewer_than_eight_comparator_failures')]
    for F,R,B,E,expected in endpoint_cases:
        fake=artificial(F,R,B,E);check('joint_exact_threshold_and_all_failure_denominators',summarize(fake)==independent_endpoint(fake) and summarize(fake)['decision']==expected)
    zero=[dict(B_exact=False,E_exact=False,B_calls=0,E_calls=0) for _ in range(8)]
    check('zero_comparator_inconclusive',summarize(zero)==independent_endpoint(zero) and summarize(zero)['decision']=='inconclusive_zero_comparator')
    result=dict(status='passed_real_prefix_and_paid_recovery_namespace_gate',checks=checks,cases=len(sources),outcome=outcome,
        full_arm_audits=audits,storage=io.finish(),synthetic_memo_states=budget.states-before_states,synthetic_work_steps=budget.splits-before_work,
        charged_candidate_board_cells=scan.cells-before_cells,charged_selection_requests=scan.requests-before_requests,
        actual_oracle_calls=0,fresh_truth_draws=0,fresh_source_draws=0,synthetic_paid_queries=oracle.calls,
        source_sha256={n:sha(SRC/n) for n in ['arm.py','audit_arm.py','oracle.py','fallback.py','audit_fallback.py',
          'recovery_pairing.py','recovery_endpoint.py','audit_recovery.py','recovery_namespace_gate.py']})
    save(folder/'sources.json',sources);save(folder/'case-results.json',paired);save(folder/'summary.json',result);print(result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);run(parser.parse_args().folder)
