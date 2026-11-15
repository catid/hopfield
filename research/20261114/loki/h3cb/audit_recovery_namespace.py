"""Independent retained real-arm recovery fixture replay, raw or exported."""
import hashlib
import json
import numpy as np
from common import SRC,load
from audit_queries import PaidQueries
from audit_arm import verify as verify_arm
from audit_recovery import pair,endpoint


def verify(folder,io,check):
    pred=np.eye(24,dtype=bool);pred[23]=pred[22];boards=np.arange(24,dtype=np.uint8)[:,None]
    specs=[(0,list(range(24))),(4,list(range(24))),(7,list(range(24))),(21,list(range(24))),(23,list(range(24))),
           (0,[0,1,2]),(0,[0]),(23,[22,23]),(4,list(range(24)))]
    canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    sources=[dict(case=i,candidate=c,initial_candidates=active,source_record_sha256=hashlib.sha256(canonical(dict(case=i,fixture='fixed_budget_recovery'))).hexdigest()) for i,(c,active) in enumerate(specs)]
    check('fixed_recovery_sources',io.read(folder/'sources.json')==sources)
    rows={a:io.read(folder/a/'arm-results.json') for a in ['B','E']}
    chain=load(SRC.parent/'h3bj/verify_public.py','h3cb_recovery_retained_chain').chain
    ledger=PaidQueries(chain(io.data(folder/'query-ledger.jsonl')),pred,boards,check,arms=('B','E'))
    audits={}
    for arm in ['B','E']:
        check('same_hierarchical_selector_and_matched_tree_fixture',io.read(folder/arm/'selector-scores.json')==io.read(folder/arm/'tree-scores.json')==['1']*24)
        audits[arm]=verify_arm(arm,folder/arm,SRC,sources,rows[arm],pred,boards,['1']*24,['1']*24,ledger,io,check)
    pairs=[]
    for b,e,stored in zip(rows['B'],rows['E'],io.read(folder/'case-results.json')):
        expected=pair(b,e,check);check('retained_fixture_pair',expected==stored==ledger.take('paired_case_result')['data']);pairs.append(expected)
    outcome=endpoint(pairs)
    check('retained_fixture_joint_endpoint_and_all_calls',outcome==ledger.take('study_complete')['data'] and ledger.at==len(ledger.events) and
          outcome['B_calls']==ledger.calls['B'] and outcome['E_calls']==ledger.calls['E'])
    check('retained_rescues_failures_and_insufficient_F',outcome['exact_transitions']==dict(both_exact=3,B_only_exact=0,E_only_exact=2,neither_exact=4) and
          outcome['decision']=='inconclusive_fewer_than_eight_comparator_failures')
    result=dict(outcome=outcome,full_arm_audits=audits,synthetic_memo_states=sum(a['tree_memo_states']+a['counterfactual_memo_states'] for a in audits.values()),
        synthetic_work_steps=sum(a['compiler_work_steps']+a['counterfactual_MAP_partitions'] for a in audits.values()),
        charged_candidate_board_cells=sum(r['fallback']['candidate_board_cells'] for arm in rows.values() for r in arm),
        charged_selection_requests=sum(r['fallback']['selection_requests'] for arm in rows.values() for r in arm),synthetic_paid_queries=sum(ledger.calls.values()))
    check('retained_fixture_totals_and_resource_seals',all(io.read(folder/'summary.json')[k]==v for k,v in result.items()))
    return result
