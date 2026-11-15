"""Full independent replay of fixed two-arm separation and no-regression fixtures."""
import hashlib
import numpy as np
from common import SRC, read, load
from ledger import canonical
from audit_queries import PaidQueries
from audit_arm import verify as verify_arm


def independent_endpoint(pairs):
    H=sum(r['H_calls'] for r in pairs);T=sum(r['T_calls'] for r in pairs)
    assert all(r['H_exact']==r['T_exact'] for r in pairs), 'Independent exact invariant violated'
    tt=sum(r['H_exact'] for r in pairs);ff=len(pairs)-tt
    cost=None if H==0 else 100*T<=98*H
    decision='inconclusive_zero_comparator' if H==0 else 'supported' if cost else 'falsified'
    return dict(decision=decision,primary=dict(left=100*T,right=98*H,threshold_percent=2,
                passing_maximum=(98*H)//100 if H else None,cost_passes=cost),
                H_calls=H,T_calls=T,actual_membership_calls=H+T,
                exact_transitions=dict(both_exact=tt,H_only_exact=0,T_only_exact=0,neither_exact=ff),H_exact=tt,T_exact=tt)


def verify(folder,io,check,memo_states,work_steps):
    read=io.read
    predictions=np.eye(20,dtype=bool);predictions[19]=predictions[18]
    boards=np.arange(20,dtype=np.uint8)[:,None]
    sources=[dict(case=i,candidate=truth,initial_candidates=candidates,
                  source_record_sha256=hashlib.sha256(canonical(dict(case=i,fixture='fixed_common_selector'))).hexdigest())
             for i,(truth,candidates) in enumerate([(0,list(range(20))),(19,list(range(20))),(0,[0,19]),(0,[0,19]),(0,[0]),(19,[18,19]),(0,list(range(6))),(5,list(range(6)))])]
    scores={'H':[str(1<<i) for i in range(20)],'T':[str(1<<(19-i)) for i in range(20)]}
    for arm in ['H','T']:
        check('fixed_fixture_source_score_identity',read(folder/arm/'selector-scores.json')==scores['H'] and read(folder/arm/'tree-scores.json')==scores[arm] and
              np.array_equal(np.load(folder/arm/'predictions-bits.npy'),np.packbits(predictions,axis=1,bitorder='little')))
    chain=load(SRC.parent/'h3bj/verify_public.py','h3cb_fixture_chain').chain
    replay=PaidQueries(chain(io.data(folder/'query-ledger.jsonl')),predictions,boards,check,arms=('H','T'))
    audits={}
    for arm in ['H','T']:
        audits[arm]=verify_arm(arm,folder/arm,SRC,sources,read(folder/arm/'arm-results.json'),
                              predictions,boards,scores['H'],scores[arm],replay,io,check)
    check('all_fixture_events_replayed',replay.at==len(replay.events))
    check('separate_full_work_charged',memo_states==sum(a['tree_memo_states']+a['counterfactual_memo_states'] for a in audits.values()) and
          work_steps==sum(a['compiler_work_steps']+a['counterfactual_MAP_partitions'] for a in audits.values()))
    from audit_pairing import verify as verify_pair
    hrows,trows=read(folder/'H/arm-results.json'),read(folder/'T/arm-results.json')
    pairs=[verify_pair(h,t,folder,scores['T'],io.read,check) for h,t in zip(hrows,trows)]
    result=independent_endpoint(pairs)
    check('common_exact_and_failure_outcomes',result['exact_transitions']==dict(
          both_exact=6,H_only_exact=0,T_only_exact=0,neither_exact=2))
    check('same_board_queries_separately_paid',sum(r['independently_paid_cross_arm_reacquisitions'] for r in trows)>0)
    check('same_arm_duplicate_cache_does_not_drop_labels',hrows[2]['membership_calls']==hrows[3]['membership_calls']==1 and
          trows[2]['membership_calls']==trows[3]['membership_calls']==1)
    check('different_tree_paths_and_equal_MAP',hrows[6]['tree_steps'][0]['query_index']!=trows[6]['tree_steps'][0]['query_index'] and
          hrows[6]['map_steps']==trows[6]['map_steps']==[])
    check('retained_ineligible_and_no_information_failures',len(hrows[1]['remaining_candidates'])==16 and
          hrows[5]['terminal_no_information'] is not None and trows[5]['terminal_no_information'] is not None)
    endpoint_fixtures=[([dict(H_calls=100,T_calls=98,H_exact=True,T_exact=True)],'supported'),
        ([dict(H_calls=100,T_calls=98,H_exact=False,T_exact=False)],'supported'),
        ([dict(H_calls=100,T_calls=99,H_exact=True,T_exact=True)],'falsified'),
        ([dict(H_calls=0,T_calls=0,H_exact=True,T_exact=True)],'inconclusive_zero_comparator')]
    for rows,decision in endpoint_fixtures:check('fixed_cost_endpoint_boundary',independent_endpoint(rows)['decision']==decision)
    for h,t in [(True,False),(False,True)]:
        try:independent_endpoint([dict(H_calls=100,T_calls=98,H_exact=h,T_exact=t)])
        except AssertionError:pass
        else:raise AssertionError('Independent endpoint accepted unequal exact flags')
    return dict(common_routing_fixture=result,full_independent_arm_audits=audits,synthetic_paid_calls=sum(replay.calls.values()),
                all_pair_common_prefix_handoff_and_exact_verified=True,matched_expected_costs=[r['matched_expected_tree_costs'] for r in pairs])
