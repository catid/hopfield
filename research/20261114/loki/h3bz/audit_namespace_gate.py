"""Full independent replay of fixed two-arm separation and no-regression fixtures."""
import hashlib
import numpy as np
from common import SRC, read, load
from ledger import canonical
from audit_queries import PaidQueries
from audit_arm import verify as verify_arm


def independent_endpoint(pairs):
    H=sum(r['H_calls'] for r in pairs);U=sum(r['U_calls'] for r in pairs)
    tt=sum(r['H_exact'] and r['U_exact'] for r in pairs)
    tf=sum(r['H_exact'] and not r['U_exact'] for r in pairs)
    ft=sum(not r['H_exact'] and r['U_exact'] for r in pairs)
    ff=len(pairs)-tt-tf-ft
    cost=None if H==0 else 100*U<=98*H
    decision='inconclusive_zero_comparator' if H==0 else 'supported' if cost and tf==0 else 'falsified'
    return dict(decision=decision,joint_primary=dict(left=100*U,right=98*H,threshold_percent=2,
                passing_maximum=(98*H)//100 if H else None,cost_passes=cost,per_case_exact_no_loss=tf==0),
                H_calls=H,U_calls=U,actual_membership_calls=H+U,
                exact_transitions=dict(both_exact=tt,H_only_exact=tf,U_only_exact=ft,neither_exact=ff),H_exact=tt+tf,U_exact=tt+ft)


def verify(folder,io,check,memo_states,work_steps):
    predictions=np.eye(20,dtype=bool)
    boards=np.arange(20,dtype=np.uint8)[:,None]
    sources=[dict(case=i,candidate=truth,initial_candidates=candidates,
                  source_record_sha256=hashlib.sha256(canonical(dict(case=i,fixture='fixed_one_hot'))).hexdigest())
             for i,(truth,candidates) in enumerate([(0,list(range(20))),(19,list(range(20))),(0,[0,19]),(0,[0,19])])]
    scores={'H':list(map(str,range(1,21))),'U':list(map(str,range(20,0,-1)))}
    for arm in ['H','U']:
        check('fixed_fixture_source_score_identity',read(folder/arm/'score-values8.json')==scores[arm] and
              np.array_equal(np.load(folder/arm/'predictions-bits.npy'),np.packbits(predictions,axis=1,bitorder='little')))
    chain=load(SRC.parent/'h3bj/verify_public.py','h3bz_fixture_chain').chain
    replay=PaidQueries(chain(io.data(folder/'query-ledger.jsonl')),predictions,boards,check,arms=('H','U'))
    audits={}
    for arm in ['H','U']:
        audits[arm]=verify_arm(arm,folder/arm,SRC,sources,read(folder/arm/'arm-results.json'),
                              predictions,boards,scores[arm],replay,io,check)
    check('all_fixture_events_replayed',replay.at==len(replay.events) and replay.calls==dict(H=7,U=7))
    check('separate_full_work_charged',memo_states==sum(a['tree_memo_states']+a['counterfactual_memo_states'] for a in audits.values()) and
          work_steps==sum(a['compiler_work_steps']+a['counterfactual_MAP_partitions'] for a in audits.values()))
    pairs=[dict(case=h['case'],H_calls=h['membership_calls'],U_calls=u['membership_calls'],
                H_exact=h['exact_true_singleton'],U_exact=u['exact_true_singleton'])
           for h,u in zip(read(folder/'H/arm-results.json'),read(folder/'U/arm-results.json'))]
    joint=independent_endpoint(pairs)
    check('different_priors_preserve_both_disagreement_directions',joint['exact_transitions']==dict(
          both_exact=2,H_only_exact=1,U_only_exact=1,neither_exact=0) and joint['decision']=='falsified')
    check('same_board_queries_separately_paid',sum(r['independently_paid_cross_arm_reacquisitions'] for r in read(folder/'U/arm-results.json'))==2)
    check('same_arm_duplicate_cache_does_not_drop_labels',sum(replay.calls.values())==14 and replay.calls==dict(H=7,U=7))
    endpoint_fixtures=[
        ([dict(H_calls=100,U_calls=98,H_exact=True,U_exact=True)],'supported'),
        ([dict(H_calls=100,U_calls=98,H_exact=False,U_exact=True)],'supported'),
        ([dict(H_calls=100,U_calls=98,H_exact=True,U_exact=False)],'falsified'),
        ([dict(H_calls=100,U_calls=99,H_exact=True,U_exact=True)],'falsified'),
        ([dict(H_calls=0,U_calls=0,H_exact=True,U_exact=True)],'inconclusive_zero_comparator')]
    for rows,decision in endpoint_fixtures:check('fixed_joint_endpoint_boundary',independent_endpoint(rows)['decision']==decision)
    return dict(exact_disagreement_fixture=joint,full_independent_arm_audits=audits,synthetic_paid_calls=sum(replay.calls.values()))
