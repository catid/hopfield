"""Independent fixed-catalog certificate and capped-arm artifact replay."""
from common import SRC,load
from verify_tree import weighted_certificate


def verify(folder,io,check):
    read=io.read;summary=read(folder/'summary.json')
    catalog=[dict(signature=s,lexfirst_board_index=s-1,board_count=1) for s in range(1,8)]
    actual=[]
    for label,weights,cost,work in [('uniform',[1]*4,8,23),('skewed',[1,2,4,8],25,28),('uniform-capped',[1]*4,8,23)]:
        name=label+'-full-proof.json' if label!='uniform-capped' else label+'-proof.json'
        proof=read(folder/name)
        check('cap_fixture_exact_inputs',proof['candidates']==list(range(4)) and proof['weights']==list(map(str,weights)) and
              proof['depth_limit']==4 and proof['feasible'] and int(proof['optimal_cost'])==cost)
        ns,nt,_=weighted_certificate(proof,catalog,check,'cap_fixture_')
        check('cap_fixture_full_counter_identity',ns==7 and nt==work)
        if label!='uniform-capped':actual.append(dict(weights=weights,states=ns,work=nt,optimal_cost=str(cost)))
    check('weight_sensitive_equal_cap_receipts',summary['fixed_full_runs']==actual and summary['fixed_cap_failures']==[
        dict(mode='work',error='Synthetic local work cap',states=7,work=23,feasible_incumbent_existed=True,complete_proof_returned=False),
        dict(mode='state',error='Synthetic local state cap',states=6,work=20,feasible_incumbent_existed=False,complete_proof_returned=False)])
    chain=load(SRC.parent/'h3bj/verify_public.py','h3ca_cap_chain').chain
    events=chain(io.data(folder/'query-ledger.jsonl'))
    check('capped_actual_arm_never_queries_or_routes',len(events)==2 and
          [r['kind'] for r in events]==['arm_namespace_bound','arm_source_bound'] and
          events[1]['data']==dict(case=0,arm='T',candidate=0,source_record_sha256='a'*64,initial_candidates=list(range(4)),inherited_queries=0))
    check('cap_is_third_state_not_infeasibility',summary['status']=='passed_weight_sensitive_cap_and_no_fallback_gate' and
          summary['integer_cap_rejected'] and summary['arm_cap_error']=='Synthetic local work cap' and
          summary['whole_study_required_status']=='failed_or_incomplete' and not summary['partial_primary_permitted'] and
          summary['no_incumbent_route_no_MAP_fallback_no_actual_query'] and
          summary['actual_oracle_calls']==summary['fresh_truth_draws']==summary['fresh_source_draws']==0 and
          summary['synthetic_memo_states']==41 and summary['synthetic_work_steps']==140)
    return dict(status='passed_independent_weight_sensitive_cap_artifact_audit',synthetic_memo_states=41,synthetic_work_steps=140)
