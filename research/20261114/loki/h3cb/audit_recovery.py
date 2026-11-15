"""Independent prefix and complete finite-cohort recovery/cost reconstruction."""


def pair(b,e,check):
    h,t=b['original_policy_result'],e['original_policy_result']
    common=['case','candidate','initial_candidates','map_calls','tree_calls','membership_calls','exact_true_singleton',
            'remaining_candidates','handoff_MAP_calls','handoff_candidates','handoff_depth']
    check('identical_original_policy_outcome',all(h[k]==t[k] for k in common))
    logical=lambda r:[(s['step'],s['query_index'],s['label'],s['candidates_before'],s['candidates_after']) for s in r]
    check('identical_actual_original_MAP_and_tree_prefix',logical(h['map_steps'])==logical(t['map_steps']) and
        logical(h['tree_steps'])==logical(t['tree_steps']) and (h['terminal_no_information'] is None)==(t['terminal_no_information'] is None))
    check('actual_additional_queries_paid_within_budget',b['fallback']['additional_calls']==0 and b['membership_calls']==h['membership_calls'] and
        e['membership_calls']==t['membership_calls']+e['fallback']['additional_calls']<=8)
    check('no_loss_or_extra_queries_on_baseline_exact',not b['exact_true_singleton'] or
        (e['exact_true_singleton'] and b['membership_calls']==e['membership_calls'] and not e['fallback']['additional_calls']))
    check('true_candidate_retained_in_both_results',b['candidate'] in b['remaining_candidates'] and e['candidate'] in e['remaining_candidates'])
    return dict(case=b['case'],candidate=b['candidate'],B_calls=b['membership_calls'],E_calls=e['membership_calls'],
        B_exact=b['exact_true_singleton'],E_exact=e['exact_true_singleton'],newly_recovered=not b['exact_true_singleton'] and e['exact_true_singleton'],
        extra_calls=e['fallback']['additional_calls'],all_actual_calls=b['membership_calls']+e['membership_calls'],
        independently_paid_cross_arm_reacquisitions=e['independently_paid_cross_arm_reacquisitions'],
        original_prefix_and_exact_no_loss_verified=True,fallback_entered=e['fallback']['entered'])


def endpoint(pairs):
    counts={(b,e):sum(r['B_exact']==b and r['E_exact']==e for r in pairs) for b in [False,True] for e in [False,True]}
    assert counts[True,False]==0, 'Original exact result lost'
    B=sum(r['B_calls'] for r in pairs);E=sum(r['E_calls'] for r in pairs)
    F=counts[False,False]+counts[False,True];R=counts[False,True]
    enabled=F>=8 and B>0
    rescue=R*4>=F*3 if enabled else None;overhead=E*100<=B*103 if enabled else None
    if F<8:decision='inconclusive_fewer_than_eight_comparator_failures'
    elif not B:decision='inconclusive_zero_comparator'
    elif rescue and overhead:decision='supported'
    else:decision='falsified'
    return dict(decision=decision,primary=dict(baseline_failure_minimum=8,baseline_failures=F,rescued=R,
        recovery_left=R*4,recovery_right=F*3,recovery_passes=rescue,overhead_left=E*100,overhead_right=B*103,
        overhead_passes=overhead,joint_passes=(rescue and overhead) if enabled else None),
        B_calls=B,E_calls=E,actual_membership_calls=B+E,B_exact=counts[True,True],E_exact=counts[True,True]+R,
        exact_transitions=dict(both_exact=counts[True,True],B_only_exact=0,E_only_exact=R,neither_exact=counts[False,False]))
