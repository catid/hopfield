"""Required original-prefix equality and retained per-case rescue accounting."""


def pair(b,e):
    left,right=b['original_policy_result'],e['original_policy_result']
    for name in ['case','candidate','initial_candidates','map_calls','tree_calls','membership_calls',
                 'exact_true_singleton','remaining_candidates','handoff_MAP_calls','handoff_candidates','handoff_depth']:
        assert left[name]==right[name], 'Original policy invariant: '+name
    fields=['step','query_index','label','candidates_before','candidates_after']
    for phase in ['map_steps','tree_steps']:
        assert [[s[k] for k in fields] for s in left[phase]]==[[s[k] for k in fields] for s in right[phase]]
    assert (left['terminal_no_information'] is None)==(right['terminal_no_information'] is None)
    assert b['fallback']['additional_calls']==0 and b['membership_calls']==left['membership_calls']
    assert e['membership_calls']==right['membership_calls']+e['fallback']['additional_calls']<=8
    if b['exact_true_singleton']:
        assert e['exact_true_singleton'] and e['fallback']['additional_calls']==0 and b['membership_calls']==e['membership_calls']
    assert b['candidate'] in b['remaining_candidates'] and e['candidate'] in e['remaining_candidates']
    return dict(case=b['case'],candidate=b['candidate'],B_calls=b['membership_calls'],E_calls=e['membership_calls'],
        B_exact=b['exact_true_singleton'],E_exact=e['exact_true_singleton'],
        newly_recovered=not b['exact_true_singleton'] and e['exact_true_singleton'],
        extra_calls=e['fallback']['additional_calls'],all_actual_calls=b['membership_calls']+e['membership_calls'],
        independently_paid_cross_arm_reacquisitions=e['independently_paid_cross_arm_reacquisitions'],
        original_prefix_and_exact_no_loss_verified=True,fallback_entered=e['fallback']['entered'])
