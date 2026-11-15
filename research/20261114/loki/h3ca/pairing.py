"""Assert paired routing invariants and the matched expected-cost inequality."""
from fractions import Fraction
from weights import integer_weights


def pair(h,t,matched_scores,proofs):
    assert h['case']==t['case'] and h['candidate']==t['candidate']
    for name in ['initial_candidates','handoff_MAP_calls','handoff_candidates','handoff_depth','map_calls','exact_true_singleton']:
        assert h[name]==t[name], 'Common routing invariant: '+name
    logical=['step','query_index','label','candidates_before','candidates_after']
    assert [[s[k] for k in logical] for s in h['map_steps']]==[[s[k] for k in logical] for s in t['map_steps']]
    assert (h['terminal_no_information'] is None)==(t['terminal_no_information'] is None)
    expected=None
    if h['handoff_MAP_calls'] is None:
        assert h['remaining_candidates']==t['remaining_candidates'] and proofs==[None,None]
    else:
        candidates=h['handoff_candidates'];weights,_=integer_weights(candidates,matched_scores)
        assert all(p['feasible'] and p['candidates']==candidates and p['depth_limit']==h['handoff_depth'] for p in proofs)
        costs=[sum(w*p['leaf_depths'][str(c)] for c,w in zip(candidates,weights)) for p in proofs]
        assert costs[1]<=costs[0], 'Matched expectation invariant violated'
        means=[Fraction(x,sum(weights)) for x in costs]
        expected=dict(candidates=candidates,H_expected=str(means[0]),T_expected=str(means[1]),
                      matched_weight_sum=str(sum(weights)),H_weighted_cost=str(costs[0]),T_weighted_cost=str(costs[1]),
                      nonincrease=True)
    return dict(case=h['case'],candidate=h['candidate'],H_calls=h['membership_calls'],T_calls=t['membership_calls'],
                H_exact=h['exact_true_singleton'],T_exact=t['exact_true_singleton'],
                all_actual_calls=h['membership_calls']+t['membership_calls'],
                independently_paid_cross_arm_reacquisitions=t['independently_paid_cross_arm_reacquisitions'],
                common_prefix_handoff_and_exact_verified=True,matched_expected_tree_costs=expected)
