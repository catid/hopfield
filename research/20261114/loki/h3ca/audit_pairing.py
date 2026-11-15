"""Independent cross-arm route and rational expected-depth reconstruction."""
import math
from fractions import Fraction


def verify(h,t,folder,matched_scores,read,check):
    check('paired_case_truth_and_initial_set', h['case']==t['case'] and h['candidate']==t['candidate'] and
          h['initial_candidates']==t['initial_candidates'])
    check('identical_paid_MAP_prefix',len(h['map_steps'])==len(t['map_steps'])==h['map_calls']==t['map_calls'] and
          all(all(a[k]==b[k] for k in ['step','query_index','label','candidates_before','candidates_after'])
              for a,b in zip(h['map_steps'],t['map_steps'])))
    check('identical_handoff_and_exact',all(h[k]==t[k] for k in ['handoff_MAP_calls','handoff_candidates','handoff_depth','exact_true_singleton']) and
          (h['terminal_no_information'] is None)==(t['terminal_no_information'] is None))
    expected=None
    if h['handoff_MAP_calls'] is None:
        check('identical_no_handoff_outcome',h['remaining_candidates']==t['remaining_candidates'])
    else:
        candidates=h['handoff_candidates']
        scores=[int(matched_scores[c]) for c in candidates]
        denominator=math.lcm(*scores);mass=[denominator//s for s in scores]
        totals=[];means=[]
        for arm,row in [('H',h),('T',t)]:
            group=row['cost_guards'][-1]['tree_group']
            proof=read(folder/arm/f'group{group}/proof.json')
            check('paired_full_tree_identity',proof['feasible'] and proof['candidates']==candidates and
                  proof['depth_limit']==row['handoff_depth'] and row['routing'][-1]['group']==group)
            mean=sum((Fraction(proof['leaf_depths'][str(c)],s) for c,s in zip(candidates,scores)),Fraction())
            mean/=sum((Fraction(1,s) for s in scores),Fraction())
            total=sum(w*proof['leaf_depths'][str(c)] for c,w in zip(candidates,mass))
            check('independent_matched_integer_rational_agreement',mean==Fraction(total,sum(mass)))
            totals.append(total);means.append(mean)
        check('matched_posterior_expected_tree_nonincrease',means[1]<=means[0])
        expected=dict(candidates=candidates,H_expected=str(means[0]),T_expected=str(means[1]),
                      matched_weight_sum=str(sum(mass)),H_weighted_cost=str(totals[0]),T_weighted_cost=str(totals[1]),nonincrease=True)
    return dict(case=h['case'],candidate=h['candidate'],H_calls=h['membership_calls'],T_calls=t['membership_calls'],
                H_exact=h['exact_true_singleton'],T_exact=t['exact_true_singleton'],
                all_actual_calls=h['membership_calls']+t['membership_calls'],
                independently_paid_cross_arm_reacquisitions=t['independently_paid_cross_arm_reacquisitions'],
                common_prefix_handoff_and_exact_verified=True,matched_expected_tree_costs=expected)
