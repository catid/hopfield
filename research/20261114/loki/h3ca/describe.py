"""Postrun descriptive reconstruction; no fresh draws or policy execution."""
from collections import Counter
from fractions import Fraction


def describe(pairs, rows, protocol):
    differences=Counter(r['T_calls']-r['H_calls'] for r in pairs)
    handoffs=[r['matched_expected_tree_costs'] for r in pairs if r['matched_expected_tree_costs'] is not None]
    failures=[]
    for pair,h,t in zip(pairs,rows['H'],rows['T']):
        if not pair['H_exact']:
            assert not pair['T_exact'] and h['remaining_candidates']==t['remaining_candidates']
            assert h['map_calls']==t['map_calls']==4 and h['tree_calls']==t['tree_calls']==0
            assert h['handoff_MAP_calls'] is t['handoff_MAP_calls'] is None
            assert len(h['remaining_candidates'])>16 and h['candidate'] in h['remaining_candidates']
            failures.append(dict(case=pair['case'],candidate=pair['candidate'],remaining_candidates=h['remaining_candidates'],
                H_calls=pair['H_calls'],T_calls=pair['T_calls'],stop='candidate_count_above16_after_four_MAP',
                true_candidate_retained=True,wrong_terminal_model_selected=False,tree_infeasibility_certified=False))
    strata=[]
    for checkpoint in [None,0,1,2,3,4]:
        subset=[p for p,h in zip(pairs,rows['H']) if h['handoff_MAP_calls']==checkpoint]
        strata.append(dict(handoff_MAP_calls=checkpoint,cases=len(subset),H_calls=sum(p['H_calls'] for p in subset),
            T_calls=sum(p['T_calls'] for p in subset),both_exact=sum(p['H_exact'] for p in subset)))
    return dict(actual_query_difference_T_minus_H={str(k):v for k,v in sorted(differences.items())},
        T_fewer=sum(v for k,v in differences.items() if k<0),T_more=sum(v for k,v in differences.items() if k>0),
        tied=differences[0],common_handoffs=len(handoffs),
        strictly_smaller_matched_expectation=sum(Fraction(r['T_expected'])<Fraction(r['H_expected']) for r in handoffs),
        equal_matched_expectation=sum(Fraction(r['T_expected'])==Fraction(r['H_expected']) for r in handoffs),
        failures=failures,by_handoff=strata,actual_oracle_calls=0,fresh_truth_draws=0,fresh_source_draws=0,
        limitation='Descriptive analysis of all sealed cases; expected-cost receipts require full proof archive to reconstruct from leaves')
