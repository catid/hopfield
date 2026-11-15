"""Finite-cohort cost endpoint after common-route correctness validation."""


def summarize(pairs):
    assert all(r['H_exact'] == r['T_exact'] for r in pairs), 'Paired exact invariant violated'
    H, T = sum(r['H_calls'] for r in pairs), sum(r['T_calls'] for r in pairs)
    exact = sum(r['H_exact'] for r in pairs)
    cost = None if H == 0 else 100*T <= 98*H
    decision = 'inconclusive_zero_comparator' if H == 0 else 'supported' if cost else 'falsified'
    return dict(decision=decision,primary=dict(left=100*T,right=98*H,threshold_percent=2,
                passing_maximum=(98*H)//100 if H else None,cost_passes=cost),
                H_calls=H,T_calls=T,actual_membership_calls=H+T,
                exact_transitions=dict(both_exact=exact,H_only_exact=0,T_only_exact=0,neither_exact=len(pairs)-exact),
                H_exact=exact,T_exact=exact)
