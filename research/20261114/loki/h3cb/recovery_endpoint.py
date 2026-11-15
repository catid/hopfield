"""One joint 75 percent rescue / 3 percent overhead finite-cohort endpoint."""


def summarize(pairs):
    assert all(not r['B_exact'] or r['E_exact'] for r in pairs)
    B,E=sum(r['B_calls'] for r in pairs),sum(r['E_calls'] for r in pairs)
    failures=sum(not r['B_exact'] for r in pairs)
    rescued=sum(not r['B_exact'] and r['E_exact'] for r in pairs)
    eligible=failures>=8 and B>0
    recovery=4*rescued>=3*failures if eligible else None
    overhead=100*E<=103*B if eligible else None
    decision='inconclusive_fewer_than_eight_comparator_failures' if failures<8 else 'inconclusive_zero_comparator' if B==0 else 'supported' if recovery and overhead else 'falsified'
    return dict(decision=decision,primary=dict(baseline_failure_minimum=8,baseline_failures=failures,rescued=rescued,
        recovery_left=4*rescued,recovery_right=3*failures,recovery_passes=recovery,
        overhead_left=100*E,overhead_right=103*B,overhead_passes=overhead,joint_passes=(recovery and overhead) if eligible else None),
        B_calls=B,E_calls=E,actual_membership_calls=B+E,
        B_exact=sum(r['B_exact'] for r in pairs),E_exact=sum(r['E_exact'] for r in pairs),
        exact_transitions=dict(both_exact=sum(r['B_exact'] for r in pairs),B_only_exact=0,E_only_exact=rescued,neither_exact=failures-rescued))
