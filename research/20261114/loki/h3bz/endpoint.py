"""Prespecified joint finite-cohort endpoint; disagreement is an outcome."""


def summarize(pairs):
    H, U = sum(r['H_calls'] for r in pairs), sum(r['U_calls'] for r in pairs)
    transitions = {name:0 for name in ['both_exact','H_only_exact','U_only_exact','neither_exact']}
    for row in pairs:
        key = ('both_exact' if row['U_exact'] else 'H_only_exact') if row['H_exact'] else ('U_only_exact' if row['U_exact'] else 'neither_exact')
        transitions[key] += 1
    no_loss = transitions['H_only_exact'] == 0
    cost_passes = None if H == 0 else 100*U <= 98*H
    decision = 'inconclusive_zero_comparator' if H == 0 else 'supported' if cost_passes and no_loss else 'falsified'
    return dict(decision=decision,joint_primary=dict(left=100*U,right=98*H,threshold_percent=2,
                passing_maximum=(98*H)//100 if H else None,cost_passes=cost_passes,per_case_exact_no_loss=no_loss),
                H_calls=H,U_calls=U,actual_membership_calls=H+U,exact_transitions=transitions,
                H_exact=transitions['both_exact']+transitions['H_only_exact'],
                U_exact=transitions['both_exact']+transitions['U_only_exact'])
