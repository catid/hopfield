"""Exhaustive finite corruption-law audit, with no source or solver outcomes."""
from fractions import Fraction
import itertools
import json
import numpy as np
from run import ROOT, HERE, sha, save, now
from calibration_estimator import fit_count


def main():
    counts = np.zeros((9,9),np.int64)
    accepted = 0
    for permutation in itertools.permutations(range(9)):
        if any(permutation[index] == index for index in range(9)):
            continue
        accepted += 1
        assert len(set(permutation)) == 9
        for index,value in enumerate(permutation):
            counts[index,value] += 1
    assert accepted == 133496
    assert not counts.diagonal().any()
    assert np.all(counts[~np.eye(9,dtype=bool)] == 16687)
    fit = fit_count(0,1000)
    eta = Fraction(fit['eta_upper_numerator'],fit['eta_upper_denominator'])
    assert eta < Fraction(1,20)
    report = {'status':'validated','utc':now(),'auditor_sha256':sha(__file__),
        'derangements':accepted,'off_diagonal_digit_mapping_count':16687,
        'marginal_wrong_probability':'1/20','conditional_wrong_digit_probability':'1/8',
        'unconditional_each_wrong_digit_probability':'1/160',
        'shared_repeat_disagreement_probability':'0',
        'iid_repeat_disagreement_probability':str(2*Fraction(1,20)-Fraction(9,8)*Fraction(1,20)**2),
        'pair_equality_invariant_for_every_bijection':True,
        'justification':'A bijection maps equal labels to equal labels and unequal labels to unequal labels. Sharing one draw between repeats makes both outputs identical. Uniform permutation rejection restricted to derangements is uniform on the accepted finite set.',
        'zero_disagreement_calibration_fit':fit,
        'calibration_undercoverage_prediction':True,'source_outcomes_accessed':0,'solver_inputs_accessed':0}
    save(ROOT/'artifacts/research/20261114/ripper/correlated_noise_law.json',report)
    print(json.dumps({'status':'validated','derangements':accepted,'eta_upper_with_zero_disagreements':fit['eta_upper']}),flush=True)


if __name__ == '__main__':
    main()
