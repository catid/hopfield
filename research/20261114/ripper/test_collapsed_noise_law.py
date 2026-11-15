"""Exact finite law for fixed pre-query cells, repeats and clean digit pairs."""
from fractions import Fraction
import json
from run import ROOT, sha, save, now
from calibration_estimator import fit_count


def main():
    burst=Fraction(9,160)
    marginal=[];pair_law=[]
    for clean in range(9):
        probability=burst*sum(target!=clean for target in range(9))/9
        assert probability==Fraction(1,20);marginal.append(str(probability))
    for left in range(9):
        for right in range(9):
            probability=burst+(1-burst)*int(left==right)
            assert probability==(1 if left==right else burst)
            pair_law.append({'left':left,'right':right,'equal_probability':str(probability)})
    fit=fit_count(0,1000);eta=Fraction(fit['eta_upper_numerator'],fit['eta_upper_denominator'])
    learned_null=eta/4-Fraction(9,64)*eta**2
    assert eta<Fraction(1,20) and burst>learned_null
    report={'status':'validated','utc':now(),'auditor_sha256':sha(__file__),'collapse_probability':str(burst),
        'fixed_prequery_cell_error_probability':'1/20','conditional_target_probability':'1/9',
        'shared_repeat_disagreement_probability':'0','unequal_clean_pair_equality_probability':str(burst),
        'equal_clean_pair_equality_probability':'1','marginal_law_checks':marginal,'all81_pair_laws':pair_law,
        'count0_calibration_fit':fit,'learned_true_pair_null':str(learned_null),'true_pair_equality_exceeds_learned_null':True,
        'scope':'Conditional on any past history, current clean source board and mask fixed before independent current noise. Realized adaptive-query errors and equality counts remain empirical diagnostics.'}
    save(ROOT/'artifacts/research/20261114/ripper/collapsed_noise_law.json',report)
    print(json.dumps({'status':'validated','marginal_error':'1/20','true_pair_equality':'9/160','learned_null':str(learned_null)}),flush=True)


if __name__=='__main__':main()
