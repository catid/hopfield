"""Analytic small posteriors, ties, empty strata and numerical decision guard."""
from decimal import Decimal
from metrics import distribution,diagnose,aggregate,decision
from common import OUT,save,now


def run():
    uniform=distribution([0,1],['1','1']);u=diagnose(uniform,1);assert u['true_probability']==dict(numerator='1',denominator='2') and u['true_rank_best']==1 and u['true_rank_worst']==2 and u['posterior_logloss_bits']==u['uniform_logloss_bits']=='1' and Decimal(u['excess_logloss_bits'])==0 and not u['MAP_lex_correct'] and u['true_in_MAP_tie'];heavy=distribution([0,1],['1','3']);a=diagnose(heavy,1);assert a['true_probability']==dict(numerator='1',denominator='4') and abs(Decimal(a['posterior_logloss_bits'])-2)<Decimal('1e-45') and Decimal(a['uniform_logloss_bits'])==1 and abs(Decimal(a['excess_logloss_bits'])-1)<Decimal('1e-45') and a['true_rank_best']==a['true_rank_worst']==2;assert aggregate([])['cases']==0 and aggregate([])['mean_excess_logloss_bits'] is None
    values=['0.250000000001','0.25','0.249999999999','0.249999999998999999'];expected=['supported','inconclusive_numerical_guard','inconclusive_numerical_guard','falsified'];assert [decision(x) for x in values]==expected;receipt=dict(utc=now(),uniform_fixture=dict(distribution=uniform,diagnostic=u),unequal_fixture=dict(distribution=heavy,diagnostic=a),guard_values=values,guard_expected=expected,empty_aggregate=aggregate([]),all_analytic_tests_passed=True);save(OUT/'synthetic-gate.json',receipt);return receipt
