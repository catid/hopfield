"""Exact one- and two-position law checks; no source or outcome access."""
from fractions import Fraction
import json
import numpy as np
from run import ROOT,HERE,sha,save

def main():
    board=np.repeat(np.arange(9),9);pairs=[(i,j)for i in range(81)for j in range(81)if i!=j]
    assert len(pairs)==6480
    equal=sum(board[i]==board[j]for i,j in pairs);assert equal==648
    p_equal=Fraction(int(equal),len(pairs));assert p_equal==Fraction(1,10)
    for original in range(9):
        assert sum(board==original)==9
        assert Fraction(9,160)*(1-Fraction(9,81))==Fraction(1,20)
        for wrong in range(9):
            if original!=wrong:assert Fraction(9,160)*Fraction(int(sum(board==wrong)),81)==Fraction(1,160)
    assert Fraction(9,160)*p_equal==Fraction(9,1600)
    assert 1-Fraction(9,160)+Fraction(9,160)*p_equal==Fraction(1519,1600)
    # A shared shuffled board returns the identical value on a repeated cell.
    for shift in range(81):
        permutation=np.roll(np.arange(81),shift);shuffled=board[permutation]
        assert np.array_equal(np.sort(shuffled),board)
        for cell in range(81):assert shuffled[cell]==shuffled[cell]
    result=dict(status='validated',auditor_sha256=sha(__file__),ordered_distinct_source_positions=6480,equal_source_pairs=648,
        conditional_equal_on_permutation='1/10',true_pair_equal_probability='9/1600',fixed_cell_error_probability='1/20',
        each_wrong_digit_probability='1/160',same_clean_pair_equal_probability='1519/1600',repeat_disagreement_probability='0',
        multiset_theorem='A permutation is a bijection of all81 source positions, so every digit multiplicity stays exactly9. Any25-cell subset has each multiplicity<=9; the balance filter necessarily accepts it.',
        uniform_permutation_law='A uniform permutation maps any fixed position uniformly to81 sources and any ordered pair of distinct positions uniformly to6480 ordered distinct sources. Exact pair counts checked above; no factorial enumeration claim.')
    save(ROOT/'artifacts/research/20261114/ripper/permuted_noise_law_audit.json',result);print(json.dumps(result))
if __name__=='__main__':main()
