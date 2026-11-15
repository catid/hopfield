"""Exact product-marginal posterior equality scores at nominal channel3/40."""
from fractions import Fraction
from itertools import combinations_with_replacement
import numpy as np
NORMALIZERS = {1: [320], 3: [8154, 265701, 25934552]}


def validate(values, given):
    assert values.dtype == np.int8 and given.dtype == bool
    assert values.ndim == 3 and values.shape[0] in [1, 3]
    assert values.shape[1:] == given.shape and given.ndim == 2
    assert (((values >= 0) & (values <= 8))[:, given]).all()
    assert (values[:, ~given] == -1).all()


def weights(values, given):
    validate(values, given)
    q = values.shape[0]
    counts = (values[:, :, :, None] == np.arange(9)[None, None, None, :]).sum(0)
    assert (counts.sum(2)[given] == q).all()
    lookup = np.array([296**n*3**(q-n) for n in range(q+1)], np.int64)
    w = lookup[counts]
    w[~given] = 0
    denominator = w.sum(2)
    assert np.isin(denominator[given], NORMALIZERS[q]).all()
    assert not denominator[~given].any()
    return w, denominator


def convert_bins(numerators, denominators, joint):
    assert numerators.dtype == denominators.dtype == joint.dtype == np.int64
    assert numerators.shape == (len(denominators), len(joint)) and (denominators > 0).all() and (numerators >= 0).all()
    costs = []
    for edge in range(len(joint)):
        total = sum((Fraction(int(numerators[k, edge]), int(denominators[k])) for k in range(len(denominators))), Fraction(0))
        assert 0 <= total <= int(joint[edge])
        costs.append(round(Fraction(1000000, int(joint[edge])+2)*(total+1)))
    return np.array(costs, np.int64)


def score(values, given):
    w, denominator = weights(values, given)
    rows, cells = given.shape
    ii, jj = np.triu_indices(cells, 1)
    normalizers = NORMALIZERS[values.shape[0]]
    denominators = np.array(sorted({a*b for a,b in combinations_with_replacement(normalizers,2)}), np.int64)
    # Each dot product is <=max_denominator**2, so this bounds any bin sum.
    assert rows*max(normalizers)**2 < np.iinfo(np.int64).max
    numerators = np.zeros((len(denominators), len(ii)), np.int64)
    joint = np.zeros(len(ii), np.int64)
    for row in range(rows):
        common = given[row, ii] & given[row, jj]
        dot = (w[row] @ w[row].T)[ii, jj]
        den = denominator[row, ii]*denominator[row, jj]
        joint += common
        for k, value in enumerate(denominators):
            numerators[k] += np.where(common & (den == value), dot, 0)
    cost = convert_bins(numerators, denominators, joint)
    return dict(cost=cost, joint_count=joint, posterior_numerators=numerators, posterior_denominators=denominators)


def independent_weights(values, given):
    validate(values, given)
    q, rows, cells = values.shape
    result = []
    for row in range(rows):
        board = []
        for cell in range(cells):
            if not given[row, cell]:
                board.append((None, None))
                continue
            observed = list(map(int, values[:, row, cell]))
            # Scale each query likelihood by320 before multiplying.
            likelihoods = []
            for digit in range(9):
                numerator = 1
                for label in observed:
                    numerator *= 296 if label == digit else 3
                likelihoods.append(numerator)
            board.append((likelihoods, sum(likelihoods)))
        result.append(board)
    return result


def independent_score(values, given, literal=False):
    posterior = independent_weights(values, given)
    rows, cells = given.shape
    costs, joint, totals = [], [], []
    for i in range(cells):
        for j in range(i+1, cells):
            groups = {}
            direct = Fraction(0)
            count = 0
            for row in range(rows):
                if not given[row, i] or not given[row, j]:
                    continue
                left, ld = posterior[row][i]
                right, rd = posterior[row][j]
                numerator = sum(a*b for a,b in zip(left,right,strict=True))
                denominator = ld*rd
                if literal:
                    direct += Fraction(numerator, denominator)
                else:
                    groups[denominator] = groups.get(denominator, 0)+numerator
                count += 1
            total = direct if literal else sum((Fraction(num,den) for den,num in groups.items()), Fraction(0))
            assert 0 <= total <= count
            joint.append(count)
            costs.append(round(1000000*(total+1)/(count+2)))
            totals.append(total)
    return np.array(costs,np.int64), np.array(joint,np.int64), totals
