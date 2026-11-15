"""H1aw exact-rational Laplace equality costs; no supplied q0 or q1."""
import numpy as np


def round_ratio_even(numerator,denominator):
    assert numerator>=0 and denominator>0
    quotient,remainder=divmod(int(numerator),int(denominator))
    if 2*remainder>denominator or (2*remainder==denominator and quotient%2):quotient+=1
    return quotient


def rate_costs(m,c):
    assert m.shape==c.shape and np.all((0<=c)&(c<=m))
    return np.array([round_ratio_even(1000000*(int(equal)+1),int(exposed)+2)
                     for exposed,equal in zip(m,c,strict=True)],np.int64)
