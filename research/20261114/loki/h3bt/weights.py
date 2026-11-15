"""Exact source posterior weights with explicit integer-resource bounds."""
import math
from compiler import ResourceCap


def integer_weights(candidates,scores,max_bits=100000):
    denominators=[int(scores[c]) for c in candidates];common=1;intermediate=[]
    for d in denominators:
        assert d>0;common=math.lcm(common,d)
        if common.bit_length()>max_bits:raise ResourceCap('Posterior LCM bit cap; whole-study incomplete')
        intermediate.append(common.bit_length())
    weights=[common//d for d in denominators]
    if any(w.bit_length()>max_bits for w in weights):raise ResourceCap('Posterior integer weight bit cap; whole-study incomplete')
    return weights,dict(score_denominators=[str(d) for d in denominators],common_denominator=str(common),LCM_intermediate_bit_lengths=intermediate,integer_weights=[str(w) for w in weights],weight_bit_lengths=[w.bit_length() for w in weights],max_bits=max_bits)
