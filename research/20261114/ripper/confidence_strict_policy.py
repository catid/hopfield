"""Fixed model-derived LR threshold 1/16; no Sudoku geometry enters assignment."""
from fractions import Fraction
from functools import lru_cache
import numpy as np
from calibrated_policy import calibration


def strong_ratio(numerator, denominator):
    assert numerator > 0 and denominator > 0
    return 16*numerator <= denominator


@lru_cache(None)
def strict_strong(eta, count, equal):
    assert isinstance(eta, Fraction) and 0 <= equal <= count
    _, a, b, _ = calibration(eta)
    numerator = pow(a.numerator, equal)*pow(b.numerator, count-equal)
    denominator = pow(a.denominator, equal)*pow(b.denominator, count-equal)
    return strong_ratio(numerator, denominator)


def weights(support, joint, equal, eta, permutation, old_rule):
    assert support.shape == joint.shape == equal.shape == old_rule.shape == (3240,)
    assert support.dtype == bool and np.array_equal(old_rule > 0, support)
    assert np.array_equal(np.sort(permutation), np.arange(3240))
    strict = np.where(support, 16, 0).astype(np.uint8)
    for index in np.flatnonzero(support):
        if not strict_strong(eta, int(joint[index]), int(equal[index])):
            strict[index] = 1
    shuffled = np.where(support, 16, 0).astype(np.uint8)
    order = permutation[support[permutation]]
    shuffled[order[:int((strict == 1).sum())]] = 1
    assert np.array_equal(np.sort(shuffled), np.sort(strict))
    assert np.array_equal(strict > 0, support) and np.array_equal(shuffled > 0, support)
    return {'old_rule': old_rule.copy(), 'strict_rule': strict, 'shuffled': shuffled}
