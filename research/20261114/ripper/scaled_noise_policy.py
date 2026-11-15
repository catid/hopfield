"""Unchanged selector and crossing law; short arms stop at their fixed prefix."""
from calibrated_policy import calibration,crossing_thresholds,learn_queried
from noisy_policy import choose_mask as frozen_mask


def choose_mask(arm,joint,excluded,k,tie_order):
    assert arm in ('exact_short','learned_short','exact_long','learned_long')
    return frozen_mask('exponential_e',joint,excluded,k,tie_order)
