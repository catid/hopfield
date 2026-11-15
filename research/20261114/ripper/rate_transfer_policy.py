"""Frozen exponential acquisition and exact thresholds, two fixed actual rates."""
from calibrated_policy import calibration,crossing_thresholds,learn_queried
from noisy_policy import choose_mask as frozen_mask


def choose_mask(arm,joint,excluded,k,tie_order):
    assert arm in ('exact_01','learned_01','exact_05','learned_05')
    return frozen_mask('exponential_e',joint,excluded,k,tie_order)
