"""Unchanged predictable exposure rule for the four equal-budget arms."""
from calibrated_policy import calibration,crossing_thresholds,learn_queried
from noisy_policy import choose_mask as frozen_mask


def choose_mask(arm,joint,excluded,k,tie_order):
    assert arm in ('exact','fixed_upper','learned_small','learned_large')
    return frozen_mask('exponential_e',joint,excluded,k,tie_order)
