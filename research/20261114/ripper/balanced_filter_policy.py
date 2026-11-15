"""Supplied global prior: no digit occurs more than nine times per complete board."""
import numpy as np


def accept_observed(values):
    assert values.shape==(25,) and np.all((values>=0)&(values<9))
    return bool(np.bincount(values,minlength=9).max()<=9)
