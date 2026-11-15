"""Exactly1000indicators, unchanged exact outward Clopper-Pearson estimator."""
from calibration_estimator import fit_count


def fit_indicators(indicators):
    values=list(indicators)
    assert len(values)==1000 and all(v in (False,True,0,1) for v in values)
    return fit_count(sum(int(v) for v in values),1000)
