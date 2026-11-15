"""Fixed-size indicator-only interfaces to the unchanged exact CP estimator."""
from calibration_estimator import fit_count


def fit_indicators(indicators,pairs):
    assert pairs in (1000,10000)
    values=list(indicators)
    assert len(values)==pairs and all(v in (False,True,0,1) for v in values)
    return fit_count(sum(int(v) for v in values),pairs)
