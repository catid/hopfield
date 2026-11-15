"""Fixed H1at mixture of four UNSTOPPED Bernoulli likelihood processes."""
import math
import numpy as np

ALTERNATIVES = (1, 2, 4, 8)  # Numerators over 36; q0 = 9 / 1600.
BOUNDARY = 648000


def log_likelihoods(m, c):
    m = np.asarray(m); c = np.asarray(c)
    return np.stack([c * math.log((a / 36) / (9 / 1600)) +
                     (m - c) * math.log(((36 - a) / 36) / (1591 / 1600))
                     for a in ALTERNATIVES], axis=0)


def mixture_log(m, c):
    logs = log_likelihoods(m, c)
    peak = logs.max(axis=0)
    return peak + np.log(np.exp(logs - peak).sum(axis=0)) - math.log(4)


def exact_crossing(m, c, mixture=True):
    """Integer certificate, including equality at the crossing boundary."""
    m, c = int(m), int(c)
    assert 0 <= c <= m
    values = ALTERNATIVES if mixture else (4,)
    numerator = sum(pow(a, c) * pow(36 - a, m - c) for a in values)
    left = numerator * pow(1600, m)
    right = len(values) * BOUNDARY * pow(36, m) * pow(9, c) * pow(1591, m - c)
    return left >= right


def thresholds(rows=1110):
    result = np.empty((2, rows + 1), np.int32)
    logbound = math.log(BOUNDARY)
    for m in range(rows + 1):
        c = np.arange(m + 1)
        curves = (log_likelihoods(m, c)[2], mixture_log(m, c))
        for arm, curve in enumerate(curves):
            crossings = np.flatnonzero(curve >= logbound)
            cut = int(crossings[0]) if len(crossings) else m + 1
            while cut <= m and not exact_crossing(m, cut, arm == 1):
                cut += 1
            while cut > 0 and exact_crossing(m, cut - 1, arm == 1):
                cut -= 1
            assert cut == m + 1 or exact_crossing(m, cut, arm == 1)
            assert cut == 0 or not exact_crossing(m, cut - 1, arm == 1)
            result[arm, m] = cut
    return result


def counts_step(joint, equal, mask, values):
    cells = np.flatnonzero(mask); view = np.ix_(cells, cells)
    labels = values[cells]
    joint[view] += 1
    equal[view] += labels[:, None] == labels[None, :]
