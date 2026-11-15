"""Separate H2bo absolute and paired endpoints, fixed 4096 maze denominator."""
from fallback_confirmation_statistics import absolute as inherited_absolute
from fallback_confirmation_statistics import noninferiority as inherited_noninferiority


def absolute(successes, n=4096):
    if n != 4096 or not isinstance(successes, int) or not 0 <= successes <= n:
        raise ValueError('Invalid fresh joint cohort')
    return inherited_absolute(successes, n)


def noninferiority(gains, losses, n=4096):
    if n != 4096 or any(not isinstance(x, int) or x < 0 for x in [gains, losses]) or gains + losses > n:
        raise ValueError('Invalid paired fresh cohort')
    return inherited_noninferiority(gains, losses, n)
