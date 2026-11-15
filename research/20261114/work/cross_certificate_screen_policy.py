"""Fixed label-free BP, cross-LP and dual numerical acceptance."""
import numpy as np


def screen(hidden_good,finite,stopped,clamps_ok,bp_margin,lp_margin):
    values=[np.asarray(v) for v in [hidden_good,finite,stopped,clamps_ok,bp_margin,lp_margin]]
    if len({v.shape for v in values})!=1:raise ValueError('All screen fields need identical shapes')
    if any(v.dtype!=np.bool_ for v in values[:4]):raise TypeError('Numerical flags must be Boolean')
    h,f,s,c,b,l=values
    own=h&f&s&c&np.isfinite(b)&(b>=-1e-6)
    cross=h&c&np.isfinite(l)&(l>=-1e-6)
    return own,cross,own&cross
