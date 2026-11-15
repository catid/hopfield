"""Count-threshold selection with a supplied equal-degree prior."""
import numpy as np


def select(counts, n=81):
    counts=np.asarray(counts)
    assert counts.shape==(n*(n-1)//2,) and counts.dtype.kind in 'iu' and (counts>=0).all()
    ii,jj=np.triu_indices(n,1)
    cuts=np.unique(counts);cuts=cuts[cuts<=np.median(counts)]
    candidates=[]
    for cut in cuts:
        keep=counts<=cut
        if not keep.any():continue
        degree=np.bincount(np.r_[ii[keep],jj[keep]],minlength=n)
        total=int(degree.sum());squared=int(np.dot(degree,degree))
        candidates.append(dict(cut=int(cut),retained=int(keep.sum()),degree_min=int(degree.min()),degree_max=int(degree.max()),
                               degree_sum=total,degree_square_sum=squared,dispersion=n*squared-total*total))
    assert candidates,'No nonempty threshold candidate; preserve as ineligible.'
    selected=min(candidates,key=lambda r:(r['dispersion'],r['retained'],r['cut']))
    support=counts<=selected['cut']
    record=dict(n=n,median_count=float(np.median(counts)),candidates=candidates,selected=dict(selected),
                minimum_dispersion_ties=sum(r['dispersion']==selected['dispersion'] for r in candidates),
                scope='Supplied equal-degree prior. No degree value, edge count, geometry or minimum-density guard.')
    return support,record
