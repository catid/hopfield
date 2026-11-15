"""Exact integer-count reformulation of paired two-way bootstrap means."""
import numpy as np


def resample_means(diff,seed,draws=5000,batch=32):
    assert diff.ndim==2 and np.all(np.isin(diff,[-1,0,1]))
    banks,puzzles=diff.shape;rng=np.random.default_rng(seed);result=[]
    data=diff.astype(np.float64)
    for start in range(0,draws,batch):
        count=min(batch,draws-start)
        rows=np.empty((count,banks));cols=np.empty((count,puzzles))
        for b in range(count):
            rows[b]=np.bincount(rng.integers(0,banks,banks),minlength=banks)
            cols[b]=np.bincount(rng.integers(0,puzzles,puzzles),minlength=puzzles)
        # All operands and sums are small integers represented exactly in FP64.
        totals=((rows@data)*cols).sum(1)
        result.extend(100.0*totals/(banks*puzzles))
    return np.asarray(result)
