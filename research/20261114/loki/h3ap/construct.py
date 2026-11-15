"""Pure unfiltered source-zero construction with optional supplied D4 prior."""
import itertools
import numpy as np
def views(board):
    square=np.asarray(board,bool).reshape(12,12);allviews=[np.ascontiguousarray(np.rot90(np.fliplr(square)if reflect else square,turn)).reshape(-1)for reflect in(False,True)for turn in range(4)];return list({v.tobytes():v for v in allviews}.values())
def construct(boards,finite_offsets,inherited_lines,augment):
    assert np.asarray(boards).shape==(16,144)
    values=[v for board in boards for v in views(board)]if augment else list(boards);unique=list({np.asarray(v,bool).tobytes():v for v in values}.values());seen={(int(b)//12-int(a)//12,int(b)%12-int(a)%12)for board in unique for a,b in itertools.combinations(np.flatnonzero(board),2)};zero=set(map(tuple,finite_offsets))-seen;F=np.zeros((169,169),np.int32)
    for a,b in itertools.combinations(range(169),2):
        dr,dc=b//13-a//13,b%13-a%13;F[a,b]=F[b,a]=(dr,dc)in zero or any(r*dr+c*dc==0 for r,c in inherited_lines)
    return F,dict(raw_labels=16,attempted_views=128 if augment else 16,unique_views=len(unique),observed_finite_offsets=sorted(seen&set(map(tuple,finite_offsets))),retained_zero_offsets=sorted(zero))
