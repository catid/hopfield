"""Pure N12 positive repair of a fixed N13 support; no file or pool access."""
import itertools
import numpy as np
def views(board):
    x=np.asarray(board,dtype=bool).reshape(12,12)
    values=[np.ascontiguousarray(np.rot90(np.fliplr(x)if reflect else x,turn)).reshape(-1)for reflect in(False,True)for turn in range(4)]
    return list({v.tobytes():v for v in values}.values())
def offsets(boards):
    return {(int(b)//12-int(a)//12,int(b)%12-int(a)%12)for board in boards for a,b in itertools.combinations(np.flatnonzero(board),2)}
def repair(boards,F,finite,augment):
    assert np.asarray(boards).shape==(32,144)
    values=[v for board in boards for v in views(board)]if augment else list(boards);unique=list({np.asarray(v,bool).tobytes():v for v in values}.values());seen=offsets(unique)&set(map(tuple,finite));answer=F.copy()
    for a,b in itertools.combinations(range(169),2):
        if(b//13-a//13,b%13-a%13)in seen:answer[a,b]=answer[b,a]=0
    return answer,dict(raw_labels=32,attempted_views=256 if augment else 32,unique_views=len(unique),witnessed_finite_offsets=sorted(seen))
