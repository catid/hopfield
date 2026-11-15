"""H2bp uniform ordered open-cell endpoints; no outcome-dependent sampling."""
import numpy as np
import study as S


def open_indices(tokens):
    tokens=np.asarray(tokens)
    if tokens.ndim!=2 or tokens.dtype.kind not in 'buif' or not np.all(np.isin(tokens,[0,1,2,3])):
        raise ValueError('Malformed maze tokens')
    if np.count_nonzero(tokens==2)!=1 or np.count_nonzero(tokens==3)!=1:
        raise ValueError('Exactly one original start and goal required')
    return np.flatnonzero(tokens.ravel()>0)


def assign(tokens,start,goal):
    candidates=open_indices(tokens)
    if any(not isinstance(x,(int,np.integer)) for x in [start,goal]) or start==goal or start not in candidates or goal not in candidates:
        raise ValueError('Endpoints must be distinct open indices')
    changed=np.asarray(tokens,dtype=np.uint8).copy()
    original=np.array([np.flatnonzero(changed.ravel()==v)[0] for v in [2,3]],np.int64)
    changed[(changed==2)|(changed==3)]=1
    changed.flat[start]=2;changed.flat[goal]=3
    assert np.array_equal(changed>0,np.asarray(tokens)>0)
    return changed,original,np.array([start,goal],np.int64)


def sample(tokens,rng):
    candidates=open_indices(tokens)
    endpoints=rng.choice(candidates,size=2,replace=False)
    return assign(tokens,int(endpoints[0]),int(endpoints[1]))


def label(tokens,rng):
    changed,original,endpoints=sample(tokens,rng)
    labels,distance,reachable=S.bfs(changed)
    if distance<=0 or reachable!=int((changed>0).sum()):
        raise ValueError('Unreachable or disconnected sampled maze; no replacement allowed')
    return changed,labels,distance,original,endpoints


def one(index,base):
    tokens,_,_,opened,eligible=S.maze(180,.8,np.random.SeedSequence([base,180,80,index]))
    rng=np.random.default_rng(np.random.SeedSequence([base,180,80,index,1]))
    changed,labels,distance,original,endpoints=label(tokens,rng)
    return changed,labels,distance,opened,eligible,original,endpoints
