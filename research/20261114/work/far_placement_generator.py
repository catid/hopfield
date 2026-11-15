"""H2bq exact four-orientation opposite-quarter open-cell sampling law."""
import numpy as np
import study as S
from endpoint_shift_generator import open_indices,assign
ORIENTATIONS=(('NW','SE'),('SE','NW'),('NE','SW'),('SW','NE'))


def regions(tokens):
    open_indices(tokens)
    tokens=np.asarray(tokens)
    if tokens.shape[0]!=tokens.shape[1] or tokens.shape[0]%4 or tokens.shape[0]<4:
        raise ValueError('Square side divisible byfour required')
    ordinary=tokens.copy();ordinary[ordinary>=2]=1
    n=tokens.shape[0];q=n//4;r,c=np.indices(tokens.shape)
    masks={'NW':(r<q)&(c<q),'NE':(r<q)&(c>=n-q),'SW':(r>=n-q)&(c<q),'SE':(r>=n-q)&(c>=n-q)}
    eligible={name:np.flatnonzero((mask&(ordinary>0)).ravel()) for name,mask in masks.items()}
    if any(not len(values) for values in eligible.values()):raise ValueError('Empty eligible region; no redraw or alternate orientation')
    return eligible


def sample(tokens,rng):
    eligible=regions(tokens)
    orientation=int(rng.integers(4))
    if not 0<=orientation<4:raise ValueError('Invalid orientation index')
    start_region,goal_region=ORIENTATIONS[orientation]
    start=int(rng.choice(eligible[start_region]));goal=int(rng.choice(eligible[goal_region]))
    changed,original,endpoints=assign(tokens,start,goal)
    a,b=np.array(np.unravel_index(endpoints,changed.shape)).T
    assert int(np.abs(a-b).sum())>=changed.shape[0]+2
    return changed,original,endpoints,orientation


def label(tokens,rng):
    changed,original,endpoints,orientation=sample(tokens,rng)
    labels,distance,reachable=S.bfs(changed)
    if distance<=0 or reachable!=int((changed>0).sum()):raise ValueError('Disconnected source; no replacement')
    return changed,labels,distance,original,endpoints,orientation


def one(index,base):
    tokens,_,_,opened,eligible=S.maze(180,.8,np.random.SeedSequence([base,180,80,index]))
    rng=np.random.default_rng(np.random.SeedSequence([base,180,80,index,1]))
    changed,labels,distance,original,endpoints,orientation=label(tokens,rng)
    return changed,labels,distance,opened,eligible,original,endpoints,orientation
