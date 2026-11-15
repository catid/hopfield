"""H2br independent50:50 endpoint mixture; no outcome/distance rejection."""
import numpy as np
import study as S
import endpoint_shift_generator as U
import far_placement_generator as F


def component(index,base):
    return int(np.random.default_rng(np.random.SeedSequence([base,180,80,index,2])).integers(2))


def sample(tokens,mixture,rng):
    if not isinstance(mixture,(int,np.integer)) or mixture not in [0,1]:raise ValueError('Mixture component must be0or1')
    if mixture==0:
        changed,old,pair=U.sample(tokens,rng);orientation=-1
    else:changed,old,pair,orientation=F.sample(tokens,rng)
    return changed,old,pair,orientation


def one(index,base):
    tokens,_,_,opened,eligible=S.maze(180,.8,np.random.SeedSequence([base,180,80,index]))
    mixture=component(index,base);rng=np.random.default_rng(np.random.SeedSequence([base,180,80,index,1]))
    changed,old,pair,orientation=sample(tokens,mixture,rng)
    labels,distance,reachable=S.bfs(changed)
    if distance<=0 or reachable!=int((changed>0).sum()):raise ValueError('Disconnected source; no replacement')
    return changed,labels,distance,opened,eligible,old,pair,orientation,mixture
