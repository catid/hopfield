"""Independent seeded H2bq source/orientation/endpoints/full-token identity."""
import numpy as np
import study as S


def check_case(index,base,actual,original):
    tokens,old_endpoints,pair,opened,eligible,orientation=actual
    original_tokens,_,_,original_opened,original_eligible=original
    expected_old=np.array([np.flatnonzero(original_tokens.ravel()==value)[0] for value in [2,3]])
    assert np.array_equal(old_endpoints,expected_old)
    assert np.array_equal(tokens>0,original_tokens>0)
    assert opened==original_opened and eligible==original_eligible
    n=original_tokens.shape[0];assert original_tokens.shape==(n,n) and n%4==0
    q=n//4;indices=np.flatnonzero(original_tokens.ravel()>0);r,c=indices//n,indices%n
    sets=[indices[(r<q)&(c<q)],indices[(r>=n-q)&(c>=n-q)],indices[(r<q)&(c>=n-q)],indices[(r>=n-q)&(c<q)]]
    assert all(len(a)>0 for a in sets)
    rng=np.random.default_rng(np.random.SeedSequence([base,180,80,int(index),1]))
    selected=int(rng.integers(4));assert isinstance(orientation,(int,np.integer)) and orientation==selected
    a,b=[(0,1),(1,0),(2,3),(3,2)][selected]
    expected_pair=np.array([rng.choice(sets[a]),rng.choice(sets[b])])
    assert np.array_equal(pair,expected_pair) and pair[0]!=pair[1]
    positions=np.array(np.unravel_index(pair,(n,n))).T
    assert np.abs(positions[0]-positions[1]).sum()>=n+2
    rewritten=original_tokens.copy();rewritten[rewritten>=2]=1
    rewritten.flat[expected_pair[0]]=2;rewritten.flat[expected_pair[1]]=3
    assert np.array_equal(tokens,rewritten)


def check_shard(args):
    path,base=args
    with np.load(path,allow_pickle=False) as z:
        for index,tokens,old,pair,opened,eligible,orientation in zip(z['indices'],z['tokens'],z['original_endpoints'],z['sampled_endpoints'],z['opened_loops'],z['eligible_walls'],z['endpoint_orientation']):
            original=S.maze(180,.8,np.random.SeedSequence([base,180,80,int(index)]))
            check_case(index,base,(tokens,old,pair,opened,eligible,orientation),original)
        return {'status':'passed','inputs_checked':len(z['indices']),'seeded_original_maze_replays':len(z['indices'])}
