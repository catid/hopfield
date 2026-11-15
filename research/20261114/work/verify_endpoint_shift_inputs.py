"""Independent H2bp input identity reconstruction; no decoder/route imports."""
import numpy as np
import study as S


def check_case(index,base,actual,original):
    tokens,old_endpoints,pair,opened,eligible=actual
    original_tokens,_,_,original_opened,original_eligible=original
    expected_old=np.array([np.flatnonzero(original_tokens.ravel()==value)[0] for value in [2,3]])
    assert np.array_equal(old_endpoints,expected_old)
    assert np.array_equal(tokens>0,original_tokens>0)
    assert opened==original_opened and eligible==original_eligible
    rng=np.random.default_rng(np.random.SeedSequence([base,180,80,int(index),1]))
    expected_pair=rng.choice(np.flatnonzero(original_tokens.ravel()>0),size=2,replace=False)
    assert np.array_equal(pair,expected_pair) and pair[0]!=pair[1]
    rewritten=original_tokens.copy();rewritten[rewritten>=2]=1
    rewritten.flat[expected_pair[0]]=2;rewritten.flat[expected_pair[1]]=3
    assert np.array_equal(tokens,rewritten)


def check_shard(args):
    path,base=args
    with np.load(path,allow_pickle=False) as z:
        for index,tokens,old,pair,opened,eligible in zip(z['indices'],z['tokens'],z['original_endpoints'],z['sampled_endpoints'],z['opened_loops'],z['eligible_walls']):
            original=S.maze(180,.8,np.random.SeedSequence([base,180,80,int(index)]))
            check_case(index,base,(tokens,old,pair,opened,eligible),original)
        return {'status':'passed','inputs_checked':len(z['indices']),'seeded_original_maze_replays':len(z['indices'])}
