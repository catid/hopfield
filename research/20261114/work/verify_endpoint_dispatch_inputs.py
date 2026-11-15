"""Independent H2br mixture coin and source/endpoints/token replay."""
import numpy as np
import study as S
from verify_endpoint_shift_inputs import check_case as uniform
from verify_far_placement_inputs import check_case as far


def check_case(index,base,actual,original):
    tokens,old,pair,opened,eligible,orientation,mixture=actual
    expected=int(np.random.default_rng(np.random.SeedSequence([base,180,80,int(index),2])).integers(2))
    assert isinstance(mixture,(int,np.integer)) and mixture==expected
    if mixture==0:
        assert orientation==-1;uniform(index,base,(tokens,old,pair,opened,eligible),original)
    else:far(index,base,(tokens,old,pair,opened,eligible,orientation),original)


def check_shard(args):
    path,base=args
    with np.load(path,allow_pickle=False) as z:
        for values in zip(z['indices'],z['tokens'],z['original_endpoints'],z['sampled_endpoints'],z['opened_loops'],z['eligible_walls'],z['endpoint_orientation'],z['mixture_component']):
            index,*actual=values;original=S.maze(180,.8,np.random.SeedSequence([base,180,80,int(index)]));check_case(index,base,actual,original)
        return {'status':'passed','inputs_checked':len(z['indices']),'seeded_original_maze_replays':len(z['indices']),'mixture_coin_replays':len(z['indices'])}
