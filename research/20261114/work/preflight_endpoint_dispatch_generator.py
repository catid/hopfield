"""H2br exact finite mixture support; synthetic inputs only, no cohort draw."""
from fractions import Fraction
from itertools import permutations,product
import json
import time
import numpy as np
import study as S
import endpoint_dispatch_generator as G
import far_placement_generator as F
from verify_endpoint_dispatch_inputs import check_case


class UniformDraw:
    def __init__(self,pair,eligible):self.pair=pair;self.eligible=eligible;self.calls=0
    def choice(self,candidates,size,replace):
        assert size==2 and not replace and np.array_equal(candidates,self.eligible);self.calls+=1;return np.array(self.pair)


class FarDraw:
    def __init__(self,orientation,pair,regions):self.orientation=orientation;self.pair=pair;self.regions=regions;self.calls=0
    def integers(self,n):assert n==4 and self.calls==0;self.calls=1;return self.orientation
    def choice(self,candidates):
        assert self.calls in [1,2];region=F.ORIENTATIONS[self.orientation][self.calls-1];assert np.array_equal(candidates,self.regions[region]);result=self.pair[self.calls-1];self.calls+=1;return result


def fails(call):
    try:call()
    except (ValueError,AssertionError):return
    raise AssertionError('Invalid mixture provenance accepted')


def main():
    started=time.perf_counter();t=np.ones((8,8),np.uint8);t[0,0]=2;t[-1,-1]=3;u=t.copy();u[0,1]=0;u[6,0]=u[6,1]=0;count=0;overlaps=0
    for tokens in [t,u,np.flip(t,0).copy()]:
        eligible=np.flatnonzero(tokens.ravel()>0);regions=F.regions(tokens);masses={};component_mass=[Fraction(0),Fraction(0)]
        for pair in permutations(map(int,eligible),2):
            rng=UniformDraw(pair,eligible);changed,old,actual,orientation=G.sample(tokens,0,rng)
            assert rng.calls==1 and tuple(actual)==pair and orientation==-1 and np.array_equal(changed>0,tokens>0)
            assert changed.flat[pair[0]]==2 and changed.flat[pair[1]]==3
            assert all(changed.flat[i]==(2 if i==pair[0] else 3 if i==pair[1] else 1) for i in old)
            probability=Fraction(1,2*len(eligible)*(len(eligible)-1));masses[pair]=probability;component_mass[0]+=probability;count+=1
        for orientation,(a,b) in enumerate(F.ORIENTATIONS):
            for pair in product(map(int,regions[a]),map(int,regions[b])):
                changed,old,actual,value=G.sample(tokens,1,FarDraw(orientation,pair,regions));assert tuple(actual)==pair and value==orientation and np.array_equal(changed>0,tokens>0)
                probability=Fraction(1,8*len(regions[a])*len(regions[b]));assert pair in masses;overlaps+=1;masses[pair]+=probability;component_mass[1]+=probability;count+=1
        assert component_mass==[Fraction(1,2),Fraction(1,2)] and sum(masses.values())==1
    seeded=0
    for index in range(16):
        base=17;mixture=G.component(index,base);rng=np.random.default_rng(np.random.SeedSequence([base,180,80,index,1]));changed,old,pair,orientation=G.sample(t,mixture,rng)
        actual=(changed,old,pair,7,9,orientation,mixture);original=(t,None,None,7,9);check_case(index,base,actual,original);seeded+=1
        fails(lambda:check_case(index,base,(*actual[:-1],1-mixture),original))
        fails(lambda:check_case(index,base,(*actual[:5],orientation+1,mixture),original))
        fails(lambda:check_case(index,base,(changed,old,pair[::-1],7,9,orientation,mixture),original))
    fails(lambda:G.sample(t,2,np.random.default_rng(0)))
    out={'status':'passed','synthetic_only':True,'exact_component_endpoint_cases':count,'far_pairs_also_in_uniform_support':overlaps,'component_masses':'exact1/2each; overlapping endpoint-pair mass adds','seeded_synthetic_coin_endpoint_cases':seeded,'corruption_or_invalid_rejections':49,'fresh_inputs_drawn':0,'model_outcomes_read':False,'seconds':time.perf_counter()-started,'source_sha256':{name:S.sha(S.HERE/name) for name in ['endpoint_dispatch_generator.py','verify_endpoint_dispatch_inputs.py','preflight_endpoint_dispatch_generator.py','endpoint_shift_generator.py','far_placement_generator.py','verify_endpoint_shift_inputs.py','verify_far_placement_inputs.py','study.py']}}
    S.save(S.OUT/'bp_first_endpoint_dispatch/generator-synthetic-gate.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
