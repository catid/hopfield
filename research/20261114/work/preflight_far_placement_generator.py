"""Exhaustive tiny-region H2bq sampling support; no research cohort draws."""
from fractions import Fraction
from itertools import product
import json
import time
import numpy as np
import study as S
import far_placement_generator as G
from audit import independent
from verify_hard_negative_feasibility import valid_shortest
from verify_far_placement_inputs import check_case


class Prescribed:
    def __init__(self,orientation,pair,eligible):self.orientation=orientation;self.pair=pair;self.eligible=eligible;self.step=0
    def integers(self,high):assert high==4 and self.step==0;self.step=1;return self.orientation
    def choice(self,candidates):
        assert self.step in [1,2]
        region=G.ORIENTATIONS[self.orientation][self.step-1]
        assert np.array_equal(candidates,self.eligible[region])
        answer=self.pair[self.step-1];self.step+=1;return answer


def fails(call):
    try:call()
    except (ValueError,AssertionError):return
    raise AssertionError('Invalid region source accepted')


def main():
    started=time.perf_counter();full=np.ones((8,8),np.uint8);full[0,0]=2;full[-1,-1]=3
    uneven=full.copy();uneven[0,1]=0;uneven[6,0]=uneven[6,1]=0
    grids=[full,uneven,np.flip(full,0).copy()];cases=old_reuse=0;orientations=[0]*4
    for tokens in grids:
        eligible=G.regions(tokens);weights={};mass=[Fraction(0) for _ in range(4)]
        for orientation,(start_region,goal_region) in enumerate(G.ORIENTATIONS):
            for pair in product(map(int,eligible[start_region]),map(int,eligible[goal_region])):
                rng=Prescribed(orientation,pair,eligible);changed,labels,distance,old,new,actual=G.label(tokens,rng)
                assert rng.step==3 and actual==orientation and tuple(new)==pair and pair[0]!=pair[1]
                assert np.array_equal(changed>0,tokens>0)
                for i in old:
                    if i not in pair:assert changed.flat[i]==1
                ok,edges=independent(changed,labels);assert ok and edges==distance and valid_shortest(changed,labels)
                positions=np.array(np.unravel_index(new,tokens.shape)).T
                assert np.abs(positions[0]-positions[1]).sum()>=10
                weight=Fraction(1,4*len(eligible[start_region])*len(eligible[goal_region]));weights[orientation,pair]=weight;mass[orientation]+=weight
                cases+=1;orientations[orientation]+=1;old_reuse+=any(i in pair for i in old)
        assert mass==[Fraction(1,4)]*4 and sum(weights.values())==1
    # Check actual180-grid boundaries with synthetic tokens only, no maze draw.
    large=np.ones((180,180),np.uint8);large[0,0]=2;large[-1,-1]=3;regions=G.regions(large)
    for name,indices in regions.items():assert len(indices)==45*45
    for orientation,(a,b) in enumerate(G.ORIENTATIONS):
        for ia in [0,-1]:
            for ib in [0,-1]:
                changed,old,pair,actual=G.sample(large,Prescribed(orientation,(int(regions[a][ia]),int(regions[b][ib])),regions))
                positions=np.array(np.unravel_index(pair,large.shape)).T
                assert np.abs(positions[0]-positions[1]).sum()>=182
    empty=full.copy();empty[:2,:2]=0;empty[3,3]=2
    fails(lambda:G.regions(empty))
    fails(lambda:G.regions(np.array([[2,1,3]],np.uint8)))
    base,index=17,3
    original=(full.copy(),None,None,7,9)
    rng=np.random.default_rng(np.random.SeedSequence([base,180,80,index,1]))
    changed,label,_,old,pair,orientation=G.label(full,rng)
    assert valid_shortest(changed,label)
    actual=(changed,old,pair,7,9,orientation);check_case(index,base,actual,original)
    wrong_old=old.copy();wrong_old[0]=3
    fails(lambda:check_case(index,base,(changed,wrong_old,pair,7,9,orientation),original))
    spare=next(i for i in range(64) if i not in pair and i not in old)
    wrong_wall=changed.copy();wrong_wall.flat[spare]=0
    assert valid_shortest(wrong_wall,S.bfs(wrong_wall)[0])
    fails(lambda:check_case(index,base,(wrong_wall,old,pair,7,9,orientation),original))
    fails(lambda:check_case(index,base,(changed,old,pair,8,9,orientation),original))
    fails(lambda:check_case(index,base,(changed,old,pair,7,10,orientation),original))
    fails(lambda:check_case(index,base,(changed,old,pair,7,9,(orientation+1)%4),original))
    fails(lambda:check_case(index,base,(changed,old,pair[::-1],7,9,orientation),original))
    wrong_token=changed.copy();wrong_token.flat[spare]=3
    fails(lambda:check_case(index,base,(wrong_token,old,pair,7,9,orientation),original))
    out={'status':'passed','synthetic_only':True,'tiny_pair_cases':cases,'orientation_case_counts':orientations,
        'cases_reusing_original_endpoints':int(old_reuse),'uniform_orientation_mass':'exact1/4 each','conditional_pair_mass':'exact1/(Na*Nb)',
        'actual180_synthetic_boundary_pairs':16,'empty_region_and_shape_rejections':2,'seeded_provenance_corruption_rejections':7,'fresh_inputs_drawn':0,'model_outcomes_read':False,
        'seconds':time.perf_counter()-started,'source_sha256':{name:S.sha(S.HERE/name) for name in ['far_placement_generator.py','preflight_far_placement_generator.py','endpoint_shift_generator.py','verify_far_placement_inputs.py','study.py','audit.py','verify_hard_negative_feasibility.py']}}
    S.save(S.OUT/'bp_first_far_placement/generator-synthetic-gate.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
