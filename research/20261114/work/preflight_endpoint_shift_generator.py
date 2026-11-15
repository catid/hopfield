"""Exhaustive tiny-grid H2bp support/label/geometry gate; no fresh cohorts."""
from itertools import permutations
import json
import time
import numpy as np
import study as S
import endpoint_shift_generator as G
from audit import independent
from audit_connectivity_screen import path_valid
from verify_hard_negative_feasibility import valid_shortest
from verify_endpoint_shift_inputs import check_case


class PrescribedPair:
    def __init__(self,pair,candidates):self.pair=pair;self.candidates=candidates;self.calls=0
    def choice(self,candidates,size,replace):
        assert np.array_equal(candidates,self.candidates) and size==2 and replace is False
        self.calls+=1
        return np.array(self.pair)


def fails(call):
    try:call()
    except (ValueError,AssertionError):return
    raise AssertionError('Malformed or disconnected grid accepted')


def main():
    started=time.perf_counter();grids=[]
    full=np.ones((3,3),np.uint8);full[0,0]=2;full[2,2]=3;grids.append(full)
    ring=full.copy();ring[1,1]=0;grids.append(ring)
    reversed_corners=np.flip(full,axis=0).copy();grids.append(reversed_corners)
    corridor=np.array([[2,1,3]],np.uint8);grids.append(corridor)
    adjacent=np.array([[2,3]],np.uint8);grids.append(adjacent)
    cases=adjacent_cases=old_endpoint_cases=leaf_cases=junction_cases=0
    for tokens in grids:
        candidates=np.flatnonzero(tokens.ravel()>0);seen=set();original=np.array([np.flatnonzero(tokens.ravel()==v)[0] for v in [2,3]])
        for pair in permutations(map(int,candidates),2):
            rng=PrescribedPair(pair,candidates)
            changed,label,distance,old,new=G.label(tokens,rng)
            assert rng.calls==1 and tuple(new)==pair and pair not in seen;seen.add(pair)
            assert np.array_equal(old,original) and np.array_equal(changed>0,tokens>0)
            assert np.count_nonzero(changed==2)==np.count_nonzero(changed==3)==1
            for i in original:
                if i not in pair:assert changed.flat[i]==1
            ok,d=independent(changed,label);assert ok and d==distance and valid_shortest(changed,label)
            assert path_valid(changed,label) and distance>0
            adjacent_cases+=distance==1;old_endpoint_cases+=any(i in pair for i in original)
            for endpoint in pair:
                r,c=np.unravel_index(endpoint,tokens.shape)
                degree=sum(0<=r+dr<tokens.shape[0] and 0<=c+dc<tokens.shape[1] and tokens[r+dr,c+dc]>0 for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)])
                leaf_cases+=degree==1;junction_cases+=degree>=3
            cases+=1
        assert len(seen)==len(candidates)*(len(candidates)-1)
    # Distinct distance2 endpoints admit both shortest and longer induced paths.
    changed,_,_=G.assign(ring,0,2)
    longer=(changed>0);longer[0,1]=False
    assert path_valid(changed,longer) and not valid_shortest(changed,longer)
    malformed=[np.array([[2,2,3]],np.uint8),np.array([[1,1]],np.uint8),np.array([[2,4,3]],np.uint8)]
    for grid in malformed:fails(lambda:G.open_indices(grid))
    for pair in [(0,0),(0,99),(0,1.5)]:fails(lambda:G.assign(full,*pair))
    disconnected=np.array([[2,0,3]],np.uint8)
    fails(lambda:G.label(disconnected,PrescribedPair((0,2),np.array([0,2]))))
    assert adjacent_cases and old_endpoint_cases and leaf_cases and junction_cases
    # These tiny synthetic tokens are not a generated research maze. Preserve a
    # valid pair/label while corrupting each independently checked source field.
    base,index=17,3
    original=(full.copy(),None,None,7,9)
    rng=np.random.default_rng(np.random.SeedSequence([base,180,80,index,1]))
    changed,fresh_label,_,old,pair=G.label(full,rng)
    assert valid_shortest(changed,fresh_label)
    actual=(changed,old,pair,7,9)
    check_case(index,base,actual,original)
    wrong_old=old.copy();wrong_old[0]=4
    fails(lambda:check_case(index,base,(changed,wrong_old,pair,7,9),original))
    changed_wall=changed.copy();spare=next(i for i in range(9) if i not in pair);changed_wall.flat[spare]=0
    assert valid_shortest(changed_wall,S.bfs(changed_wall)[0])
    fails(lambda:check_case(index,base,(changed_wall,old,pair,7,9),original))
    fails(lambda:check_case(index,base,(changed,old,pair,6,9),original))
    fails(lambda:check_case(index,base,(changed,old,pair,7,8),original))
    wrong_token=changed.copy();wrong_token.flat[spare]=3
    fails(lambda:check_case(index,base,(wrong_token,old,pair,7,9),original))
    out={'status':'passed','synthetic_only':True,'ordered_distinct_pairs':cases,'adjacent_pairs':int(adjacent_cases),
        'pairs_including_old_endpoints':int(old_endpoint_cases),'leaf_endpoint_cases':int(leaf_cases),'junction_endpoint_cases':int(junction_cases),
        'valid_longer_alternative_retained_by_geometry':True,'seeded_source_metadata_corruption_rejections':5,'zero_distance_and_malformed_rejections':7,
        'fresh_inputs_drawn':0,'model_outcomes_read':False,'seconds':time.perf_counter()-started,
        'source_sha256':{name:S.sha(S.HERE/name) for name in ['endpoint_shift_generator.py','preflight_endpoint_shift_generator.py','verify_endpoint_shift_inputs.py','study.py','audit.py','audit_connectivity_screen.py','verify_hard_negative_feasibility.py']}}
    S.save(S.OUT/'bp_first_endpoint_shift/generator-synthetic-gate.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
