"""Separate polynomial arithmetic and exhaustive incidence checks."""
import itertools
import numpy as np


def polynomial_product(u,v):
    coefficients=[0,0,0]
    for i in range(2):
        for j in range(2):coefficients[i+j]+=(u//(3**i)%3)*(v//(3**j)%3)
    coefficients[0]-=coefficients[2]
    return coefficients[0]%3+3*(coefficients[1]%3)


def polynomial_sum(u,v):return sum(((u//3**i%3+v//3**i%3)%3)*3**i for i in range(2))


def reference_lines():
    lines=[]
    for slope in range(9):
        for intercept in range(9):
            line=set()
            for point in range(81):
                x,y=divmod(point,9)
                if polynomial_sum(polynomial_product(slope,x),intercept)==y:line.add(point)
            lines.append(line)
    for vertical in range(9):lines.append({point for point in range(81) if point//9==vertical})
    return lines


def reference_balanced(seed):
    rng=np.random.default_rng(seed);lines=reference_lines();mask=np.ones((180,81),bool)
    for cycle in range(2):
        permutation=list(map(int,rng.permutation(81)));order=list(map(int,rng.permutation(90)))
        for offset,lineindex in enumerate(order):
            for cell in lines[lineindex]:mask[90*cycle+offset,permutation[cell]]=False
    return mask


def verify_incidence():
    from balanced_mask_math import add,multiply,line_complements,make_balanced_masks
    for x,y in itertools.product(range(9),repeat=2):
        assert add(x,y)==polynomial_sum(x,y) and multiply(x,y)==polynomial_product(x,y)
        assert add(x,y)==add(y,x) and multiply(x,y)==multiply(y,x)
    for x in range(1,9):assert len([y for y in range(1,9) if multiply(x,y)==1])==1
    for x,y,z in itertools.product(range(9),repeat=3):
        assert multiply(multiply(x,y),z)==multiply(x,multiply(y,z))
        assert multiply(x,add(y,z))==add(multiply(x,y),multiply(x,z))
    lines=reference_lines();assert len(set(map(frozenset,lines)))==90 and all(len(line)==9 for line in lines)
    omitted=np.array([[cell in line for cell in range(81)] for line in lines],np.int32)
    assert np.array_equal(~omitted.astype(bool),line_complements())
    incidence=omitted.T@omitted;assert np.array_equal(incidence,np.ones((81,81),np.int32)+9*np.eye(81,dtype=np.int32))
    observed=1-omitted;assert np.array_equal(observed.T@observed,71*np.ones((81,81),np.int32)+9*np.eye(81,dtype=np.int32))
    for seed in [980801,980802,980803]:
        masks=make_balanced_masks(seed);assert np.array_equal(masks,reference_balanced(seed));assert (masks.sum(1)==72).all() and (masks.sum(0)==160).all()
        counts=masks.astype(np.int32).T@masks.astype(np.int32);assert np.array_equal(counts,142*np.ones((81,81),np.int32)+18*np.eye(81,dtype=np.int32))
        for cycle in range(2):assert len({row.tobytes() for row in masks[90*cycle:90*(cycle+1)]})==90
    return dict(field_elements=9,field_associativity_distributivity_triples=729,lines=90,points=81,omitted_cells_per_line=9,lines_per_point=10,lines_per_distinct_pair=1,one_cycle_pair_coqueries=71,two_cycle_pair_coqueries=142,two_cycle_queries_per_cell=160,independent_replay_seeds=3)


if __name__=='__main__':
    import json
    print(json.dumps(dict(status='passed',**verify_incidence())))
