"""Bounded synthetic H1bz posterior identities and exact aggregation benchmark."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib, json, statistics, time, platform
import numpy as np
from posterior_query_math import weights, score, independent_score, convert_bins


def main():
    q, r = Fraction(37,40), Fraction(3,320)
    labels = np.array(list(product(range(9),repeat=3)),np.int8).T[:, :, None]
    given = np.ones((729,1),bool)
    w, den = weights(labels,given)
    for row in range(729):
        literal = []
        for digit in range(9):
            likelihood = Fraction(1)
            for label in labels[:,row,0]:
                likelihood *= q if int(label)==digit else r
            literal.append(likelihood)
        total = sum(literal)
        assert [value/total for value in literal] == [Fraction(int(value),int(den[row,0])) for value in w[row,0]]
    assert np.array_equal(weights(labels[::-1],given)[0],w)
    for digit in range(8):
        permutation=np.arange(9,dtype=np.int8);permutation[digit],permutation[digit+1]=permutation[digit+1],permutation[digit]
        mapped,_=weights(permutation[labels],given)
        assert np.array_equal(mapped[:,:,permutation],w)
    # One-query posterior pair equality is affine in observed equality.
    a=q*q+8*r*r;b=2*q*r+7*r*r
    for left,right in product(range(9),repeat=2):
        one=np.array([[[left,right]]],np.int8);mask=np.ones((1,2),bool)
        result=score(one,mask);_,_,total=independent_score(one,mask,True)
        assert total==[a if left==right else b]
        assert result['cost'][0]==round(1000000*(total[0]+1)/3)
    # Degenerate one-hot posteriors recover literal equality counts.
    counts=np.array([0,1,2,3],np.int64);exposed=np.full(4,3,np.int64)
    onehot=convert_bins(counts[None],np.ones(1,np.int64),exposed)
    assert onehot.tolist()==[round(Fraction(1000000*(int(c)+1),5)) for c in counts]
    # Exact half-even ties at fractional posterior sums, withjoint=1.
    assert convert_bins(np.array([[1,7]],np.int64),np.array([2000000],np.int64),np.array([1,1],np.int64)).tolist()==[333334,333334]
    rng=np.random.default_rng(989001)
    checks=[]
    for queries,rows,cells in [(1,17,9),(3,17,9),(3,90,81)]:
        clean=rng.integers(0,9,(rows,cells),dtype=np.int8)
        values=np.stack([np.where(rng.random((rows,cells))<.075,(clean+rng.integers(1,9,(rows,cells),dtype=np.int8))%9,clean).astype(np.int8) for _ in range(queries)])
        mask=rng.random((rows,cells))<.9;values[:,~mask]=-1
        produced=score(values,mask)
        reference,joint,total=independent_score(values,mask)
        assert np.array_equal(produced['cost'],reference) and np.array_equal(produced['joint_count'],joint)
        literal,_,_=independent_score(values,mask,True)
        assert np.array_equal(literal,reference)
        # Query-order invariance, arbitrary label permutation and all-missing rows.
        assert np.array_equal(score(values[::-1],mask)['cost'],reference)
        permutation=rng.permutation(9);permuted=values.copy();valid=permuted>=0;permuted[valid]=permutation[permuted[valid]]
        assert np.array_equal(score(permuted,mask)['cost'],reference)
        checks.append(dict(queries=queries,rows=rows,cells=cells,exact_parity=True))
    empty=np.full((3,2,9),-1,np.int8);missing=np.zeros((2,9),bool)
    missing_result=score(empty,missing)
    assert not missing_result['joint_count'].any() and (missing_result['cost']==500000).all()
    timings={}
    for name,fn in [('binned',lambda:score(values,mask)['cost']),('dictionary',lambda:independent_score(values,mask)[0]),('literal_fraction',lambda:independent_score(values,mask,True)[0])]:
        elapsed=[]
        for _ in range(3):
            started=time.perf_counter();result=fn();elapsed.append(time.perf_counter()-started);assert np.array_equal(result,reference)
        timings[name]=dict(samples_wall_seconds=elapsed,median_wall_seconds=statistics.median(elapsed))
    location=Path(__file__).resolve().parent
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    receipt=dict(status='passed',host=platform.node(),scope='Only unrelated synthetic labels; no cohort scores or observations read.',
                 exact_posterior_truth_label_cases=9*729,query_permutation=True,adjacent_digit_swaps=8,
                 single_query_pair_cases=81,onehot_limit=True,half_even_boundaries=True,aggregation_fixtures=checks,
                 full_90x81_toy_timings=timings,int64_bound=270*25934552**2,maximum_int64=int(np.iinfo(np.int64).max),
                 source_sha256={path.name:sha(path) for path in [Path(__file__).resolve(),location/'posterior_query_math.py']})
    (location/'posterior-query-math-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
