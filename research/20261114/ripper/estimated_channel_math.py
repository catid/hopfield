"""H1ce exact estimated-channel likelihoods on a fixed positive percentage grid.

The logarithm interval and independent rational-product routines are immutable
H1cc imports; only the exact assumed channel is generalized here.
"""
from decimal import Decimal
from functools import lru_cache
from itertools import product
import numpy as np
from posterior_query_math import validate
from pair_likelihood_math import (SCALE,WORK_PRECISIONS,REFERENCE_DIGITS,REFERENCE_TERMS,
    UncertifiedCost,contexts,neglog_decimal_interval,rounded_decimal_interval,reference_product_cost)
SUPPORTED=tuple((k,100) for k in range(1,89))+((3,40),)


def likelihood_units(channel):
    assert channel in SUPPORTED
    a,b=channel
    return 8*(b-a),a


def weights(values,given,channel):
    validate(values,given);q=values.shape[0];match,wrong=likelihood_units(channel)
    counts=(values[:,:,:,None]==np.arange(9)[None,None,None,:]).sum(0)
    assert (counts.sum(2)[given]==q).all()
    assert 9*max(match,wrong)**q<np.iinfo(np.int64).max
    lookup=np.array([match**n*wrong**(q-n) for n in range(q+1)],np.int64)
    w=lookup[counts];w[~given]=0;denominator=w.sum(2)
    assert (int(denominator.max()) if denominator.size else 0)**2<np.iinfo(np.int64).max
    assert not denominator[~given].any()
    return w,denominator


def independent_weights(values,given,channel):
    validate(values,given);assert channel in SUPPORTED
    q,rows,cells=values.shape;a,b=channel;result=[]
    # Literal likelihood product, with channel algebra written separately.
    for row in range(rows):
        board=[]
        for cell in range(cells):
            if not given[row,cell]:board.append((None,None));continue
            likelihoods=[]
            for digit in range(9):
                numerator=1
                for query in range(q):numerator*=8*b-8*a if int(values[query,row,cell])==digit else a
                likelihoods.append(numerator)
            board.append((likelihoods,sum(likelihoods)))
        result.append(board)
    return result


@lru_cache(maxsize=None)
def type_table(q,channel):
    """Enumerate all possible ordered query patterns; no cohort observations."""
    assert q in [1,3]
    labels=np.array(list(product(range(9),repeat=q)),np.int8)
    counts=(labels[:,:,None]==np.arange(9)[None,None,:]).sum(1)
    match,wrong=likelihood_units(channel)
    lookup=np.array([match**n*wrong**(q-n) for n in range(q+1)],np.int64)
    w=lookup[counts];normalizer=w.sum(1)
    assert int(normalizer.max())**2<np.iinfo(np.int64).max
    den=np.outer(normalizer,normalizer);dot=w@w.T
    den_values=np.unique(den);stride=int(den.max())+1
    assert len(den_values)*stride+int(dot.max())<np.iinfo(np.int64).max
    keys=np.unique(np.searchsorted(den_values,den)*stride+dot)
    numerators=keys%stride;denominators=den_values[keys//stride]
    assert ((numerators>0)&(numerators<denominators)).all()
    logs={precision:[neglog_decimal_interval(int(d-n),int(d),precision) for n,d in zip(numerators,denominators,strict=True)] for precision in WORK_PRECISIONS}
    return dict(q=q,keys=keys,numerators=numerators,denominators=denominators,den_values=den_values,stride=stride,logs=logs)


def type_counts(values,given,channel):
    table=type_table(values.shape[0],channel);w,denominator=weights(values,given,channel)
    rows,cells=given.shape;ii,jj=np.triu_indices(cells,1)
    counts=np.zeros((len(table['keys']),len(ii)),np.int64);joint=np.zeros(len(ii),np.int64)
    for row in range(rows):
        common=given[row,ii]&given[row,jj];edges=np.flatnonzero(common)
        dot=(w[row]@w[row].T)[ii[common],jj[common]]
        den=denominator[row,ii[common]]*denominator[row,jj[common]]
        den_index=np.searchsorted(table['den_values'],den)
        assert np.array_equal(table['den_values'][den_index],den)
        key=den_index*table['stride']+dot;index=np.searchsorted(table['keys'],key)
        assert np.array_equal(table['keys'][index],key)
        counts[index,edges]+=1;joint+=common
    assert np.array_equal(counts.sum(0),joint)
    return counts,joint


def interval_from_counts(counts,table,edge,precision):
    down,up,_=contexts(precision);lo=hi=Decimal(0)
    for kind in np.flatnonzero(counts[:,edge]):
        count=Decimal(int(counts[kind,edge]));a,b=table['logs'][precision][int(kind)]
        lo=down.add(lo,down.multiply(count,a));hi=up.add(hi,up.multiply(count,b))
    lo=down.multiply(lo,Decimal(SCALE));hi=up.multiply(hi,Decimal(SCALE))
    return rounded_decimal_interval(lo,hi),lo,hi


def score(values,given,channel):
    table=type_table(values.shape[0],channel);counts,joint=type_counts(values,given,channel)
    costs=[];precisions=[];lowers=[];uppers=[];certified=[]
    for edge in range(len(joint)):
        answer=None
        for precision in WORK_PRECISIONS:
            answer,lo,hi=interval_from_counts(counts,table,edge,precision)
            if answer is not None:break
        costs.append(0 if answer is None else answer);precisions.append(precision);certified.append(answer is not None)
        lowers.append(str(lo));uppers.append(str(hi))
    return dict(cost=np.array(costs,np.int64),joint_count=joint,likelihood_type_counts=counts,
                likelihood_type_numerators=table['numerators'],likelihood_type_denominators=table['denominators'],
                cost_certified=np.array(certified,bool),cost_precision=np.array(precisions,np.int64),
                cost_lower=np.array(lowers),cost_upper=np.array(uppers))


def independent_score(values,given,channel):
    """Literal cell likelihoods; exact per-pair likelihood product; fixed rational log bounds."""
    posterior=independent_weights(values,given,channel);rows,cells=given.shape
    costs=[];joint=[];bounds=[]
    for i in range(cells):
        for j in range(i+1,cells):
            multiplicities={};count=0
            for row in range(rows):
                if not given[row,i] or not given[row,j]:continue
                left,ld=posterior[row][i];right,rd=posterior[row][j]
                dot=sum(a*b for a,b in zip(left,right,strict=True));den=ld*rd
                pair=(den-dot,den);multiplicities[pair]=multiplicities.get(pair,0)+1;count+=1
            terms=[(n,d,c) for (n,d),c in sorted(multiplicities.items())]
            cost,lo,hi,scale=reference_product_cost(terms)
            costs.append(cost);joint.append(count);bounds.append((lo,hi,scale))
    return np.array(costs,np.int64),np.array(joint,np.int64),bounds


def moment(numerator,denominator):
    """Exact numerator/denominator of2p-9p*p/8; unreduced integers."""
    assert 0<=9*numerator<=8*denominator
    return 16*numerator*denominator-9*numerator*numerator,8*denominator*denominator


def estimate_from_counts(disagreements,total):
    assert type(disagreements) is type(total) is int and 0<=disagreements<=total
    base=dict(disagreements=disagreements,total_comparisons=total)
    if total==0 or 9*disagreements>8*total:
        return dict(base,fit_allowed=False,reason='no_comparisons' if total==0 else 'empirical_disagreement_above8over9',
                    rate_numerator=None,rate_denominator=None,lower_clipped=None,upper_clipped=None)
    k=88
    for lower in range(1,88):
        numerator,denominator=moment(2*lower+1,200)
        difference=disagreements*denominator-total*numerator
        if difference<0:k=lower;break
        if difference==0:k=lower if lower%2==0 else lower+1;break
    low_n,low_d=moment(1,100);high_n,high_d=moment(88,100)
    return dict(base,fit_allowed=True,reason=None,rate_numerator=k,rate_denominator=100,
                lower_clipped=disagreements*low_d<total*low_n,upper_clipped=disagreements*high_d>total*high_n)


def estimate(values,given):
    validate(values,given);assert values.shape[0]==3
    total=3*int(given.sum());disagreements=sum(int(np.count_nonzero((values[a]!=values[b])&given)) for a,b in [(0,1),(0,2),(1,2)])
    return estimate_from_counts(disagreements,total)


def independent_estimate_from_counts(disagreements,total):
    from fractions import Fraction
    assert type(disagreements) is type(total) is int and 0<=disagreements<=total
    result=dict(disagreements=disagreements,total_comparisons=total)
    if total==0 or Fraction(disagreements,total)>Fraction(8,9):
        return dict(result,fit_allowed=False,reason='no_comparisons' if total==0 else 'empirical_disagreement_above8over9',
                    rate_numerator=None,rate_denominator=None,lower_clipped=None,upper_clipped=None)
    value=Fraction(disagreements,total)
    def f(p):return 2*p-Fraction(9,8)*p*p
    candidates=[]
    for k in range(1,89):
        lower=None if k==1 else f(Fraction(2*k-1,200));upper=None if k==88 else f(Fraction(2*k+1,200))
        left=lower is None or value>lower or (value==lower and k%2==0)
        right=upper is None or value<upper or (value==upper and k%2==0)
        if left and right:candidates.append(k)
    assert len(candidates)==1
    return dict(result,fit_allowed=True,reason=None,rate_numerator=candidates[0],rate_denominator=100,
                lower_clipped=value<f(Fraction(1,100)),upper_clipped=value>f(Fraction(88,100)))


def independent_estimate(values,given):
    validate(values,given);assert values.shape[0]==3
    disagreements=total=0
    for row in range(given.shape[0]):
        for cell in range(given.shape[1]):
            if not given[row,cell]:continue
            for first in range(3):
                for second in range(first+1,3):
                    total+=1;disagreements+=int(values[first,row,cell]!=values[second,row,cell])
    return independent_estimate_from_counts(disagreements,total)
