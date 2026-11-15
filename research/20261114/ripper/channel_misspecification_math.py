"""H1cd exact posterior likelihoods under prespecified positive assumed channels.

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
CHANNELS={'assumed_low_3_80':(3,80),'assumed_high_3_20':(3,20)}
SUPPORTED=((3,80),(3,40),(3,20))


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
