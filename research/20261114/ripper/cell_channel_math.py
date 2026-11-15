"""H1cg supplied per-cell channels with exact mixed posterior products.

Only the channel map generalizes the immutable H1cd composite likelihood.
All integer multiplication bounds are checked with Python integers first.
"""
from decimal import Decimal
from functools import lru_cache
from itertools import product
import numpy as np
from posterior_query_math import validate
from pair_likelihood_math import (SCALE,WORK_PRECISIONS,REFERENCE_DIGITS,REFERENCE_TERMS,
    UncertifiedCost,contexts,neglog_decimal_interval,rounded_decimal_interval,reference_product_cost)
RATES=((3,80),(3,40),(3,20))
LIMIT=np.iinfo(np.int64).max


def validate_channels(classes,cells):
    assert isinstance(classes,np.ndarray) and classes.shape==(cells,) and classes.dtype==np.int8
    assert ((classes>=0)&(classes<3)).all()


def likelihood_units(channel):
    assert channel in RATES
    a,b=channel;return 8*(b-a),a


def pattern_weights(q,channel):
    assert q in [1,3];match,wrong=likelihood_units(channel)
    assert 9*max(match,wrong)**q<LIMIT
    labels=np.array(list(product(range(9),repeat=q)),np.int8)
    counts=(labels[:,:,None]==np.arange(9)[None,None,:]).sum(1)
    lookup=np.array([match**n*wrong**(q-n) for n in range(q+1)],np.int64)
    w=lookup[counts];return w,w.sum(1)


def weights(values,given,classes):
    validate(values,given);validate_channels(classes,given.shape[1]);q=values.shape[0];assert q in [1,3]
    counts=(values[:,:,:,None]==np.arange(9)[None,None,None,:]).sum(0)
    assert (counts.sum(2)[given]==q).all()
    lookup=[]
    for channel in RATES:
        match,wrong=likelihood_units(channel);assert 9*max(match,wrong)**q<LIMIT
        lookup.append([match**n*wrong**(q-n) for n in range(q+1)])
    w=np.array(lookup,np.int64)[classes[None,:,None],counts];w[~given]=0;den=w.sum(2)
    assert int(den.max())**2<LIMIT and 9*int(w.max())**2<LIMIT and not den[~given].any()
    return w,den


@lru_cache(maxsize=None)
def type_table(q):
    patterns=[pattern_weights(q,c) for c in RATES];pairs=set();bounds=[]
    for left,(lw,ld) in enumerate(patterns):
        for right,(rw,rd) in enumerate(patterns):
            maximum_den=int(ld.max())*int(rd.max());maximum_dot_bound=9*int(lw.max())*int(rw.max())
            assert maximum_den<LIMIT and maximum_dot_bound<LIMIT
            den=np.outer(ld,rd);dot=lw@rw.T;dv=np.unique(den);stride=int(den.max())+1
            key_bound=len(dv)*stride+int(dot.max());assert key_bound<LIMIT
            keys=np.unique(np.searchsorted(dv,den)*stride+dot)
            pairs.update((int(dv[int(k)//stride]),int(k)%stride) for k in keys)
            bounds.append(dict(left=left,right=right,maximum_den=maximum_den,maximum_dot_bound=maximum_dot_bound,packed_key_bound=key_bound,unique_unreduced_types=len(keys)))
    den_values=np.array(sorted({d for d,n in pairs}),np.int64);stride=int(den_values.max())+1
    key_bound=len(den_values)*stride+max(n for d,n in pairs);assert key_bound<LIMIT
    keys=np.array(sorted(int(np.searchsorted(den_values,d))*stride+n for d,n in pairs),np.int64)
    numerators=keys%stride;denominators=den_values[keys//stride]
    assert len(keys)==len(pairs) and ((numerators>0)&(numerators<denominators)).all()
    logs={precision:[neglog_decimal_interval(int(d-n),int(d),precision) for n,d in zip(numerators,denominators,strict=True)] for precision in WORK_PRECISIONS}
    return dict(q=q,keys=keys,numerators=numerators,denominators=denominators,den_values=den_values,stride=stride,logs=logs,
        overflow_bounds=bounds,global_packed_key_bound=key_bound,int64_limit=int(LIMIT))


def type_counts(values,given,classes):
    table=type_table(values.shape[0]);w,denominator=weights(values,given,classes)
    rows,cells=given.shape;ii,jj=np.triu_indices(cells,1);counts=np.zeros((len(table['keys']),len(ii)),np.int64);joint=np.zeros(len(ii),np.int64)
    for row in range(rows):
        common=given[row,ii]&given[row,jj];edges=np.flatnonzero(common)
        dot=(w[row]@w[row].T)[ii[common],jj[common]];den=denominator[row,ii[common]]*denominator[row,jj[common]]
        den_index=np.searchsorted(table['den_values'],den);assert np.array_equal(table['den_values'][den_index],den)
        key=den_index*table['stride']+dot;index=np.searchsorted(table['keys'],key);assert np.array_equal(table['keys'][index],key)
        counts[index,edges]+=1;joint+=common
    assert np.array_equal(counts.sum(0),joint);return counts,joint


def score(values,given,classes):
    table=type_table(values.shape[0]);counts,joint=type_counts(values,given,classes);costs=[];precisions=[];lowers=[];uppers=[];certified=[]
    for edge in range(len(joint)):
        answer=None
        for precision in WORK_PRECISIONS:
            down,up,_=contexts(precision);lo=hi=Decimal(0)
            for kind in np.flatnonzero(counts[:,edge]):
                count=Decimal(int(counts[kind,edge]));a,b=table['logs'][precision][int(kind)]
                lo=down.add(lo,down.multiply(count,a));hi=up.add(hi,up.multiply(count,b))
            lo=down.multiply(lo,Decimal(SCALE));hi=up.multiply(hi,Decimal(SCALE));answer=rounded_decimal_interval(lo,hi)
            if answer is not None:break
        costs.append(0 if answer is None else answer);precisions.append(precision);certified.append(answer is not None);lowers.append(str(lo));uppers.append(str(hi))
    return dict(cost=np.array(costs,np.int64),joint_count=joint,likelihood_type_counts=counts,
        likelihood_type_numerators=table['numerators'],likelihood_type_denominators=table['denominators'],cost_certified=np.array(certified,bool),
        cost_precision=np.array(precisions,np.int64),cost_lower=np.array(lowers),cost_upper=np.array(uppers))


def independent_weights(values,given,classes):
    validate(values,given);validate_channels(classes,given.shape[1]);q,rows,cells=values.shape;assert q in [1,3];result=[]
    for row in range(rows):
        board=[]
        for cell in range(cells):
            if not given[row,cell]:board.append((None,None));continue
            a,b=RATES[int(classes[cell])];likelihoods=[]
            for digit in range(9):
                numerator=1
                for query in range(q):numerator*=8*b-8*a if int(values[query,row,cell])==digit else a
                likelihoods.append(numerator)
            board.append((likelihoods,sum(likelihoods)))
        result.append(board)
    return result


def independent_score(values,given,classes):
    posterior=independent_weights(values,given,classes);rows,cells=given.shape;costs=[];joint=[];bounds=[]
    for i in range(cells):
        for j in range(i+1,cells):
            multiplicities={};count=0
            for row in range(rows):
                if not given[row,i] or not given[row,j]:continue
                left,ld=posterior[row][i];right,rd=posterior[row][j]
                dot=sum(a*b for a,b in zip(left,right,strict=True));den=ld*rd
                pair=(den-dot,den);multiplicities[pair]=multiplicities.get(pair,0)+1;count+=1
            terms=[(n,d,c) for (n,d),c in sorted(multiplicities.items())];cost,lo,hi,scale=reference_product_cost(terms)
            costs.append(cost);joint.append(count);bounds.append((lo,hi,scale))
    return np.array(costs,np.int64),np.array(joint,np.int64),bounds
