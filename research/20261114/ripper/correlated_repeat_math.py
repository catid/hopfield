"""H1ch exact posterior for a latent shared-corruption/IID-repeat mixture.

Three query responses only. No realized branch indicators enter this module.
The rho=0 and rho=1 cases exist only for declared synthetic limit gates.
"""
from decimal import Decimal
from functools import lru_cache
from itertools import product
import numpy as np
from posterior_query_math import validate
from pair_likelihood_math import (SCALE,WORK_PRECISIONS,REFERENCE_DIGITS,REFERENCE_TERMS,
    UncertifiedCost,contexts,neglog_decimal_interval,rounded_decimal_interval,reference_product_cost)
PARAMETERS=(3,40,1,2)
SUPPORTED=((3,40,0,1),PARAMETERS,(3,40,1,1))
LIMIT=int(np.iinfo(np.int64).max)


def likelihood_units(parameters):
    assert parameters in SUPPORTED
    a,b,r,s=parameters;assert 0<a<b and 0<=r<=s
    return 8*(b-a),a,8*b,r,s


def joint_weights(counts,first,unanimous,parameters):
    match,wrong,K,r,s=likelihood_units(parameters)
    maximum=r*max(match,wrong)*K*K+(s-r)*max(match,wrong)**3
    assert 9*maximum<LIMIT and 9*maximum*maximum<LIMIT
    lookup=np.array([match**n*wrong**(3-n) for n in range(4)],np.int64)
    iid=lookup[counts]
    single=np.where(first[...,None]==np.arange(9),match,wrong)
    return r*unanimous[...,None]*(K*K)*single+(s-r)*iid


def weights(values,given,parameters):
    validate(values,given);assert values.shape[0]==3
    counts=(values[:,:,:,None]==np.arange(9)[None,None,None,:]).sum(0)
    assert (counts.sum(2)[given]==3).all()
    unanimous=(values==values[0:1]).all(0)
    w=joint_weights(counts,values[0],unanimous,parameters);w[~given]=0;denominator=w.sum(2)
    assert (denominator[given]>0).all(),'impossible_observation_under_supplied_joint_model'
    assert int(denominator.max())**2<LIMIT and not denominator[~given].any()
    return w,denominator


def independent_weights(values,given,parameters):
    validate(values,given);assert values.shape[0]==3 and parameters in SUPPORTED
    a,b,r,s=parameters;rows,cells=given.shape;result=[]
    for row in range(rows):
        board=[]
        for cell in range(cells):
            if not given[row,cell]:board.append((None,None));continue
            observed=[int(values[q,row,cell]) for q in range(3)];likelihoods=[]
            for digit in range(9):
                factors=[8*b-8*a if v==digit else a for v in observed]
                independent=1
                for factor in factors:independent*=factor
                shared=factors[0]*(8*b)**2 if observed[0]==observed[1]==observed[2] else 0
                likelihoods.append(r*shared+(s-r)*independent)
            denominator=sum(likelihoods);assert denominator>0,'impossible_observation_under_supplied_joint_model'
            board.append((likelihoods,denominator))
        result.append(board)
    return result


@lru_cache(maxsize=None)
def type_table(q,channel):
    """Enumerate all possible ordered query patterns; no cohort observations."""
    assert q==3 and channel in SUPPORTED
    all_labels=np.array(list(product(range(9),repeat=q)),np.int8)
    counts=(all_labels[:,:,None]==np.arange(9)[None,None,:]).sum(1)
    unanimous=(all_labels==all_labels[:,0:1]).all(1)
    all_w=joint_weights(counts,all_labels[:,0],unanimous,channel);normalizer=all_w.sum(1)
    possible=normalizer>0;w=all_w[possible];normalizer=normalizer[possible]
    assert int(normalizer.max())**2<LIMIT
    den=np.outer(normalizer,normalizer);dot=w@w.T
    den_values=np.unique(den);stride=int(den.max())+1
    assert len(den_values)*stride+int(dot.max())<np.iinfo(np.int64).max
    keys=np.unique(np.searchsorted(den_values,den)*stride+dot)
    numerators=keys%stride;denominators=den_values[keys//stride]
    assert ((numerators>0)&(numerators<denominators)).all()
    logs={precision:[neglog_decimal_interval(int(d-n),int(d),precision) for n,d in zip(numerators,denominators,strict=True)] for precision in WORK_PRECISIONS}
    return dict(q=q,keys=keys,numerators=numerators,denominators=denominators,den_values=den_values,stride=stride,logs=logs,
                possible_patterns=int(possible.sum()),impossible_patterns=int((~possible).sum()),maximum_denominator_product=int(den.max()),
                maximum_dot=int(dot.max()),packed_key_bound=len(den_values)*stride+int(dot.max()),int64_limit=LIMIT)


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
