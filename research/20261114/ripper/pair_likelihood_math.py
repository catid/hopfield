"""Certified integer composite-likelihood costs, with exact independent log bounds.

Decimal.ln is documented by Python as correctly rounded in ROUND_HALF_EVEN.
We enclose each call by a full last-place unit, and use directed division,
products and sums. The independent path uses integer fixed-point bounds on
2*atanh(z), z=(x-1)/(x+1), with an explicit geometric remainder.
"""
from decimal import Decimal,Context,ROUND_FLOOR,ROUND_CEILING,ROUND_HALF_EVEN
from fractions import Fraction
from functools import lru_cache
from itertools import product
import numpy as np
from posterior_query_math import weights,independent_weights
SCALE=1000000
WORK_PRECISIONS=(80,160)
REFERENCE_DIGITS=200
REFERENCE_TERMS=240


class UncertifiedCost(Exception):
    pass


def contexts(precision):
    assert type(precision) is int and precision>=20
    return (Context(prec=precision,rounding=ROUND_FLOOR),Context(prec=precision,rounding=ROUND_CEILING),
            Context(prec=precision,rounding=ROUND_HALF_EVEN))


def last_unit(value,precision):
    return Decimal((0,(1,),value.adjusted()-precision+1))


def neglog_decimal_interval(numerator,denominator,precision):
    """Bound -ln(numerator/denominator), 0<numerator<=denominator, by Decimal."""
    assert type(numerator) is type(denominator) is int and 0<numerator<=denominator
    if numerator==denominator:return Decimal(0),Decimal(0)
    down,up,nearest=contexts(precision)
    xlo=down.divide(Decimal(numerator),Decimal(denominator));xhi=up.divide(Decimal(numerator),Decimal(denominator))
    assert 0<xlo<=xhi<=1
    llo=xlo.ln(context=nearest);lhi=xhi.ln(context=nearest)
    # Monotonic ln plus correctly rounded endpoint values; one ulp safely
    # exceeds the half-ulp rounding error, including binade/decade boundaries.
    lower=down.subtract(llo,last_unit(llo,precision));upper=up.add(lhi,last_unit(lhi,precision))
    return upper.copy_negate(),lower.copy_negate()


def rounded_decimal_interval(lower,upper):
    assert lower<=upper
    a=int(lower.to_integral_value(rounding=ROUND_HALF_EVEN));b=int(upper.to_integral_value(rounding=ROUND_HALF_EVEN))
    return a if a==b else None


def rounded_log_terms(terms,precision):
    """terms=(positive numerator,denominator,multiplicity), for -sum count*ln(n/d)."""
    down,up,_=contexts(precision);lower=upper=Decimal(0)
    for numerator,denominator,count in terms:
        assert type(count) is int and count>=0
        lo,hi=neglog_decimal_interval(numerator,denominator,precision)
        lower=down.add(lower,down.multiply(Decimal(count),lo))
        upper=up.add(upper,up.multiply(Decimal(count),hi))
    lower=down.multiply(lower,Decimal(SCALE));upper=up.multiply(upper,Decimal(SCALE))
    return rounded_decimal_interval(lower,upper),lower,upper


def certified_log_terms(terms):
    for precision in WORK_PRECISIONS:
        value,lower,upper=rounded_log_terms(terms,precision)
        if value is not None:return value,precision,lower,upper
    return None,WORK_PRECISIONS[-1],lower,upper


def ceil_div(a,b):
    assert a>=0 and b>0
    return (a+b-1)//b


def atanh_log_bounds(numerator,denominator,digits=REFERENCE_DIGITS,terms=REFERENCE_TERMS):
    """Exact integer bounds /10**digits for ln(n/d), when 1<=n/d<=2.

    Each fixed-point operation rounds outwards. After M terms, the positive
    remainder in atanh(z) is <=z**(2M+1)/((2M+1)*(1-z*z)).
    """
    assert type(numerator) is type(denominator) is int and 0<denominator<=numerator<=2*denominator
    assert type(digits) is int and digits>=20 and type(terms) is int and terms>0
    scale=10**digits
    if numerator==denominator:return 0,0,scale
    a,b=numerator-denominator,numerator+denominator
    zlo=a*scale//b;zhi=ceil_div(a*scale,b)
    squarelo=zlo*zlo//scale;squarehi=ceil_div(zhi*zhi,scale)
    assert 0<=zlo<=zhi<scale and 0<=squarelo<=squarehi<scale
    powerlo,powerhi=zlo,zhi;lower=upper=0
    for index in range(terms):
        odd=2*index+1
        lower+=powerlo//odd;upper+=ceil_div(powerhi,odd)
        powerlo=powerlo*squarelo//scale;powerhi=ceil_div(powerhi*squarehi,scale)
    tail=ceil_div(powerhi*scale,(2*terms+1)*(scale-squarehi))
    return 2*lower,2*(upper+tail),scale


@lru_cache(maxsize=None)
def ln2_reference(digits=REFERENCE_DIGITS,terms=REFERENCE_TERMS):
    return atanh_log_bounds(2,1,digits,terms)


def log_fraction_reference(numerator,denominator,digits=REFERENCE_DIGITS,terms=REFERENCE_TERMS):
    """Bound ln(N/D)>=0 by powers of2 and an independently bounded atanh series."""
    assert type(numerator) is type(denominator) is int and numerator>=denominator>0
    k=numerator.bit_length()-denominator.bit_length()
    if numerator<(denominator<<k):k-=1
    assert k>=0 and denominator<<k<=numerator<denominator<<(k+1)
    lo,hi,scale=atanh_log_bounds(numerator,denominator<<k,digits,terms)
    twolo,twohi,two_scale=ln2_reference(digits,terms);assert scale==two_scale
    return lo+k*twolo,hi+k*twohi,scale


def reference_product_cost(terms):
    numerator=denominator=1
    for n,d,count in terms:
        assert 0<n<=d and type(count) is int and count>=0
        numerator*=pow(d,count);denominator*=pow(n,count)
    lower,upper,scale=log_fraction_reference(numerator,denominator)
    a=round(Fraction(lower*SCALE,scale));b=round(Fraction(upper*SCALE,scale))
    if a!=b:raise UncertifiedCost('Fixed200digit240term exact-product interval did not certify integer cost')
    return a,lower,upper,scale


@lru_cache(maxsize=None)
def type_table(q):
    """Enumerate all possible ordered query patterns; no cohort observations."""
    assert q in [1,3]
    labels=np.array(list(product(range(9),repeat=q)),np.int8)
    counts=(labels[:,:,None]==np.arange(9)[None,None,:]).sum(1)
    lookup=np.array([296**n*3**(q-n) for n in range(q+1)],np.int64)
    w=lookup[counts];normalizer=w.sum(1);den=np.outer(normalizer,normalizer);dot=w@w.T
    den_values=np.unique(den);stride=int(den.max())+1
    assert len(den_values)*stride+int(dot.max())<np.iinfo(np.int64).max
    keys=np.unique(np.searchsorted(den_values,den)*stride+dot)
    numerators=keys%stride;denominators=den_values[keys//stride]
    assert ((numerators>0)&(numerators<denominators)).all()
    logs={precision:[neglog_decimal_interval(int(d-n),int(d),precision) for n,d in zip(numerators,denominators,strict=True)] for precision in WORK_PRECISIONS}
    return dict(q=q,keys=keys,numerators=numerators,denominators=denominators,den_values=den_values,stride=stride,logs=logs)


def type_counts(values,given):
    table=type_table(values.shape[0]);w,denominator=weights(values,given)
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


def score(values,given):
    table=type_table(values.shape[0]);counts,joint=type_counts(values,given)
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


def independent_score(values,given):
    """Literal cell likelihoods; exact per-pair likelihood product; fixed rational log bounds."""
    posterior=independent_weights(values,given);rows,cells=given.shape
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
