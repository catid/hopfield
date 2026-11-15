"""Clean-label-free, outward-rounded Clopper-Pearson noise calibration."""
from decimal import Decimal,localcontext
from fractions import Fraction
from functools import lru_cache
import math

ALPHA=Fraction(1,200)
GRID=1000000
MAX_ETA=Fraction(8,9)


def disagreement_probability(eta):
    return 2*eta-Fraction(9,8)*eta*eta


def log_binomial_cdf(k,n,p):
    if k>=n:return 0.
    if p<=0:return 0.
    if p>=1:return -math.inf
    log_mass=math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+k*math.log(p)+(n-k)*math.log1p(-p)
    total=term=1.
    for j in range(k,0,-1):
        term*=j/(n-j+1)*(1-p)/p
        total+=term
    return log_mass+math.log(total)


@lru_cache(maxsize=4096)
def exact_cdf_comparison(k,n,probability,alpha=ALPHA):
    """Compare the exact rational binomial lower tail with alpha."""
    if k>=n:return 1
    a,b=probability.numerator,probability.denominator;c=b-a
    if a==0:return 1
    if c==0:return -1
    term=pow(c,n);total=term
    for j in range(k):
        numerator=term*(n-j)*a;denominator=(j+1)*c
        term,remainder=divmod(numerator,denominator)
        assert remainder==0
        total+=term
    left=alpha.denominator*total;right=alpha.numerator*pow(b,n)
    return (left>right)-(left<right)


@lru_cache(maxsize=4096)
def decimal_cdf(k,n,probability):
    with localcontext() as context:
        context.prec=80;p=Decimal(probability.numerator)/probability.denominator
        if p==0:return str(Decimal(1))
        if p==1:return str(Decimal(int(k>=n)))
        term=(1-p)**n;total=term
        for j in range(k):
            term=term*(n-j)/(j+1)*p/(1-p);total+=term
        return str(total)


@lru_cache(maxsize=1024)
def fit_count(k,n=10000,grid=GRID):
    assert isinstance(k,int) and isinstance(n,int) and 0<=k<=n and n>0 and grid>0
    if k==n:raw_d=1.
    else:
        lo,hi=k/n,1.
        for _ in range(100):
            mid=(lo+hi)/2
            if mid==lo or mid==hi:break
            if log_binomial_cdf(k,n,mid)>math.log(float(ALPHA)):lo=mid
            else:hi=mid
        raw_d=hi
    fallback=raw_d>=8/9
    raw_eta=None if fallback else 8*raw_d/(8+math.sqrt(64-72*raw_d))
    tick=None if fallback else max(0,math.ceil(raw_eta*grid))
    if tick is not None and Fraction(tick,grid)>=MAX_ETA:fallback=True
    endpoint=None;previous=None
    if not fallback:
        while exact_cdf_comparison(k,n,disagreement_probability(Fraction(tick,grid)))>0:
            tick+=1
            if Fraction(tick,grid)>=MAX_ETA:fallback=True;break
        if not fallback:
            while tick>0 and exact_cdf_comparison(k,n,disagreement_probability(Fraction(tick-1,grid)))<=0:tick-=1
            eta=Fraction(tick,grid);d=disagreement_probability(eta)
            assert exact_cdf_comparison(k,n,d)<=0
            assert tick==0 or exact_cdf_comparison(k,n,disagreement_probability(Fraction(tick-1,grid)))>0
            endpoint=decimal_cdf(k,n,d)
            previous=decimal_cdf(k,n,disagreement_probability(Fraction(tick-1,grid))) if tick else None
    if fallback:eta=MAX_ETA
    return {'disagreements':k,'pairs':n,'alpha_numerator':1,'alpha_denominator':200,
        'cp_disagreement_upper_float':raw_d,'inverted_upper_float':raw_eta,'grid':grid,
        'eta_upper_numerator':eta.numerator,'eta_upper_denominator':eta.denominator,'eta_upper':float(eta),
        'no_pruning_fallback':fallback,'endpoint_exact_integer_verified':not fallback,
        'previous_grid_exactly_rejected':not fallback,'endpoint_cdf_decimal80':endpoint,
        'previous_grid_cdf_decimal80':previous,
        'fallback_reason':'Outward-rounded or inverted upper bound reaches eta=8/9.' if fallback else None}


def fit_disagreements(indicators):
    values=list(indicators)
    assert len(values)==10000 and all(v in (False,True,0,1) for v in values)
    return fit_count(sum(int(v) for v in values),10000)
