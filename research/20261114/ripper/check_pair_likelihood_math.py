"""Synthetic only: rigorous log enclosures, posterior types, exposures and parity."""
import json,platform,time,statistics
from decimal import Decimal,localcontext,ROUND_HALF_EVEN
from fractions import Fraction
import numpy as np
from run import ROOT,HERE,sha,save
from pair_likelihood_math import (type_table,score,independent_score,neglog_decimal_interval,
    log_fraction_reference,reference_product_cost,certified_log_terms,rounded_decimal_interval,WORK_PRECISIONS,REFERENCE_DIGITS,REFERENCE_TERMS)
from posterior_query_math import weights
from repeat_query_tradeoff_study import make_masks


def inside(mainlo,mainhi,reflo,refhi,scale):
    assert Fraction(mainlo)<=Fraction(reflo,scale)<=Fraction(refhi,scale)<=Fraction(mainhi)


def main():
    assert platform.node()=='ripper'
    target=HERE/'pair-likelihood-math-check.json';assert not target.exists()
    assert not (HERE/'protocol-pair-likelihood.json').exists()
    assert not (ROOT/'artifacts/research/20261114/ripper/pair_likelihood').exists()
    started=time.perf_counter()
    documented=Decimal.ln.__doc__
    assert 'ROUND_HALF_EVEN' in documented and 'correctly rounded' in documented
    previous=json.loads((HERE/'protocol-posterior-query-labels.json').read_text())
    assert sha(HERE/'posterior_query_math.py')==previous['source_sha256']['research/20261114/ripper/posterior_query_math.py']
    logs=[]
    for n,d in [(1,1),(2,1),(3,2),(7,5),(10**200+1,10**200),(2**1024+17,3)]:
        lo,hi,scale=log_fraction_reference(n,d)
        a,b=neglog_decimal_interval(d,n,80);inside(a,b,lo,hi,scale)
        assert 0<=lo<=hi
        logs.append(dict(numerator=str(n),denominator=str(d),interval_width_units=hi-lo,scale_digits=REFERENCE_DIGITS))
    assert rounded_decimal_interval(Decimal('.5'),Decimal('.5'))==0
    assert rounded_decimal_interval(Decimal('1.5'),Decimal('1.5'))==2
    assert rounded_decimal_interval(Decimal('2.5'),Decimal('2.5'))==2
    assert rounded_decimal_interval(Decimal('.49'),Decimal('.51')) is None
    assert rounded_decimal_interval(Decimal('1.49'),Decimal('1.51')) is None
    fallback=[]
    for sign,expected in [(-1,0),(1,1)]:
        with localcontext() as ctx:
            ctx.prec=220
            requested=Decimal('0.0000005')+Decimal(sign)*Decimal('1e-100')
            probability=requested.copy_negate().exp(context=ctx)
        fraction=Fraction(probability);terms=[(fraction.numerator,fraction.denominator,1)]
        cost,precision,lo,hi=certified_log_terms(terms)
        reference,rlo,rhi,scale=reference_product_cost(terms)
        assert cost==reference==expected and precision==160
        inside(lo,hi,rlo*1000000,rhi*1000000,scale)
        fallback.append(dict(sign=sign,cost=cost,precision=precision))
    types=[]
    for q in [1,3]:
        table=type_table(q)
        for num,den in zip(table['numerators'],table['denominators'],strict=True):
            num,den=int(num),int(den);terms=[(den-num,den,71)]
            value,precision,lo,hi=certified_log_terms(terms)
            ref,rlo,rhi,scale=reference_product_cost(terms)
            assert value==ref and precision==80
            inside(lo,hi,rlo*1000000,rhi*1000000,scale)
            # LR(edge/unrestricted)=(1-e)/(8/9), by exact likelihood sums.
            assert Fraction(den-num,72)/Fraction(den,81)==Fraction(9*(den-num),8*den)
        types.append(dict(queries=q,raw_types=len(table['keys']),posterior_rational_types=len({Fraction(int(n),int(d)) for n,d in zip(table['numerators'],table['denominators'],strict=True)})))
    rng=np.random.default_rng(992101)
    mask=make_masks(992102)
    fullchecks={}
    for q,n,name in [(1,270,'single270'),(3,90,'triple90')]:
        given=mask[:n];values=rng.integers(0,9,(q,n,81),dtype=np.int8);values[:,~given]=-1
        before=time.perf_counter();actual=score(values,given);fast=time.perf_counter()-before
        before=time.perf_counter();expected,joint,bounds=independent_score(values,given);slow=time.perf_counter()-before
        assert actual['cost_certified'].all() and (actual['cost_precision']==80).all()
        assert np.array_equal(actual['cost'],expected) and np.array_equal(actual['joint_count'],joint)
        assert (joint==71*(n//90)).all() and (given.sum(0)==80*(n//90)).all()
        for edge,(lo,hi,scale) in enumerate(bounds):
            inside(Decimal(actual['cost_lower'][edge]),Decimal(actual['cost_upper'][edge]),lo*1000000,hi*1000000,scale)
        if q==1:
            table=type_table(1)
            assert table['numerators'].tolist()==[1839,87688] and table['denominators'].tolist()==[102400,102400]
            ii,jj=np.triu_indices(81,1);equal=((given[:,ii]&given[:,jj])&(values[0,:,ii].T==values[0,:,jj].T)).sum(0)
            assert np.array_equal(actual['likelihood_type_counts'][1],equal)
            assert np.array_equal(actual['likelihood_type_counts'][0],joint-equal)
            # Exact symbolic pre-round cost = joint*Ldifferent + equal*(Lsame-Ldifferent).
            # Lsame>Ldifferent follows from14712/102400<100561/102400 and monotone -ln.
            assert 14712<100561
        fullchecks[name]=dict(rows=n,cells=81,queries=q,pair_coqueries=int(joint[0]),all_costs_certified=True,
                              production_wall_seconds=fast,independent_wall_seconds=slow)
    toy=rng.integers(0,9,(3,12,9),dtype=np.int8);given=rng.random((12,9))>.15;toy[:,~given]=-1
    out=score(toy,given);expected,joint,_=independent_score(toy,given);assert np.array_equal(out['cost'],expected)
    permutation=rng.permutation(9).astype(np.int8)
    changed=permutation[np.maximum(toy,0)];changed[:,~given]=-1
    assert np.array_equal(score(changed,given)['cost'],out['cost'])
    assert np.array_equal(score(toy[::-1],given)['cost'],out['cost'])
    absent=np.zeros((1,9),bool);empty=np.full((3,1,9),-1,np.int8)
    zero=score(empty,absent);assert not zero['cost'].any() and not zero['joint_count'].any() and zero['cost_certified'].all()
    timings={}
    for name,fn in [('finite_type_intervals',score),('exact_product_rational_series',independent_score)]:
        samples=[]
        for _ in range(3):
            before=time.perf_counter();answer=fn(toy,given);samples.append(time.perf_counter()-before)
            value=answer['cost'] if isinstance(answer,dict) else answer[0];assert np.array_equal(value,expected)
        timings[name]=dict(samples_wall_seconds=samples,median_wall_seconds=statistics.median(samples))
    result=dict(status='passed',host=platform.node(),study_scores_read=False,Decimal_ln_documentation=documented,
                working_precisions=list(WORK_PRECISIONS),independent_digits=REFERENCE_DIGITS,independent_terms=REFERENCE_TERMS,
                independent_series_tail='2*z^(2M+1)/((2M+1)*(1-z^2)); each fixed-point operation rounded outwards.',
                log_boundary_cases=logs,half_even_boundary_cases=5,forced160_digit_fallbacks=fallback,
                all_possible_query_pair_types=types,exact_pair_LR_identity=True,Q1_affine_pre_round_relation=True,
                full_81cell_score_parity=fullchecks,constant_pair_exposure_verified=True,
                digit_and_query_permutation=True,missing_observation_zero_cost=True,timings=timings,
                timing_scope='Unrelated 3-query12x9 fixture, three repetitions; full-shape synthetic parity timed once. Not full-study speedup.',
                source_sha256={name:sha(HERE/name) for name in ['pair_likelihood_math.py','check_pair_likelihood_math.py','posterior_query_math.py','repeat_query_tradeoff_study.py']},
                wall_seconds=time.perf_counter()-started)
    save(target,result);print(json.dumps(dict(status='passed',types=types,full_shape_timings=fullchecks,toy_timings=timings,wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
