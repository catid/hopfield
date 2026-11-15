"""H1ce synthetic only: channel likelihoods, exhaustive types and interval gates."""
import json,platform,time,statistics
from decimal import Decimal
from fractions import Fraction
import numpy as np
from run import ROOT,HERE,sha,save
from estimated_channel_math import (SUPPORTED,weights,independent_weights,type_table,score,independent_score,estimate,independent_estimate,estimate_from_counts,independent_estimate_from_counts,moment)
from pair_likelihood_math import score as nominal_score,certified_log_terms,reference_product_cost
from check_pair_likelihood_math import inside
from repeat_query_tradeoff_study import make_masks


def check_estimator(rng,given):
    assert given.shape==(90,81) and (given.sum(0)==80).all() and (given.sum(1)==72).all()
    assert int(given.sum())==6480
    moment_checks=[]
    for k in range(1,89):
        p=Fraction(k,100);probabilities=[1-p]+[p/8]*8
        disagreement=sum((probabilities[a]*probabilities[b] for a in range(9) for b in range(9) if a!=b),Fraction(0))
        n,d=moment(k,100);assert disagreement==Fraction(n,d)==2*p-Fraction(9,8)*p*p
        moment_checks.append(k)
    boundaries=[]
    for lower in range(1,88):
        n,d=moment(2*lower+1,200)
        for offset,want in [(-1,lower),(0,lower if lower%2==0 else lower+1),(1,lower+1)]:
            a,b=10*n+offset,10*d;out=estimate_from_counts(a,b);ref=independent_estimate_from_counts(a,b)
            assert out==ref and out['fit_allowed'] and out['rate_numerator']==want
            boundaries.append(dict(lower=lower,offset=offset,rounded=want))
    endpoints=[]
    for a,b,allowed,want in [(0,1,True,1),(8,9,True,88),(9,10,False,None),(1,1,False,None),(0,0,False,None)]:
        out=estimate_from_counts(a,b);assert out==independent_estimate_from_counts(a,b)
        assert out['fit_allowed']==allowed and out['rate_numerator']==want;endpoints.append(out)
    for k in [1,88]:
        n,d=moment(k,100);out=estimate_from_counts(n,d)
        assert out==independent_estimate_from_counts(n,d) and out['rate_numerator']==k
        assert out['lower_clipped'] is False and out['upper_clipped'] is False
    values=rng.integers(0,9,(3,90,81),dtype=np.int8);values[:,~given]=-1
    actual=estimate(values,given);assert actual==independent_estimate(values,given) and actual['total_comparisons']==19440
    permutation=rng.permutation(9).astype(np.int8);changed=permutation[np.maximum(values,0)];changed[:,~given]=-1
    assert estimate(changed,given)==actual and estimate(values[::-1],given)==actual
    same=np.zeros((3,90,81),np.int8);same[:,~given]=-1
    allsame=estimate(same,given);assert allsame['disagreements']==0 and allsame['rate_numerator']==1 and allsame['lower_clipped']
    distinct=np.stack([np.full((90,81),k,np.int8) for k in range(3)]);distinct[:,~given]=-1
    alldifferent=estimate(distinct,given);assert alldifferent['disagreements']==19440 and not alldifferent['fit_allowed']
    return dict(exact_moment_rates=moment_checks,midpoint_and_neighbor_cases=boundaries,invalid_and_clipping_cases=endpoints,
                exact_comparison_count=19440,observed_positions=6480,vector_literal_count_parity=True,digit_query_invariance=True)


def main():
    assert platform.node()=='ripper'
    target=HERE/'estimated-channel-math-check.json';assert not target.exists()
    assert not (HERE/'protocol-estimated-channel.json').exists()
    assert not (ROOT/'artifacts/research/20261114/ripper/estimated_channel').exists()
    started=time.perf_counter();parent=json.loads((HERE/'protocol-pair-likelihood.json').read_text())
    for name in ['pair_likelihood_math.py','posterior_query_math.py','check_pair_likelihood_math.py']:
        assert sha(HERE/name)==parent['source_sha256']['research/20261114/ripper/'+name]
    oldgate=json.loads((HERE/'pair-likelihood-math-check.json').read_text())
    assert oldgate['status']=='passed' and oldgate['half_even_boundary_cases']==5 and len(oldgate['forced160_digit_fallbacks'])==2
    assert 'correctly rounded' in Decimal.ln.__doc__ and 'ROUND_HALF_EVEN' in Decimal.ln.__doc__
    rng=np.random.default_rng(992301);masks=make_masks(992302);types=[];fullchecks={};timings={}
    estimator_gate=check_estimator(rng,masks[:90])
    for channel in SUPPORTED:
        for q in [1,3]:
            table=type_table(q,channel)
            assert 1<=len(table['keys'])<=(2 if q==1 else 21)
            for num,den in zip(table['numerators'],table['denominators'],strict=True):
                num,den=int(num),int(den);terms=[(den-num,den,71)]
                value,precision,lo,hi=certified_log_terms(terms);ref,rlo,rhi,scale=reference_product_cost(terms)
                assert value==ref and precision==80;inside(lo,hi,rlo*1000000,rhi*1000000,scale)
                assert Fraction(den-num,72)/Fraction(den,81)==Fraction(9*(den-num),8*den)
            if q==1:
                a,b=channel;match=8*(b-a);wrong=a;den=(match+8*wrong)**2
                assert table['numerators'].tolist()==[2*match*wrong+7*wrong**2,match**2+8*wrong**2]
                assert table['denominators'].tolist()==[den,den]
                assert table['numerators'][1]>table['numerators'][0]
            types.append(dict(channel=list(channel),queries=q,types=len(table['keys']),maximum_type_denominator=int(table['denominators'].max())))
        if channel not in [(1,100),(7,100),(8,100),(20,100),(88,100),(3,40)]:continue
        for q,n,name in [(1,270,'single270'),(3,90,'triple90')]:
            given=masks[:n];values=rng.integers(0,9,(q,n,81),dtype=np.int8);values[:,~given]=-1
            before=time.perf_counter();actual=score(values,given,channel);fast=time.perf_counter()-before
            before=time.perf_counter();expected,joint,bounds=independent_score(values,given,channel);slow=time.perf_counter()-before
            assert actual['cost_certified'].all() and (actual['cost_precision']==80).all()
            assert np.array_equal(actual['cost'],expected) and np.array_equal(actual['joint_count'],joint)
            assert (joint==71*(n//90)).all() and (given.sum(0)==80*(n//90)).all()
            for edge,(lo,hi,scale) in enumerate(bounds):
                inside(Decimal(actual['cost_lower'][edge]),Decimal(actual['cost_upper'][edge]),lo*1000000,hi*1000000,scale)
            if channel==(3,40):
                old=nominal_score(values,given)
                assert set(old)==set(actual) and all(np.array_equal(actual[k],v) for k,v in old.items())
            if q==1:
                ii,jj=np.triu_indices(81,1);equal=((given[:,ii]&given[:,jj])&(values[0,:,ii].T==values[0,:,jj].T)).sum(0)
                assert np.array_equal(actual['likelihood_type_counts'][1],equal)
                assert np.array_equal(actual['likelihood_type_counts'][0],joint-equal)
            fullchecks[str(channel)+'-'+name]=dict(rows=n,cells=81,queries=q,pair_coqueries=int(joint[0]),production_wall_seconds=fast,independent_wall_seconds=slow)
        toy=rng.integers(0,9,(3,12,9),dtype=np.int8);given=rng.random((12,9))>.15;toy[:,~given]=-1
        prod,den=weights(toy,given,channel);literal=independent_weights(toy,given,channel)
        for row,cell in zip(*np.where(given)):
            assert prod[row,cell].tolist()==literal[row][cell][0] and den[row,cell]==literal[row][cell][1]
        out=score(toy,given,channel);expected,joint,_=independent_score(toy,given,channel);assert np.array_equal(out['cost'],expected)
        permutation=rng.permutation(9).astype(np.int8);changed=permutation[np.maximum(toy,0)];changed[:,~given]=-1
        assert np.array_equal(score(changed,given,channel)['cost'],out['cost'])
        assert np.array_equal(score(toy[::-1],given,channel)['cost'],out['cost'])
        empty=np.full((3,1,9),-1,np.int8);zero=score(empty,np.zeros((1,9),bool),channel)
        assert not zero['cost'].any() and not zero['joint_count'].any() and zero['cost_certified'].all()
        for label,fn in [('finite_type',score),('independent_product',independent_score)]:
            samples=[]
            for _ in range(3):
                before=time.perf_counter();answer=fn(toy,given,channel);samples.append(time.perf_counter()-before)
                val=answer['cost'] if isinstance(answer,dict) else answer[0];assert np.array_equal(val,expected)
            timings[str(channel)+'-'+label]=dict(samples_wall_seconds=samples,median_wall_seconds=statistics.median(samples))
    result=dict(status='passed',host=platform.node(),study_scores_read=False,parent_numerical_gate_sha256=sha(HERE/'pair-likelihood-math-check.json'),
        inherited_immutable_rounding_and_fallback_gates=True,all_possible_types=types,estimator_gate=estimator_gate,Q1_affine_identities_all_rates=True,nominal_full_array_parity=True,
        literal_cell_likelihood_parity=True,exact_pair_LR_identity=True,digit_query_permutation_and_missingness=True,
        full81cell_parity=fullchecks,timings=timings,timing_scope='Unrelated full-shape inputs timed once and12x9 toys timed three times; not whole-study speedups.',
        source_sha256={name:sha(HERE/name) for name in ['estimated_channel_math.py','check_estimated_channel_math.py','pair_likelihood_math.py','posterior_query_math.py','check_pair_likelihood_math.py','repeat_query_tradeoff_study.py']},wall_seconds=time.perf_counter()-started)
    save(target,result);print(json.dumps(dict(status='passed',types=types,fullchecks=fullchecks,wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
