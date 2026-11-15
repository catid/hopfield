"""H1ch exact latent-mixture probabilities and directed-log synthetic gates."""
from fractions import Fraction
from itertools import product
import math,time
import numpy as np
import correlated_repeat_math as M
from channel_misspecification_math import score as iid_score
from heterogeneous_channel_study import ROOT,HERE,sha,save,make_masks


def rejected(fn,*args):
    try:fn(*args)
    except AssertionError:return
    raise AssertionError('Impossible synthetic observation accepted')


def main():
    target=HERE/'correlated-repeat-math-check.json';assert not target.exists() and not (HERE/'protocol-correlated-repeat.json').exists();started=time.perf_counter()
    labels=np.array(list(product(range(9),repeat=3)),np.int8);tables=[];truth_checks=0;pair_checks=0;certified=0
    for parameters in M.SUPPORTED:
        a,b,r,s=parameters;K=8*b;exact=[];possible=[]
        for pattern in labels:
            likelihoods=[]
            for digit in range(9):
                probabilities=[Fraction(b-a,b) if int(y)==digit else Fraction(a,8*b) for y in pattern]
                shared=Fraction(r,s)*probabilities[0] if int(pattern[0])==int(pattern[1])==int(pattern[2]) else Fraction(0)
                independent=Fraction(s-r,s)*math.prod(probabilities);probability=shared+independent
                scaled=probability*s*K**3;assert scaled.denominator==1;likelihoods.append(scaled.numerator);truth_checks+=1
            denominator=sum(likelihoods);possible.append(denominator>0);exact.append(likelihoods)
        exact=np.array(exact,np.int64);possible=np.array(possible,bool);kept=labels[possible]
        values=kept.T[:, :,None];given=np.ones((len(kept),1),bool);actual,den=M.weights(values,given,parameters)
        assert np.array_equal(actual[:,0],exact[possible]) and np.array_equal(den[:,0],exact[possible].sum(1))
        reference=M.independent_weights(values,given,parameters)
        assert all(reference[i][0]==(list(map(int,exact[possible][i])),int(exact[possible][i].sum())) for i in range(len(kept)))
        table=M.type_table(3,parameters);w=exact[possible];den=np.outer(w.sum(1),w.sum(1));dot=w@w.T
        key=np.searchsorted(table['den_values'],den)*table['stride']+dot;indices=np.searchsorted(table['keys'],key)
        assert np.array_equal(table['keys'][indices],key) and np.array_equal(table['numerators'][indices],dot) and np.array_equal(table['denominators'][indices],den)
        assert set(map(int,np.unique(indices)))==set(range(len(table['keys'])));pair_checks+=dot.size
        for num,den,interval in zip(table['numerators'],table['denominators'],table['logs'][80],strict=True):
            num,den=int(num),int(den);cost,lo,hi,scale=M.reference_product_cost([(den-num,den,1)]);lower,upper=interval
            assert Fraction(lower)<=Fraction(lo,scale)<=Fraction(hi,scale)<=Fraction(upper)
            down,up,_=M.contexts(80);assert M.rounded_decimal_interval(down.multiply(lower,M.Decimal(M.SCALE)),up.multiply(upper,M.Decimal(M.SCALE)))==cost;certified+=1
        tables.append(dict(parameters=list(parameters),possible_patterns=int(possible.sum()),impossible_patterns=int((~possible).sum()),unique_types=len(table['keys']),
            maximum_denominator_product=table['maximum_denominator_product'],maximum_dot=table['maximum_dot'],packed_key_bound=table['packed_key_bound'],int64_limit=table['int64_limit']))
        if r==s:
            assert possible.sum()==9 and all(np.array_equal(pattern,np.repeat(pattern[0],3)) for pattern in kept)
            impossible=np.array([0,0,1],np.int8).reshape(3,1,1);mask=np.ones((1,1),bool)
            rejected(M.weights,impossible,mask,parameters);rejected(M.independent_weights,impossible,mask,parameters);rejected(M.score,impossible,mask,parameters);rejected(M.independent_score,impossible,mask,parameters)
    rng=np.random.default_rng(994901);mask=make_masks(994902);values=rng.integers(0,9,(3,90,81),dtype=np.int8);values[:,~mask]=-1
    before=time.perf_counter();actual=M.score(values,mask,M.PARAMETERS);producer_seconds=time.perf_counter()-before
    before=time.perf_counter();expected,joint,bounds=M.independent_score(values,mask,M.PARAMETERS);reference_seconds=time.perf_counter()-before
    assert actual['cost_certified'].all() and np.array_equal(actual['cost'],expected) and np.array_equal(actual['joint_count'],joint) and (joint==71).all()
    for edge,(lo,hi,scale) in enumerate(bounds):assert Fraction(M.Decimal(actual['cost_lower'][edge]))<=Fraction(lo*M.SCALE,scale)<=Fraction(hi*M.SCALE,scale)<=Fraction(M.Decimal(actual['cost_upper'][edge]))
    zero=M.score(values,mask,(3,40,0,1));iid=iid_score(values,mask,(3,40));assert np.array_equal(zero['cost'],iid['cost']) and zero['cost_certified'].all()
    unanimous=np.repeat(values[:1],3,axis=0);one=M.score(unanimous,mask,(3,40,1,1));single=iid_score(unanimous[:1],mask,(3,40));assert np.array_equal(one['cost'],single['cost']) and one['cost_certified'].all()
    permutation=np.array([4,0,8,1,7,2,6,3,5],np.int8);mapped=values.copy();mapped[:,mask]=permutation[values[:,mask]];assert np.array_equal(M.score(mapped,mask,M.PARAMETERS)['cost'],actual['cost'])
    order=np.arange(81)[::-1];swapped=M.score(values[:,:,order],mask[:,order],M.PARAMETERS);ii,jj=np.triu_indices(81,1);matrix=np.zeros((81,81),np.int64);matrix[ii,jj]=matrix[jj,ii]=actual['cost'];assert np.array_equal(swapped['cost'],matrix[np.ix_(order,order)][ii,jj])
    old=M.rounded_decimal_interval;calls=[]
    def fallback(lo,hi):calls.append(1);return None if len(calls)%2 else old(lo,hi)
    M.rounded_decimal_interval=fallback
    try:
        result=M.score(values[:,:2,:4],mask[:2,:4],M.PARAMETERS);assert result['cost_certified'].all() and (result['cost_precision']==160).all() and len(calls)==12
        M.rounded_decimal_interval=lambda lo,hi:None
        result=M.score(values[:,:2,:4],mask[:2,:4],M.PARAMETERS);assert not result['cost_certified'].any() and not result['cost'].any() and (result['cost_precision']==160).all()
    finally:M.rounded_decimal_interval=old
    inherited=__import__('json').loads((HERE/'channel-misspecification-math-check.json').read_text());assert inherited['status']=='passed'
    for name,h in inherited['source_sha256'].items():assert sha(HERE/name)==h
    sources=dict(inherited['source_sha256']);sources.update({name:sha(HERE/name) for name in ['correlated_repeat_math.py','preflight_correlated_repeat_math.py']})
    record=dict(status='passed',study_scores_read=False,proposal_sha256=sha(HERE/'correlated-repeat-proposal.json'),exact_fraction_mixture_truth_checks=truth_checks,ordered_pattern_pair_checks=pair_checks,
        every_unique_table_value_independently_certified=certified,tables=tables,fullshape_mixture_reference_parity=True,rho0_immutable_IID_parity=True,rho1_single_response_parity=True,
        rho1_impossible_pattern_rejections=4,digit_and_cell_permutation_invariance=True,producer160_fallback_exercised=True,uncertified_placeholder_mode_exercised=True,
        broader_query_count_support=False,inherited_rounding_gate_sha256=sha(HERE/'channel-misspecification-math-check.json'),fullshape_production_seconds=producer_seconds,
        fullshape_independent_seconds=reference_seconds,source_sha256=sources,wall_seconds=time.perf_counter()-started)
    save(target,record);print(__import__('json').dumps({k:v for k,v in record.items() if k not in ['tables','source_sha256']}))


if __name__=='__main__':main()
