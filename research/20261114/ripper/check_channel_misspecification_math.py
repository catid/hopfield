"""H1cd synthetic only: channel likelihoods, exhaustive types and interval gates."""
import json,platform,time,statistics
from decimal import Decimal
from fractions import Fraction
import numpy as np
from run import ROOT,HERE,sha,save
from channel_misspecification_math import (CHANNELS,SUPPORTED,weights,independent_weights,type_table,score,independent_score)
from pair_likelihood_math import score as nominal_score,certified_log_terms,reference_product_cost
from check_pair_likelihood_math import inside
from repeat_query_tradeoff_study import make_masks


def main():
    assert platform.node()=='ripper'
    target=HERE/'channel-misspecification-math-check.json';assert not target.exists()
    assert not (HERE/'protocol-channel-misspecification.json').exists()
    assert not (ROOT/'artifacts/research/20261114/ripper/channel_misspecification').exists()
    started=time.perf_counter();parent=json.loads((HERE/'protocol-pair-likelihood.json').read_text())
    for name in ['pair_likelihood_math.py','posterior_query_math.py','check_pair_likelihood_math.py']:
        assert sha(HERE/name)==parent['source_sha256']['research/20261114/ripper/'+name]
    oldgate=json.loads((HERE/'pair-likelihood-math-check.json').read_text())
    assert oldgate['status']=='passed' and oldgate['half_even_boundary_cases']==5 and len(oldgate['forced160_digit_fallbacks'])==2
    assert 'correctly rounded' in Decimal.ln.__doc__ and 'ROUND_HALF_EVEN' in Decimal.ln.__doc__
    rng=np.random.default_rng(992201);masks=make_masks(992202);types=[];fullchecks={};timings={}
    for channel in SUPPORTED:
        for q in [1,3]:
            table=type_table(q,channel)
            assert len(table['keys'])==(2 if q==1 else 21)
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
        inherited_immutable_rounding_and_fallback_gates=True,all_possible_types=types,Q1_affine_identities_all_rates=True,nominal_full_array_parity=True,
        literal_cell_likelihood_parity=True,exact_pair_LR_identity=True,digit_query_permutation_and_missingness=True,
        full81cell_parity=fullchecks,timings=timings,timing_scope='Unrelated full-shape inputs timed once and12x9 toys timed three times; not whole-study speedups.',
        source_sha256={name:sha(HERE/name) for name in ['channel_misspecification_math.py','check_channel_misspecification_math.py','pair_likelihood_math.py','posterior_query_math.py','check_pair_likelihood_math.py','repeat_query_tradeoff_study.py']},wall_seconds=time.perf_counter()-started)
    save(target,result);print(json.dumps(dict(status='passed',types=types,fullchecks=fullchecks,wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
