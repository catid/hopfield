"""H1cg data-independent mixed-channel rational/numerical certification gate."""
from pathlib import Path
from fractions import Fraction
from itertools import product
import json,time
import numpy as np
import cell_channel_math as M
from channel_misspecification_math import score as homogeneous_score
from heterogeneous_channel_study import ROOT,HERE,sha,save,make_masks


def rejected(fn,*args):
    try:fn(*args)
    except AssertionError:return
    raise AssertionError('Invalid synthetic channels accepted')


def main():
    target=HERE/'cell-channel-math-check.json';assert not target.exists() and not (HERE/'protocol-cell-channel.json').exists();started=time.perf_counter()
    tables=[];certified=0;patternchecks=0;pairchecks=0
    for q in [1,3]:
        labels=list(product(range(9),repeat=q));independent=[]
        for rate in M.RATES:
            a,b=rate;expected=np.array([[__import__('math').prod(8*b-8*a if x==digit else a for x in pattern) for digit in range(9)] for pattern in labels],np.int64)
            weights,normalizers=M.pattern_weights(q,rate);assert np.array_equal(weights,expected) and np.array_equal(normalizers,expected.sum(1));independent.append(expected);patternchecks+=len(labels)
        table=M.type_table(q);seen=set()
        for left,right in product(range(3),repeat=2):
            lw,rw=independent[left],independent[right];den=np.outer(lw.sum(1),rw.sum(1));dot=lw@rw.T
            # Independently formed exact pair integers cover every ordered query-pattern pair.
            key=np.searchsorted(table['den_values'],den)*table['stride']+dot;indices=np.searchsorted(table['keys'],key)
            assert np.array_equal(table['keys'][indices],key) and np.array_equal(table['numerators'][indices],dot) and np.array_equal(table['denominators'][indices],den)
            assert np.array_equal(dot,(rw@lw.T).T);seen.update(map(int,np.unique(indices)));pairchecks+=dot.size
        assert seen==set(range(len(table['keys'])))
        for num,den,lohi in zip(table['numerators'],table['denominators'],table['logs'][80],strict=True):
            num,den=int(num),int(den);cost,lo,hi,scale=M.reference_product_cost([(den-num,den,1)])
            a,b=lohi;assert Fraction(a)<=Fraction(lo,scale)<=Fraction(hi,scale)<=Fraction(b)
            down,up,_=M.contexts(80);assert M.rounded_decimal_interval(down.multiply(a,M.Decimal(M.SCALE)),up.multiply(b,M.Decimal(M.SCALE)))==cost;certified+=1
        tables.append(dict(q=q,patterns=len(labels),ordered_channel_pairs=9,unique_unreduced_types=len(table['keys']),overflow_bounds=table['overflow_bounds'],global_packed_key_bound=table['global_packed_key_bound'],int64_limit=table['int64_limit']))
    rng=np.random.default_rng(992901);mask=make_masks(992902);classes=np.repeat(np.arange(3,dtype=np.int8),27);classes=classes[rng.permutation(81)]
    values=rng.integers(0,9,(3,90,81),dtype=np.int8);values[:,~mask]=-1;before=time.perf_counter();mixed=M.score(values,mask,classes);production_seconds=time.perf_counter()-before
    before=time.perf_counter();expected,joint,bounds=M.independent_score(values,mask,classes);reference_seconds=time.perf_counter()-before
    assert mixed['cost_certified'].all() and np.array_equal(mixed['cost'],expected) and np.array_equal(mixed['joint_count'],joint) and (joint==71).all()
    for i,(lo,hi,scale) in enumerate(bounds):assert Fraction(M.Decimal(mixed['cost_lower'][i]))<=Fraction(lo*M.SCALE,scale)<=Fraction(hi*M.SCALE,scale)<=Fraction(M.Decimal(mixed['cost_upper'][i]))
    for ci,rate in enumerate(M.RATES):
        actual=M.score(values,mask,np.full(81,ci,np.int8));expected=homogeneous_score(values,mask,rate)
        assert np.array_equal(actual['cost'],expected['cost']) and np.array_equal(actual['joint_count'],expected['joint_count']) and actual['cost_certified'].all()
    permutation=np.array([4,0,8,1,7,2,6,3,5],np.int8);mapped=values.copy();mapped[:,mask]=permutation[values[:,mask]]
    assert np.array_equal(M.score(mapped,mask,classes)['cost'],mixed['cost'])
    order=np.arange(81)[::-1];swapped=M.score(values[:,:,order],mask[:,order],classes[order]);ii,jj=np.triu_indices(81,1);matrix=np.zeros((81,81),np.int64);matrix[ii,jj]=matrix[jj,ii]=mixed['cost'];assert np.array_equal(swapped['cost'],matrix[np.ix_(order,order)][ii,jj])
    for bad in [classes.astype(np.int64),classes[:-1],np.full(81,-1,np.int8),np.full(81,3,np.int8)]:rejected(M.score,values,mask,bad)
    old=M.rounded_decimal_interval;calls=[]
    def fallback(lo,hi):calls.append(1);return None if len(calls)%2 else old(lo,hi)
    M.rounded_decimal_interval=fallback
    try:
        result=M.score(values[:,:2,:4],mask[:2,:4],classes[:4]);assert result['cost_certified'].all() and (result['cost_precision']==160).all() and len(calls)==12
        M.rounded_decimal_interval=lambda lo,hi:None
        result=M.score(values[:,:2,:4],mask[:2,:4],classes[:4]);assert not result['cost_certified'].any() and not result['cost'].any() and (result['cost_precision']==160).all()
    finally:M.rounded_decimal_interval=old
    inherited=json.loads((HERE/'channel-misspecification-math-check.json').read_text());assert inherited['status']=='passed'
    for name,h in inherited['source_sha256'].items():assert sha(HERE/name)==h
    sources=dict(inherited['source_sha256']);sources.update({name:sha(HERE/name) for name in ['cell_channel_math.py','preflight_cell_channel_math.py']})
    record=dict(status='passed',study_scores_read=False,proposal_sha256=sha(HERE/'cell-channel-proposal.json'),pattern_weight_checks=patternchecks,ordered_pattern_pair_checks=pairchecks,
        every_unique_table_value_independently_certified=certified,tables=tables,fullshape_mixed_reference_parity=True,all3_homogeneous_controls_match=True,digit_permutation_and_cell_swap_invariance=True,
        invalid_channel_maps_rejected=4,producer160_fallback_exercised=True,uncertified_placeholder_mode_exercised=True,inherited_rounding_gate_sha256=sha(HERE/'channel-misspecification-math-check.json'),
        fullshape_production_seconds=production_seconds,fullshape_independent_seconds=reference_seconds,source_sha256=sources,wall_seconds=time.perf_counter()-started)
    save(target,record);print(json.dumps({k:v for k,v in record.items() if k not in ['tables','source_sha256']}))


if __name__=='__main__':main()
