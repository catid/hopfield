"""H1bs synthetic null replay, exact ratios, joint endpoint and label gates."""
import json,platform,itertools
from collections import Counter
import numpy as np
from marginal_shuffle_study import ROOT,HERE,ARMS,sha,save,observed_histograms,shuffle_observed,observe,concentration,strict_greater
from audit_marginal_shuffle import reference_histograms,reference_shuffle,reference_observe,reference_concentration,reference_compare,literal_account,label_gate,aggregate
from balanced_mask_math import make_balanced_masks
from audit_balanced_mask_math import reference_balanced,verify_incidence


def main():
    assert platform.node()=='ripper';path=HERE/'marginal-shuffle-preflight.json';assert not path.exists();incidence=verify_incidence()
    rng=np.random.default_rng(981501);base=rng.integers(0,9,(180,81),dtype=np.int8);uniform=np.zeros((180,81),bool)
    for row in range(180):uniform[row,rng.permutation(81)[:72]]=True
    masks=dict(uniform=uniform,balanced=make_balanced_masks(981502));assert np.array_equal(masks['balanced'],reference_balanced(981502));expected={};labels={}
    for ai,arm in enumerate(ARMS):
        given=masks[arm];original=np.where(given,base,-1).astype(np.int8);hist=observed_histograms(given,original);assert np.array_equal(hist,reference_histograms(given,original))
        labels[arm]=shuffle_observed(given,hist,981503+ai);assert np.array_equal(labels[arm],reference_shuffle(given,hist,981503+ai)) and np.array_equal(hist,reference_histograms(given,labels[arm])) and (labels[arm][~given]==-1).all()
        assert not np.array_equal(labels[arm],shuffle_observed(given,hist,981505+ai))
        reordered=original.copy()
        for cell in range(81):
            rows=np.flatnonzero(given[:,cell]);reordered[rows,cell]=original[rows[::-1],cell]
        assert np.array_equal(observed_histograms(given,reordered),hist) and np.array_equal(shuffle_observed(given,observed_histograms(given,reordered),981503+ai),labels[arm])
        expected[arm]=observe(given,labels[arm]);other=reference_observe(given,labels[arm])
        for key,value in other.items():assert np.array_equal(value,expected[arm][key])
        assert expected[arm]['joint_count'].sum()==460080
    assert (expected['balanced']['joint_count']==142).all();shared=masks['uniform']&masks['balanced'];assert not np.array_equal(labels['uniform'][shared],labels['balanced'][shared])
    q=literal_account(masks);assert q==dict(uniform_queries=12960,balanced_queries=12960,shared_synthetic_queries=0,total_shuffled_assignments=25920,reused_real_shared_queries=int(shared.sum()),reused_real_union_queries=int((masks['uniform']|masks['balanced']).sum()))
    gates=label_gate(masks,labels,expected)
    permutation_cases=[]
    for multiset in [[0,0,1],[0,1,2,2]]:
        outputs=Counter(tuple(multiset[i] for i in permutation) for permutation in itertools.permutations(range(len(multiset))));assert len(set(outputs.values()))==1
        permutation_cases.append(dict(multiset=multiset,distinct_arrangements=len(outputs),multiplicity_per_arrangement=next(iter(outputs.values()))))
    curves=[]
    for curvature in [[0]*39,[1]*39,[1]*19+[40]+[1]*19]:
        values=[0,5]
        for value in curvature:values.append(2*values[-1]-values[-2]+value)
        a=concentration(values);b=reference_concentration(values);assert a==b;curves.append((values,a))
    assert curves[0][1]==dict(numerator=0,denominator=1,all_zero_curvature=True) and curves[1][1]['numerator']==1 and curves[1][1]['denominator']==39 and curves[2][1]['numerator']==40 and curves[2][1]['denominator']==78
    comparisons=[]
    for _,a in curves:
        for _,b in curves:
            assert strict_greater(a,b)==reference_compare(a,b);comparisons.append(strict_greater(a,b))
    huge=dict(numerator=10**40+1,denominator=10**40,all_zero_curvature=False);one=dict(numerator=1,denominator=1,all_zero_curvature=False);assert strict_greater(huge,one) and reference_compare(huge,one)
    bad=curves[2][0].copy();bad[7]=None;assert concentration(bad) is None and reference_concentration(bad) is None and strict_greater(None,one) is None and reference_compare(one,None) is None
    cases=[]
    for successes,unresolved,decision in [(29,0,'supported'),(28,1,'inconclusive'),(28,0,'falsified'),(0,2,'falsified')]:
        rows=[]
        for bank in range(32):
            pair=[True,True] if bank<successes else [True,None] if bank<successes+unresolved else [False,None]
            for ai,arm in enumerate(ARMS):rows.append(dict(index=2*bank+ai,seed=bank,arm=arm,real_greater=pair[ai],null_grid=dict(selected_degree=1),null_outcome=dict(classification='resolved_failure_fractional'),null_fixed20=dict(classification='resolved_failure_nonexact'),null_top810=dict(classification='resolved_failure_nonexact')))
        result=aggregate(rows);assert result['endpoint']==dict(lower=successes,upper=successes+unresolved,required=29,total=32,decision=decision) and len(result['joint_banks']['resolved_failure'])==32-successes-unresolved;cases.append(result['endpoint'])
    files=[HERE/name for name in ['marginal_shuffle_study.py','audit_marginal_shuffle.py','preflight_marginal_shuffle.py','run_marginal_shuffle.py','balanced_mask_math.py','audit_balanced_mask_math.py','partial_degree_study.py','audit_partial_degree.py','extra_coverage_study.py','audit_extra_coverage.py','run.py','degree_prior_solver.py','degree_prior_math.py','audit_degree_prior.py','audit_laplace_rate.py','laplace_rate_math.py']]
    result=dict(status='passed',host=platform.node(),scope='Synthetic labels/masks/histograms/objectivevectors only; no real H1bs histograms or shuffle outcomes read.',incidence=incidence,independent_histogram_and_shuffle_replay=True,canonical_input_order_invariance=True,exact_multiset_permutation_law_cases=permutation_cases,all_counts_match_independent_reference=True,independent_arm_shuffles_not_deduplicated=True,complete_label_gates=gates,query_accounting=q,ratio_cases=[v for _,v in curves],all_nine_ratio_comparisons_match=comparisons,integer_beyond_binary64_compare=True,zero_denominator_and_unresolved_rules=True,endpoint_bound_cases=cases,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files});save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
