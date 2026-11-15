"""H1br synthetic null replay, exact ratios, joint endpoint and label gates."""
import json,platform
import numpy as np
from null_signal_study import ROOT,HERE,ARMS,sha,save,sample_null,observe,concentration,strict_greater
from audit_null_signal import reference_null,reference_observe,reference_concentration,reference_compare,literal_account,label_gate,aggregate
from balanced_mask_math import make_balanced_masks
from audit_balanced_mask_math import reference_balanced,verify_incidence


def main():
    assert platform.node()=='ripper';path=HERE/'null-signal-preflight.json';assert not path.exists();incidence=verify_incidence()
    labels=sample_null(981301);assert labels.shape==(180,81) and labels.dtype==np.int8 and ((labels>=0)&(labels<9)).all() and np.array_equal(labels,reference_null(981301))
    rng=np.random.default_rng(981302);uniform=np.zeros((180,81),bool)
    for row in range(180):uniform[row,rng.permutation(81)[:72]]=True
    masks=dict(uniform=uniform,balanced=make_balanced_masks(981303));assert np.array_equal(masks['balanced'],reference_balanced(981303));expected={}
    for arm in ARMS:
        expected[arm]=observe(masks[arm],labels);other=reference_observe(masks[arm],labels)
        for key,value in other.items():assert np.array_equal(value,expected[arm][key])
        assert expected[arm]['joint_count'].sum()==460080
    assert (expected['balanced']['joint_count']==142).all();shared=masks['uniform']&masks['balanced'];union=masks['uniform']|masks['balanced'];assert np.array_equal(expected['uniform']['values'][shared],expected['balanced']['values'][shared])
    q=literal_account(masks);assert q==dict(uniform_queries=12960,balanced_queries=12960,shared_queries=int(shared.sum()),new_union_queries=int(union.sum())) and q['shared_queries']+q['new_union_queries']==25920
    gates=label_gate(masks,labels,expected)
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
    files=[HERE/name for name in ['null_signal_study.py','audit_null_signal.py','preflight_null_signal.py','run_null_signal.py','balanced_mask_math.py','audit_balanced_mask_math.py','partial_degree_study.py','audit_partial_degree.py','extra_coverage_study.py','audit_extra_coverage.py','run.py','degree_prior_solver.py','degree_prior_math.py','audit_degree_prior.py','audit_laplace_rate.py','laplace_rate_math.py']]
    result=dict(status='passed',host=platform.node(),scope='Synthetic null/masks/objectivevectors only; no official H1br null streams or archivedreal concentration ratios read.',incidence=incidence,independent_null_replay=True,all_counts_match_independent_reference=True,shared_null_values_identical=True,complete_label_gates=gates,query_accounting=q,ratio_cases=[v for _,v in curves],all_nine_ratio_comparisons_match=comparisons,integer_beyond_binary64_compare=True,zero_denominator_and_unresolved_rules=True,endpoint_bound_cases=cases,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files});save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
