"""H1bk synthetic mask, corruption, query-union and bound checks."""
import json,platform
import numpy as np
from uniform32_study import ROOT,HERE,sha,save,make_masks,observe
from audit_uniform32 import reference_masks,reference_observe,complete_gate,literal_account,aggregate


def main():
    assert platform.node()=='ripper';path=HERE/'uniform32-preflight.json';assert not path.exists()
    rng=np.random.default_rng(950201);rows=37;clean=rng.integers(0,9,(rows,81),dtype=np.int8);noise=dict(uniforms=rng.random(rows),permutations=np.stack([rng.permutation(81).astype(np.uint8) for _ in range(rows)]));noise['uniforms'][:4]=[0.,9/160,np.nextafter(9/160,0),1.]
    feedback=rng.integers(0,9,(81,81),dtype=np.int32);feedback=np.triu(feedback,1);feedback+=feedback.T
    actual=make_masks(feedback,950202,950203,rows);reference=reference_masks(feedback,950202,950203,rows)
    for key,value in actual.items():assert np.array_equal(value,reference[key])
    outcomes={}
    for arm in ['uniform','donor']:
        outcomes[arm]=observe(actual[arm],clean,noise);ref=reference_observe(actual[arm],clean,noise)
        for key,value in ref.items():assert np.array_equal(value,outcomes[arm][key])
    gates=complete_gate(feedback,950202,950203,clean,noise,outcomes);q=literal_account(actual['uniform'],actual['donor'])
    assert q['uniform_queries']==q['donor_queries']==rows*32 and q['new_union_queries']==int((actual['uniform']|actual['donor']).sum())
    shared=actual['uniform']&actual['donor'];assert np.array_equal(outcomes['uniform']['values'][shared],outcomes['donor']['values'][shared])
    # Every added donor cell appears in the shared rank prefix32, since at most25 are excluded.
    assert not ((actual['donor']&~actual['donor_old25'])&~actual['uniform']).any()
    cases=[]
    for unresolved in [False,True]:
        fake=[]
        for seed in range(32):
            for ai,arm in enumerate(['uniform','donor']):
                category='success' if arm=='donor' or seed<30 else ('unresolved' if unresolved and seed==30 else 'resolved_failure_nonexact')
                fake.append(dict(index=2*seed+ai,seed=seed,arm=arm,classification=category,selected_degree=20 if category!='unresolved' else None,missing_true=0 if category=='success' else None,extra=0 if category=='success' else None,fixed_degree20={'classification':category},top810={'classification':category}))
        _,bounds,_=aggregate(fake)
        assert bounds['uniform_procedure']['decision']==('inconclusive' if unresolved else 'falsified') and bounds['donor_benefit']['decision']==('inconclusive' if unresolved else 'supported')
        cases.append(bounds)
    pool=np.arange(10000);excluded=np.unique(rng.integers(0,10000,2700));eligible=np.setdiff1d(pool,excluded);draws=np.stack([np.random.default_rng(950300+bank).choice(eligible,900,replace=False) for bank in range(32)])
    assert all(len(set(row))==900 for row in draws) and not np.isin(draws,excluded).any()
    files=[HERE/name for name in ['uniform32_study.py','audit_uniform32.py','preflight_uniform32.py','run_uniform32.py','partial_degree_study.py','audit_partial_degree.py','extra_coverage_study.py','audit_extra_coverage.py','run.py','degree_prior_solver.py','degree_prior_math.py','audit_degree_prior.py','audit_laplace_rate.py','laplace_rate_math.py','transfer_feedback_policy.py','soft_feedback_policy.py','prefix_rate_features.py','audit_transfer_feedback.py','audit_soft_feedback.py']]
    result=dict(status='passed',host=platform.node(),scope='Synthetic masks/labels and toyIDpool only; no official H1bk IDs/masks/labels beforefreeze.',rows=rows,all_masks_match_independent_reference=True,all_values_and_counts_match_reference=True,complete_label_gates=gates,query_accounting=q,
                separate_unresolved_bound_cases=cases,toy_source_draws_disjoint=True,donor_extra_cells_in_uniform_rank_prefix=True,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files})
    save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
