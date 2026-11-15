"""H1bm synthetic nested masks, dense prefix, matched pair coqueries and endpoint bounds."""
import json,platform
import numpy as np
from matched_coquery_study import ROOT,HERE,sha,save,make_masks,observe
from audit_matched_coquery import reference_masks,reference_observe,complete_gate,literal_account,aggregate


def main():
    assert platform.node()=='ripper';path=HERE/'matched-coquery-preflight.json';assert not path.exists()
    rng=np.random.default_rng(960401);rows=639;clean=rng.integers(0,9,(rows,81),dtype=np.int8);noise=dict(uniforms=rng.random(rows),permutations=np.stack([rng.permutation(81).astype(np.uint8) for _ in range(rows)]));noise['uniforms'][:4]=[0.,9/160,np.nextafter(9/160,0),1.]
    actual=make_masks(960402,rows);reference=reference_masks(960402,rows)
    for key,value in actual.items():assert np.array_equal(value,reference[key])
    assert actual['sparse'].shape==(639,81) and actual['dense'].shape==(124,81)
    assert not (actual['sparse'][:124]&~actual['dense']).any() and (actual['sparse'].sum(1)==32).all() and (actual['dense'].sum(1)==72).all()
    outcomes={}
    for arm in ['sparse','dense']:
        n=len(actual[arm]);outcomes[arm]=observe(actual[arm],clean[:n],{k:v[:n] for k,v in noise.items()});ref=reference_observe(actual[arm],clean[:n],{k:v[:n] for k,v in noise.items()})
        for key,value in ref.items():assert np.array_equal(value,outcomes[arm][key])
    assert outcomes['sparse']['joint_count'].sum()==316944 and outcomes['dense']['joint_count'].sum()==316944
    gates=complete_gate(960402,clean,noise,outcomes);q=literal_account(actual['sparse'],actual['dense'])
    assert q==dict(sparse_queries=20448,dense_queries=8928,shared_queries=3968,new_union_queries=25408)
    shared=actual['sparse'][:124]&actual['dense'];assert np.array_equal(outcomes['sparse']['values'][:124][shared],outcomes['dense']['values'][shared])
    # Dense uses no suffix observations or latent draws; sparse retains the common prefix.
    changed=clean.copy();changed[124:]=(changed[124:]+1)%9;changed_noise={k:v.copy() for k,v in noise.items()};changed_noise['uniforms'][124:]=1-changed_noise['uniforms'][124:];changed_noise['permutations'][124:]=np.roll(changed_noise['permutations'][124:],11,axis=1)
    dense_after=observe(actual['dense'],changed[:124],{k:v[:124] for k,v in changed_noise.items()});sparse_after=observe(actual['sparse'],changed,changed_noise)
    for key,value in outcomes['dense'].items():assert np.array_equal(value,dense_after[key])
    assert np.array_equal(sparse_after['values'][:124],outcomes['sparse']['values'][:124])
    cases=[]
    for sc,su,dc,du,expect_d,expect_a in [(30,0,32,0,'supported','supported'),(30,1,32,0,'supported','inconclusive'),(32,0,30,1,'inconclusive','falsified'),(32,0,30,0,'falsified','falsified')]:
        fake=[]
        for seed in range(32):
            for ai,(arm,count,unresolved) in enumerate([('sparse',sc,su),('dense',dc,du)]):
                category='success' if seed<count else 'unresolved' if seed<count+unresolved else 'resolved_failure_nonexact'
                fake.append(dict(index=2*seed+ai,seed=seed,arm=arm,classification=category,selected_degree=20 if category!='unresolved' else None,missing_true=0 if category=='success' else None,extra=0 if category=='success' else None,fixed_degree20={'classification':category},top810={'classification':category}))
        _,bounds,_=aggregate(fake);assert bounds['dense_procedure']['decision']==expect_d and bounds['dense_advantage']['decision']==expect_a;cases.append(bounds)
    pool=np.arange(10000);excluded=np.unique(rng.integers(0,10000,2700));eligible=np.setdiff1d(pool,excluded);draws=np.stack([np.random.default_rng(960500+bank).choice(eligible,639,replace=False) for bank in range(32)])
    assert all(len(set(row))==639 for row in draws) and not np.isin(draws,excluded).any()
    files=[HERE/name for name in ['matched_coquery_study.py','audit_matched_coquery.py','preflight_matched_coquery.py','run_matched_coquery.py','partial_degree_study.py','audit_partial_degree.py','extra_coverage_study.py','audit_extra_coverage.py','run.py','degree_prior_solver.py','degree_prior_math.py','audit_degree_prior.py','audit_laplace_rate.py','laplace_rate_math.py']]
    result=dict(status='passed',host=platform.node(),scope='639 syntheticrows and toyIDpool only; no official H1bm IDs/masks/labels beforefreeze.',all_masks_and_counts_match_independent_reference=True,nested_prefix_masks=True,dense_suffix_label_and_latent_gate=True,complete_label_gates=gates,query_accounting=q,
                separate_endpoint_bound_cases=cases,toy_source_draws_disjoint=True,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files})
    save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
