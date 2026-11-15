"""H1bl synthetic nested masks, dense prefix, equal costs and endpoint bounds."""
import json,platform
import numpy as np
from coverage_tradeoff_study import ROOT,HERE,sha,save,make_masks,observe
from audit_coverage_tradeoff import reference_masks,reference_observe,complete_gate,literal_account,aggregate


def main():
    assert platform.node()=='ripper';path=HERE/'coverage-tradeoff-preflight.json';assert not path.exists()
    benchmark=json.loads((HERE/'coverage-tradeoff-csv-benchmark.json').read_text())
    assert benchmark['status']=='passed' and benchmark['exact_selected_answer_parity'] and benchmark['producer_sha256']==sha(HERE/'coverage_tradeoff_study.py') and benchmark['benchmark_sha256']==sha(HERE/'benchmark_coverage_csv.py')
    rng=np.random.default_rng(950401);rows=450;clean=rng.integers(0,9,(rows,81),dtype=np.int8);noise=dict(uniforms=rng.random(rows),permutations=np.stack([rng.permutation(81).astype(np.uint8) for _ in range(rows)]));noise['uniforms'][:4]=[0.,9/160,np.nextafter(9/160,0),1.]
    actual=make_masks(950402,rows);reference=reference_masks(950402,rows)
    for key,value in actual.items():assert np.array_equal(value,reference[key])
    assert actual['sparse'].shape==(450,81) and actual['dense'].shape==(200,81)
    assert not (actual['sparse'][:200]&~actual['dense']).any() and (actual['sparse'].sum(1)==32).all() and (actual['dense'].sum(1)==72).all()
    outcomes={}
    for arm in ['sparse','dense']:
        n=len(actual[arm]);outcomes[arm]=observe(actual[arm],clean[:n],{k:v[:n] for k,v in noise.items()});ref=reference_observe(actual[arm],clean[:n],{k:v[:n] for k,v in noise.items()})
        for key,value in ref.items():assert np.array_equal(value,outcomes[arm][key])
    assert outcomes['sparse']['joint_count'].sum()==223200 and outcomes['dense']['joint_count'].sum()==511200
    gates=complete_gate(950402,clean,noise,outcomes);q=literal_account(actual['sparse'],actual['dense'])
    assert q==dict(sparse_queries=14400,dense_queries=14400,shared_queries=6400,new_union_queries=22400)
    shared=actual['sparse'][:200]&actual['dense'];assert np.array_equal(outcomes['sparse']['values'][:200][shared],outcomes['dense']['values'][shared])
    # Dense uses no suffix observations or latent draws; sparse retains the common prefix.
    changed=clean.copy();changed[200:]=(changed[200:]+1)%9;changed_noise={k:v.copy() for k,v in noise.items()};changed_noise['uniforms'][200:]=1-changed_noise['uniforms'][200:];changed_noise['permutations'][200:]=np.roll(changed_noise['permutations'][200:],11,axis=1)
    dense_after=observe(actual['dense'],changed[:200],{k:v[:200] for k,v in changed_noise.items()});sparse_after=observe(actual['sparse'],changed,changed_noise)
    for key,value in outcomes['dense'].items():assert np.array_equal(value,dense_after[key])
    assert np.array_equal(sparse_after['values'][:200],outcomes['sparse']['values'][:200])
    cases=[]
    for sc,su,dc,du,expect_d,expect_a in [(30,0,32,0,'supported','supported'),(30,1,32,0,'supported','inconclusive'),(32,0,30,1,'inconclusive','falsified'),(32,0,30,0,'falsified','falsified')]:
        fake=[]
        for seed in range(32):
            for ai,(arm,count,unresolved) in enumerate([('sparse',sc,su),('dense',dc,du)]):
                category='success' if seed<count else 'unresolved' if seed<count+unresolved else 'resolved_failure_nonexact'
                fake.append(dict(index=2*seed+ai,seed=seed,arm=arm,classification=category,selected_degree=20 if category!='unresolved' else None,missing_true=0 if category=='success' else None,extra=0 if category=='success' else None,fixed_degree20={'classification':category},top810={'classification':category}))
        _,bounds,_=aggregate(fake);assert bounds['dense_procedure']['decision']==expect_d and bounds['dense_advantage']['decision']==expect_a;cases.append(bounds)
    pool=np.arange(10000);excluded=np.unique(rng.integers(0,10000,2700));eligible=np.setdiff1d(pool,excluded);draws=np.stack([np.random.default_rng(950500+bank).choice(eligible,450,replace=False) for bank in range(32)])
    assert all(len(set(row))==450 for row in draws) and not np.isin(draws,excluded).any()
    files=[HERE/name for name in ['coverage_tradeoff_study.py','audit_coverage_tradeoff.py','preflight_coverage_tradeoff.py','run_coverage_tradeoff.py','benchmark_coverage_csv.py','partial_degree_study.py','audit_partial_degree.py','extra_coverage_study.py','audit_extra_coverage.py','run.py','degree_prior_solver.py','degree_prior_math.py','audit_degree_prior.py','audit_laplace_rate.py','laplace_rate_math.py']]
    result=dict(status='passed',host=platform.node(),scope='450 syntheticrows and toyIDpool only; no official H1bl IDs/masks/labels beforefreeze.',all_masks_and_counts_match_independent_reference=True,nested_prefix_masks=True,dense_suffix_label_and_latent_gate=True,complete_label_gates=gates,query_accounting=q,
                separate_endpoint_bound_cases=cases,toy_source_draws_disjoint=True,archived_csv_benchmark=benchmark,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files})
    save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
