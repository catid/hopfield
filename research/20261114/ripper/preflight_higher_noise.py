"""H1bq synthetic masks/noise, exact incidence, budget and endpoint gates."""
import json,platform
from fractions import Fraction
import numpy as np
from higher_noise_study import ROOT,HERE,ARMS,sha,save,make_masks,observe
from audit_higher_noise import reference_masks,reference_observe,complete_gate,literal_account,aggregate,realized_noise
from audit_balanced_mask_math import verify_incidence


def main():
    assert platform.node()=='ripper';path=HERE/'higher-noise-preflight.json';assert not path.exists();incidence=verify_incidence()
    rng=np.random.default_rng(981101);clean=rng.integers(0,9,(180,81),dtype=np.int8);noise=dict(iid_uniforms=rng.random((180,81)),iid_offsets=rng.integers(1,9,(180,81),dtype=np.int8));noise['iid_uniforms'][:5]=np.array([0,np.nextafter(.075,0),.075,np.nextafter(.075,1),1])[:,None]
    masks=make_masks(981102,981103);refm=reference_masks(981102,981103);ii,jj=np.triu_indices(81,1)
    for key,value in masks.items():assert np.array_equal(value,refm[key]) and value.shape==(180,81) and (value.sum(1)==72).all()
    assert np.array_equal(make_masks(981104,981103)['balanced'],masks['balanced']) and np.array_equal(make_masks(981102,981105)['uniform'],masks['uniform'])
    assert not np.array_equal(make_masks(981104,981103)['uniform'],masks['uniform']) and not np.array_equal(make_masks(981102,981105)['balanced'],masks['balanced'])
    expected={}
    for arm in ARMS:
        given=masks[arm];expected[arm]=observe(given,clean,noise);reference=reference_observe(given,clean,noise)
        for key,value in reference.items():assert np.array_equal(value,expected[arm][key])
        errors=noise['iid_uniforms']<.075;assert realized_noise(given,clean,noise)==dict(full_cell_errors=int(errors.sum()),queried_cell_errors=int(errors[given].sum()),full_cells=14580,queried_cells=12960)
        assert expected[arm]['joint_count'].sum()==460080
    assert (expected['balanced']['joint_count']==142).all() and np.var(expected['uniform']['joint_count'])>0
    shared=masks['uniform']&masks['balanced'];union=masks['uniform']|masks['balanced'];assert np.array_equal(expected['uniform']['values'][shared],expected['balanced']['values'][shared])
    q=literal_account(masks['uniform'],masks['balanced']);assert q==dict(uniform_queries=12960,balanced_queries=12960,shared_queries=int(shared.sum()),new_union_queries=int(union.sum())) and q['shared_queries']+q['new_union_queries']==25920 and q['new_union_queries']<=14580
    gates=complete_gate(981102,981103,clean,noise,expected)
    cases=[]
    for su,uu,sb,ub,recovery,advantage in [(20,0,24,0,'supported','supported'),(21,0,24,0,'supported','falsified'),(20,1,23,1,'inconclusive','inconclusive'),(24,0,23,0,'falsified','falsified'),(0,4,0,4,'falsified','inconclusive'),(0,0,32,0,'supported','supported')]:
        rows=[]
        for bank in range(32):
            for ai,arm in enumerate(ARMS):
                count,u=(su,uu) if ai==0 else (sb,ub);category='success' if bank<count else 'unresolved' if bank<count+u else 'resolved_failure_nonexact'
                rows.append(dict(index=2*bank+ai,seed=bank,arm=arm,classification=category,selected_degree=20 if category!='unresolved' else None,missing_true=0 if category=='success' else None,extra=0 if category=='success' else None,fixed_degree20={'classification':category},top810={'classification':category}))
        _,endpoints,paired=aggregate(rows);assert set(endpoints)=={'balanced_recovery','balanced_advantage'} and endpoints['balanced_recovery']['decision']==recovery and endpoints['balanced_advantage']['decision']==advantage
        assert endpoints['balanced_advantage']['lower']==sb-su-uu and endpoints['balanced_advantage']['upper']==sb+ub-su and sum(map(len,paired.values()))==32;cases.append(endpoints)
    pool=np.arange(10000);excluded=np.unique(rng.integers(0,10000,2700));eligible=np.setdiff1d(pool,excluded);draws=np.stack([np.random.default_rng(981200+bank).choice(eligible,180,replace=False) for bank in range(32)]);assert all(len(set(row))==180 for row in draws) and not np.isin(draws,excluded).any()
    pair_probability=Fraction(72*71,81*80);assert 180*pair_probability==142 and 180*pair_probability*(1-pair_probability)==Fraction(1349,45)
    p=Fraction(3,40);unequal=sum((1-p if z==0 else p/8)*(1-p if z==1 else p/8) for z in range(9));same=sum((1-p if z==0 else p/8)**2 for z in range(9))
    assert unequal==Fraction(1839,102400) and same-unequal==Fraction(85849,102400)>0
    assert float(.075).hex()=='0x1.3333333333333p-4'
    files=[HERE/name for name in ['higher_noise_study.py','audit_higher_noise.py','preflight_higher_noise.py','run_higher_noise.py','balanced_mask_math.py','audit_balanced_mask_math.py','partial_degree_study.py','audit_partial_degree.py','extra_coverage_study.py','audit_extra_coverage.py','run.py','degree_prior_solver.py','degree_prior_math.py','audit_degree_prior.py','audit_laplace_rate.py','laplace_rate_math.py']]
    result=dict(status='passed',host=platform.node(),scope='Synthetic180rows, finite-field incidence and toyIDpool only; no H1bq official draws.',incidence=incidence,ideal_noise_probability='3/40',implemented_noise_hex=float(.075).hex(),ideal_true_edge_equality=str(unequal),ideal_affine_signal_scale=str(same-unequal),independent_mask_streams=True,all_counts_match_independent_reference=True,shared_latent_values_identical=True,realized_noise_counts_match_thresholds=True,complete_label_gates=gates,query_accounting=q,endpoint_bound_cases=cases,uniform_pair_expected_exposure='142',uniform_pair_marginal_variance='1349/45',toy_source_draws_disjoint=True,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files});save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
