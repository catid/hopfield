"""H1bn toy corruption probabilities, boundaries, masks and endpoint bounds."""
from decimal import Decimal,localcontext
from fractions import Fraction
import json,platform
import numpy as np
from corruption_law_study import ROOT,HERE,ARMS,PAIR_P,sha,save,make_masks,make_noise,corrupt,observe
from audit_corruption_law import reference_masks,reference_corrupt,reference_observe,noise_audit,complete_gate,literal_account,aggregate


def pair_collision(p):
    def probability(original,observed):return 1-p if original==observed else p/8
    return sum(probability(0,digit)*probability(1,digit) for digit in range(9))


def main():
    assert platform.node()=='ripper';path=HERE/'corruption-law-preflight.json';assert not path.exists()
    with localcontext() as context:
        context.prec=80;parameter=Decimal('.72')/(Decimal(16)+Decimal('243.04').sqrt())
    assert float(parameter)==PAIR_P and PAIR_P.hex()=='0x1.756d7561abf76p-6'
    labels=np.repeat(np.arange(9),9);ordered_pairs=[(i,j) for i in range(81) for j in range(81) if i!=j]
    permuted_equality=Fraction(int(sum(labels[i]==labels[j] for i,j in ordered_pairs)),len(ordered_pairs))
    bursty_collision=Fraction(9,160)*permuted_equality;bursty_error=Fraction(9,160)*Fraction(int(sum(labels!=0)),81)
    assert permuted_equality==Fraction(1,10) and bursty_collision==Fraction(9,1600) and bursty_error==Fraction(1,20)
    assert pair_collision(Fraction(1,20))==Fraction(311,25600)
    actual_pair_collision=pair_collision(Fraction(PAIR_P));assert abs(actual_pair_collision-bursty_collision)<Fraction(1,10**18)
    rng=np.random.default_rng(970401);clean=rng.integers(0,9,(124,81),dtype=np.int8);bases=dict(bursty=970500,iid5=970600,iid_pair=970700);noise=make_noise(bases,0);noise_audit(noise,bases,0)
    for arm,threshold in [('bursty',9/160),('iid5',.05),('iid_pair',PAIR_P)]:
        values=np.array([0.,np.nextafter(threshold,0),threshold,np.nextafter(threshold,1),1.])
        if arm=='bursty':noise[arm+'_uniforms'][:5]=values
        else:noise[arm+'_uniforms'][:5]=values[:,None]
    original_noise={k:v.copy() for k,v in noise.items()};masks=make_masks(970402);reference=reference_masks(970402)
    assert np.array_equal(masks['given'],reference['given']) and masks['given'].shape==(124,81) and (masks['given'].sum(1)==72).all()
    expected={};boundary_cases=[]
    for arm in ARMS:
        actual=corrupt(clean,noise,arm);literal=reference_corrupt(clean,noise,arm);assert np.array_equal(actual,literal)
        if arm!='bursty':
            assert (actual[:2]!=clean[:2]).all() and np.array_equal(actual[2:5],clean[2:5]);bad={k:v.copy() for k,v in noise.items()};bad[arm+'_offsets'][0,0]=0
            try:corrupt(clean,bad,arm)
            except AssertionError:pass
            else:raise AssertionError('Zero replacement offset accepted')
        expected[arm]=observe(masks['given'],clean,noise,arm);ref=reference_observe(masks['given'],clean,noise,arm)
        for key,value in ref.items():assert np.array_equal(value,expected[arm][key])
        assert expected[arm]['joint_count'].sum()==316944
        boundary_cases.append(dict(arm=arm,threshold=9/160 if arm=='bursty' else .05 if arm=='iid5' else PAIR_P,matched_reference=True))
    for key,value in noise.items():assert np.array_equal(value,original_noise[key])
    changed_masks=masks['given'][:,::-1].copy()
    for arm in ARMS:
        alternate=observe(changed_masks,clean,noise,arm);common=changed_masks&masks['given'];assert np.array_equal(alternate['values'][common],expected[arm]['values'][common])
    gates=complete_gate(970402,clean,noise,expected);account=literal_account(masks['given'])
    assert account==dict(bursty_queries=8928,iid5_queries=8928,iid_pair_queries=8928,shared_noisy_queries=0,new_union_queries=26784)
    cases=[]
    for success,unresolved,expected_decision in [(31,0,'supported'),(30,0,'falsified'),(30,1,'inconclusive'),(0,2,'falsified')]:
        fake=[]
        for seed in range(32):
            for ai,arm in enumerate(ARMS):
                category='success' if seed<success else 'unresolved' if seed<success+unresolved else 'resolved_failure_nonexact'
                fake.append(dict(index=3*seed+ai,seed=seed,arm=arm,classification=category,selected_degree=20 if category!='unresolved' else None,missing_true=0 if category=='success' else None,extra=0 if category=='success' else None,fixed_degree20={'classification':category},top810={'classification':category}))
        _,endpoints,_=aggregate(fake);assert set(endpoints)=={'iid5','iid_pair'} and all(v['decision']==expected_decision for v in endpoints.values());cases.append(endpoints)
    pool=np.arange(10000);excluded=np.unique(rng.integers(0,10000,2700));eligible=np.setdiff1d(pool,excluded);draws=np.stack([np.random.default_rng(970800+bank).choice(eligible,124,replace=False) for bank in range(32)])
    assert all(len(set(row))==124 for row in draws) and not np.isin(draws,excluded).any()
    files=[HERE/name for name in ['corruption_law_study.py','audit_corruption_law.py','preflight_corruption_law.py','run_corruption_law.py','partial_degree_study.py','audit_partial_degree.py','extra_coverage_study.py','audit_extra_coverage.py','run.py','degree_prior_solver.py','degree_prior_math.py','audit_degree_prior.py','audit_laplace_rate.py','laplace_rate_math.py']]
    result=dict(status='passed',host=platform.node(),scope='Synthetic124row labels/masks/noise and toyIDs only; no official H1bn draws.',pair_p_float=PAIR_P,pair_p_hex=PAIR_P.hex(),pair_probability_error=float(actual_pair_collision-bursty_collision),enumerated_bursty_wrong_cell=str(bursty_error),enumerated_bursty_true_edge_collision=str(bursty_collision),enumerated_iid5_true_edge_collision=str(pair_collision(Fraction(1,20))),threshold_boundary_cases=boundary_cases,nonzero_offsets_required=True,independent_noise_replay=True,mask_latent_invariance=True,all_observation_counts_match=True,complete_label_gates=gates,query_accounting=account,endpoint_bound_cases=cases,all96_arm_indices_checked=True,toy_sources_disjoint=True,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files})
    save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
