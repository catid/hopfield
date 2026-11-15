"""H1bo toy iid prefix, budget accounting, bounds and affine-cost algebra."""
import itertools,json,platform
from fractions import Fraction
import numpy as np
from iid_row_budget_study import ROOT,HERE,ARMS,sha,save,make_masks,observe
from audit_iid_row_budget import reference_masks,reference_observe,complete_gate,literal_account,aggregate,realized_noise


def main():
    assert platform.node()=='ripper';path=HERE/'iid-row-budget-preflight.json';assert not path.exists()
    rng=np.random.default_rng(980401);clean=rng.integers(0,9,(248,81),dtype=np.int8);noise=dict(iid5_uniforms=rng.random((248,81)),iid5_offsets=rng.integers(1,9,(248,81),dtype=np.int8));noise['iid5_uniforms'][:5]=np.array([0,np.nextafter(.05,0),.05,np.nextafter(.05,1),1])[:,None]
    masks=make_masks(980402);refm=reference_masks(980402)
    for key,value in masks.items():assert np.array_equal(value,refm[key])
    assert masks['prefix124'].shape==(124,81) and masks['full248'].shape==(248,81) and np.array_equal(masks['prefix124'],masks['full248'][:124]) and (masks['full248'].sum(1)==72).all()
    expected={}
    for arm in ARMS:
        n=len(masks[arm]);expected[arm]=observe(masks[arm],clean[:n],{k:v[:n] for k,v in noise.items()});reference=reference_observe(masks[arm],clean[:n],{k:v[:n] for k,v in noise.items()})
        for key,value in reference.items():assert np.array_equal(value,expected[arm][key])
    assert np.array_equal(expected['prefix124']['values'],expected['full248']['values'][:124])
    assert expected['prefix124']['joint_count'].sum()==316944 and expected['full248']['joint_count'].sum()==633888
    for arm in ARMS:
        given=masks[arm];n=len(given);part={k:v[:n] for k,v in noise.items()};measured=realized_noise(given,clean[:n],part);errors=part['iid5_uniforms']<.05
        assert measured==dict(full_cell_errors=int(errors.sum()),queried_cell_errors=int(errors[given].sum()),full_cells=81*n,queried_cells=72*n)
    gates=complete_gate(980402,clean,noise,expected);q=literal_account(masks['prefix124'],masks['full248']);assert q==dict(prefix124_queries=8928,full248_queries=17856,shared_queries=8928,new_union_queries=17856)
    changed=clean.copy();changed[124:]=(changed[124:]+1)%9;changed_noise={k:v.copy() for k,v in noise.items()};changed_noise['iid5_uniforms'][124:]=1-changed_noise['iid5_uniforms'][124:];changed_noise['iid5_offsets'][124:]=1+changed_noise['iid5_offsets'][124:]%8
    for key,value in expected['prefix124'].items():assert np.array_equal(value,observe(masks['prefix124'],changed[:124],{k:v[:124] for k,v in changed_noise.items()})[key])
    full_after=observe(masks['full248'],changed,changed_noise);assert np.array_equal(full_after['values'][:124],expected['prefix124']['values'])
    cases=[]
    for successes,unresolved,decision in [(31,0,'supported'),(30,0,'falsified'),(30,1,'inconclusive'),(0,2,'falsified')]:
        rows=[]
        for bank in range(32):
            for ai,arm in enumerate(ARMS):
                count=successes if ai else 16;u=unresolved if ai else 0;category='success' if bank<count else 'unresolved' if bank<count+u else 'resolved_failure_nonexact'
                rows.append(dict(index=2*bank+ai,seed=bank,arm=arm,classification=category,selected_degree=20 if category!='unresolved' else None,missing_true=0 if category=='success' else None,extra=0 if category=='success' else None,fixed_degree20={'classification':category},top810={'classification':category}))
        _,endpoints,_=aggregate(rows);assert set(endpoints)=={'full248'} and endpoints['full248']['decision']==decision;cases.append(endpoints)
    # Independent exhaustive four-node regular graph costs verify the affine identity.
    ii,jj=np.triu_indices(4,1);cost=np.array([7,1,5,2,9,3],np.int64);a,b=3,-4;values=[];transformed=[];feasible=0
    for degree in range(4):
        original=[];changed=[]
        for bits in itertools.product([0,1],repeat=6):
            x=np.array(bits,np.int64);degrees=np.bincount(np.r_[ii,jj],weights=np.r_[x,x],minlength=4)
            if not (degrees==degree).all():continue
            feasible+=1;v=int(cost@x);w=int((a*cost+b)@x);assert w==a*v+b*4*degree//2;original.append(v);changed.append(w)
        values.append(min(original));transformed.append(min(changed))
    assert np.array_equal(np.diff(transformed,2),a*np.diff(values,2))
    p=Fraction(1,20);collision=p/4-9*p*p/64;scale=1-9*collision
    assert collision==Fraction(311,25600) and scale==Fraction(22801,25600)>0
    pool=np.arange(10000);excluded=np.unique(rng.integers(0,10000,2700));eligible=np.setdiff1d(pool,excluded);draws=np.stack([np.random.default_rng(980500+bank).choice(eligible,248,replace=False) for bank in range(32)]);assert all(len(set(row))==248 for row in draws) and not np.isin(draws,excluded).any()
    files=[HERE/name for name in ['iid_row_budget_study.py','audit_iid_row_budget.py','preflight_iid_row_budget.py','run_iid_row_budget.py','corruption_law_study.py','audit_corruption_law.py','partial_degree_study.py','audit_partial_degree.py','extra_coverage_study.py','audit_extra_coverage.py','run.py','degree_prior_solver.py','degree_prior_math.py','audit_degree_prior.py','audit_laplace_rate.py','laplace_rate_math.py']]
    result=dict(status='passed',host=platform.node(),scope='Synthetic248rows and toyIDpool only; no H1bo official draws.',literal_prefix_masks_values_noise=True,prefix_suffix_invariance=True,all_counts_match_independent_reference=True,realized_noise_counts_match_thresholds=True,complete_label_gates=gates,query_accounting=q,endpoint_bound_cases=cases,affine_identity_four_node_feasible_graphs=feasible,affine_second_difference_scaling=True,ideal_iid_collision=str(collision),ideal_iid_scale=str(scale),toy_source_draws_disjoint=True,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files});save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
