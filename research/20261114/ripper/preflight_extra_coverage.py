"""Synthetic mask, label gate and paired query accounting before H1bj."""
import json,platform
import numpy as np
from extra_coverage_study import ROOT,HERE,sha,save,extend,acquire,accounting
from audit_extra_coverage import reference_masks,reference_observe,literal_account,complete_gate


def main():
    assert platform.node()=='ripper';path=HERE/'extra-coverage-preflight.json';assert not path.exists()
    rng=np.random.default_rng(950101);rows=37;clean=rng.integers(0,9,(rows,81),dtype=np.int8);noise=dict(uniforms=rng.random(rows),permutations=np.stack([rng.permutation(81).astype(np.uint8) for _ in range(rows)]));noise['uniforms'][:4]=[0.,9/160,np.nextafter(9/160,0),1.]
    old=[];new=[];observed=[];gates=[]
    for arm in range(2):
        mask=np.zeros((rows,81),bool)
        for row in range(rows):mask[row,rng.choice(81,25,replace=False)]=True
        given=extend(mask,950102);assert np.array_equal(given,reference_masks(mask,950102));o=acquire(mask,950102,clean,noise);ref=reference_observe(given,clean,noise)
        for key,value in ref.items():assert np.array_equal(o[key],value)
        gates.append(complete_gate(mask,950102,clean,noise,o));old.append(mask);new.append(given);observed.append(o)
    production=accounting(old[0],old[1],new[0],new[1]);literal=literal_account(old[0],old[1],new[0],new[1]);assert production==literal
    assert production['aligned_added_to_own']==production['permuted_added_to_own']==rows*7
    shared=new[0]&new[1];assert np.array_equal(observed[0]['values'][shared],observed[1]['values'][shared]);assert production['actual_increment']<=2*rows*7
    for which in range(2):
        unchanged=extend(old[which],950102);assert np.array_equal(unchanged,new[which])
    files=[HERE/name for name in ['extra_coverage_study.py','audit_extra_coverage.py','preflight_extra_coverage.py','run_extra_coverage.py','partial_degree_study.py','audit_partial_degree.py','run.py','degree_prior_solver.py','degree_prior_math.py','audit_degree_prior.py','audit_laplace_rate.py','laplace_rate_math.py','transfer_feedback_policy.py','soft_feedback_policy.py','prefix_rate_features.py']]
    result=dict(status='passed',host=platform.node(),scope='37 synthetic rows only; no official additional labels or mask draws before freeze.',rows=rows,paired_masks_match_independent_reference=True,raw_values_and_counts_match_independent_reference=True,complete_label_gates=gates,literal_query_accounting=literal,
                old_coordinates_retained=True,exactly_seven_added_per_row=True,common_latent_values_match=True,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files})
    save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
