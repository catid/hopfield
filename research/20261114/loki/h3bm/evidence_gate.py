"""Synthetic eligibility/filter/likelihood boundary gate before fresh draws."""
import itertools
import numpy as np
from evidence import score,draw_pair
from common import OUT,save,now


def run():
    populations=[7,8,15,16,20];orbits=[0,1,2,0,2];rows=[]
    for m in [8,16]:
        observed=set(range(m));eligible=[];scores=[]
        for cid,(M,k) in enumerate(zip(populations,orbits)):
            compatible=observed<=set(range(M));value=score(M,k,m);combinations=sum(1 for _ in itertools.combinations(range(M),m)) if M>=m else 0;prior_choices=sum(1 for _ in itertools.combinations(range(55),k));assert compatible==(M>=m) and value==(prior_choices*combinations if compatible else None)
            if compatible:eligible.append(cid)
            scores.append(str(value) if value is not None else None)
        toy_predictions=np.arange(20)[None,:]<np.array(populations)[:,None];actual=np.flatnonzero(toy_predictions[:,sorted(observed)].all(1)).tolist();assert actual==eligible
        rows.append(dict(positive_count=m,source_indices=sorted(observed),compatible_candidates=eligible,vectorized_filter_candidates=actual,exact_scores=scores))
    assert rows[0]['compatible_candidates']==[1,2,3,4] and rows[1]['compatible_candidates']==[3,4]
    # Same class prior: eight-source likelihood can rank the small-population model
    # ahead of the ordinary model; at sixteen it is necessarily incompatible.
    assert int(rows[0]['exact_scores'][1])<int(rows[0]['exact_scores'][3]) and rows[1]['exact_scores'][1] is None
    population=np.arange(20,dtype=np.uint8).reshape(20,1);indices,subset,sixteen,eight=draw_pair(population,811,812);assert len(set(indices.tolist()))==16 and len(set(subset.tolist()))==8 and np.array_equal(eight,sixteen[subset]) and subset.tolist()!=list(range(8));result=dict(utc=now(),toy_populations=populations,toy_orbit_counts=orbits,likelihood_filter_fixtures=rows,all8_15_population_boundary_cases_retained=True,exact_combinations_independently_enumerated=True,uniform_subset_mechanism='Independent default_rng(812).choice(16,8,replace=False); no lexical subset',draw_fixture=dict(seed16=811,seed8=812,indices16=indices.tolist(),subset_indices8=subset.tolist(),sample16=sixteen.tolist(),sample8=eight.tolist()));save(OUT/'evidence-gate.json',result);return result
