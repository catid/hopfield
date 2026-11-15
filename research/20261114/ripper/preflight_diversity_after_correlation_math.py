"""H1ci immutable q1/q3 methods on unrelated full-size synthetic arrays."""
from fractions import Fraction
from pathlib import Path
import json,time
import numpy as np
from correlated_repeat_study import ROOT,HERE,sha,save,make_masks
import correlated_repeat_math as joint
import channel_misspecification_math as single


def main():
    target=HERE/'diversity-after-correlation-math-check.json';assert not target.exists() and not (HERE/'protocol-diversity-after-correlation.json').exists()
    start=time.perf_counter();sources={};inherited={}
    for name in ['correlated-repeat-math-check.json','channel-misspecification-math-check.json']:
        gate=json.loads((HERE/name).read_text());assert gate['status']=='passed' and not gate['study_scores_read'];inherited[name]=sha(HERE/name)
        for file,h in gate['source_sha256'].items():assert sha(HERE/file)==h;sources[file]=h
    rng=np.random.default_rng(995401);mask=np.concatenate([make_masks(995410+i) for i in range(3)])
    values=rng.integers(0,9,(1,270,81),dtype=np.int8);values[:,~mask]=-1
    repeated=rng.integers(0,9,(3,90,81),dtype=np.int8);repeated[0]=values[0,:90];repeated[:,~mask[:90]]=-1
    checks=[]
    for arm,v,g,method,params in [('supplied_correlated90',repeated,mask[:90],joint,joint.PARAMETERS),('supplied_single270',values,mask,single,(3,40))]:
        before=time.perf_counter();actual=method.score(v,g,params);production=time.perf_counter()-before
        before=time.perf_counter();expected,joint_count,bounds=method.independent_score(v,g,params);reference=time.perf_counter()-before
        assert actual['cost_certified'].all() and (actual['cost_precision']==80).all() and np.array_equal(actual['cost'],expected) and np.array_equal(actual['joint_count'],joint_count)
        assert (joint_count==(71 if arm=='supplied_correlated90' else 213)).all()
        for edge,(lo,hi,scale) in enumerate(bounds):
            assert Fraction(method.Decimal(actual['cost_lower'][edge]))<=Fraction(lo*1000000,scale)<=Fraction(hi*1000000,scale)<=Fraction(method.Decimal(actual['cost_upper'][edge]))
        permutation=np.array([4,0,8,1,7,2,6,3,5],np.int8);mapped=v.copy();mapped[:,g]=permutation[v[:,g]]
        assert np.array_equal(method.score(mapped,g,params)['cost'],actual['cost'])
        order=np.arange(g.shape[0])[::-1];assert np.array_equal(method.score(v[:,order],g[order],params)['cost'],actual['cost'])
        checks.append(dict(arm=arm,shape=list(v.shape),joint_count=int(joint_count[0]),integer_costs=len(expected),production_seconds=production,independent_seconds=reference,all_certified=True))
    full=single.score(values,mask,(3,40));parts=[single.score(values[:,90*k:90*(k+1)],mask[90*k:90*(k+1)],(3,40)) for k in range(3)]
    assert np.array_equal(full['likelihood_type_counts'],sum(x['likelihood_type_counts'] for x in parts))
    assert np.array_equal(full['joint_count'],sum(x['joint_count'] for x in parts))
    assert np.array_equal(values[0,:90],repeated[0])
    sources['preflight_diversity_after_correlation_math.py']=sha(__file__)
    result=dict(status='passed',study_scores_read=False,proposal_sha256=sha(HERE/'diversity-after-correlation-proposal.json'),
        immutable_method_receipts=inherited,fullshape_checks=checks,literal_shared_first90=True,q1_three_block_integer_count_additivity=True,
        digit_permutation_and_row_order_invariance=True,source_sha256=sources,wall_seconds=time.perf_counter()-start)
    save(target,result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256']}))


if __name__=='__main__':main()
