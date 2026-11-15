"""Toy-only sampling and endpoint gates before actual H4am draws."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import ast,itertools,json,time,statistics
import numpy as np
from cycle_power_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,CONDITIONS,bind,corrupt,sha,save
from cycle_power_sampling import graph_order,graph_edges,sample
from audit_cycle_power_controls import independent_graph,independent_colorings
from audit_cycle_power_lp import aggregate,literal_curvature
from cycle_power_lp import curvature
from audit_regularity_intensity import literal_corruption


def main():
    path=HERE/'cycle-power-preflight.json';assert not path.exists() and not PROTOCOL.exists() and not OUT.exists()
    start=time.monotonic();module=bind()
    assert module.OUT==OUT and module.PUBLIC==PUBLIC and module.PROTOCOL==PROTOCOL and module.CONDITIONS==CONDITIONS
    # Every open path-power coloring is equally likely; enumerate toy support.
    uniform_cases=[]
    for n,k,q in [(6,2,3),(7,2,4),(8,3,4),(8,3,5)]:
        pairs=[(i,j) for i in range(n) for j in range(i+1,n) if min(j-i,n-j+i)<=k]
        support={}
        def visit(row):
            if len(row)==n:
                if all(row[i]!=row[j] for i,j in pairs):support[tuple(row)]=support.get(tuple(row),0)+1
                return
            for digit in range(q):
                if digit not in row[-k:]:visit(row+[digit])
        for initial in itertools.permutations(range(q),k):visit(list(initial))
        brute={row for row in itertools.product(range(q),repeat=n) if all(row[i]!=row[j] for i,j in pairs)}
        assert support and set(support)==brute and set(support.values())=={1}
        uniform_cases.append(dict(n=n,k=k,q=q,proper_colorings=len(brute),equal_path_multiplicity=1))
    cases=0
    for family,n,q in [('power2',6,3),('power2',7,4),('power3',8,4),('power3',8,5)]:
        for seed in [19,38,71]:
            order=graph_order(seed,n);truth=graph_edges(order,family,q);reference=independent_graph(order,family,q)
            assert np.array_equal(truth,reference)
            a,record=sample(order,family,seed+700,rows=37,q=q,candidate_cap=10000,batch=16)
            b,expected=independent_colorings(order,family,seed+700,rows=37,q=q,cap=10000,batch=16)
            assert np.array_equal(a,b) and record==expected
            ii,jj=np.triu_indices(n,1);degree=np.zeros(n,np.int64);np.add.at(degree,ii,truth);np.add.at(degree,jj,truth)
            assert (degree==2*int(family[-1])).all() and (a[:,ii[truth]]!=a[:,jj[truth]]).all()
            cases+=1
    failures=0
    for sampler in [lambda:sample(np.arange(7),'power2',2,rows=1,q=3,candidate_cap=16,batch=8),
                    lambda:independent_colorings(np.arange(7),'power2',2,rows=1,q=3,cap=16,batch=8)]:
        try:sampler()
        except RuntimeError as error:assert str(error)=='proper_coloring_candidate_cap';failures+=1
    assert failures==2
    # Measure the lookup optimization on unrelated toy inputs; include table build.
    from cycle_power_sampling import allowed_table
    allowed_table.cache_clear();measurements=[];reference_times=[]
    for trial in range(3):
        order=np.arange(15);seed=7281+trial
        start_a=time.monotonic();a,ar=sample(order,'power2',seed,rows=512,q=5,candidate_cap=5000,batch=128);measurements.append(time.monotonic()-start_a)
        start_b=time.monotonic();b,br=independent_colorings(order,'power2',seed,rows=512,q=5,cap=5000,batch=128);reference_times.append(time.monotonic()-start_b)
        assert np.array_equal(a,b) and ar==br
    rng=np.random.default_rng(4187);labels=rng.integers(0,9,(31,81),dtype=np.uint8);free=np.ones_like(labels,bool)
    uniforms=1-rng.random((31,81));offsets=rng.integers(1,9,(31,81),dtype=np.int64)
    for hot in [np.arange(18),np.arange(63,81)]:
        a,c=corrupt(np,labels,free,uniforms,offsets,hot,'hot18_weight32');b,d=literal_corruption(labels,free,uniforms,offsets,hot,'hot18_weight32')
        assert np.array_equal(a,b) and np.array_equal(c,d) and ((a!=labels).sum(1)==9).all()
    endpoint_cases=[]
    for successes,unresolved,expected in [(61,0,'supported'),(60,0,'falsified'),(60,1,'inconclusive'),(0,3,'falsified')]:
        rows=[]
        for family in CONDITIONS:
            true_degree=4 if family=='power2' else 6
            for i in range(64):
                cls='success' if i<successes else 'unresolved' if i<successes+unresolved else 'resolved_failure_fractional'
                rows.append(dict(condition=family,classification=cls,true_degree_selected=cls=='success',selected_degree=true_degree if cls=='success' else None if cls=='unresolved' else 1,missing_true=0 if cls=='success' else None,extra=0 if cls=='success' else None))
        _,decisions=aggregate(rows)
        assert all(d['graph']['decision']==d['degree']['decision']==expected for d in decisions.values())
        endpoint_cases.append(dict(successes=successes,unresolved=unresolved,expected=expected))
    for values in [np.arange(41,dtype=np.int64),np.arange(41,dtype=np.int64)**2,np.maximum(np.arange(41,dtype=np.int64)-8,0)]:
        a,c=curvature(values);b,d=literal_curvature(values);assert a==b and np.array_equal(c,d)
    jobs=[(g,f) for g in range(64) for f in CONDITIONS]
    assert len(jobs)==128 and all(jobs[i]==(i//2,CONDITIONS[i%2]) for i in range(128))
    assert sorted(i for w in range(8) for i in range(w,128,8))==list(range(128))
    dependencies=set()
    for name in ['varying_degree.json','degree_curvature.json']:
        p=json.loads((HERE/name).read_text())
        for rel,digest in p['source_sha256'].items():assert sha(ROOT/rel)==digest;dependencies.add(rel)
    dependencies.update(str((HERE/name).relative_to(ROOT)) for name in ['cycle_power_sampling.py','cycle_power_study.py','cycle_power_lp.py','audit_cycle_power_controls.py','audit_cycle_power_lp.py','preflight_cycle_power.py'])
    for rel in dependencies:ast.parse((ROOT/rel).read_text())
    result=dict(status='passed',no_actual_study_draws=True,toy_uniform_support=uniform_cases,toy_sampler_replays=cases,candidate_cap_failure_cases=failures,corruption_cases=2,worker_binding=True,job_ordering=128,endpoint_cases=endpoint_cases,source_sha256={rel:sha(ROOT/rel) for rel in sorted(dependencies)},toy_lookup_seconds=measurements,toy_literal_seconds=reference_times,toy_median_speed_ratio=statistics.median(reference_times)/statistics.median(measurements),timing_scope='Unrelated toy15vertex q5 k2 generation,three trials,lookup build included firsttrial; no actual study outcomes or timing guarantee.',wall_seconds=time.monotonic()-start)
    save(path,result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
