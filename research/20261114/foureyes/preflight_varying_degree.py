"""Toy-only sampling and endpoint gates before actual H4al draws."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import ast,itertools,json,time
import numpy as np
from varying_degree_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,CONDITIONS,bind,corrupt,sha,save
from varying_degree_sampling import graph_order,graph_edges,sample
from audit_varying_degree_controls import independent_graph,independent_colorings
from audit_varying_degree_lp import aggregate,literal_curvature
from varying_degree_lp import curvature
from audit_regularity_intensity import literal_corruption


def main():
    path=HERE/'varying-degree-preflight.json';assert not path.exists() and not PROTOCOL.exists() and not OUT.exists()
    start=time.monotonic();module=bind()
    assert module.OUT==OUT and module.PUBLIC==PUBLIC and module.PROTOCOL==PROTOCOL and module.CONDITIONS==CONDITIONS
    # Enumerate every equiprobable path on a toy 5-cycle, q=3. Each accepted
    # proper coloring must occur exactly once, proving conditional uniformity.
    n=5;q=3;support={}
    for first in range(q):
        for steps in itertools.product(range(1,q),repeat=n-1):
            row=[first]
            for step in steps:row.append((row[-1]+step)%q)
            if row[-1]!=row[0]:support[tuple(row)]=support.get(tuple(row),0)+1
    brute={row for row in itertools.product(range(q),repeat=n) if all(row[i]!=row[(i+1)%n] for i in range(n))}
    assert set(support)==brute and set(support.values())=={1} and len(brute)==30
    cases=0
    for family,n,q in [('cycle5',5,3),('cycle7',7,4),('cliques3',9,3),('cliques4',12,4)]:
        for seed in [19,38,71]:
            order=graph_order(seed,n);truth=graph_edges(order,family,q);reference=independent_graph(order,family,q)
            assert np.array_equal(truth,reference)
            a,record=sample(order,family,seed+700,rows=37,q=q,candidate_cap=1000,batch=16)
            b,expected=independent_colorings(order,family,seed+700,rows=37,q=q,cap=1000,batch=16)
            assert np.array_equal(a,b) and record==expected
            ii,jj=np.triu_indices(n,1);degree=np.zeros(n,np.int64);np.add.at(degree,ii,truth);np.add.at(degree,jj,truth)
            assert (degree==(2 if family.startswith('cycle') else q-1)).all()
            assert (a[:,ii[truth]]!=a[:,jj[truth]]).all()
            if family.startswith('cliques'):
                assert all(((a==digit).sum(1)==n//q).all() for digit in range(q))
            cases+=1
    failures=0
    for sampler in [lambda:sample(np.arange(5),'cycle5',2,rows=1,q=2,candidate_cap=16,batch=8),
                    lambda:independent_colorings(np.arange(5),'cycle5',2,rows=1,q=2,cap=16,batch=8)]:
        try:sampler()
        except RuntimeError as error:assert str(error)=='proper_coloring_candidate_cap';failures+=1
    assert failures==2
    rng=np.random.default_rng(4187);labels=rng.integers(0,9,(31,81),dtype=np.uint8);free=np.ones_like(labels,bool)
    uniforms=1-rng.random((31,81));offsets=rng.integers(1,9,(31,81),dtype=np.int64)
    for hot in [np.arange(18),np.arange(63,81)]:
        a,c=corrupt(np,labels,free,uniforms,offsets,hot,'hot18_weight32');b,d=literal_corruption(labels,free,uniforms,offsets,hot,'hot18_weight32')
        assert np.array_equal(a,b) and np.array_equal(c,d) and ((a!=labels).sum(1)==9).all()
    endpoint_cases=[]
    for successes,unresolved,expected in [(61,0,'supported'),(60,0,'falsified'),(60,1,'inconclusive'),(0,3,'falsified')]:
        rows=[]
        for family in CONDITIONS:
            true_degree=2 if family=='cycle81' else 8
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
    for name in ['fresh_curvature.json','degree_curvature.json']:
        p=json.loads((HERE/name).read_text())
        for rel,digest in p['source_sha256'].items():assert sha(ROOT/rel)==digest;dependencies.add(rel)
    dependencies.update(str((HERE/name).relative_to(ROOT)) for name in ['varying_degree_sampling.py','varying_degree_study.py','varying_degree_lp.py','audit_varying_degree_controls.py','audit_varying_degree_lp.py','preflight_varying_degree.py'])
    for rel in dependencies:ast.parse((ROOT/rel).read_text())
    result=dict(status='passed',no_actual_study_draws=True,toy_uniform_support=30,toy_sampler_replays=cases,candidate_cap_failure_cases=failures,corruption_cases=2,worker_binding=True,job_ordering=128,endpoint_cases=endpoint_cases,source_sha256={rel:sha(ROOT/rel) for rel in sorted(dependencies)},wall_seconds=time.monotonic()-start)
    save(path,result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
