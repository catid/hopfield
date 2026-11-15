"""Toy-only prefix counts, selectors, worker binding and endpoint gates."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import ast,json,time,statistics
import numpy as np
from prefix_budget import ROOT,HERE,OUT,PUBLIC,PROTOCOL,FAMILIES,SIZES,CONDITIONS,prefix_counts,sha,save
from audit_prefix_budget import reference_counts,aggregate
from regularity_gap import select
from audit_regularity_gap import independent_selection


def main():
    dest=HERE/'prefix-budget-preflight.json';assert not dest.exists() and not PROTOCOL.exists() and not OUT.exists()
    start=time.monotonic();rng=np.random.default_rng(82417);labels=rng.integers(0,9,(1333,81),dtype=np.uint8)
    times=[];references=[]
    for n in SIZES:
        t=time.monotonic();a=prefix_counts(labels,n);times.append(time.monotonic()-t)
        t=time.monotonic();b=reference_counts(labels,n);references.append(time.monotonic()-t)
        assert np.array_equal(a,b) and a.dtype==np.int64 and a.shape==(3240,)
        changed=labels.copy();changed[n:]=(changed[n:]+1)%9
        assert np.array_equal(a,prefix_counts(changed,n)) and np.array_equal(b,reference_counts(changed,n))
        s,r=select(a);ss,rr=independent_selection(b);assert np.array_equal(s,ss)
        for k,v in rr.items():assert r[k]==v
    assert np.array_equal(prefix_counts(labels,600)-prefix_counts(labels,300),reference_counts(labels[300:600],300))
    ownership=[[12*g+k for g in range(w,64,8) for k in range(12)] for w in range(8)]
    assert all(len(v)==96 for v in ownership) and sorted(sum(ownership,[]))==list(range(768))
    def fake(successes,unresolved):
        rows=[]
        for g in range(64):
            for fi,f in enumerate(FAMILIES):
                for ni,n in enumerate(SIZES):
                    cls='success' if g<successes else 'unresolved' if g<successes+unresolved else 'resolved_failure_fractional'
                    rows.append(dict(index=12*g+3*fi+ni,group=g,family=f,rows=n,condition=CONDITIONS[3*fi+ni],classification=cls,selected_degree=2 if cls!='unresolved' else None,true_degree_selected=cls=='success',missing_true=0 if cls=='success' else None,extra=0 if cls=='success' else None,raw_control=dict(exact=False),centered_control=dict(exact=False),restricted_control=dict(exact=False)))
        return rows
    cases=[]
    for successes,unresolved,expected in [(61,0,'supported'),(60,0,'falsified'),(60,1,'inconclusive'),(0,3,'falsified')]:
        _,decisions,nonmonotonic=aggregate(fake(successes,unresolved));assert all(d['decision']==expected for d in decisions.values()) and not nonmonotonic
        cases.append(dict(successes=successes,unresolved=unresolved,expected=expected))
    rows=fake(64,0);rows[1].update(classification='resolved_failure_fractional',true_degree_selected=False,missing_true=None,extra=None)
    _,decisions,nonmonotonic=aggregate(rows);assert nonmonotonic==[dict(group=0,family='cycle81',lower_rows=300,higher_rows=600)] and decisions['cycle81']['lower']==63
    dependencies=set()
    for parent in ['varying_degree','cycle_power']:
        p=json.loads((HERE/(parent+'.json')).read_text())
        for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==h;dependencies.add(rel)
    dependencies.update(str((HERE/name).relative_to(ROOT)) for name in ['prefix_budget.py','audit_prefix_budget.py','preflight_prefix_budget.py','audit_cycle_power_lp.py','audit_cycle_power_controls.py','audit_varying_degree_controls.py'])
    for rel in dependencies:ast.parse((ROOT/rel).read_text())
    result=dict(status='passed',no_actual_study_prefix_outcomes=True,synthetic_count_cases=3,prefix_suffix_invariance=True,incremental_count_identity=True,worker_partition=[len(a) for a in ownership],endpoint_cases=cases,nonmonotonic_failure_retained=True,toy_production_count_seconds=times,toy_independent_count_seconds=references,timing_scope='Three unrelated toy prefixes of1333syntheticrows; no general speedclaim.',source_sha256={rel:sha(ROOT/rel) for rel in sorted(dependencies)},wall_seconds=time.monotonic()-start)
    save(dest,result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
