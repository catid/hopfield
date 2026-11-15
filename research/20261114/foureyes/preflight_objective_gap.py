"""Toy strict/degenerate/fractional witnesses and frozen endpoint boundaries."""
import json,time
import numpy as np
from objective_gap import ROOT,HERE,evaluate,classify,decision,sha,save
from audit_objective_gap import certify
start=time.perf_counter();cases=[]
for n,fractional in [(4,False),(6,True)]:
    ii,jj=np.triu_indices(n,1);t=np.zeros(len(ii),np.int64);truth=np.zeros(len(ii),bool)
    if fractional:t=np.array([int(a//3==b//3) for a,b in zip(ii,jj)],np.int64);truth=(jj-ii)==3
    else:t=np.array([2*int((a,b) in [(0,1),(2,3)]) for a,b in zip(ii,jj)],np.int64);truth=np.array([(a,b) in [(0,2),(1,3)] for a,b in zip(ii,jj)],bool)
    for tied in [False,True]:
        c=np.zeros(len(ii),np.int64) if tied else (t==0).astype(np.int64);y=np.zeros(n,np.int64);z=np.zeros(len(ii),np.int64)
        r=evaluate(n,1,c,t,y,z,truth);assert certify(n,1,c,t,y,z)==0 and r['gap_twice']==(0 if tied else n);cases.append(dict(n=n,fractional=fractional,tied=tied,result=r))
        for which in range(3):
            bad=[t.copy(),y.copy(),z.copy()];bad[which][0]+=1
            for checker in [lambda: evaluate(n,1,c,*bad,truth),lambda: certify(n,1,c,*bad)]:
                try:checker()
                except AssertionError:pass
                else:raise AssertionError('Mutation survived')
try:classify(0,1)
except AssertionError:pass
else:raise AssertionError('Negative gap accepted')
assert decision(354)=='falsified' and decision(355)=='supported' and decision(0)=='falsified' and decision(394)=='supported'
paths=[HERE/x for x in ['objective_gap.py','audit_objective_gap.py','preflight_objective_gap.py']]+[HERE.parent/'ripper/audit_degree_prior.py']
save(HERE/'objective-gap-preflight.json',dict(status='passed',cases=cases,mutations_rejected=24,negative_gap_rejected=True,endpoint_boundaries=[0,354,355,394],actual_gap_outcomes_read=False,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},wall_seconds=time.perf_counter()-start))
print('TOY_GATES_PASSED')
