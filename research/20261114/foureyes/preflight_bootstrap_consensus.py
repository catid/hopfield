"""H4ap toy count/proof/decision gates and frozen-registry seed audit."""
import itertools,json,time
import numpy as np
from bootstrap_consensus import ROOT,HERE,RIPPER,SEED,SOLVER_BASE,fast_counts,decision,sha,read,save
from audit_bootstrap_consensus import reference_counts,reference_decisions
from audit_degree_prior import check_certificate
from degree_prior_math import attempt_certificate


def main():
    start=time.monotonic();rng=np.random.default_rng(937107);labels=rng.integers(0,9,(600,81),dtype=np.int8);draws=rng.integers(0,600,(32,600),dtype=np.int16);draws[0]=0;draws[1]=599;draws[2]=np.arange(600,dtype=np.int16)
    a=fast_counts(labels,draws);b=reference_counts(labels,draws);assert np.array_equal(a,b)
    # Maximal multiplicity, identical labels and forced inequality remain exact.
    constant=np.zeros((600,81),np.int8);assert (fast_counts(constant,draws)==600).all()
    constant[:,1]=1;result=fast_counts(constant,draws);assert (result[:,0]==0).all() and (result[:,1:80]==600).all()
    boundary=[]
    for good,unknown,loss,oldunknown,expected in [(145,0,0,0,('supported','supported')),(144,0,0,0,('falsified','supported')),(144,1,8,1,('inconclusive','inconclusive')),(145,1,9,0,('supported','falsified')),(0,256,0,129,('inconclusive','inconclusive'))]:
        d=decision(good,unknown,loss,oldunknown);assert (d['recovery']['decision'],d['paired_loss']['decision'])==expected;boundary.append(dict(success=good,unresolved=unknown,losses=loss,unresolved_old=oldunknown,expected=list(expected)))
    # Independent aggregate endpoint path on synthetic rows.
    rows=[dict(baseline_exact=i<129,classification='success' if i<145 else 'resolved_failure_nonexact') for i in range(256)];assert reference_decisions(rows)==decision(145,0,0,0)
    rows[0]['classification']='unresolved';assert reference_decisions(rows)==decision(144,1,0,1)
    n=6;ii,jj=np.triu_indices(n,1);half=np.array([int(x//3==y//3) for x,y in zip(ii,jj)],np.int64);cost=(half==0).astype(np.int64);zero=np.zeros(n,np.float64)
    cert,w=attempt_certificate(n,1,ii,jj,cost,half/2,zero);assert cert['certified'] and not cert['integral'] and check_certificate(n,1,ii,jj,cost,*[w[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']])
    votes=np.stack([half]*32).sum(0);consensus=64-votes;assert set(votes)=={0,32} and set(consensus)=={32,64}
    # A fractional optimum is used as half-votes; it is never rounded to a graph.
    invalid=half/2;invalid=invalid.copy();invalid[0]+=.01;bad,_=attempt_certificate(n,1,ii,jj,cost,invalid,zero);assert not bad['certified']
    old={};files={}
    def walk(v,context,rel):
        if isinstance(v,dict):
            for k,value in v.items():walk(value,context+'/'+k,rel)
        elif isinstance(v,list):
            for value in v:walk(value,context,rel)
        elif isinstance(v,int) and not isinstance(v,bool) and 'seed' in context.lower():old.setdefault(v,[]).append(rel+':'+context)
    for path in sorted((ROOT/'research').rglob('*.json')):
        if path.name.startswith('bootstrap-consensus'):continue
        try:obj=read(path)
        except (ValueError,UnicodeError):continue
        if not isinstance(obj,dict) or 'frozen_utc' not in obj:continue
        rel=str(path.relative_to(ROOT));files[rel]=sha(path);walk(obj,'',rel)
    boots=list(range(SEED,SEED+8192));solvers=list(range(SOLVER_BASE,SOLVER_BASE+8448));new=boots+solvers;collisions={str(v):old[v] for v in new if v in old};assert not collisions and len(new)==len(set(new)) and max(solvers)<2**31
    seedgate=dict(scope='All explicit integer seed-valued fields in research JSON objects with frozen_utc. Not undocumented streams or implied formula expansions.',bootstrap_range=[min(boots),max(boots)],solver_range=[min(solvers),max(solvers)],unique_bootstrap_streams=8192,unique_solver_seeds=8448,historical_explicit_seed_count=len(old),collisions=collisions,frozen_protocol_sha256=files)
    save(HERE/'bootstrap-consensus-seed-audit.json',seedgate)
    sources=[HERE/name for name in ['bootstrap_consensus.py','audit_bootstrap_consensus.py','preflight_bootstrap_consensus.py','audit_objective_gap.py']]+[RIPPER/name for name in ['degree_prior_solver.py','degree_prior_math.py','audit_degree_prior.py']]
    benchmark=read(ROOT/'artifacts/research/20261114/foureyes/bootstrap-consensus-count-toy-v2.json')
    save(HERE/'bootstrap-consensus-preflight.json',dict(status='passed',actual_bootstrap_outcomes_read=False,toy_seed=937107,all32_resampled_counts_match=True,extreme_multiplicities_and_zero600_counts=True,fractional_votes_exact_and_no_rounding=True,uncertified_primal_rejected=True,endpoint_cases=boundary,seed_audit_sha256=sha(HERE/'bootstrap-consensus-seed-audit.json'),count_benchmark=benchmark,source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},wall_seconds=time.monotonic()-start))
    print('PREFLIGHT_PASSED',len(files),'FROZEN_PROTOCOLS',len(old),'HISTORIC_EXPLICIT_SEEDS',flush=True)
if __name__=='__main__':main()
