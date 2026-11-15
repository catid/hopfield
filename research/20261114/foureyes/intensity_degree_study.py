"""H4ah: fixed-degree LP on the unchanged H4ag count matrices."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,resource,signal,subprocess,sys,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from weak_weight_study import ROOT,HERE,RIPPER,sha,save
sys.path.insert(0,str(RIPPER))
OUT=ROOT/'artifacts/research/20261114/foureyes/intensity_degree'
PUBLIC=ROOT/'results/research/foureyes/intensity_degree'
PROTOCOL=HERE/'intensity_degree.json'
PARENT=ROOT/'results/research/foureyes/regularity_intensity'
PYTHON=ROOT/'artifacts/degree-lp-venv/bin/python'
CONDITIONS=['hot18_weight8','hot18_weight32']


def verify(p):
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel


def preflight():
    import itertools,numpy as np
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate
    from audit_degree_prior import check_certificate
    path=HERE/'intensity-degree-preflight.json';assert not path.exists();d=OUT.parent/'intensity_degree_preflight';d.mkdir(exist_ok=False)
    start=time.monotonic();records=[]
    for case,(n,degree) in enumerate([(4,2),(6,1)]):
        ii,jj=np.triu_indices(n,1)
        cost=(np.arange(len(ii),dtype=np.int64)*7)%11 if n==4 else np.where((ii<3)==(jj<3),0,10).astype(np.int64)
        x,y,solver=solve(n,degree,ii,jj,cost,10000+case,d/f'{case}.log')
        cert,w=attempt_certificate(n,degree,ii,jj,cost,x,y)
        assert solver['solver_optimal'] and cert['certified']
        assert check_certificate(n,degree,ii,jj,cost,w['primal_twice'],w['node_dual_twice'],w['upper_dual_twice'])
        altered=0
        for key in ['primal_twice','node_dual_twice','upper_dual_twice']:
            bad={k:v.copy() for k,v in w.items()};bad[key][0]+=1
            assert not check_certificate(n,degree,ii,jj,cost,bad['primal_twice'],bad['node_dual_twice'],bad['upper_dual_twice']);altered+=1
        if n==4:
            feasible=[]
            for bits in itertools.product([0,1],repeat=len(ii)):
                degree_actual=[0]*n
                for i,j,v in zip(ii,jj,bits):degree_actual[i]+=v;degree_actual[j]+=v
                if degree_actual==[degree]*n:feasible.append(sum(int(c)*v for c,v in zip(cost,bits)))
            assert cert['primal_objective_twice']==2*min(feasible)
        else:assert not cert['integral'] and cert['primal_objective_twice']==0
        records.append(dict(n=n,degree=degree,solver=solver,certificate=cert,mutations=altered))
    files=[Path(__file__),HERE/'audit_intensity_degree.py',RIPPER/'degree_prior_solver.py',RIPPER/'degree_prior_math.py',RIPPER/'audit_degree_prior.py']
    save(path,dict(status='passed',cases=records,wall_seconds=time.monotonic()-start,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files}))
    print(json.dumps(dict(status='passed',cases=2,mutation_rejections=6)))


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    gate=HERE/'intensity-degree-preflight.json';g=json.loads(gate.read_text());assert g['status']=='passed'
    for rel,digest in g['source_sha256'].items():assert sha(ROOT/rel)==digest
    parent=json.loads((PARENT/'summary.json').read_text());assert parent['status']=='validated'
    source=[Path(__file__),HERE/'audit_intensity_degree.py',HERE/'weak_weight_study.py',RIPPER/'degree_prior_solver.py',RIPPER/'degree_prior_math.py',RIPPER/'audit_degree_prior.py']
    inputs=[gate,HERE/'regularity_intensity.json',PARENT/'models.npz',PARENT/'summary.json']
    p=dict(id='H4ah-fixed-degree-intensity-repair',frozen_utc=datetime.now(timezone.utc).isoformat(),groups=64,conditions=CONDITIONS,
           scope='Adaptive follow-up on all128 already examined H4ag count matrices; no new sources, noise, labels, optimizer fits or solver inputs.',
           hypothesis='Per-cell degree constraints can recover strong-noise graphs despite scalar count overlap.',
           primary='Weight32: at least61/64 certified integral exact graphs. Fractional or integral nonexact outputs are resolved failures; uncertified/time/status failure unresolved.',
           decision='Support if exact_lower>=61; falsify if exact_lower+unresolved<61; otherwise inconclusive. Whole-study runtime failure incomplete. No replacement or filtered banks.',
           diagnostics='Weight8 same finite rule as descriptive paired control. Top810 exact same rawcountcost with pairindex ties. Archived BP32 first, canonical256, restrictedgap and equaldegree threshold clearly reused. Missing/extra totals apply only to integral outputs; never label fractional edges absent or present.',
           prior='Supply degree exactly20 at each of81cells, implying810edges. All3240pairs eligible. No row/column/box geometry in optimizer/costs.',
           objective='Minimize sum raw equality_count[e]*x[e],0<=x<=1,each node degree20. Raw integercounts0..2400, no quantization/no noise parameters/no learned weights.',
           ties='Top810 by(rawcount,pairindex). LP frozen deterministic dualsimplex seed2026121502+inputindex; no objectiveperturbation or optimumuniqueness claim.',
           certificate='Inherited strict scale-two primal/dual proof; no primal rounding. Certified fractional result counts procedurefailure and does not exclude an integral co-optimum.',
           solver=dict(highspy='1.12.0',numpy='2.3.4',threads=1,parallel='off',solver='simplex',simplex_strategy=1,presolve='off',time_limit=1.,random_seed_base=2026121502,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9),
           budget=dict(workers=8,aggregate_cpu_seconds=240,wall_seconds=120,lp_wall_seconds=1.,gpu_seconds=0),
           budget_scope='All128LPs,workerproofs,publicaggregation andindependentproof replay monitored together;syntheticpreflight andpublicationfigure preparation separate.',
           source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def worker(worker_id):
    import numpy as np
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate,top_count_support
    from audit_degree_prior import check_certificate
    p=json.loads(PROTOCOL.read_text());verify(p)
    with np.load(PARENT/'models.npz',allow_pickle=False) as z:counts=z['counts'].copy();conditions=z['conditions'].copy();groups=z['groups'].copy()
    ii,jj=np.triu_indices(81,1)
    for index in range(worker_id,128,p['budget']['workers']):
        d=OUT/f'j{index:03d}';d.mkdir(exist_ok=False);start=time.monotonic();cost=counts[index]
        assert cost.dtype==np.int64 and cost.shape==(3240,) and ((cost>=0)&(cost<=2400)).all()
        x,y,solver=solve(81,20,ii,jj,cost,10000+index,d/'highs.log')
        if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:certificate,w=attempt_certificate(81,20,ii,jj,cost,x,y)
        else:
            certificate=dict(certified=False,reason='solver_status_or_solution_validity')
            w={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
        assert check_certificate(81,20,ii,jj,cost,w['primal_twice'],w['node_dual_twice'],w['upper_dual_twice'])==certificate['certified']
        top=top_count_support(np.ones(3240,bool),cost)
        np.savez_compressed(d/'model.npz',cost=cost,primal_values=x,node_dual_values=y,top810_support=top,**w)
        save(d/'receipt.json',dict(index=index,group=int(groups[index]),condition=str(conditions[index]),protocol_sha256=sha(PROTOCOL),
                                  solver=solver,certificate=certificate,model_sha256=sha(d/'model.npz'),log_sha256=sha(d/'highs.log'),wall_seconds=time.monotonic()-start))
        print(json.dumps(dict(index=index,complete=True)),flush=True)


def launch():
    p=json.loads(PROTOCOL.read_text());verify(p);assert not OUT.exists() and not PUBLIC.exists();OUT.mkdir(parents=True)
    start=time.monotonic();workers=[];logs=[];samples=[];status='failed';error=None;tick=os.sysconf('SC_CLK_TCK')
    def cpu_seconds():
        total=sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)])
        for w in workers:
            if w.poll() is None:
                try:
                    v=Path(f'/proc/{w.pid}/stat').read_text().split();total+=(int(v[13])+int(v[14]))/tick
                except FileNotFoundError:pass
        return total
    baseline=cpu_seconds()
    def monitor():
        count=len(list(OUT.glob('j*/receipt.json')));active=[w.pid for w in workers if w.poll() is None]
        sample=dict(elapsed=time.monotonic()-start,cpu_seconds=cpu_seconds()-baseline,completed=count,active_pids=active);samples.append(sample)
        assert sample['elapsed']<p['budget']['wall_seconds'] and sample['cpu_seconds']<p['budget']['aggregate_cpu_seconds'],'runtime_ceiling'
    try:
        for i in range(p['budget']['workers']):
            f=(OUT/f'worker{i}.log').open('x');logs.append(f)
            workers.append(subprocess.Popen([str(PYTHON),str(Path(__file__)),'worker','--worker',str(i)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
        while any(w.poll() is None for w in workers):
            monitor();assert all(w.poll() in [None,0] for w in workers),'worker_failure';time.sleep(.1)
        assert all(w.returncode==0 for w in workers);monitor()
        from audit_intensity_degree import export_and_audit
        export_and_audit();monitor();status='complete'
    except BaseException as exc:error=repr(exc);raise
    finally:
        for w in workers:
            if w.poll() is None:
                os.killpg(w.pid,signal.SIGTERM)
                try:w.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
        for f in logs:f.close()
        result=dict(status=status,error=error,wall_seconds=time.monotonic()-start,aggregate_cpu_seconds=cpu_seconds()-baseline,
                    returncodes=[w.returncode for w in workers],samples=samples,host=os.uname().nodename,protocol_sha256=sha(PROTOCOL))
        save(OUT/'runtime.json',result)
        if PUBLIC.exists():save(PUBLIC/'runtime.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='samples'}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','freeze','worker','launch']);ap.add_argument('--worker',type=int);a=ap.parse_args()
    if a.action=='worker':worker(a.worker)
    else:globals()[a.action]()
