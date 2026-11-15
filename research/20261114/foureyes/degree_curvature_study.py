"""H4aj: infer bounded common degree from exact LP value curvature."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,resource,signal,subprocess,sys,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from weak_weight_study import ROOT,HERE,RIPPER,sha,save
sys.path.insert(0,str(RIPPER))
OUT=ROOT/'artifacts/research/20261114/foureyes/degree_curvature'
PUBLIC=ROOT/'results/research/foureyes/degree_curvature'
PARENT=ROOT/'results/research/foureyes/regularity_intensity'
PROTOCOL=HERE/'degree_curvature.json'
PYTHON=ROOT/'artifacts/degree-lp-venv/bin/python'
CONDITIONS=['hot18_weight8','hot18_weight32']


def curvature(values):
    values=np.asarray(values);assert values.dtype==np.int64 and values.ndim==1 and len(values)>=3
    second=values[2:]-2*values[1:-1]+values[:-2]
    selected=int(np.argmax(second))+1
    return selected,second


def zero_witness(n,cost):
    assert cost.dtype==np.int64 and (cost>=0).all()
    return dict(primal_twice=np.zeros(len(cost),np.int64),node_dual_twice=np.zeros(n,np.int64),upper_dual_twice=np.zeros(len(cost),np.int64))


def verify(p):
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel


def preflight():
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate
    from audit_degree_prior import check_certificate
    from audit_degree_curvature import literal_curvature
    path=HERE/'degree-curvature-preflight.json';assert not path.exists();folder=OUT.parent/'degree_curvature_preflight';folder.mkdir(exist_ok=False)
    start=time.monotonic();results=[]
    for case,n in enumerate([4,5]):
        ii,jj=np.triu_indices(n,1);cost=np.ones(len(ii),np.int64)
        if n==4:cost=np.where((jj-ii==1)|((ii==0)&(jj==3)),1,10).astype(np.int64)
        b=np.arange(n,dtype=np.int64)+1;allvalues=[];selected_integral=[]
        for variant,weights in enumerate([cost,cost+b[ii]+b[jj]]):
            z=zero_witness(n,weights);assert check_certificate(n,0,ii,jj,weights,**dict(t=z['primal_twice'],y=z['node_dual_twice'],z=z['upper_dual_twice']))
            values=[0];integral=[True]
            for degree in range(1,n):
                x,y,solver=solve(n,degree,ii,jj,weights,30000+case*100+variant*10+degree,folder/f'{case}-{variant}-{degree}.log')
                cert,w=attempt_certificate(n,degree,ii,jj,weights,x,y)
                assert solver['solver_optimal'] and cert['certified'] and check_certificate(n,degree,ii,jj,weights,w['primal_twice'],w['node_dual_twice'],w['upper_dual_twice'])
                assert solver['solver_objective']==cert['primal_objective_twice']/2
                values.append(cert['primal_objective_twice']);integral.append(cert['integral'])
            values=np.array(values,np.int64);chosen,second=curvature(values);ref,expected=literal_curvature(values)
            assert chosen==ref and np.array_equal(second,expected) and (second>=0).all()
            allvalues.append(values);selected_integral.append(integral[chosen])
        assert np.array_equal(allvalues[1]-allvalues[0],2*np.arange(n,dtype=np.int64)*b.sum())
        assert np.array_equal(curvature(allvalues[0])[1],curvature(allvalues[1])[1])
        assert curvature(allvalues[0])[0]==(2 if n==4 else 1)
        assert selected_integral==([True,True] if n==4 else [False,False])
        results.append(dict(n=n,objective_twice=allvalues[0].tolist(),selected_degree=curvature(allvalues[0])[0],selected_integral=selected_integral[0],additive_curvature_invariance=True))
    files=[Path(__file__),HERE/'audit_degree_curvature.py',HERE/'weak_weight_study.py',RIPPER/'degree_prior_solver.py',RIPPER/'degree_prior_math.py',RIPPER/'audit_degree_prior.py']
    save(path,dict(status='passed',cases=results,wall_seconds=time.monotonic()-start,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files}))
    print(json.dumps(dict(status='passed',cases=results)))


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate=HERE/'degree-curvature-preflight.json';g=json.loads(gate.read_text());assert g['status']=='passed'
    for rel,digest in g['source_sha256'].items():assert sha(ROOT/rel)==digest
    inputs=[gate,HERE/'regularity_intensity.json',PARENT/'models.npz',PARENT/'summary.json',HERE/'intensity_degree.json',ROOT/'results/research/foureyes/intensity_degree/summary.json',
            HERE/'centered_regularity.json',ROOT/'results/research/foureyes/centered_regularity/summary.json']
    p=dict(id='H4aj-degree-from-LP-curvature',frozen_utc=datetime.now(timezone.utc).isoformat(),groups=64,conditions=CONDITIONS,degrees=list(range(41)),candidate_degrees=list(range(1,40)),
           scope='Adaptive follow-up on all128 examined H4ag rawcount matrices, after H4ah/H4ai results. No new sources, corruptions, labels or puzzle inputs. Fresh LPs for every degree including20.',
           prior='Common degree and a bounded grid0..40 on81nodes. No degree20, edge count810, row/column/box geometry or preferred-degree tie rule supplied.',
           objective='All3240 pairs eligible; minimize raw equality_count*x with0<=x<=1 and each node degree d. Same deterministic solver/proof rule at every d.',
           selector='Exact doubled optimal values Vd=2F(d), analytic V0=0. For d1..39 maximize V[d+1]-2V[d]+V[d-1]; ties smallest d, including zero curvature. No filtering odd degrees.',
           primary='Weight32 at least61/64 certified integral exact selected graphs.',
           decision='Exact_lower>=61 supported; exact_lower+unresolved<61 falsified; otherwise inconclusive. Any uncertified grid degree makes that bank unresolved. Selected certified fractional/nonexact is a procedure failure; intermediate fractional grid solutions remain valid objective evidence. Whole resource failure incomplete.',
           diagnostics='All41 objectives,39 curvatures, all minimum-degree ties, selected degree/primal, intermediate fractional counts, missing/extras for integral selections only. Controls archived BP first, fixeddegree20, centering and thresholds. Weight8 same threshold descriptive.',
           algebra='Any additive cell cost bi+bj changes F(d) by d*sum(b), so exact second differences are unchanged. This is not an assertion that the knee identifies a universal true degree.',
           proof='Strict inherited half-integral primal/dual proof, no primal rounding; exact complementarity and objective equality. Analytic d0 primal/dual allzero valid because costs nonnegative. No uniqueness claim.',
           solver=dict(highspy='1.12.0',numpy='2.3.4',threads=1,parallel='off',solver='simplex',simplex_strategy=1,presolve='off',time_limit=1.,random_seed_base=2026141502,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9),
           seed_rule='Seed = base + input_index*41 + d; independent new solver instance every positive degree, increasing d execution order.',
           budget=dict(workers=8,new_LPs=5120,aggregate_cpu_seconds=900,wall_seconds=180,lp_wall_seconds=1.,gpu_seconds=0),
           budget_scope='All LPs, worker certificates, independent all-proof replay, selection and export monitored together. Synthetic preflight and later publication separate.',
           source_sha256=g['source_sha256'],input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def worker(worker_id):
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate
    p=json.loads(PROTOCOL.read_text());verify(p)
    with np.load(PARENT/'models.npz',allow_pickle=False) as z:counts=z['counts'].copy()
    ii,jj=np.triu_indices(81,1)
    for index in range(worker_id,128,p['budget']['workers']):
        start=time.monotonic();d=OUT/f'j{index:03d}';d.mkdir(exist_ok=False);cost=counts[index];arrays=dict(cost=cost);records=[]
        assert cost.dtype==np.int64 and cost.shape==(3240,) and ((cost>=0)&(cost<=2400)).all()
        arrays.update({f'd00_{key}':value for key,value in zero_witness(81,cost).items()})
        for degree in range(1,41):
            x,y,solver=solve(81,degree,ii,jj,cost,30000+index*41+degree,d/f'd{degree:02d}.log')
            if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:cert,w=attempt_certificate(81,degree,ii,jj,cost,x,y)
            else:cert=dict(certified=False,reason='solver_status_or_validity');w={key:np.empty(0,np.int64) for key in ['primal_twice','node_dual_twice','upper_dual_twice']}
            arrays.update({f'd{degree:02d}_{key}':value for key,value in dict(primal_values=x,node_dual_values=y,**w).items()})
            records.append(dict(degree=degree,solver=solver,certificate=cert,log_sha256=sha(d/f'd{degree:02d}.log')))
            save(d/'progress.json',dict(index=index,complete_degrees=degree,pid=os.getpid()))
        np.savez_compressed(d/'models.npz',**arrays)
        save(d/'receipt.json',dict(index=index,group=index//2,condition=CONDITIONS[index%2],protocol_sha256=sha(PROTOCOL),records=records,models_sha256=sha(d/'models.npz'),wall_seconds=time.monotonic()-start))
        print(json.dumps(dict(index=index,complete=True)),flush=True)


def launch():
    p=json.loads(PROTOCOL.read_text());verify(p);assert not OUT.exists();OUT.mkdir(parents=True)
    start=time.monotonic();children=[];logs=[];samples=[];status='failed';error=None;tick=os.sysconf('SC_CLK_TCK')
    def cpu_seconds():
        total=sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)])
        for child in children:
            if child.poll() is None:
                try:
                    a=Path(f'/proc/{child.pid}/stat').read_text().split();total+=(int(a[13])+int(a[14]))/tick
                except FileNotFoundError:pass
        return total
    baseline=cpu_seconds()
    def monitor(phase):
        partial=[]
        for f in OUT.glob('j*/progress.json'):
            try:partial.append(json.loads(f.read_text())['complete_degrees'])
            except FileNotFoundError:pass
        sample=dict(phase=phase,wall_seconds=time.monotonic()-start,cpu_seconds=cpu_seconds()-baseline,complete_groups=len(list(OUT.glob('j*/receipt.json'))),complete_degrees=sum(partial),active_pids=[c.pid for c in children if c.poll() is None])
        samples.append(sample);assert sample['wall_seconds']<p['budget']['wall_seconds'] and sample['cpu_seconds']<p['budget']['aggregate_cpu_seconds'],'runtime_ceiling'
    try:
        for worker_id in range(8):
            f=(OUT/f'worker{worker_id}.log').open('x');logs.append(f)
            children.append(subprocess.Popen([str(PYTHON),str(Path(__file__)),'worker','--worker',str(worker_id)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
        while any(c.poll() is None for c in children):
            monitor('LP');assert all(c.poll() in [None,0] for c in children),'worker_failure';time.sleep(.2)
        assert all(c.returncode==0 for c in children);monitor('LP')
        f=(OUT/'audit.log').open('x');logs.append(f)
        children.append(subprocess.Popen([str(PYTHON),str(HERE/'audit_degree_curvature.py'),'--export'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
        while children[-1].poll() is None:monitor('audit');time.sleep(.2)
        assert children[-1].returncode==0,'audit_failure';monitor('audit');status='complete'
    except BaseException as exc:error=repr(exc);raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for f in logs:f.close()
        r=dict(status=status,error=error,protocol_sha256=sha(PROTOCOL),wall_seconds=time.monotonic()-start,aggregate_cpu_seconds=cpu_seconds()-baseline,returncodes=[c.returncode for c in children],samples=samples)
        save(OUT/'runtime.json',r)
        if PUBLIC.exists():save(PUBLIC/'runtime.json',r)
        print(json.dumps({k:v for k,v in r.items() if k!='samples'}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','freeze','worker','launch']);ap.add_argument('--worker',type=int);a=ap.parse_args()
    if a.action=='worker':worker(a.worker)
    else:globals()[a.action]()
