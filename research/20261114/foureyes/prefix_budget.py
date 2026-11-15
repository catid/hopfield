"""H4an: unchanged LP inference across synthetic graph families."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,resource,signal,subprocess,sys,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from weak_weight_study import ROOT,HERE,RIPPER,sha,save
sys.path.insert(0,str(RIPPER))
OUT=ROOT/'artifacts/research/20261114/foureyes/prefix_budget/degree'
PUBLIC=ROOT/'results/research/foureyes/prefix_budget/degree'
PROPOSAL=HERE/'prefix-budget-proposal.json'
FAMILIES=['cycle81','power2','power3','cliques9']
SIZES=[300,600,1200]
PROTOCOL=HERE/'prefix_budget.json'
PYTHON=ROOT/'artifacts/degree-lp-venv/bin/python'
CONDITIONS=[f'{f}-n{n}' for f in FAMILIES for n in SIZES]


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


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate_path=HERE/'prefix-budget-preflight.json';gate=json.loads(gate_path.read_text());assert gate['status']=='passed'
    for rel,h in gate['source_sha256'].items():assert sha(ROOT/rel)==h
    design=json.loads(PROPOSAL.read_text());parent=json.loads((HERE/'cycle_power.json').read_text())
    for rel,h in design['scientific_inputs_sha256'].items():assert sha(ROOT/rel)==h
    manifests={name:json.loads((ROOT/f'artifacts/research/20261114/foureyes/{name}/inputs.json').read_text()) for name in ['varying_degree','cycle_power']}
    jobs=[]
    for group in range(64):
        for fi,family in enumerate(FAMILIES):
            source=design['families'][fi];index=2*group+source['parent_condition_index'];entry=manifests[source['parent']]['entries'][index]
            assert entry['group']==group and entry['condition']==family
            for ni,n in enumerate(SIZES):jobs.append(dict(index=12*group+3*fi+ni,group=group,family=family,rows=n,condition=CONDITIONS[3*fi+ni],parent=source['parent'],parent_index=index,source_path=entry['path'],source_sha256=entry['sha256']))
    p=dict(id=design['id'],frozen_utc=datetime.now(timezone.utc).isoformat(),scope=design['scope'],jobs=jobs,conditions=CONDITIONS,families=FAMILIES,prefixes=SIZES,
        primary=design['primary'],decision=design['decision'],controls=design['controls'],budget_scope=design['budget_scope'],lp_budget=design['budget'],
        degrees=parent['degrees'],candidate_degrees=parent['candidate_degrees'],selector=parent['selector'],prior='Common degree and bounded0..40 grid; no true degree2/4/6/8/20,edgecount,family,order or geometry supplied to inference.',objective=parent['objective'],proof=parent['proof'],solver=dict(parent['solver'],random_seed_base=2026231502),
        source_sha256=gate['source_sha256'],input_sha256=design['scientific_inputs_sha256']|{str(f.relative_to(ROOT)):sha(f) for f in [PROPOSAL,gate_path]})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),grids=len(jobs))))


def prefix_counts(labels,n):
    ii,jj=np.triu_indices(81,1)
    return (labels[:n,ii]==labels[:n,jj]).sum(0,dtype=np.int64)


def worker(worker_id):
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate
    from centered_regularity_study import center
    from regularity_gap import select
    p=json.loads(PROTOCOL.read_text());verify(p);ii,jj=np.triu_indices(81,1)
    cached=None;labels=None
    for index in [12*g+k for g in range(worker_id,64,p['lp_budget']['workers']) for k in range(12)]:
        start=time.monotonic();job=p['jobs'][index];assert job['index']==index
        path=ROOT/job['source_path']
        if cached!=str(path):
            assert sha(path)==job['source_sha256']
            with np.load(path,allow_pickle=False) as z:labels=z['noisy'].copy()
            cached=str(path)
        assert labels.shape==(2400,81) and labels.dtype==np.uint8 and ((labels>=0)&(labels<9)).all()
        cost=prefix_counts(labels,job['rows']);d=OUT/f'j{index:03d}';d.mkdir(exist_ok=False);arrays=dict(cost=cost);records=[]
        arrays['centered_support'],center_record=select(center(cost)[0]);arrays['raw_support'],raw_record=select(cost)
        values=np.unique(cost);eligible=np.flatnonzero(values[1:]<=np.median(cost));k=int(eligible[np.argmax(np.diff(values)[eligible])]) if len(eligible) else None
        cut=int((values[k]+values[k+1])//2) if k is not None else 0;arrays['restricted_support']=cost<=cut
        arrays.update({f'd00_{key}':value for key,value in zero_witness(81,cost).items()})
        for degree in range(1,41):
            x,y,solver=solve(81,degree,ii,jj,cost,120000+index*41+degree,d/f'd{degree:02d}.log')
            if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:cert,w=attempt_certificate(81,degree,ii,jj,cost,x,y)
            else:cert=dict(certified=False,reason='solver_status_or_validity');w={key:np.empty(0,np.int64) for key in ['primal_twice','node_dual_twice','upper_dual_twice']}
            arrays.update({f'd{degree:02d}_{key}':value for key,value in dict(primal_values=x,node_dual_values=y,**w).items()})
            records.append(dict(degree=degree,solver=solver,certificate=cert,log_sha256=sha(d/f'd{degree:02d}.log')))
        save(d/'progress.json',dict(index=index,complete_degrees=40,pid=os.getpid()))
        np.savez_compressed(d/'models.npz',**arrays)
        save(d/'receipt.json',dict(index=index,group=job['group'],condition=job['condition'],family=job['family'],rows=job['rows'],source_path=job['source_path'],source_sha256=job['source_sha256'],protocol_sha256=sha(PROTOCOL),records=records,models_sha256=sha(d/'models.npz'),centered_selection=center_record['selected'],raw_selection=raw_record['selected'],restricted_cut=cut,wall_seconds=time.monotonic()-start))
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
        samples.append(sample);assert sample['wall_seconds']<p['lp_budget']['wall_seconds'] and sample['cpu_seconds']<p['lp_budget']['aggregate_cpu_seconds'],'runtime_ceiling'
    try:
        for worker_id in range(8):
            f=(OUT/f'worker{worker_id}.log').open('x');logs.append(f)
            children.append(subprocess.Popen([str(PYTHON),str(Path(__file__)),'worker','--worker',str(worker_id)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
        while any(c.poll() is None for c in children):
            monitor('LP');assert all(c.poll() in [None,0] for c in children),'worker_failure';time.sleep(.2)
        assert all(c.returncode==0 for c in children);monitor('LP')
        previous=len(children)
        for worker_id in range(8):
            f=(OUT/f'audit{worker_id}.log').open('x');logs.append(f)
            children.append(subprocess.Popen([str(PYTHON),str(HERE/'audit_prefix_budget.py'),'--worker',str(worker_id)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
        while any(c.poll() is None for c in children[previous:]):
            monitor('audit');assert all(c.poll() in [None,0] for c in children[previous:]),'audit_worker_failure';time.sleep(.2)
        assert all(c.returncode==0 for c in children[previous:]);monitor('audit')
        f=(OUT/'export.log').open('x');logs.append(f)
        children.append(subprocess.Popen([str(PYTHON),str(HERE/'audit_prefix_budget.py'),'--export'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
        while children[-1].poll() is None:monitor('export');time.sleep(.2)
        assert children[-1].returncode==0,'export_failure';monitor('export');status='complete'
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
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','launch','worker']);ap.add_argument('--worker',type=int);args=ap.parse_args()
    if args.action=='worker':worker(args.worker)
    elif args.action=='freeze':freeze()
    else:launch()
