"""H4ai: remove additive cell effects before the unchanged regularity selector."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,resource,signal,subprocess,sys,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from weak_weight_study import ROOT,HERE,sha,save
from regularity_gap import select
OUT=ROOT/'artifacts/research/20261114/foureyes/centered_regularity'
PUBLIC=ROOT/'results/research/foureyes/centered_regularity'
PARENT=ROOT/'results/research/foureyes/regularity_intensity'
PROTOCOL=HERE/'centered_regularity.json'
CONDITIONS=['hot18_weight8','hot18_weight32']
COLUMNS=['cut','retained','degree_min','degree_max','degree_sum','degree_square_sum','dispersion']


def center(counts,n=81):
    counts=np.asarray(counts)
    assert counts.dtype==np.int64 and counts.shape==(n*(n-1)//2,)
    assert abs(counts).max()<2**40 and 2<n<=81
    ii,jj=np.triu_indices(n,1);sums=np.zeros(n,np.int64)
    np.add.at(sums,ii,counts);np.add.at(sums,jj,counts)
    raw=(n-2)*counts-sums[ii]-sums[jj]
    return raw-raw.min(),sums,int(raw.min())


def verify(p):
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel


def preflight():
    from audit_centered_regularity import literal_center,incremental_selection
    path=HERE/'centered-regularity-preflight.json';assert not path.exists();start=time.monotonic();cases=[]
    rng=np.random.default_rng(202611900001)
    for n in [4,11,81]:
        ii,jj=np.triu_indices(n,1)
        for counts in [np.zeros(len(ii),np.int64),rng.integers(0,2401,len(ii),dtype=np.int64)]:
            score,sums,shift=center(counts,n);reference,rs,rm=literal_center(counts,n)
            assert np.array_equal(score,reference) and np.array_equal(sums,rs) and shift==rm
            b=rng.integers(-100,101,n,dtype=np.int64);modified=counts+b[ii]+b[jj]
            changed,_,_=center(modified,n);assert np.array_equal(changed,score)
            # Independent least-squares additive fit, only synthetic arrays.
            design=np.zeros((len(ii),n));design[np.arange(len(ii)),ii]=1;design[np.arange(len(ii)),jj]=1
            fit=np.linalg.lstsq(design,counts.astype(float),rcond=None)[0]
            residual=(n-2)*(counts-design@fit);residual-=residual.min()
            assert abs(residual-score).max()<1e-8
            actual,record=select(score,n);expected,rows,chosen,ties=incremental_selection(score,n)
            assert np.array_equal(actual,expected) and record['selected']==chosen and record['minimum_dispersion_ties']==ties
            assert np.array_equal(np.array([[r[k] for k in COLUMNS] for r in record['candidates']],np.int64),rows)
            cases.append(dict(n=n,candidates=len(rows),least_squares_max_error=float(abs(residual-score).max())))
    files=[Path(__file__),HERE/'audit_centered_regularity.py',HERE/'regularity_gap.py',HERE/'weak_weight_study.py']
    save(path,dict(status='passed',cases=cases,additive_cell_invariance=True,wall_seconds=time.monotonic()-start,
                   source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files}))
    print(json.dumps(dict(status='passed',cases=len(cases))))


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate=HERE/'centered-regularity-preflight.json';g=json.loads(gate.read_text());assert g['status']=='passed'
    for rel,digest in g['source_sha256'].items():assert sha(ROOT/rel)==digest
    sources=[Path(__file__),HERE/'audit_centered_regularity.py',HERE/'regularity_gap.py',HERE/'weak_weight_study.py']
    inputs=[gate,HERE/'regularity_intensity.json',PARENT/'summary.json',PARENT/'models.npz',
            HERE/'intensity_degree.json',ROOT/'results/research/foureyes/intensity_degree/summary.json']
    p=dict(id='H4ai-additive-cell-centering',frozen_utc=datetime.now(timezone.utc).isoformat(),groups=64,conditions=CONDITIONS,n=81,
           source_scope='Same128 already examined H4ag count matrices. H4ah results motivated this adaptive follow-up; no new source, corruption, label, fit or puzzle input.',
           score='si=sum of80 offdiagonal counts incident on cell i; rij=79*cij-si-sj; score=rij-min(r). Exact int64. No fitted coefficient, degree value, geometry or edge count enters scores.',
           algebra='Additive changes bi+bj in pair counts alter rij only by a common constant. This identity is exact; additive effects as the actual corruption model remain a hypothesis.',
           selector='Unchanged H4ae equal-degree threshold selector: every distinct score<=median, minimize81*sum(degree^2)-sum(degree)^2; tie fewer edges then lower cut. No density guard.',
           primary='Weight32 centered regularity at least61/64 exact graphs AND zero missing true edges across all64. Both required.',
           decision='Supported if both requirements pass; falsified after all128 complete if either fails; runtime/numerical failure incomplete. Retain every bank and sparse output.',
           controls='Archived BP32 first, canonical256, raw restricted gap, raw regularity, raw top810 and degree20LP. New centered top810 uses supplied cardinality810 and index tie breaks; descriptive. Weight8 same criterion descriptive.',
           diagnostics='Raw and adjusted separating margins, every candidate and degree dispersion, ties, sparse outcomes, missing/extra counts. Geometry only in independent postselection audit.',
           candidate_columns=COLUMNS,budget=dict(selection_wall_seconds=30,audit_wall_seconds=120,aggregate_cpu_seconds=240,gpu_seconds=0),
           runtime_scope='Selection plus public independent replay monitored together; synthetic preflight and publication separate.',
           source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def selection():
    p=json.loads(PROTOCOL.read_text());verify(p);PUBLIC.mkdir(parents=True,exist_ok=False);start=time.monotonic();cpu=time.process_time()
    with np.load(PARENT/'models.npz',allow_pickle=False) as z:counts=z['counts'].copy();groups=z['groups'].copy();conditions=z['conditions'].copy()
    assert counts.shape==(128,3240) and np.array_equal(groups,np.repeat(np.arange(64),2)) and conditions.tolist()==CONDITIONS*64
    scores=[];sums=[];shifts=[];supports=[];tops=[];candidates=[];offsets=[0];records=[]
    for index,count in enumerate(counts):
        score,total,shift=center(count);support,record=select(score)
        top=np.zeros(3240,bool);top[np.lexsort((np.arange(3240),score))[:810]]=True
        table=np.array([[row[key] for key in COLUMNS] for row in record['candidates']],np.int64)
        records.append(dict(index=index,group=int(groups[index]),condition=str(conditions[index]),selected=record['selected'],
                            minimum_dispersion_ties=record['minimum_dispersion_ties'],median_score=record['median_count'],candidates=len(table)))
        scores.append(score);sums.append(total);shifts.append(shift);supports.append(support);tops.append(top);candidates.append(table);offsets.append(offsets[-1]+len(table))
        assert time.monotonic()-start<p['budget']['selection_wall_seconds'],'selection_budget'
        print(json.dumps(dict(index=index,candidates=len(table))),flush=True)
    np.savez_compressed(PUBLIC/'models.npz',scores=np.stack(scores),row_sums=np.stack(sums),raw_min=np.array(shifts,np.int64),
                        regularity_supports=np.stack(supports),top810_supports=np.stack(tops),groups=groups,conditions=conditions)
    np.savez_compressed(PUBLIC/'candidates.npz',rows=np.concatenate(candidates),offsets=np.array(offsets,np.int64),columns=np.array(COLUMNS))
    receipt=dict(status='complete',protocol_sha256=sha(PROTOCOL),models_sha256=sha(PUBLIC/'models.npz'),candidates_sha256=sha(PUBLIC/'candidates.npz'),
                 records=records,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
    assert receipt['wall_seconds']<p['budget']['selection_wall_seconds'];save(PUBLIC/'selection.json',receipt)


def launch():
    p=json.loads(PROTOCOL.read_text());verify(p);assert not OUT.exists();OUT.mkdir(parents=True)
    start=time.monotonic();base=sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)])
    child=None;samples=[];status='failed';error=None;tick=os.sysconf('SC_CLK_TCK');returns=[]
    def cpu_seconds():
        total=sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)])-base
        if child and child.poll() is None:
            try:
                a=Path(f'/proc/{child.pid}/stat').read_text().split();total+=(int(a[13])+int(a[14]))/tick
            except FileNotFoundError:pass
        return total
    try:
        for phase,script in [('selection',Path(__file__)),('audit',HERE/'audit_centered_regularity.py')]:
            phase_start=time.monotonic();args=[sys.executable,str(script)]+(['selection'] if phase=='selection' else ['--export'])
            with (OUT/f'{phase}.log').open('x') as f:
                child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    elapsed=time.monotonic()-phase_start;cpu=cpu_seconds();samples.append(dict(phase=phase,elapsed=elapsed,aggregate_cpu_seconds=cpu,pid=child.pid))
                    assert elapsed<p['budget'][phase+'_wall_seconds'] and cpu<p['budget']['aggregate_cpu_seconds'],'runtime_budget'
                    time.sleep(.1)
                returns.append(child.returncode);assert child.returncode==0,phase
        assert cpu_seconds()<p['budget']['aggregate_cpu_seconds'];status='complete'
    except BaseException as exc:error=repr(exc);raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        result=dict(status=status,error=error,protocol_sha256=sha(PROTOCOL),wall_seconds=time.monotonic()-start,aggregate_cpu_seconds=cpu_seconds(),returncodes=returns,samples=samples)
        save(OUT/'runtime.json',result)
        if PUBLIC.exists():save(PUBLIC/'runtime.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='samples'}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','freeze','selection','launch']);globals()[ap.parse_args().action]()
