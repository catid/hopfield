"""H4ao: distinguish empirical-objective exclusion from co-optimal truth."""
from pathlib import Path
import argparse,hashlib,json,resource,signal,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'ripper'))
from audit_degree_prior import check_certificate
PARENT=ROOT/'results/research/foureyes/prefix_budget/degree'
PUBLIC=ROOT/'results/research/foureyes/objective_gap'
PROTOCOL=HERE/'objective-gap.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def classify(truth_twice,optimum_twice):
    gap=truth_twice-optimum_twice
    assert gap>=0,'Feasible truth cannot beat a certified minimum'
    return gap,'cooptimal' if gap==0 else 'strictly_suboptimal'
def decision(strict,denominator=394):
    assert denominator==394 and 0<=strict<=denominator
    return 'supported' if 10*strict>=9*denominator else 'falsified'
def evaluate(n,d,c,t,y,z,truth):
    ii,jj=np.triu_indices(n,1);assert check_certificate(n,d,ii,jj,c,t,y,z)
    assert truth.dtype==bool and truth.shape==c.shape
    deg=np.zeros(n,np.int64);np.add.at(deg,ii,truth);np.add.at(deg,jj,truth);assert (deg==d).all()
    optimum=sum(int(a)*int(b) for a,b in zip(c,t,strict=True));actual=2*sum(int(a) for a,b in zip(c,truth,strict=True) if b)
    gap,kind=classify(actual,optimum)
    return dict(optimum_twice=optimum,truth_cost_twice=actual,gap_twice=gap,truth_objective_status=kind)
def aggregate(rows):
    assert len(rows)==768 and [r['index'] for r in rows]==list(range(768))
    failures=[r for r in rows if r['parent_classification']!='success'];success=[r for r in rows if r['parent_classification']=='success']
    assert len(failures)==394 and len(success)==374 and all(r['gap_twice']==0 for r in success)
    strict=sum(r['gap_twice']>0 for r in failures);groups=[]
    for family in ['cycle81','power2','power3','cliques9']:
        for n in [300,600,1200]:
            part=[r for r in rows if r['family']==family and r['rows']==n];assert len(part)==64
            fail=[r for r in part if r['parent_classification']!='success']
            groups.append(dict(family=family,rows=n,exact_graphs=64-len(fail),failures=len(fail),strictly_suboptimal_truth=sum(r['gap_twice']>0 for r in fail),cooptimal_truth=sum(r['gap_twice']==0 for r in fail),fractional_failures=sum(r['parent_classification']=='resolved_failure_fractional' for r in fail),maximum_gap_twice=max(r['gap_twice'] for r in part)))
    return dict(denominator=394,strictly_suboptimal_truth=strict,cooptimal_truth=394-strict,required_strict=355,decision=decision(strict),groups=groups,successes_with_zero_gap=374)
def freeze():
    assert not PROTOCOL.exists() and not PUBLIC.exists()
    gate=HERE/'objective-gap-preflight.json';g=json.loads(gate.read_text());assert g['status']=='passed'
    for rel,h in g['source_sha256'].items():assert sha(ROOT/rel)==h
    s=json.loads((PARENT/'summary.json').read_text());assert len(s['rows'])==768 and sum(r['classification']!='success' for r in s['rows'])==394
    paths=[PARENT/'summary.json',HERE/'prefix_budget.json',gate,HERE/'objective-gap-proposal.json']
    paths+=[PARENT/f'j{i:03d}'/name for i in range(768) for name in ['models.npz','receipt.json']]
    paths+=[ROOT/f'results/research/foureyes/{name}/controls/graphs.npz' for name in ['varying_degree','cycle_power']]
    proposal=json.loads((HERE/'objective-gap-proposal.json').read_text())
    p=dict(proposal,status='frozen_before_gap_outcomes',frozen_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),source_sha256=g['source_sha256'],input_sha256={str(x.relative_to(ROOT)):sha(x) for x in paths})
    save(PROTOCOL,p);print('FROZEN',sha(PROTOCOL),flush=True)
def main():
    start=time.perf_counter();cpu=time.process_time();p=json.loads(PROTOCOL.read_text());assert not PUBLIC.exists();PUBLIC.mkdir(parents=True)
    def expired(*_):raise TimeoutError('H4ao frozen60CPU/90wall limit')
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGXCPU,expired);signal.setitimer(signal.ITIMER_REAL,90);used=int(resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime);resource.setrlimit(resource.RLIMIT_CPU,(used+60,used+61))
    rows=[]
    try:
        for rel,h in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==h,rel
        parents={}
        for name in ['varying_degree','cycle_power']:
            with np.load(ROOT/f'results/research/foureyes/{name}/controls/graphs.npz',allow_pickle=False) as z:parents[name]={k:z[k] for k in z.files}
        old=json.loads((PARENT/'summary.json').read_text());jobs=json.loads((HERE/'prefix_budget.json').read_text())['jobs']
        for i,prior in enumerate(old['rows']):
            job=jobs[i];d=prior['selected_degree'];assert d==prior['true_degree'];receipt=json.loads((PARENT/f'j{i:03d}/receipt.json').read_text());assert receipt['models_sha256']==sha(PARENT/f'j{i:03d}/models.npz');assert receipt['family']==job['family'] and receipt['rows']==job['rows']
            with np.load(PARENT/f'j{i:03d}/models.npz',allow_pickle=False) as z:c,t,y,u=[z[key] for key in ['cost',f'd{d:02d}_primal_twice',f'd{d:02d}_node_dual_twice',f'd{d:02d}_upper_dual_twice']]
            graph=parents[job['parent']];assert graph['conditions'][job['parent_index']]==job['family'] and graph['groups'][job['parent_index']]==job['group'];truth=graph['truths'][job['parent_index']]
            result=evaluate(81,d,c,t,y,u,truth);assert result['optimum_twice']==prior['objective_twice'][d]==receipt['records'][d-1]['certificate']['primal_objective_twice']
            exact=bool(np.array_equal(t,2*truth));integral=bool(np.isin(t,[0,2]).all());assert prior['classification']==('success' if exact else 'resolved_failure_nonexact' if integral else 'resolved_failure_fractional')
            rows.append(dict(index=i,group=job['group'],family=job['family'],rows=job['rows'],degree=d,parent_classification=prior['classification'],**result))
            if (i+1)%64==0:save(PUBLIC/'progress.json',dict(completed=i+1,planned=768,cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-start));print('GAPS_VERIFIED',i+1,flush=True)
        summary=aggregate(rows);save(PUBLIC/'rows.json',rows)
        result=dict(status='complete',protocol_sha256=sha(PROTOCOL),rows_sha256=sha(PUBLIC/'rows.json'),selected_certificates=768,new_LPs=0,new_labels=0,new_draws=0,new_GPU_seconds=0,**summary)
        save(PUBLIC/'summary.json',result);result['scientific_cpu_seconds']=time.process_time()-cpu;result['scientific_wall_seconds']=time.perf_counter()-start;assert result['scientific_cpu_seconds']<60 and result['scientific_wall_seconds']<90;save(PUBLIC/'summary.json',result);print(json.dumps(summary),flush=True)
    except BaseException as exc:
        save(PUBLIC/'failure.json',dict(error=repr(exc),completed=len(rows),partial_rows=rows,partial_results_usable=False));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['freeze','run']);a=ap.parse_args();freeze() if a.command=='freeze' else main()
