"""H4ap: fixed-degree bootstrap votes with bounded exact LP certificates."""
from pathlib import Path
import argparse,hashlib,json,os,resource,signal,subprocess,sys,time
WALL_ORIGIN=time.monotonic()
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;RIPPER=HERE.parent/'ripper';sys.path.insert(0,str(RIPPER))
PARENT=ROOT/'results/research/foureyes/prefix_budget/degree';OUT=ROOT/'artifacts/research/20261114/foureyes/bootstrap_consensus';PUBLIC=ROOT/'results/research/foureyes/bootstrap_consensus';PROTOCOL=HERE/'bootstrap-consensus.json';PYTHON=ROOT/'artifacts/degree-lp-venv/bin/python'
SEED=2030010000;SOLVER_BASE=2028011502;REPS=32;FAMILIES=['cycle81','power2','power3','cliques9']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def indices(index):return np.stack([np.random.default_rng(SEED+32*index+r).integers(0,600,600,dtype=np.int16) for r in range(REPS)])
def fast_counts(labels,draws):
    assert labels.shape==(600,81) and draws.shape==(32,600) and draws.dtype==np.int16 and ((draws>=0)&(draws<600)).all()
    ii,jj=np.triu_indices(81,1);weights=np.stack([np.bincount(x,minlength=600) for x in draws]);assert (weights.sum(1)==600).all()
    counts=weights.astype(np.float64)@(labels[:,ii]==labels[:,jj]).astype(np.float64);result=counts.astype(np.int64)
    assert np.array_equal(counts,result) and ((result>=0)&(result<=600)).all()
    return result

def decision(success,unresolved,losses,unresolved_baseline_success):
    a,b=success,success+unresolved;l,u=losses,losses+unresolved_baseline_success
    return dict(recovery=dict(lower=a,upper=b,required=145,decision='supported' if a>=145 else 'falsified' if b<145 else 'inconclusive'),paired_loss=dict(lower=l,upper=u,maximum=8,decision='supported' if u<=8 else 'falsified' if l>8 else 'inconclusive'))
def verify(p,public=False):
    missing=[]
    for rel,h in (p['source_sha256']|p['input_sha256']).items():
        path=ROOT/rel
        if not path.exists():assert public and rel.startswith(('artifacts/','upstream/')),rel;missing.append(rel)
        else:assert sha(path)==h,rel
    return missing

def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists();gate=read(HERE/'bootstrap-consensus-preflight.json');assert gate['status']=='passed'
    for rel,h in gate['source_sha256'].items():assert sha(ROOT/rel)==h
    parent=read(HERE/'prefix_budget.json');summary=read(PARENT/'summary.json');jobs=[]
    inputs=[HERE/'prefix_budget.json',PARENT/'summary.json',HERE/'bootstrap-consensus-proposal.json',HERE/'bootstrap-consensus-preflight.json',HERE/'bootstrap-consensus-seed-audit.json']
    for group in range(64):
        for fi,family in enumerate(FAMILIES):
            index=4*group+fi;oldindex=12*group+3*fi+1;old=summary['rows'][oldindex];job=parent['jobs'][oldindex]
            assert old['family']==family and old['rows']==600 and old['true_degree_selected'] and old['classification']!='unresolved'
            jobs.append(dict(job,index=index,parent_prefix_index=oldindex,inferred_degree=old['selected_degree'],baseline_exact=old['classification']=='success'))
            inputs.extend([ROOT/job['source_path'],PARENT/f'j{oldindex:03d}/models.npz',PARENT/f'j{oldindex:03d}/receipt.json'])
    assert sum(j['baseline_exact'] for j in jobs)==129
    for name in ['cycle_power','varying_degree']:inputs.extend([HERE/f'{name}.json',ROOT/f'results/research/foureyes/{name}/controls/graphs.npz',ROOT/f'results/research/foureyes/{name}/controls/generation-manifest.json'])
    proposal=read(HERE/'bootstrap-consensus-proposal.json');source=parent['source_sha256']|gate['source_sha256']
    p=dict(proposal,status='frozen_before_bootstrap_indices_and_outcomes',frozen_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),jobs=jobs,bootstrap_seed_base=SEED,bootstrap_rng='32separate default_rng(base+32*index+rep).integers(0,600,600,dtype=int16); all256indexarrayssealedbeforeLPs',solver=dict(parent['solver'],random_seed_base=SOLVER_BASE),solver_seed_rule='base+33*index+rep;rep0..31bootstrap,rep32consensus',counts='Multiplicityweights and0/1equalityindicators multipliedinfloat64; allnonnegativeintegerproductsandpartials<=600exactlyrepresentable; castchecked. Independentdirectresampledrowcounts. BLASthreads1 beforeimports.',source_sha256=source,input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs})
    save(PROTOCOL,p);print('FROZEN',sha(PROTOCOL),flush=True)

def prepare(p):
    entries=[]
    for job in p['jobs']:
        d=OUT/f'b{job["index"]:03d}';d.mkdir();np.savez_compressed(d/'indices.npz',indices=indices(job['index']));entries.append(dict(index=job['index'],indices_sha256=sha(d/'indices.npz')))
    save(OUT/'indices-seal.json',dict(protocol_sha256=sha(PROTOCOL),all256_before_any_LPs=True,entries=entries));print('ALL_INDICES_SEALED',flush=True)

def worker(worker_id):
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate
    from audit_bootstrap_consensus import audit_bank
    p=read(PROTOCOL);verify(p);seal=read(OUT/'indices-seal.json');assert seal['protocol_sha256']==sha(PROTOCOL) and seal['all256_before_any_LPs']
    ii,jj=np.triu_indices(81,1)
    for index in range(worker_id,256,8):
        start=time.monotonic();job=p['jobs'][index];d=OUT/f'b{index:03d}';source=ROOT/job['source_path'];assert sha(source)==job['source_sha256'];labels=npz(source)['noisy'][:600];draws=npz(d/'indices.npz')['indices'];assert seal['entries'][index]==dict(index=index,indices_sha256=sha(d/'indices.npz'));costs=fast_counts(labels,draws);degree=job['inferred_degree'];arrays=dict(costs=costs);records=[];votes=np.zeros(3240,np.int64);all_valid=True
        for rep in range(33):
            if rep==32 and not all_valid:records.append(dict(rep=rep,skipped_due_to_uncertified_bootstrap=True));break
            cost=costs[rep] if rep<32 else 64-votes
            if rep==32:arrays['consensus_cost']=cost;arrays['votes_twice']=votes
            log=d/f'r{rep:02d}.log';x,y,meta=solve(81,degree,ii,jj,cost,SOLVER_BASE-2026111502+33*index+rep,log)
            if meta['solver_optimal'] and meta['value_valid'] and meta['dual_valid']:cert,w=attempt_certificate(81,degree,ii,jj,cost,x,y)
            else:cert=dict(certified=False,reason='solver_status_or_validity');w={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
            arrays.update({f'r{rep:02d}_'+k:v for k,v in dict(primal_values=x,node_dual_values=y,**w).items()});records.append(dict(rep=rep,solver=meta,certificate=cert,log_sha256=sha(log)))
            if rep<32:
                if cert['certified']:votes+=w['primal_twice']
                else:all_valid=False
            save(d/'progress.json',dict(index=index,complete_LPs=rep+1,pid=os.getpid()))
        np.savez_compressed(d/'models.npz',**arrays);save(d/'receipt.json',dict(index=index,degree=degree,protocol_sha256=sha(PROTOCOL),source_sha256=sha(source),indices_sha256=sha(d/'indices.npz'),models_sha256=sha(d/'models.npz'),records=records,wall_seconds=time.monotonic()-start))
        checked=audit_bank(index,p,False,labels);save(d/'audit.json',checked);print('BANK_AUDITED',index,flush=True)

def export(p):
    import shutil
    from audit_bootstrap_consensus import aggregate
    PUBLIC.mkdir();rows=[];files={}
    for index in range(256):
        d=OUT/f'b{index:03d}';audit=read(d/'audit.json');assert audit['protocol_sha256']==sha(PROTOCOL) and audit['auditor_sha256']==sha(HERE/'audit_bootstrap_consensus.py')
        for name,h in audit['artifact_sha256'].items():assert sha(d/name)==h
        rows.append(audit['row']);target=PUBLIC/d.name;target.mkdir()
        for name in ['models.npz','receipt.json','indices.npz','audit.json']:shutil.copyfile(d/name,target/name);files[str((target/name).relative_to(PUBLIC))]=sha(target/name)
    shutil.copyfile(OUT/'indices-seal.json',PUBLIC/'indices-seal.json');save(PUBLIC/'summary.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),rows=rows,**aggregate(rows),artifact_sha256=files,new_labels=0,new_source_draws=0,new_bootstrap_samples=8192,new_bootstrap_index_draws=4915200,new_GPU_seconds=0,scope=p['scope'],baseline='BPAdamWoutsidecountLPintervention; fixedH4an600rowLPresults pairedbaseline.'))

def launch():
    p=read(PROTOCOL);assert not OUT.exists();OUT.mkdir(parents=True);children=[];logs=[];samples=[];status='failed';error=None;tick=os.sysconf('SC_CLK_TCK')
    def cpu():
        total=sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)])
        for c in children:
            if c.poll() is None:
                try:a=Path(f'/proc/{c.pid}/stat').read_text().split();total+=(int(a[13])+int(a[14]))/tick
                except FileNotFoundError:pass
        return total
    def monitor(phase):
        partial=[read(f)['complete_LPs'] for f in OUT.glob('b*/progress.json')];sample=dict(phase=phase,wall_seconds=time.monotonic()-WALL_ORIGIN,aggregate_cpu_seconds=cpu(),complete_LPs=sum(partial),audited_banks=len(list(OUT.glob('b*/audit.json'))),active_pids=[c.pid for c in children if c.poll() is None]);samples.append(sample);assert sample['wall_seconds']<180 and sample['aggregate_cpu_seconds']<900,'frozen_resource_ceiling'
    try:
        verify(p);prepare(p);monitor('prepare')
        for worker_id in range(8):
            f=(OUT/f'worker{worker_id}.log').open('x');logs.append(f);children.append(subprocess.Popen([str(PYTHON),str(Path(__file__)),'worker','--worker',str(worker_id)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
        while any(c.poll() is None for c in children):monitor('LP_and_independent_audit');assert all(c.poll() in [None,0] for c in children),'worker_failure';time.sleep(.2)
        assert all(c.returncode==0 for c in children);monitor('LP_and_independent_audit');export(p);monitor('export');status='complete'
    except BaseException as exc:error=repr(exc);raise
    finally:
        for c in children:
            if c.poll() is None:
                os.killpg(c.pid,signal.SIGTERM)
                try:c.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
        for f in logs:f.close()
        result=dict(status=status,error=error,protocol_sha256=sha(PROTOCOL),wall_seconds=time.monotonic()-WALL_ORIGIN,aggregate_cpu_seconds=cpu(),returncodes=[c.returncode for c in children],samples=samples);save(OUT/'runtime.json',result)
        if PUBLIC.exists():save(PUBLIC/'runtime.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','launch','worker']);ap.add_argument('--worker',type=int);a=ap.parse_args();worker(a.worker) if a.action=='worker' else freeze() if a.action=='freeze' else launch()
