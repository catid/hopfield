"""H3w: frozen positive weights with permutation initialization/proposal prior."""
import argparse,datetime as dt,fcntl,hashlib,importlib.util,json,math,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3w';OUT=ROOT/'artifacts/research/20261114/loki/h3w';OLD=ROOT/'artifacts/research/20261114/loki/h3v';UP=ROOT/'upstream'
sys.path[:0]=[str(SRC),str(SRC.parent),str(UP)];import backend;import study as S
_spec=importlib.util.spec_from_file_location('h3v_arbitrary',SRC.parent/'h3v/backend.py');arbitrary_backend=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(arbitrary_backend)
ARMS={'bp-adamw-permutation':'positive-bp-adamw-weights','bp-adamw-arbitrary':'positive-bp-adamw-weights','uniform-filtered-permutation':'uniform-filtered','oracle-permutation':'supplied-rule-oracle'}
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):S.save(p,v)
def protocol():return read(SRC/'protocol.json')
def frozen():
    p=protocol()
    for group in ['source_sha256','data_sha256']:
        for path,h in p[group].items():assert sha(ROOT/path)==h,path
    return p

def freeze():
    assert not(SRC/'protocol.json').exists();gates=[read(OUT/f'kernel-validation-{x}.json')for x in ['cpu','cuda']];assert all(x['status']=='passed'for x in gates)
    sources=[Path(__file__),SRC/'backend.py',SRC/'prepare.py',SRC/'preflight.py',SRC/'validate_kernel.py',SRC.parent/'h3v/backend.py',SRC.parent/'h3v/protocol.json',SRC.parent/'study.py',UP/'cram/cram_nqueens.py',UP/'cram/nqueens_data.py']
    data=[OUT/'data.npz',OUT/'dataset.json',OUT/'kernel-validation-cpu.json',OUT/'kernel-validation-cuda.json',OLD/'independent-preflight.json',ROOT/'artifacts/research/20261114/loki/h3u/source-design.npz',*[OLD/f'models/{a}/model.npz'for a in sorted(set(ARMS.values()))]]
    p=dict(id='H3w-permutation-initialization-and-proposal-prior',frozen_utc=now(),
      hypothesis='With identical frozen positive BP AdamW weights and source temperature, permutation initialization and row-pair column swaps improve certified valid success by at least50percentage points over H3v arbitrary-cell initialization/swaps on the same fresh cohort, with zero wrongstops.',
      antecedent='H3v weighted arbitrary-cell search was much worse than uniform on identical support. This adaptive follow-up changes BOTH initialization and proposal constraints; it cannot isolate a pure move-barrier cause.',
      arms=[dict(arm=k,model=v,kernel='arbitrary'if k=='bp-adamw-arbitrary'else'permutation')for k,v in ARMS.items()],
      models='Byte-identical copies of all H3v source-produced positive/uniform/oracle models. No refit, coefficient change, filtering change, rescaling or source-temperature recalibration. Weighted/uniformF still carries308false target pairs. Oracle explicitlysupplied.',
      added_prior='Require the original source-inferred line family to include row andcolumn lines. Initialize onequeen perrow/column while preserving givens, then preserve this inferred row/column prior through every permutation proposal. This is a stronger supplied representation prior than merely fixedq13.',
      initialization='For each input/chain keep given rows and their columns fixed; uniformly permute unused columns among nongiven rows. The arbitrary control reruns the frozen H3v initialization: uniformly add nongiven cells untilq13, which need not have unique rows/columns.',
      proposal='Permutation arms choose an unordered pair of nongiven rows uniformly via top2 IIDuniform scores, exchange their queen columns, and use Metropolis acceptance min(1,exp(-DeltaE/T)). This symmetric transposition preserves q13,allrows,allcolumnsandgivens. Arbitrary control uses unchanged H3v occupied-nongiven→emptycell swap.',
      temperature='Exactly frozen H3v schedule T(k)=u*0.01**(k/19999),k0..19999;u=3.0367188453674316weighted,u=1uniform/oracle. Retain row/column contributions to the source unit even though they are zero on permutation states; no target tuning.',
      certificate='Exact integer count of occupied F-edges=0, including given–given pairs; independent of float tolerance. All weights retained positive; no native inherited-margin certificate.',
      search=dict(q=13,unique_inputs=4096,chains=20,cap=20000,seeds=[0,1,2],end_temperature_ratio=.01,block=65536),
      primary=dict(arm='bp-adamw-permutation',control='bp-adamw-arbitrary',metric='Any valid zero-conflict stopped completion among20chains, abstain otherwise.',gain_pp=50.,wrong_stops=0),
      decision='Support iff paired-input bootstrap one-sided95% lower primary gain>=50pp and no permutation-weighted wrongstops; falsify iff uppergain<50ppORany suchwrongstop; otherwise inconclusive.2000draws,RNG20261114,all3searchseeds clusteredwithininput. Uniform/oracle permutation and coverage comparisons descriptive.',
      secondary='Original lowest-running-energy and certificate-first fallback accuracy; distinct valid completion coverage; exact supportcapacity; proposal/accepted counts; per-proposal runtime; allcap returns. Equalproposalcounts do not imply equalFLOPs or equally expensive transitions.',
      validation='Exhaustive24smallpermutations×alltranspositions with0/1/2clues; direct fullenergy/countdelta and fixedtemperature detailedbalance; initialization bijection counts24/6/2; CPU/GPU traces verifyq,rows,columns,givens,fields,energy,integerconflicts; native source denseenergy parity; bounded source-only pilot. Confirm source inferredaxes, allcopied model hashes and fresh cohort before targetsearch.',
      numeric_audit='Every saved state checked forq/givens, permutation states additionallyonerow/column. Reconstruct integerFconflicts and float64 energies. Strictstop requires exact0conflicts; maxenergyerror>.1prevents support without overriding wrong-stop falsification.',
      identity='The two BP arms have identicalmodelbytes but differentinitialization/proposal kernels, so they are NOT outcome-identical. All12jobs are actualsearches; no mappings.',
      budget=dict(search_gpu_minutes_envelope=20,wall_minutes_envelope=10,logical_model_seed_jobs=12,optimizer_refits=0,policy='Shared absolute10minute wall deadline and perworkerSIGALRM bound aggregate at20GPUminutes. On expiry retain completed evidence and report incomplete; no cap or recipe retuning.'),
      dataset=read(OUT/'dataset.json'),source_sha256={str(x.relative_to(ROOT)):sha(x)for x in sources},data_sha256={str(x.relative_to(ROOT)):sha(x)for x in data},
      limits=['Joint initialization+proposal+representation intervention, not isolated causalproof that barriers alone explain H3v.','Fixed-q permutation solver is a new derivative with inferred row/column prior; not paper reproduction.','Unchanged source support still excludes somevalid targetsolutions and limits abstainingprimary.','One complete source universe and inherited fitted trajectory; repeated searchseeds are not independentlearning replicas.','No optimizer refit, targettemperature adjustment or targetdependentmask edit.'])
    save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',sha256=sha(SRC/'protocol.json'))))

def construct():
    import numpy as np
    frozen();d=np.load(ROOT/'artifacts/research/20261114/loki/h3u/source-design.npz');lines={tuple(x)for x in d['lines']};assert(0,1)in lines and(1,0)in lines
    rows=[]
    for arm,parent in ARMS.items():
        dest=OUT/f'models/{arm}';dest.mkdir(parents=True,exist_ok=True);source=OLD/f'models/{parent}/model.npz';shutil.copy2(source,dest/'model.npz');m=np.load(source);assert(m['J'][m['F'].astype(bool)]>0).all()
        r=dict(status='complete',arm=arm,parent_arm=parent,kernel='arbitrary'if arm=='bp-adamw-arbitrary'else'permutation',unit=float(m['unit']),minimum_positive_coefficient=float(m['J'][m['F'].astype(bool)].min()),target_pairs=int(m['F'].sum()//2),model_sha256=sha(dest/'model.npz'),source_model_sha256=sha(source),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__));save(dest/'model.json',r);rows.append(r)
    save(OUT/'construction.json',dict(status='complete',models=rows));print(json.dumps(rows))

def search(arm,seed,data,progress):
    import numpy as np
    path=OUT/f'models/{arm}/model.npz';m=np.load(path);inputs=data['test_inputs'];progress(arm=arm,seed=seed);start=time.perf_counter()
    chosen=backend if arm!='bp-adamw-arbitrary' else arbitrary_backend
    z=chosen.solve(m['J'],m['h'],m['F'],inputs,q=13,unit=float(m['unit']),chains=20,cap=20000,seed=seed,device='cuda');seconds=time.perf_counter()-start
    keys={row.tobytes() for row in data['all_solutions']};boards=z['boards'];valid=np.array([[r.tobytes() in keys and r[given].all() for r in samples] for given,samples in zip(inputs,boards)])
    stopped=z['stopped'];original=z['energy'].argmin(1);selected=np.where(stopped.any(1),np.where(stopped,z['energy'],np.inf).argmin(1),original);ids=np.arange(len(inputs));denom=np.minimum(data['test_solution_counts'],20);found=np.array([min(len({r.tobytes() for r in samples[ok]}),limit) for samples,ok,limit in zip(boards,valid,denom)])
    dest=OUT/f'search/{arm}/s{seed}';dest.mkdir(parents=True,exist_ok=True);z.update(boards=np.packbits(boards,-1),inputs=inputs,valid=valid,n20=valid[ids,original],n20_certificate=valid[ids,selected],certified_success=(stopped&valid).any(1),original_index=original,certificate_index=selected,found=found,denom=denom,solution_counts=data['test_solution_counts'])
    np.savez_compressed(dest/'records.npz',**z);rec=dict(status='complete',utc=now(),arm=arm,seed=seed,certified_success_pct=100*float(z['certified_success'].mean()),original_accuracy_pct=100*float(z['n20'].mean()),certificate_accuracy_pct=100*float(z['n20_certificate'].mean()),coverage_pct=100*float(found.sum()/denom.sum()),wrong_stops=int((stopped&~valid).sum()),capped_chains=int((~stopped).sum()),proposal_moves=int(z['moves'].sum()),accepted_swaps=int(z['accepted'].sum()),search_seconds=seconds,model_sha256=sha(path),records_sha256=sha(dest/'records.npz'),source_sha256=sha(__file__),backend_sha256=sha(SRC/'backend.py' if arm!='bp-adamw-arbitrary' else SRC.parent/'h3v/backend.py'),kernel='permutation' if arm!='bp-adamw-arbitrary' else 'arbitrary',protocol_sha256=sha(SRC/'protocol.json'));save(dest/'result.json',rec);print(json.dumps(rec),flush=True)

def worker(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=frozen();assert read(OUT/'independent-preflight.json')['status']=='passed'
    query=lambda *x:subprocess.check_output(x,text=True,timeout=20).strip();assert query('hostname')=='loki';device=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader');assert device.split(',')[1].strip() not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    budget_handle=(OUT/'global-budget.json').open('a+');fcntl.flock(budget_handle,fcntl.LOCK_EX);budget_handle.seek(0);budget_text=budget_handle.read()
    if budget_text:deadline=json.loads(budget_text)['deadline_unix']
    else:
        deadline=time.time()+600;budget_handle.write(json.dumps(dict(started_utc=now(),deadline_unix=deadline,global_wall_seconds=600,maximum_aggregate_worker_seconds=1200,protocol_sha256=sha(SRC/'protocol.json')))+'\n');budget_handle.flush()
    budget_handle.close()
    def expired(signum,frame):raise TimeoutError('Frozen global10minute wall /20GPUminute envelope expired; retain completed evidence')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    data=dict(np.load(OUT/'data.npz'));start=time.monotonic();base=dict(started_utc=now(),host='loki',gpu=gpu,pid=os.getpid(),device=device,protocol_sha256=sha(SRC/'protocol.json'));receipt=OUT/f'worker-g{gpu}.json'
    def progress(**v):save(receipt,dict(base,status='running',utc=now(),progress=v))
    progress(stage='starting')
    try:
        for cfg in p['arms']:
            arm=cfg['arm']
            for seed in [0,1,2]:
                dest=OUT/f'search/{arm}/s{seed}';dest.mkdir(parents=True,exist_ok=True)
                if(dest/'result.json').exists():continue
                claim=(dest/'job.lock').open('a')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                try:
                    if time.monotonic()-start>600:raise RuntimeError('Frozen10minute worker wall envelope reached; retain partial evidence')
                    if not(dest/'result.json').exists():search(arm,seed,data,progress)
                finally:claim.close()
        signal.setitimer(signal.ITIMER_REAL,0);save(receipt,dict(base,status='complete',completed_utc=now()))
    except BaseException as e:save(receipt,dict(base,status='failed',completed_utc=now(),error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);s.add_parser('freeze');s.add_parser('construct');w=s.add_parser('worker');w.add_argument('--gpu',type=int,choices=[0,1],required=True);a=p.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='construct':construct()
    else:worker(a.gpu)
