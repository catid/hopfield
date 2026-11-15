"""H3v positive-weight fixed-support search; no optimizer refit."""
import argparse,datetime as dt,fcntl,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3v';OUT=ROOT/'artifacts/research/20261114/loki/h3v';OLD=ROOT/'artifacts/research/20261114/loki/h3u';UP=ROOT/'upstream'
sys.path[:0]=[str(SRC),str(SRC.parent),str(UP)];import backend;import study as S

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
    assert not(SRC/'protocol.json').exists();validations=[read(OUT/f'kernel-validation-{x}.json') for x in ['cpu','cuda']];assert all(v['status']=='passed' for v in validations)
    sources=[Path(__file__),SRC/'backend.py',SRC/'prepare.py',SRC/'preflight.py',SRC/'validate_kernel.py',SRC.parent/'study.py',UP/'cram/cram_nqueens.py',UP/'cram/nqueens_data.py',SRC.parent/'h3u/protocol.json']
    data=[OUT/'data.npz',OUT/'dataset.json',OLD/'source-design.npz',OLD/'models/bp-adamw-grouped/model.pt',OUT/'kernel-validation-cpu.json',OUT/'kernel-validation-cuda.json']
    p=dict(id='H3v-fixed-support-positive-weight-swap-search',frozen_utc=now(),
      hypothesis='Positive learned weights improve valid zero-conflict completion found among20chains by at least5percentage points over uniform weights on the identical filtered support, with zero wrong stopped chains.',
      antecedent='H3u source-native soft margins admitted false stops. This new derivative tests pure search quality on a fixed zero set, not repair of source-global exclusions. H3u coefficients are frozen and no source refit occurs.',
      arms=[dict(arm=a) for a in ['positive-bp-adamw-weights','uniform-filtered','supplied-rule-oracle']],
      producer='Use frozen H3u source counts/exposures and inferred lines. F is finite source zero with E>=ln(264/.01), union all originally inferred lines. Weighted arm retains the positive original BP AdamW coupling on F, zero elsewhere; unary0. Uniform arm uses1on exactly F,0elsewhere. Oracle supplies true queen+camel support with1weights. No target label enters weighted/uniform construction.',
      positivity_gate='Every source-retained positive coupling and every transferred Fedge must be finite and strictly>0. Record minimum. No clipping, radius cutoff or dropping failed edges. A failed gate stops research searches and requires a transparent amendment; it is not repaired silently.',
      temperature='T(k)=u*(0.01**(k/19999)),k=0..19999. u is the arithmetic mean retained positive coefficient across unique canonical SOURCE offsets after line union, computed in float32 exactly as the source-only pilot. Uniform and oracleu=1. No target-temperature tuning.',
      kernel='Initialize givens plus uniformly chosen empty cells to exactlyq13. At each step choose uniformly one occupied nongiven celli and one emptycellj via independent maxima over disjoint sets of IIDuniform scores. DeltaE=field_j-field_i-Jij. Accept iffDelta<=0orU<exp(-Delta/T). Cardinality/givens preserved. Same proposal and annealing schedule for allarms. Matched seed/initial random stream; adaptive completion changes later RNG streams, so not claimed common post-initial random numbers.',
      certificate='Stop iff exact integer count of occupied Fedges is0. Count maintained with integer deltas independently of floatenergy; q13 preserved. All weights onF strictlypositive, so this is precisely the common zero-energy set. No native inherited margin certificate.',
      search=dict(q=13,unique_inputs=4096,chains=20,cap=20000,seeds=[0,1,2],end_temperature_ratio=.01,block=65536),
      primary=dict(arm='positive-bp-adamw-weights',control='uniform-filtered',metric='Any valid stopped zero-conflict completion among20chains; abstain ifnone.',gain_pp=5.,wrong_stops=0),
      decision='Support iff paired-input bootstrap one-sided95% lower primary gain>=5pp and zero weighted wrongstops. Falsify iff uppergain<5ppORany wrongstop; otherwise inconclusive. Bootstrap2000draws,RNG20261114,all3searchseeds clustered byinput. Alloracle/coverage/fallback comparisons descriptive.',
      secondary='Original lowest-running-energy and certificate-first-with-best-energy-fallback accuracy, valid distinct-solution coverage, zero-set capacity, allcap returns, moves/acceptances, exact-source support and target diagnostic attack recovery. No target diagnostic changes model.',
      capacity='Weighted/uniform support Fidentical, so common exact valid zero-conflict completion capacity is an actual ceiling only for the abstaining certified primary. It is not a bound on fallback accuracy. Supplied target-geometry audit is diagnostic only.',
      validation='Exhaustive84toyq3states and3024swaps across0/1/2clues; native direct energy parity, symmetric proposal and detailed balance at3fixed temperatures; CPU/GPU independent trace checks forq,givens,energy,fields,integercounts; source-only runtime pilot. Then independently rebuild full source support, weights, temperature, target expansion and exact cohort counts before fresh target searches.',
      numeric_audit='Independently reconstruct each saved board energy and integer conflict count. All stopped counts must beexact0, all boardsq13/givens. Record running energy drift and exact-energy selector differences; energy discrepancy>.1prevents support without changing wrong-stop falsification.',
      budget=dict(search_gpu_minutes_envelope=20,wall_minutes_envelope=10,logical_model_seed_jobs=9,optimizer_fits=0,source_only_pilot='128sourceinputs×20chains×2000swaps onCPU/GPU; feasibility only, no retuning.'),
      dataset=read(OUT/'dataset.json'),source_sha256={str(x.relative_to(ROOT)):sha(x) for x in sources},data_sha256={str(x.relative_to(ROOT)):sha(x) for x in data},
      limits=['New fixed-q Metropolis/annealing derivative, not paper solver reproduction.','Unchanged filtered support keeps source-global false exclusions. Positive weighting changes search but not exact zero-set capacity.','Learned projection discards nonuniform off-support couplings and unary, rather than only subtracting a constant from the original variable-cardinality energy.','One complete finite source universe and one inherited fitted trajectory; three seeds repeatsearch,not training.','Oracle target rules privileged; source-only producers never inspect targetgeometry or solutions.'])
    save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',sha256=sha(SRC/'protocol.json'))))

def construct():
    import numpy as np
    import torch
    frozen();d=np.load(OLD/'source-design.npz');m=torch.load(OLD/'models/bp-adamw-grouped/model.pt',weights_only=False);offsets=d['offsets'];mask=(d['counts']==0)&(d['exposure']>=math.log(264/.01));mask|=(offsets@d['lines'].T==0).any(1);coef=m['A'].numpy()[d['groups']]
    assert np.isfinite(coef[mask]).all() and (coef[mask]>0).all();u=float(coef[mask].mean());lookup={tuple(o):(bool(v),float(a)) for o,v,a in zip(offsets,mask,coef)};i,j=np.triu_indices(169,1);dr=j//13-i//13;dc=j%13-i%13;receipts=[]
    for arm in [c['arm'] for c in protocol()['arms']]:
        f=[];weights=[]
        for r,c in zip(dr,dc):
            if arm=='supplied-rule-oracle':inside=r==0 or c==0 or abs(r)==abs(c) or (r,abs(c)) in [(1,3),(3,1)];w=1.
            else:
                line=next((k for k,(a,b) in enumerate(d['lines']) if a*r+b*c==0),None)
                inside,w=(True,float(m['A'][line])) if line is not None else lookup.get((r,c),(False,0.))
                if arm=='uniform-filtered':w=1.
            f.append(inside);weights.append(w if inside else 0.)
        F=np.zeros((169,169),np.int32);F[i,j]=f;F[j,i]=F[i,j];J=np.zeros((169,169),np.float32);J[i,j]=weights;J[j,i]=J[i,j];assert np.isfinite(J).all() and (J[F.astype(bool)]>0).all()
        dest=OUT/f'models/{arm}';dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'model.npz',J=J,h=np.zeros(169,np.float32),F=F,unit=np.array(u if arm=='positive-bp-adamw-weights' else 1.),q=np.array(13))
        rec=dict(status='complete',arm=arm,source_patterns=176,optimizer_refits=0,unit=u if arm=='positive-bp-adamw-weights' else 1.,source_retained_offsets=int(mask.sum()),minimum_retained_positive_weight=float(J[F.astype(bool)].min()),target_forbidden_pairs=int(F.sum()//2),model_sha256=sha(dest/'model.npz'),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'));save(dest/'model.json',rec);receipts.append(rec)
    save(OUT/'construction.json',dict(status='complete',models=receipts));print(json.dumps(receipts))

def search(arm,seed,data,progress):
    import numpy as np
    path=OUT/f'models/{arm}/model.npz';m=np.load(path);inputs=data['test_inputs'];progress(arm=arm,seed=seed);start=time.perf_counter()
    z=backend.solve(m['J'],m['h'],m['F'],inputs,q=13,unit=float(m['unit']),chains=20,cap=20000,seed=seed,device='cuda');seconds=time.perf_counter()-start
    keys={row.tobytes() for row in data['all_solutions']};boards=z['boards'];valid=np.array([[r.tobytes() in keys and r[given].all() for r in samples] for given,samples in zip(inputs,boards)])
    stopped=z['stopped'];original=z['energy'].argmin(1);selected=np.where(stopped.any(1),np.where(stopped,z['energy'],np.inf).argmin(1),original);ids=np.arange(len(inputs));denom=np.minimum(data['test_solution_counts'],20);found=np.array([min(len({r.tobytes() for r in samples[ok]}),limit) for samples,ok,limit in zip(boards,valid,denom)])
    dest=OUT/f'search/{arm}/s{seed}';dest.mkdir(parents=True,exist_ok=True);z.update(boards=np.packbits(boards,-1),inputs=inputs,valid=valid,n20=valid[ids,original],n20_certificate=valid[ids,selected],certified_success=(stopped&valid).any(1),original_index=original,certificate_index=selected,found=found,denom=denom,solution_counts=data['test_solution_counts'])
    np.savez_compressed(dest/'records.npz',**z);rec=dict(status='complete',utc=now(),arm=arm,seed=seed,certified_success_pct=100*float(z['certified_success'].mean()),original_accuracy_pct=100*float(z['n20'].mean()),certificate_accuracy_pct=100*float(z['n20_certificate'].mean()),coverage_pct=100*float(found.sum()/denom.sum()),wrong_stops=int((stopped&~valid).sum()),capped_chains=int((~stopped).sum()),proposal_moves=int(z['moves'].sum()),accepted_swaps=int(z['accepted'].sum()),search_seconds=seconds,model_sha256=sha(path),records_sha256=sha(dest/'records.npz'),source_sha256=sha(__file__),backend_sha256=sha(SRC/'backend.py'),protocol_sha256=sha(SRC/'protocol.json'));save(dest/'result.json',rec);print(json.dumps(rec),flush=True)

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
