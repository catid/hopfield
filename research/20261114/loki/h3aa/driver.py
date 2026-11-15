"""Three-arm initialization/proposal triangle with a shared bounded GPU queue."""
import argparse,datetime as dt,fcntl,hashlib,json,os,signal,socket,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3aa';OUT=ROOT/'artifacts/research/20261114/loki/h3aa'
ARMS={'arbitrary-init-arbitrary-swaps':('arbitrary','arbitrary'),'permutation-init-arbitrary-swaps':('permutation','arbitrary'),'permutation-init-permutation-swaps':('permutation','permutation')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def frozen():
    p=read(SRC/'protocol.json')
    for group in['source_sha256','data_sha256']:
        for path,h in p[group].items():assert sha(ROOT/path)==h,path
    return p
def search(arm,seed,data,progress):
    import numpy as np
    import arbitrary_backend,permutation_backend
    initial_kind,kernel=ARMS[arm];path=OUT/f'initial/{initial_kind}/s{seed}.npz';initial=np.unpackbits(np.load(path)['boards'],axis=-1)[...,:169].astype(bool)
    m=np.load(OUT/'model.npz');backend=arbitrary_backend if kernel=='arbitrary'else permutation_backend;progress(arm=arm,seed=seed,initial_file_sha256=sha(path));start=time.perf_counter()
    z=backend.solve(m['J'],m['h'],m['F'],data['test_inputs'],q=13,unit=float(m['unit']),chains=20,cap=20000,seed=seed,device='cuda',block=65536,end_ratio=.01,initial=initial)
    seconds=time.perf_counter()-start;inputs=data['test_inputs'];keys={row.tobytes()for row in data['all_solutions']};boards=z['boards'];valid=np.array([[row.tobytes()in keys and row[given].all()for row in samples]for given,samples in zip(inputs,boards)])
    stopped=z['stopped'];original=z['energy'].argmin(1);selected=np.where(stopped.any(1),np.where(stopped,z['energy'],np.inf).argmin(1),original);ids=np.arange(len(inputs));denom=np.minimum(data['test_solution_counts'],20);found=np.array([min(len({row.tobytes()for row in samples[ok]}),limit)for samples,ok,limit in zip(boards,valid,denom)])
    folder=OUT/f'search/{arm}/s{seed}';z.update(boards=np.packbits(boards,axis=-1),inputs=inputs,valid=valid,n20=valid[ids,original],n20_certificate=valid[ids,selected],certified_success=(stopped&valid).any(1),original_index=original,certificate_index=selected,found=found,denom=denom,solution_counts=data['test_solution_counts'])
    np.savez_compressed(folder/'records.npz',**z);record=dict(status='complete',utc=now(),arm=arm,seed=seed,initialization=initial_kind,kernel=kernel,
        certified_success_pct=100*float(z['certified_success'].mean()),original_accuracy_pct=100*float(z['n20'].mean()),certificate_accuracy_pct=100*float(z['n20_certificate'].mean()),coverage_pct=100*float(found.sum()/denom.sum()),
        wrong_stops=int((stopped&~valid).sum()),capped_chains=int((~stopped).sum()),proposal_moves=int(z['moves'].sum()),accepted_swaps=int(z['accepted'].sum()),search_seconds=seconds,
        model_sha256=sha(OUT/'model.npz'),initial_file_sha256=sha(path),initial_unpacked_bytes_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),records_sha256=sha(folder/'records.npz'),
        protocol_sha256=sha(SRC/'protocol.json'),driver_sha256=sha(__file__),backend_sha256=sha(SRC/f'{kernel}_backend.py'))
    save(folder/'result.json',record);print(json.dumps(record),flush=True)
def worker(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;assert socket.gethostname()=='loki';lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=frozen();pre=read(OUT/'preflight.json');assert pre['status']=='passed'
    for path,h in pre['input_sha256'].items():assert sha(ROOT/path)==h,path
    query=lambda *a:subprocess.check_output(a,text=True,timeout=20).strip();device=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader');assert device.split(',')[1].strip()not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    with(OUT/'global-budget.json').open('a+')as budget:
        fcntl.flock(budget,fcntl.LOCK_EX);budget.seek(0);text=budget.read()
        if text:deadline=json.loads(text)['deadline_unix']
        else:deadline=time.time()+600;budget.write(json.dumps(dict(started_utc=now(),deadline_unix=deadline,global_wall_seconds=600,aggregate_gpu_worker_envelope=1200,protocol_sha256=sha(SRC/'protocol.json')))+'\n');budget.flush()
    def expired(*args):raise TimeoutError('Frozen shared10minute wall/20GPUminute budget reached')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()));data=dict(np.load(OUT/'data.npz'));base=dict(started_utc=now(),host='loki',gpu=gpu,pid=os.getpid(),device=device,protocol_sha256=sha(SRC/'protocol.json'));receipt=OUT/f'worker-g{gpu}.json'
    def progress(**values):save(receipt,dict(base,status='running',utc=now(),progress=values))
    progress(stage='starting')
    try:
        for arm in p['queue_order']:
            for seed in[0,1,2]:
                folder=OUT/f'search/{arm}/s{seed}';folder.mkdir(parents=True,exist_ok=True)
                if(folder/'result.json').exists():continue
                claim=(folder/'job.lock').open('a')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                try:
                    if not(folder/'result.json').exists():search(arm,seed,data,progress)
                finally:claim.close()
        signal.setitimer(signal.ITIMER_REAL,0);save(receipt,dict(base,status='complete',completed_utc=now()))
    except BaseException as exc:save(receipt,dict(base,status='failed',completed_utc=now(),error=repr(exc)));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=[0,1],required=True);args=a.parse_args();worker(args.gpu)
