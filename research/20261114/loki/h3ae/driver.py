"""Fresh six-search comparison after source-only timing calibration."""
import argparse,fcntl,hashlib,os,signal,socket,subprocess,time
from common import ROOT,SRC,OUT,OLD,sha,read,save,now,frozen,budget

def worker(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    import torch
    import arbitrary_backend,permutation_backend
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;assert socket.gethostname()=='loki';p=frozen(target=True);lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pre=read(OUT/'preflight.json');assert pre['status']=='passed'
    for rel,h in pre['input_sha256'].items():assert sha(ROOT/rel)==h
    query=lambda *a:subprocess.check_output(a,text=True,timeout=20).strip();device=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader');assert device.split(',')[1].strip()not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    deadline=budget();base=dict(started_utc=now(),gpu=gpu,device=device,pid=os.getpid(),phase='target',bank=BANK,protocol_sha256=sha(SRC/'protocol.json'));receipt=OUT/f'target-worker-g{gpu}.json';dataset=read(OUT/'dataset.json');data=dict(np.load(OUT/'data.npz'));m=np.load(OUT/'model.npz');keys={row.tobytes()for row in data['all_solutions']}
    def progress(**v):save(receipt,dict(base,status='running',utc=now(),progress=v))
    try:
        for seed in[0,1,2]:
            kernel='arbitrary'if(gpu+seed)%2==0 else'permutation';cap=dataset['selected_caps'][kernel];folder=OUT/f'search/{kernel}/s{seed}';folder.mkdir(parents=True,exist_ok=True);assert not(folder/'result.json').exists()
            path=OUT/f'initial/s{seed}.npz';initial=np.unpackbits(np.load(path)['boards'],axis=-1)[...,:169].astype(bool);backend=arbitrary_backend if kernel=='arbitrary'else permutation_backend
            # Untimed short warmup counts in overall worker budget, not solve-call timing.
            progress(stage='warmup',kernel=kernel,cap=cap,seed=seed);torch.cuda.synchronize();warmstart=time.perf_counter();backend.solve(m['J'],m['h'],m['F'],data['test_inputs'],q=13,unit=float(m['unit']),chains=20,cap=32,seed=100,device='cuda',initial=initial);torch.cuda.synchronize();warmseconds=time.perf_counter()-warmstart
            progress(stage='timed_search',kernel=kernel,cap=cap,seed=seed);torch.cuda.synchronize();start=time.perf_counter();z=backend.solve(m['J'],m['h'],m['F'],data['test_inputs'],q=13,unit=float(m['unit']),chains=20,cap=cap,seed=seed,device='cuda',block=65536,end_ratio=.01,initial=initial);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            boards=z['boards'];inputs=data['test_inputs'];valid=np.array([[row.tobytes()in keys and row[given].all()for row in samples]for given,samples in zip(inputs,boards)]);stopped=z['stopped'];original=z['energy'].argmin(1);selected=np.where(stopped.any(1),np.where(stopped,z['energy'],np.inf).argmin(1),original);ids=np.arange(len(inputs));denom=np.minimum(data['test_solution_counts'],20);found=np.array([min(len({row.tobytes()for row in samples[ok]}),limit)for samples,ok,limit in zip(boards,valid,denom)])
            z.update(boards=np.packbits(boards,axis=-1),inputs=inputs,valid=valid,n20=valid[ids,original],n20_certificate=valid[ids,selected],certified_success=(stopped&valid).any(1),original_index=original,certificate_index=selected,found=found,denom=denom,solution_counts=data['test_solution_counts'])
            np.savez_compressed(folder/'records.npz',**z);record=dict(status='complete',utc=now(),bank=BANK,kernel=kernel,seed=seed,gpu=gpu,cap=cap,worker_solve_seconds=elapsed,warmup_seconds=warmseconds,
                certified_success_pct=100*float(z['certified_success'].mean()),original_accuracy_pct=100*float(z['n20'].mean()),certificate_accuracy_pct=100*float(z['n20_certificate'].mean()),coverage_pct=100*float(found.sum()/denom.sum()),wrong_stops=int((stopped&~valid).sum()),
                proposal_moves=int(z['moves'].sum()),accepted_swaps=int(z['accepted'].sum()),capped_chains=int((~stopped).sum()),initial_file_sha256=sha(path),initial_unpacked_bytes_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                model_sha256=sha(OUT/'model.npz'),records_sha256=sha(folder/'records.npz'),calibration_selection_sha256=sha(OUT/'calibration-selection.json'),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__),backend_sha256=sha(SRC.parent/f'h3aa/{kernel}_backend.py'))
            save(folder/'result.json',record);print(str(gpu)+' '+kernel+' seed'+str(seed)+' cap'+str(cap)+' '+str(round(elapsed,3))+'s success'+str(record['certified_success_pct']),flush=True)
        signal.setitimer(signal.ITIMER_REAL,0);save(receipt,dict(base,status='complete',completed_utc=now(),deadline_unix=deadline))
    except BaseException as e:save(receipt,dict(base,status='failed',completed_utc=now(),error=repr(e)));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--bank',type=int,choices=range(2),required=True);a.add_argument('--gpu',type=int,choices=[0,1],required=True);args=a.parse_args();BANK=args.bank;OUT=OUT/f'b{BANK}';worker(args.gpu)
