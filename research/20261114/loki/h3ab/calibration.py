"""Source-derived target-shaped timing calibration, never target accuracy."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse
import fcntl
import hashlib
import itertools
import signal
import socket
import subprocess
import time
import numpy as np
from common import ROOT,SRC,OUT,BASE,OLD,sha,read,save,now,frozen,budget

def prepare():
    import torch
    from initialization import make
    torch.set_num_threads(1);p=frozen();assert not(OUT/'calibration-data.json').exists();source=np.load(BASE/'h3t/solutions-n12.npy');assert source.shape==(176,144)
    possible={k:set()for k in[1,2,3]}
    for board in source:
        cells=np.flatnonzero(board);positions=(cells//12*13+cells%12).tolist()
        for clues in[1,2,3]:
            possible[clues].update(sum((cell+1)*170**(2-j)for j,cell in enumerate(subset))for subset in itertools.combinations(positions,clues))
    rng=np.random.default_rng(2026112602);keys=np.sort(np.concatenate([rng.choice(np.array(sorted(possible[k]),np.int64),count,replace=False)for k,count in[(1,2),(2,205),(3,3889)]]))
    assert len(keys)==4096==len(np.unique(keys));inputs=np.zeros((4096,169),bool)
    for power in[2,1,0]:
        cells=keys//170**power%170;ok=cells>0;inputs[np.flatnonzero(ok),cells[ok]-1]=True
    grid=inputs.reshape(-1,13,13);assert(grid.sum(1)<=1).all()and(grid.sum(2)<=1).all()and not grid[:,12].any()and not grid[:,:,12].any()
    np.savez_compressed(OUT/'calibration-data.npz',keys=keys,inputs=inputs);receipts=[]
    for seed in[101,102,103]:
        initial=make(inputs,q=13,chains=20,seed=seed,kind='permutation');path=OUT/f'calibration-initial/s{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,boards=np.packbits(initial,axis=-1))
        receipts.append(dict(seed=seed,path=str(path.relative_to(ROOT)),file_sha256=sha(path),unpacked_bytes_sha256=hashlib.sha256(initial.tobytes()).hexdigest()))
    save(OUT/'calibration-data.json',dict(status='prepared_from_N12_source_only',utc=now(),source_patterns=176,source_sha256=sha(BASE/'h3t/solutions-n12.npy'),
        no_target_bank_labels_or_heldout_keys_accessed=True,unique_inputs=4096,by_clues={'1':2,'2':205,'3':3889},available_unique_source_subsets={str(k):len(v)for k,v in possible.items()},data_sha256=sha(OUT/'calibration-data.npz'),initial_states=receipts,protocol_sha256=sha(SRC/'protocol.json')))
    print('SOURCE_CALIBRATION_PREPARED',sha(OUT/'calibration-data.json'),flush=True)

def worker(gpu,phase):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    import torch
    import arbitrary_backend,permutation_backend
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;assert socket.gethostname()=='loki';p=frozen();lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    query=lambda *args:subprocess.check_output(args,text=True,timeout=20).strip();device=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader');assert device.split(',')[1].strip()not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    deadline=budget();started=now();receipt=OUT/f'calibration-worker-{phase}-g{gpu}.json';data=np.load(OUT/'calibration-data.npz');inputs=data['inputs'];metadata=read(OUT/'calibration-data.json');assert metadata['data_sha256']==sha(OUT/'calibration-data.npz')
    m=np.load(OLD/'model.npz');assert sha(OLD/'model.npz')==p['model_sha256'];initial={}
    for r in metadata['initial_states']:
        path=ROOT/r['path'];assert sha(path)==r['file_sha256'];states=np.unpackbits(np.load(path)['boards'],axis=-1)[...,:169].astype(bool);assert hashlib.sha256(states.tobytes()).hexdigest()==r['unpacked_bytes_sha256'];initial[r['seed']]=states
    base=dict(started_utc=started,host='loki',gpu=gpu,phase=phase,device=device,pid=os.getpid(),protocol_sha256=sha(SRC/'protocol.json'))
    def progress(**values):save(receipt,dict(base,status='running',utc=now(),progress=values))
    try:
        # Full-shape short warmups instantiate both kernels before measured trials.
        warmups=[]
        for name,backend in[('arbitrary',arbitrary_backend),('permutation',permutation_backend)]:
            progress(stage='warmup',kernel=name);torch.cuda.synchronize();start=time.perf_counter();z=backend.solve(m['J'],m['h'],m['F'],inputs,q=13,unit=float(m['unit']),chains=20,cap=32,seed=100,device='cuda',initial=initial[101]);torch.cuda.synchronize();seconds=time.perf_counter()-start
            y=z['boards'];assert(y.sum(-1)==13).all()and(y>=inputs[:,None,:]).all();sample=y[:32].astype(np.int64);assert np.array_equal(z['conflicts'][:32],(sample*(sample@m['F'].astype(np.int64))).sum(-1)//2)
            if name=='permutation':assert(y.reshape(-1,13,13).sum(1)==1).all()and(y.reshape(-1,13,13).sum(2)==1).all()
            warmups.append(dict(kernel=name,seconds=seconds,proposals=int(z['moves'].sum()),checked_before_timing=True))
        configurations=[('arbitrary',20000),*[('permutation',cap)for cap in[4000,8000,12000,16000]]]
        if phase=='refinement':configurations=[('permutation',read(OUT/'refinement-request.json')['cap'])]
        for index,seed in enumerate([101,102,103]):
            ordered=configurations if(gpu+index)%2==0 else configurations[::-1]
            for name,cap in ordered:
                folder=OUT/f'calibration/{name}-c{cap}/g{gpu}-s{seed}';folder.mkdir(parents=True,exist_ok=True);assert not(folder/'result.json').exists();progress(stage='timed_trial',kernel=name,cap=cap,seed=seed)
                backend=arbitrary_backend if name=='arbitrary'else permutation_backend;torch.cuda.synchronize();start=time.perf_counter();z=backend.solve(m['J'],m['h'],m['F'],inputs,q=13,unit=float(m['unit']),chains=20,cap=cap,seed=seed,device='cuda',block=65536,end_ratio=.01,initial=initial[seed]);torch.cuda.synchronize();elapsed=time.perf_counter()-start
                record=dict(status='complete',utc=now(),phase=phase,kernel=name,cap=cap,gpu=gpu,seed=seed,worker_solve_seconds=elapsed,proposal_moves=int(z['moves'].sum()),accepted_swaps=int(z['accepted'].sum()),stopped_chains=int(z['stopped'].sum()),
                    no_target_accuracy_computed=True,initial_bytes_sha256=hashlib.sha256(initial[seed].tobytes()).hexdigest(),calibration_data_sha256=sha(OUT/'calibration-data.npz'),model_sha256=sha(OLD/'model.npz'),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'))
                save(folder/'result.json',record);print(str(gpu)+' '+name+' cap'+str(cap)+' seed'+str(seed)+' '+str(round(elapsed,4))+'s '+str(record['proposal_moves'])+'proposals',flush=True)
        signal.setitimer(signal.ITIMER_REAL,0);save(receipt,dict(base,status='complete',completed_utc=now(),warmups=warmups,deadline_unix=deadline))
    except BaseException as e:save(receipt,dict(base,status='failed',completed_utc=now(),error=repr(e)));raise

def selection():
    p=frozen();assert not(OUT/'calibration-selection.json').exists();groups={}
    for folder in sorted((OUT/'calibration').iterdir()):
        trials=[read(path)for path in sorted(folder.glob('*/result.json'))];assert len(trials)==6 and {(r['gpu'],r['seed'])for r in trials}=={(gpu,seed)for gpu in[0,1]for seed in[101,102,103]}
        for r in trials:assert r['status']=='complete'and r['no_target_accuracy_computed']and r['protocol_sha256']==sha(SRC/'protocol.json')
        times=[r['worker_solve_seconds']for r in trials];groups[folder.name]=dict(kernel=trials[0]['kernel'],cap=trials[0]['cap'],median_seconds=float(np.median(times)),minimum_seconds=min(times),maximum_seconds=max(times),
            per_gpu_median_seconds={str(g):float(np.median([r['worker_solve_seconds']for r in trials if r['gpu']==g]))for g in[0,1]},trials=trials)
    if(OUT/'refinement-request.json').exists():assert f"permutation-c{read(OUT/'refinement-request.json')['cap']}"in groups
    reference=groups['arbitrary-c20000'];permutation=[v for v in groups.values()if v['kernel']=='permutation'];best=min(permutation,key=lambda v:(abs(v['median_seconds']/reference['median_seconds']-1),v['cap']));ratio=best['median_seconds']/reference['median_seconds'];matched=abs(ratio-1)<=.05
    if not matched and not(OUT/'refinement-request.json').exists():
        ordered=sorted(permutation,key=lambda v:v['cap']);brackets=[(a,b)for a,b in zip(ordered[:-1],ordered[1:])if a['median_seconds']<reference['median_seconds']<b['median_seconds']]
        if brackets:
            a,b=brackets[0];raw=a['cap']+(reference['median_seconds']-a['median_seconds'])/(b['median_seconds']-a['median_seconds'])*(b['cap']-a['cap']);cap=int(np.rint(raw/128)*128)
            if cap not in[v['cap']for v in permutation]and a['cap']<=cap<=b['cap']:
                save(OUT/'refinement-request.json',dict(status='one_source_time_only_interpolation',utc=now(),cap=cap,unrounded_cap=raw,low=a['cap'],high=b['cap'],reference_seconds=reference['median_seconds'],initial_grid_groups=groups,protocol_sha256=sha(SRC/'protocol.json')));print('REFINEMENT_REQUIRED',cap,flush=True);return
    rng=np.random.default_rng(2026112604);ratios=[]
    for _ in range(2000):
        seeds=rng.choice([101,102,103],3,replace=True);a=[r['worker_solve_seconds']for seed in seeds for r in reference['trials']if r['seed']==seed];b=[r['worker_solve_seconds']for seed in seeds for r in best['trials']if r['seed']==seed];ratios.append(float(np.median(b)/np.median(a)))
    record=dict(status='matched_source_median_times'if matched else'inconclusive_source_timing_match',utc=now(),selected_arbitrary_cap=20000,selected_permutation_cap=best['cap'],source_median_time_ratio=ratio,
        source_matching_tolerance=.05,source_ratio_seed_cluster_bootstrap_5_95=np.quantile(ratios,[.05,.95]).tolist(),uncertainty='Three source search streams; each repeated on bothphysicalGPUs. Bootstrap resamples streams retaining bothdevice measurements. Six timings are not six independent searchstreams.',
        interpolation_used=(OUT/'refinement-request.json').exists(),all_calibration_groups=groups,no_target_accuracy_used=True,counts_frozen_before_target_cohort=True,protocol_sha256=sha(SRC/'protocol.json'),
        calibration_result_sha256={str(path.relative_to(ROOT)):sha(path)for path in sorted((OUT/'calibration').rglob('result.json'))})
    save(OUT/'calibration-selection.json',record);print('CALIBRATION_SELECTION',record['status'],best['cap'],ratio,'SHA',sha(OUT/'calibration-selection.json'),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();s=a.add_subparsers(dest='mode',required=True);s.add_parser('prepare');s.add_parser('select');w=s.add_parser('worker');w.add_argument('--gpu',type=int,choices=[0,1],required=True);w.add_argument('--phase',choices=['grid','refinement'],default='grid');args=a.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='select':selection()
    else:worker(args.gpu,args.phase)
