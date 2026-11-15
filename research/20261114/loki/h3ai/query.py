"""Pure model-cache query: reads input clues, never target solution labels."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import hashlib
import time
import numpy as np
from common import ROOT,SRC,OUT,sha,read,save,now,frozen

def solve(boards,index,inputs,bank,seed):
    rng=np.random.default_rng(2026113100+100*bank+seed);q=15;full=(1<<len(boards))-1;out=np.zeros((len(inputs),20,225),bool);available=np.zeros(len(inputs),np.int64);ids=np.full((len(inputs),20),-1,np.int64)
    for t,given in enumerate(inputs):
        mask=full
        for cell in np.flatnonzero(given):mask&=index[int(cell)]
        matches=[]
        while mask:bit=mask&-mask;mask^=bit;matches.append(bit.bit_length()-1)
        available[t]=len(matches)
        if not matches:continue
        take=rng.choice(matches,min(20,len(matches)),replace=False)
        if len(take)<20:take=np.concatenate([take,rng.choice(matches,20-len(take),replace=True)])
        ids[t]=take;out[t]=boards[take]
    return dict(boards=out,available=available,indices=ids,abstained=available==0,stopped=np.repeat((available>0)[:,None],20,axis=1))

def main():
    frozen();assert read(OUT/'compilation-gate.json')['status']=='complete_verified';assert not(OUT/'query-worker.json').exists();base=dict(started_utc=now(),pid=os.getpid(),no_target_labels_opened=True,protocol_sha256=sha(SRC/'protocol.json'));timings=[]
    for bank in range(2):
        folder=OUT/f'b{bank}';comp=read(folder/'compilation.json');assert comp['cache_sha256']==sha(folder/'cache.npz');start=time.perf_counter();cache=np.load(folder/'cache.npz');boards=cache['boards'];index=[int.from_bytes(row.tobytes(),'little')for row in cache['index']];load_seconds=time.perf_counter()-start;inputs=np.load(folder/'query-inputs.npy');jobs=[]
        for seed in[0,1,2]:
            start=time.perf_counter();z=solve(boards,index,inputs,bank,seed);elapsed=time.perf_counter()-start;dest=folder/f'exact/s{seed}';dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'records.npz',boards=np.packbits(z.pop('boards'),axis=-1),**z);r=dict(status='complete',utc=now(),bank=bank,seed=seed,query_seconds=elapsed,source_cache_sha256=sha(folder/'cache.npz'),inputs_bytes_sha256=hashlib.sha256(inputs.tobytes()).hexdigest(),records_sha256=sha(dest/'records.npz'),abstained_inputs=int(z['abstained'].sum()),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__),no_target_label_access_or_accuracy_computed=True);save(dest/'result.json',r);jobs.append(r);print('QUERY',bank,seed,elapsed,r['abstained_inputs'],flush=True)
        graph_seconds=read(OUT/'external-graph-gate.json')['families'][bank]['wall_seconds'];timings.append(dict(bank=bank,external_graph_gate_seconds=graph_seconds,cache_load_seconds=load_seconds,shared_build_and_parity_seconds=read(OUT/'compilation-gate.json')['shared_prerequisite_wall_seconds'],compile_wall_seconds=comp['compile_wall_seconds'],three_query_seconds=sum(j['query_seconds']for j in jobs),cpu_compile_load_three_query_seconds=read(OUT/'compilation-gate.json')['shared_prerequisite_wall_seconds']+comp['compile_wall_seconds']+graph_seconds+load_seconds+sum(j['query_seconds']for j in jobs),queries=jobs))
    save(OUT/'query-worker.json',dict(base,status='complete',completed_utc=now(),families=timings))
if __name__=='__main__':main()
