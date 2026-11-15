"""Fresh independently sampled source banks, then unchanged count-gap fits."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import time
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/20261114/foureyes/noise_source_replication'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def claim(gpu,job_id=None,status=None):
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);jobs=json.loads((OUT/'queue.json').read_text())
        if job_id is None:
            job=next((j for j in jobs if j['status']=='pending'),None)
            if job is not None:job.update(status='running',gpu=gpu,pid=os.getpid(),started=time.time())
        else:
            job=next(j for j in jobs if j['shard']==job_id);job.update(status=status,finished=time.time())
        write(OUT/'queue.json',jobs);return job

def prepare(p,protocol):
    import numpy as np
    start=time.monotonic();assert not OUT.exists();OUT.mkdir(parents=True)
    source=ROOT/p['source'];assert sha(source)==p['source_sha256']
    excluded=np.zeros(p['pool_size'],bool);evidence={}
    for spec in p['exclusions']:
        path=ROOT/spec['path'];assert sha(path)==spec['sha256'];manifest=json.loads(path.read_text())
        for item in manifest['inputs']:
            file=ROOT/item['completion_path'];assert sha(file)==item['completion_sha256']
            excluded[np.load(file)['source_row']]=True;evidence[str(file.relative_to(ROOT))]=sha(file)
    eligible=np.flatnonzero(~excluded)
    selected=np.stack([np.random.default_rng(p['bank_rng_base']+b).choice(eligible,p['rows_per_bank'],replace=False) for b in range(p['banks'])])
    assert not excluded[selected].any() and all(len(set(r))==p['rows_per_bank'] for r in selected)
    # Dense corpus buffers avoid a Python object for each requested occurrence.
    answers=np.empty((p['pool_size'],81),dtype=np.uint8);givens=np.empty_like(answers,dtype=bool)
    with source.open(newline='') as f:
        for index,row in enumerate(csv.DictReader(f)):
            answer=np.frombuffer(row['answer'].encode(),dtype=np.uint8)-ord('1')
            question=np.frombuffer(row['question'].encode(),dtype=np.uint8)
            assert answer.shape==question.shape==(81,) and (answer<9).all()
            mask=(question>=ord('1'))&(question<=ord('9'))
            assert np.array_equal(question[mask]-ord('1'),answer[mask])
            answers[index]=answer;givens[index]=mask
    assert index+1==p['pool_size']
    entries=[];shard_size=p['shard_banks']
    for shard,begin in enumerate(range(0,p['banks'],shard_size)):
        rows=selected[begin:begin+shard_size];path=OUT/f'inputs-{shard:02d}.npz'
        np.savez_compressed(path,solutions=answers[rows],given=givens[rows],source_row=rows,banks=np.arange(begin,begin+len(rows)))
        entries.append(dict(shard=shard,banks=list(map(int,range(begin,begin+len(rows)))),path=str(path.relative_to(ROOT)),sha256=sha(path)))
    write(OUT/'inputs.json',dict(status='prepared',protocol_sha256=sha(protocol),producer_sha256=sha(__file__),source_sha256=sha(source),numpy=np.__version__,exclusion_inputs=evidence,excluded_rows=int(excluded.sum()),eligible_rows=len(eligible),selected_occurrences=selected.size,unique_selected_rows=len(np.unique(selected)),shards=entries,wall_s=time.monotonic()-start))
    write(OUT/'queue.json',[dict(shard=e['shard'],status='pending') for e in entries])
    print(json.dumps(dict(status='prepared',banks=p['banks'],wall_s=time.monotonic()-start)),flush=True)

def worker(p,protocol,gpu):
    assert platform.node()=='foureyes'
    lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True);assert uuid not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    manifest=json.loads((OUT/'inputs.json').read_text());assert manifest['protocol_sha256']==sha(protocol) and manifest['producer_sha256']==sha(__file__)
    ii,jj=np.triu_indices(81,1);it=torch.as_tensor(ii,device='cuda');jt=torch.as_tensor(jj,device='cuda')
    def execute(job):
        start=time.monotonic();entry=manifest['shards'][job['shard']];path=ROOT/entry['path'];assert sha(path)==entry['sha256']
        z=np.load(path);clean=torch.as_tensor(z['solutions'].astype(np.int64),device='cuda');free=torch.as_tensor(~z['given'],device='cuda');assert (free.sum(2)>=9).all()
        dest=OUT/f'shard-{job["shard"]:02d}';dest.mkdir(exist_ok=False)
        cells=[];values=[];ties=0
        for k,bank in enumerate(z['banks']):
            gen=torch.Generator(device='cuda').manual_seed(p['noise_rng_base']+int(bank))
            keys=torch.rand((3000,81),generator=gen,dtype=torch.float64,device='cuda');keys.masked_fill_(~free[k],float('inf'))
            v,indices=keys.topk(10,dim=1,largest=False,sorted=True);tied=(v.diff(dim=1)==0).any(1)
            while bool(tied.any()):
                ties+=int(tied.sum());new=torch.rand((int(tied.sum()),81),generator=gen,dtype=torch.float64,device='cuda');new.masked_fill_(~free[k,tied],float('inf'));keys[tied]=new
                v,indices=keys.topk(10,dim=1,largest=False,sorted=True);tied=(v.diff(dim=1)==0).any(1)
            cc=indices[:,:9];vv=(clean[k].gather(1,cc)+torch.randint(1,9,cc.shape,generator=gen,device='cuda'))%9
            cells.append(cc);values.append(vv)
        cells=torch.stack(cells);values=torch.stack(values);noisy=clean.clone().scatter_(2,cells,values)
        assert (noisy!=clean).sum(2).eq(9).all() and torch.equal(noisy[~free],clean[~free])
        arrays={};preflight=[]
        sample=noisy[0].cpu().numpy()
        for n in p['sample_sizes']:
            encoded=torch.nn.functional.one_hot(noisy[:,:n],9).float();features=encoded.permute(0,2,1,3).reshape(len(clean),81,n*9)
            matrix=torch.bmm(features,features.transpose(1,2));assert torch.equal(matrix,matrix.round())
            counts=matrix[:,it,jt].to(torch.int16).cpu().numpy().astype(np.uint16)
            assert np.array_equal(counts[0],(sample[:n,ii]==sample[:n,jj]).sum(0));preflight.append(n)
            thresholds=[]
            for row in counts:
                distinct=np.unique(row.astype(np.int64));ix=int(np.diff(distinct).argmax()) if len(distinct)>1 else None
                thresholds.append(int((distinct[ix]+distinct[ix+1])//2) if ix is not None else 0)
            arrays[f'n{n}']=counts;arrays[f't{n}']=np.asarray(thresholds)
        torch.cuda.synchronize();assert time.monotonic()-start<p['budget']['max_producer_job_seconds']
        write(dest/'progress.json',dict(shard=job['shard'],models_done=len(clean)*len(p['sample_sizes']),wall_s=time.monotonic()-start,pid=os.getpid(),gpu=gpu))
        np.savez_compressed(dest/'counts.npz',**arrays);np.savez_compressed(dest/'corruptions.npz',cells=cells.cpu().numpy().astype(np.uint8),values=values.cpu().numpy().astype(np.uint8))
        write(dest/'receipt.json',dict(status='complete',shard=job['shard'],banks=z['banks'].tolist(),gpu=gpu,uuid=uuid,protocol_sha256=sha(protocol),producer_sha256=sha(__file__),input_sha256={entry['path']:sha(path)},output_sha256={name:sha(dest/name) for name in ['counts.npz','corruptions.npz']},cpu_preflight_one_bank_sizes=preflight,tie_rows=ties,wall_s=time.monotonic()-start,torch=torch.__version__,numpy=np.__version__,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        print(json.dumps(dict(shard=job['shard'],status='complete',wall_s=time.monotonic()-start)),flush=True)
    while True:
        job=claim(gpu)
        if job is None:break
        try:execute(job)
        except BaseException:claim(gpu,job['shard'],'failed');raise
        claim(gpu,job['shard'],'complete')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--gpu',type=int);args=parser.parse_args()
    protocol=HERE/'noise_source_replication.json';p=json.loads(protocol.read_text())
    if args.prepare:prepare(p,protocol)
    else:worker(p,protocol,args.gpu)

if __name__=='__main__':main()
