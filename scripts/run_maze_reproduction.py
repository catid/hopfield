"""Fresh maze fits and single-pass full-test evaluation in task-owned output folders.

Uses the pinned upstream model/optimizer/inference without changing numerical logic.
Captures predictions from that one evaluation for independent NumPy/BFS validation.
Timing is explicit: MazeUnique loop vs complete cold/warm fit; MazeHard complete cold.
"""
from __future__ import annotations
import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def utc():return datetime.now(timezone.utc).isoformat()


def run(cmd):
    return subprocess.run(cmd,text=True,capture_output=True,check=True).stdout.strip()


@contextmanager
def patched(obj,name,value):
    old=getattr(obj,name);setattr(obj,name,value)
    try:yield
    finally:setattr(obj,name,old)


def path_audit(tokens,predictions,labels):
    """Independent CPU check: exact labels and BFS shortest/simple-path validity.

    BFS neighbor order is down,left,right,up. This also checks the claimed MazeHard
    target tie convention independently of the released loader and evaluator.
    """
    import numpy as np
    exact=np.all(predictions.astype(bool)==labels.astype(bool),axis=(1,2))
    proper=[];shortest=[];bfs_label=[];lengths=[]
    for tok,pred,label in zip(tokens,predictions.astype(bool),labels.astype(bool)):
        starts=np.argwhere(tok==2);goals=np.argwhere(tok==3)
        if len(starts)!=1 or len(goals)!=1:raise ValueError('expected one start and one goal')
        st=tuple(starts[0]);goal=tuple(goals[0]);h,w=tok.shape
        parent={st:None};queue=deque([st]);distance={st:0}
        while queue and goal not in parent:
            i,j=queue.popleft()
            for di,dj in [(1,0),(0,-1),(0,1),(-1,0)]:
                q=(i+di,j+dj)
                if 0<=q[0]<h and 0<=q[1]<w and tok[q]!=0 and q not in parent:
                    parent[q]=(i,j);distance[q]=distance[(i,j)]+1;queue.append(q)
        expected=np.zeros((h,w),bool)
        if goal in parent:
            q=goal
            while q is not None:expected[q]=True;q=parent[q]
        bfs_label.append(bool(np.array_equal(expected,label)))
        degree=np.zeros((h,w),int)
        degree[1:]+=pred[:-1];degree[:-1]+=pred[1:];degree[:,1:]+=pred[:,:-1];degree[:,:-1]+=pred[:,1:]
        valid=bool(pred[st] and pred[goal] and not np.any(pred & (tok==0)))
        valid &= bool(degree[st]==1 and degree[goal]==1 and np.all(degree[pred & (tok!=2) & (tok!=3)]==2))
        reached={st};queue=deque([st])
        while queue:
            i,j=queue.popleft()
            for di,dj in [(1,0),(0,-1),(0,1),(-1,0)]:
                q=(i+di,j+dj)
                if 0<=q[0]<h and 0<=q[1]<w and pred[q] and q not in reached:
                    reached.add(q);queue.append(q)
        valid &= len(reached)==int(pred.sum())
        proper.append(valid);shortest.append(valid and goal in distance and int(pred.sum())==distance[goal]+1)
        lengths.append(int(pred.sum()))
    return dict(n=len(exact),exact_solved=int(exact.sum()),wrong_indices=np.flatnonzero(~exact).tolist(),
                valid_simple_paths=int(sum(proper)),shortest_paths=int(sum(shortest)),
                labels_match_independent_BFS=int(sum(bfs_label)),labels_not_matching_BFS=np.flatnonzero(~np.asarray(bfs_label)).tolist(),
                prediction_path_lengths=lengths,
                protocol='NumPy exact match over ALL cells; independent deque BFS down,left,right,up; walls/endpoints/degree/connectivity/shortest length checked')


def compare_weights(model,reference):
    import torch
    ref=torch.load(reference,map_location='cpu',weights_only=True)
    cur={k:v.detach().cpu() for k,v in model.state_dict().items()}
    if set(ref)!=set(cur):raise ValueError('checkpoint parameter names differ')
    per={k:dict(max_abs_diff=float((cur[k].double()-ref[k].double()).abs().max()),reference_max_abs=float(ref[k].abs().max()),bitwise_equal=bool(torch.equal(cur[k],ref[k]))) for k in ref}
    return dict(reference=str(reference),reference_sha256=sha256(reference),parameters=per,
                all_bitwise=all(v['bitwise_equal'] for v in per.values()),
                note='Reference was trained on H200; Blackwell numerical or RNG differences are reported, not silently treated as exact reproduction.')


def provenance(args):
    import torch
    files=sorted(p for d in ['cram','experiments'] for p in (args.upstream/d).glob('*.py'))
    code={str(p.relative_to(args.upstream)):sha256(p) for p in files}
    data=Path(os.environ.get('CRAM_DATA',args.upstream/'data'))
    paths=sorted((data/'eqr/maze-30x30-unique-1k').glob('*/*.npy')) if args.task.startswith('unique') else sorted((data/'maze-hard').glob('*.csv'))
    return dict(started_utc=utc(),hostname=platform.node(),python=sys.version,interpreter=sys.executable,
                torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                driver_inventory=run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader']),
                argv=sys.argv,wrapper_sha256=sha256(__file__),upstream_commit='62bbe830c7d6f87f2ba18f30a28039258b986454',source_files_sha256=code,
                dataset_files_sha256={str(p.relative_to(data)):sha256(p) for p in paths},
                source_digest=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest(),
                timing_hardware='Blackwell local measurement; not H200 paper timing')


def unique_fit(args):
    import numpy as np
    import torch
    import maze_unique as U
    from cram import maze_unique as M
    cfg=U.Train(split='train',seed=args.seed,kernel='free',free_init=0.,opt='sgd',momentum=.5,cosine=True,lr=1.,updates=20,l1=0.,score=(),out=str(args.out))
    save(args.out/'config.json',dataclasses.asdict(cfg))
    print(json.dumps({'event':'fit_start','task':'maze_unique','seed':args.seed,'updates':20}),flush=True)
    model,info=U.train(cfg)
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite parameters')
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},args.out/'model.pt')
    save(args.out/'train.json',info)
    compare=compare_weights(model,args.upstream/'checkpoints/maze-unique'/f'seed{args.seed}'/'model.pt')
    save(args.out/'weight_comparison.json',compare)
    captured=[];orig=M.gibbs
    def capture(*a,**kw):
        result=orig(*a,**kw)
        captured.append({'inputs':a[1].detach().cpu().numpy(),'labels':a[2].detach().cpu().numpy(),
                         'predictions':result['pred'].detach().cpu().numpy(),'stopped':result['stopped'].detach().cpu().numpy()})
        return result
    with torch.no_grad(),patched(M,'gibbs',capture):
        result=U.score(model,'test',U.Search(search='gibbs',T=0.0,seed=0))
    arrays={k:np.concatenate([c[k] for c in captured]) for k in captured[0]}
    audit=path_audit(arrays['inputs'],arrays['predictions'],arrays['labels'])
    assert audit['exact_solved']==result['solved'] and audit['n']==1000
    assert int((arrays['stopped'] & ~np.all(arrays['predictions']==arrays['labels'],axis=(1,2))).sum())==result['wrong_stops']
    np.savez_compressed(args.out/'test_predictions.npz',**arrays)
    save(args.out/'test.json',result);save(args.out/'independent_test_audit.json',audit)
    return dict(task='maze_unique',seed=args.seed,n=1000,solved=result['solved'],wrong_stops=result['wrong_stops'],budget_failures=result['budget_failures'],
                train_loop_s=info['loop_s'],mean_flop=result['mean_flop'],all_weights_bitwise_reference=compare['all_bitwise'],independent_shortest_paths=audit['shortest_paths'],
                inference_timing_note='Prediction capture adds CPU transfer to score seconds; use algorithm FLOPs, not this instrumented elapsed time for speed claims.')


def hard_fit(args):
    import numpy as np
    import torch
    import maze_hard_latent as H
    from cram import maze_h3 as M
    recipe_path=args.upstream/'experiments/recipes/maze-hard-latent.json'
    frozen=json.loads(recipe_path.read_text());recipe=H.Recipe(**frozen['recipe'])
    save(args.out/'config.json',dataclasses.asdict(recipe))
    tok,y=M.load('train')
    print(json.dumps({'event':'fit_start','task':'maze_hard','seed':args.seed,'updates':recipe.updates}),flush=True)
    torch.cuda.init();torch.cuda.synchronize()
    model,times,curve=H.fit_recipe(tok,y,args.seed,recipe,ckpt=args.out/'rolling.pt',log=args.out/'train.jsonl',triton=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite parameters')
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},args.out/'model.pt')
    save(args.out/'train.json',dict(times=times,curve=curve,params=H.count_params(model,recipe),latent=H.latent_summary(model)))
    compare=compare_weights(model,args.upstream/'checkpoints/maze-hard'/f'seed{args.seed}'/'model.pt')
    save(args.out/'weight_comparison.json',compare)
    # Official final scorer rebuilds the model, reverting settle to PyTorch. Match it.
    scored_model=H.build(recipe).cuda()
    scored_model.load_state_dict(torch.load(args.out/'model.pt',map_location='cuda',weights_only=True))
    for p in scored_model.parameters():p.requires_grad_(False)
    ttok,ty=M.load('test')
    captured=[];orig=M.descend
    def capture(*a,**kw):
        pred,used,stopped=orig(*a,**kw)
        captured.append({'predictions':pred.detach().cpu().numpy(),'steps':used.detach().cpu().numpy(),'stopped':stopped.detach().cpu().numpy()})
        return pred,used,stopped
    with patched(M,'descend',capture):result=H.evaluate(scored_model,ttok,ty)
    arrays={k:np.concatenate([c[k] for c in captured]) for k in captured[0]}
    arrays.update(inputs=ttok,labels=ty)
    audit=path_audit(ttok,arrays['predictions'],ty)
    exact=np.all(arrays['predictions'].astype(bool)==ty,axis=(1,2))
    assert int((exact & arrays['stopped']).sum())==result['solved'] and audit['n']==1000
    assert int((~exact & arrays['stopped']).sum())==result['wrong_stops']
    np.savez_compressed(args.out/'test_predictions.npz',**arrays)
    save(args.out/'test.json',result);save(args.out/'independent_test_audit.json',audit)
    return dict(task='maze_hard',seed=args.seed,n=1000,solved=result['solved'],wrong_stops=result['wrong_stops'],budget_failures=result['budget_failures'],
                cold_fit_s=times['total_s'],train_loop_s=times['train_s'],mean_flop=result['flop_mean'],
                flop_convention='upstream maze_hard_latent.evaluate legacy component count; paper Table8 uses broader FC2 count (~1.41e7). Do not compare these counts directly.',
                all_weights_bitwise_reference=compare['all_bitwise'],independent_shortest_paths=audit['shortest_paths'])


def unique_time(args):
    import maze_unique_fit_time as T
    T.OUT=args.out/'timing'
    T.ROWS['maze-unique__cram-sgd-fg']['run']=str(args.out)
    # Preserve the timing scope while correcting the paper's hardcoded hardware label.
    import torch
    hw = torch.cuda.get_device_name() if args.device == 'cuda' else 'not used for this CPU-only run'
    T.DEFINITION = T.DEFINITION.replace('GPU: one H200.', f'GPU: {hw}.')
    T.cmd_run(T.Run(row='maze-unique__cram-sgd-fg',seed=args.seed,device=args.device,repeats=7))
    return json.loads((T.OUT/'maze-unique__cram-sgd-fg'/f's{args.seed}-{args.device}.json').read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['unique','hard','unique-time']);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--upstream',type=Path,default=Path(__file__).resolve().parents[1]/'upstream')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args();args.upstream=args.upstream.resolve();args.out=args.out.resolve()
    sys.path[:0]=[str(args.upstream),str(args.upstream/'experiments')]
    import torch
    torch.set_num_threads(1)
    # The official CPU timing command sets interop threads itself, once.
    if not (args.task == 'unique-time' and args.device == 'cpu'):
        torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    final=args.out/(f'timing_summary_{args.device}.json' if args.task=='unique-time' else 'summary.json')
    if final.exists():raise SystemExit(f'Refusing to overwrite finished run {final}')
    save(args.out/(f'provenance_time_{args.device}.json' if args.task=='unique-time' else 'provenance.json'),provenance(args))
    result={'unique':unique_fit,'hard':hard_fit,'unique-time':unique_time}[args.task](args)
    result['completed_utc']=utc();result['model_sha256']=sha256(args.out/'model.pt')
    save(final,result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
