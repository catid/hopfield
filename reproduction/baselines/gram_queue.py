"""Persistent main-table GRAM-tree queue, including its HRM/TRM controls.

All final fits use frozen shell recipes. Deterministic controls score one sample,
matching their paper rows; GRAM records20 samples and all declared inference depths.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,json,os,socket,subprocess,time
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1];CODE=ROOT/'code';ART=ROOT/'artifacts';PY=ROOT/'.venv/bin/python'
ART.mkdir(parents=True,exist_ok=True)

def event(**kw):
 kw['utc']=datetime.now(timezone.utc).isoformat()
 with (ART/'gram-queue.jsonl').open('a') as f:f.write(json.dumps(kw)+'\n')
 print(json.dumps(kw),flush=True)

def command(name,argv,cwd=CODE,extra=None):
 env=dict(os.environ);env.update(extra or {})
 event(stage=name,status='started',argv=[str(x) for x in argv]);begin=time.monotonic()
 with (ART/f'{name}.log').open('a') as f:
  p=subprocess.Popen([str(x) for x in argv],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT);event(stage=name,status='running',pid=p.pid);rc=p.wait()
 event(stage=name,status='complete' if not rc else 'failed',exit_code=rc,wall_s=time.monotonic()-begin)
 if rc:raise SystemExit(f'{name} failed; see log')

def options(task):
 cmd=f'source {ROOT}/env.sh; source {CODE}/gram/configs.sh; printf "%s\\0" "${{GRAM_{task.upper()}[@]}}"'
 return subprocess.check_output(['bash','-c',cmd]).decode().strip('\0').split('\0')

def score(task,seed):
 fit=CODE/f'runs/gram/{task}-s{seed}';out=CODE/f'runs/gram/score/{task}-s{seed}';out.mkdir(parents=True,exist_ok=True)
 ckpt=max((p for p in fit.glob('step_*') if p.is_dir()),key=lambda p:int(p.name.split('_')[1]))/'checkpoint.pt'
 cfg=options(task)+[f'seed={seed}'];name=f'gram-{task}-s{seed}';steps='1,2,4,8,12,16,24,32'
 if task in ['sudoku','mazeunique']:
  per=2 if task=='sudoku' else 20
  for first in range(0,20,per):
   shard=out/f'shard-{first}-{first+per}'
   if not shard.with_suffix('.json').exists():command(name+f'-score-{first}',[PY,CODE/'gram/score.py','run',shard.with_suffix('.npz'),ckpt,first,first+per,steps,*cfg],Path(os.environ['GRAM_SRC']),dict(TORCHINDUCTOR_CACHE_DIR=str(fit/'.inductor-score')))
  if not (out/'score.json').exists():command(name+'-merge',[PY,CODE/'gram/score.py','merge',out/'score.json',*sorted(out.glob('shard-*.npz'))],extra=dict(CUDA_VISIBLE_DEVICES=''))
  if task=='sudoku' and not (out/'score-unseen.json').exists():command(name+'-unseen',[PY,CODE/'gram/unseen.py',out],extra=dict(CUDA_VISIBLE_DEVICES=''))
 else:
  samples=1 if task.startswith(('hrm_','trm_')) else 20
  shard=out/f'shard-0-{samples}'
  if not shard.with_suffix('.json').exists():command(name+'-score',[PY,CODE/'gram/score_multi.py','run',shard.with_suffix('.npz'),ckpt,0,samples,steps,*cfg],Path(os.environ['GRAM_SRC']),dict(TORCHINDUCTOR_CACHE_DIR=str(fit/'.inductor-score')))
  bare=task.split('_')[-1];kind='nqueens' if bare.startswith('nqueens') else 'gc';n=bare[len('nqueens'):] if kind=='nqueens' else bare[2:]
  data=Path(os.environ['GRAM_DATA'])/(f'nqueens-{n}' if kind=='nqueens' else f'graph-coloring-{n}')
  if not (out/'score.json').exists():command(name+'-merge',[PY,CODE/'gram/score_multi.py','merge',kind,n,data,out/'score.json',shard.with_suffix('.npz')],Path(os.environ['GRAM_SRC']),dict(CUDA_VISIBLE_DEVICES='',CHECKERS=str(CODE/'checkers')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--wait-pid',type=int);a=p.parse_args()
 if a.wait_pid:
  while Path(f'/proc/{a.wait_pid}').exists():time.sleep(15)
 lock=(WORK/'artifacts'/f'gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 if any(r.split(',')[0].strip()==uuid for r in active.splitlines()):raise SystemExit('GPU has a holder despite the acquired cooperative lock')
 os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CODE=str(CODE),SLURM_SUBMIT_DIR=str(CODE),SLURM_GPUS_ON_NODE='1',SLURM_RESTART_COUNT='0')
 os.environ.pop('OUT_NAME',None);os.environ.pop('EXTRA_OVERRIDES',None)
 smoke=CODE/'runs/gram/smoke-nqueens8-s0/fit.json'
 if not smoke.exists():raise SystemExit('Production-batch smoke did not finish')
 sm=json.loads(smoke.read_text());assert sm['steps']>0 and sm['steps']==sm['total_steps'] and sm['global_batch_size']==768
 event(status='exclusive-gpu-acquired',gpu=a.gpu,host=socket.gethostname(),smoke_fit=sm)
 # All seed0 task configurations first, then the remaining training seeds.
 tasks=['nqueens8','gc8','nqueens10','gc10','trm_nqueens8','hrm_nqueens8','trm_gc8','hrm_gc8','trm_nqueens10','hrm_nqueens10','trm_gc10','hrm_gc10','sudoku','mazeunique']
 for seed in range(3):
  for task in tasks:
   os.environ['SLURM_JOB_ID']=f'local-gram-{task}-s{seed}'
   fit=CODE/f'runs/gram/{task}-s{seed}'
   if not (fit/'fit.json').exists():command(f'gram-{task}-s{seed}-fit',['bash',ROOT/'launch.sh','gram',task,seed])
   score(task,seed)
   # The count depends on architecture/configuration, not fitted weights; count seed0 once per task.
   if seed==0:
    count=CODE/f'runs/flops-full/gram-{task}.json';count.parent.mkdir(parents=True,exist_ok=True)
    if not count.exists():command(f'gram-{task}-count',[PY,ROOT/'count_gram.py',count,1 if task.startswith(('hrm_','trm_')) else 20,*options(task),'seed=0'],Path(os.environ['GRAM_SRC']),dict(TORCHINDUCTOR_CACHE_DIR=str(fit/'.inductor-count')))
   event(status='final-fit-and-score-complete',task=task,seed=seed,fit=str(fit))
 event(status='queue-complete')
