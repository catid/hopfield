"""Delegate untouched DE jobs while only legacy scheduler parents are paused.

Original fit/search children are never signaled. An independent watchdog imports
finished or safely resumable outputs and resumes exact scheduler identities.
"""
from pathlib import Path
import argparse,contextlib,datetime,fcntl,hashlib,json,os,shlex,signal,socket,subprocess,sys,time,traceback
import data_efficiency_queue as Q
ROOT=Q.ROOT;OUT=Q.OUT;EXTRA=OUT/'extra';MANIFEST=EXTRA/'delegation.json'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10']

@contextlib.contextmanager
def transaction():
 EXTRA.mkdir(parents=True,exist_ok=True)
 with (EXTRA/'transaction.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  yield

def identity(pid):
 p=Path(f'/proc/{pid}')
 if not p.exists():return None
 raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 return {'pid':pid,'ppid':int(fields[1]),'startticks':int(fields[19]),'state':fields[0],
         'argv':(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
         'cpu_ticks':int(fields[11])+int(fields[12])}

def same(saved):
 r=identity(saved['pid'])
 return bool(r and all(r[k]==saved[k] for k in ('pid','ppid','startticks','argv')))

def proc_snapshot(host,pid):
 code='import sys,json;sys.path.insert(0,"/home/catid/hopfield/scripts");from data_efficiency_extra import identity;print(json.dumps(identity('+str(pid)+')))' 
 argv=['python3','-c',code]
 if host!='foureyes':argv=SSH+[host,shlex.join(argv)]
 return json.loads(subprocess.check_output(argv,text=True))

def plan():
 groups=[('foureyes',3,[(10000,0),(10000,5)]),('ripper',0,[(3000,0),(3000,4)]),
         ('ripper',1,[(3000,1),(3000,5)]),('ripper',2,[(3000,2)]),('ripper',3,[(3000,3)]),
         ('work',0,[(10000,1)]),('work',1,[(10000,2)]),('loki',0,[(10000,3)]),('loki',1,[(10000,4)])]
 config=json.loads(Q.CONFIG.read_text());lookup={(j['k'],j['seed']):j for j in config['jobs']}
 return [dict(host=h,gpu=g,jobs=[lookup[k] for k in keys]) for h,g,keys in groups]

def freeze():
 assert socket.gethostname().split('.')[0]=='foureyes'
 EXTRA.mkdir(parents=True,exist_ok=True)
 assert not MANIFEST.exists(),'Existing delegation must be recovered, never replaced'
 groups=plan();parents=[]
 for lane in range(3):
  state=json.loads((OUT/f'gpu{lane}-worker.json').read_text());assert state['worker_pid']=={0:1137265,1:1128993,2:1130719}[lane];parent=identity(state['worker_pid']);child=identity(state['child_pid'])
  assert parent['argv']==['python3','-u',str(ROOT/'scripts/data_efficiency_queue.py'),'worker','--lane',str(lane)]
  assert child and child['ppid']==parent['pid'] and child['argv'][2:4]==['data_efficiency.py','search']
  assert parent['state'] not in ('T','t','Z') and child['state'] not in ('T','t','Z')
  parents.append(dict(lane=lane,parent=parent,child=child,job=state['job']))
 for group in groups:
  for job in group['jobs']:
   assert not (OUT/'jobs'/f"{job['id']}.json").exists(),f"Already owned: {job['id']}"
   assert not (Q.SOURCE/'experiments'/json.loads(Q.CONFIG.read_text())['run_root']/f"k{job['k']}-s{job['seed']}").exists()
 manifest=dict(status='preparing',created_utc=Q.now(),config_sha256=Q.sha(Q.CONFIG),groups=groups,parents=parents,
               recovery_deadline_unix=time.time()+600,launches_closed=False,
               policy='Only exact scheduler parent PIDs paused; never process groups or existing fit/search children.')
 Q.write(MANIFEST,manifest)
 with (EXTRA/'watchdog.log').open('ab',buffering=0) as log:
  watcher=subprocess.Popen(['python3','-u',str(Path(__file__).resolve()),'watch'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 manifest['watchdog']=identity(watcher.pid);Q.write(MANIFEST,manifest)
 try:
  for r in parents:
   assert same(r['parent']) and same(r['child'])
   os.kill(r['parent']['pid'],signal.SIGSTOP)
  time.sleep(.1)
  for r in parents:
   assert same(r['parent']) and identity(r['parent']['pid'])['state']=='T'
   assert same(r['child']) and identity(r['child']['pid'])['state'] not in ('T','t','Z')
  # Selectivity is verified again after all parents stop, before any delegation.
  for group in groups:
   for job in group['jobs']:assert not (OUT/'jobs'/f"{job['id']}.json").exists()
  manifest.update(status='paused_preparing',paused_utc=Q.now());Q.write(MANIFEST,manifest)
 except BaseException:
  for r in parents:
   if same(r['parent']):os.kill(r['parent']['pid'],signal.SIGCONT)
  manifest.update(status='freeze_failed_rolled_back',error=traceback.format_exc());Q.write(MANIFEST,manifest);raise
 print(json.dumps({'status':manifest['status'],'watchdog_pid':watcher.pid,'paused_parents':[r['parent']['pid'] for r in parents],'untouched_children':[r['child']['pid'] for r in parents]}),flush=True)

def worker(host,gpu):
 assert socket.gethostname().split('.')[0]==host
 config=json.loads(Q.CONFIG.read_text());manifest=json.loads(MANIFEST.read_text())
 assert Q.sha(Q.CONFIG)==manifest['config_sha256']
 group=next(g for g in manifest['groups'] if g['host']==host and g['gpu']==gpu)
 state_path=EXTRA/f'worker-{host}-g{gpu}.json'
 state=dict(host=host,gpu=gpu,identity=identity(os.getpid()),status='initializing',started_utc=Q.now(),jobs=group['jobs'],completed=[])
 def save(**kw):state.update(**kw,utc=Q.now());Q.write(state_path,state)
 save()
 try:
  for rel,h in config['source_sha256'].items():assert Q.sha(Q.SOURCE/rel)==h,rel
  assert Q.sha(Q.DATA/'sudoku-extreme-1k-aug-1000/build_provenance.json')==config['reference_provenance_sha256']
  # Acquire both lock conventions, since objective lanes used their own path.
  locks=[]
  for p in [ROOT/f'artifacts/gpu{gpu}.lock',ROOT/f'artifacts/objectives/gpu-{gpu}.lock']:
   p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  assert not Q.gpu(gpu)['owners'],'GPU has a live owner'
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
           CRAM_DATA=str(Q.UP/'data'),V1_DATA=str(Q.UP/'data'),PYTHONPATH=str(Q.SOURCE),TRM_DATA=str(Q.DATA),DE_VARIANTS='99',CRAM_E_ROOT='DE2-v99-recreated42')
  for job in group['jobs']:
   directory=Q.DATA/f"sudoku-smalldata/s{job['seed']}-n{job['k']}";marker=directory/'subset.json'
   data_manifest=json.loads(marker.read_text());hashes={name:Q.sha(directory/'train'/f'all__{name}.npy') for name in ('inputs','labels','group_indices')}
   assert hashes=={name:data_manifest['sha256'][name] for name in hashes} and data_manifest['seed']==job['seed'] and data_manifest['size']==job['k']
   run=Q.SOURCE/'experiments'/config['run_root']/f"k{job['k']}-s{job['seed']}";meta_path=OUT/'jobs'/f"{job['id']}.json"
   assert not meta_path.exists() and not run.exists(),'Delegated job already has outputs; refusing duplicate dispatch'
   meta=job|dict(started_utc=Q.now(),config_sha256=Q.sha(Q.CONFIG),data_sha256=hashes,data_manifest_sha256=Q.sha(marker),phases=[],host=host,gpu=Q.gpu(gpu),
        environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','TRM_DATA','DE_VARIANTS','CRAM_E_ROOT')},protocol=config['protocol'],
        delegation={'manifest_sha256':Q.sha(MANIFEST),'worker_sha256':Q.sha(__file__),'original_lane':job['lane'],'actual_host':host,'actual_gpu':gpu})
   for phase in ('fit','search'):
    assert not Q.gpu(gpu)['owners'],'GPU acquired an unexpected owner'
    argv=[str(Q.UP/'.venv/bin/python'),'-u','data_efficiency.py',phase,'--k',str(job['k']),'--seed',str(job['seed'])]
    if phase=='search':argv+=['--data','test10k']
    Q.validate_saved_outputs(run,job,directory);meta['status']=phase;Q.write(meta_path,meta)
    log_path=OUT/f"{job['id']}-{phase}.log";log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('ab',buffering=0) as log:
     child=subprocess.Popen(argv,cwd=Q.SOURCE/'experiments',env=env,stdout=log,stderr=subprocess.STDOUT)
     save(status=phase,job=job['id'],child_identity=identity(child.pid),argv=argv)
     Q.wait_with_progress(child,lambda:save(gpu_state=Q.gpu(gpu)))
    meta['phases'].append(dict(phase=phase,argv=argv,exit_code=child.returncode,completed_utc=Q.now()));Q.write(meta_path,meta)
    if child.returncode:raise RuntimeError(f"{job['id']} {phase} failed {child.returncode}")
    Q.validate_saved_outputs(run,job,directory)
    if phase=='fit':meta['training_archive_sha256']=Q.sha(run/'train.npz');Q.write(meta_path,meta)
   result=json.loads((run/'test10k.json').read_text());meta.update(status='complete',completed_utc=Q.now(),summary=result.get('summary',result),output_sha256={name:Q.sha(run/name) for name in ('model.pt','fit.json','test10k.json')});Q.write(meta_path,meta)
   state['completed'].append(job['id']);save(status='between_jobs',child_identity=None)
  save(status='complete',child_identity=None)
 except BaseException:
  save(status='failed',error=traceback.format_exc());raise


def host_launch(host,gpu):
 assert socket.gethostname().split('.')[0]==host
 with (EXTRA/f'host-launch-{host}-g{gpu}.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  receipt=EXTRA/f'host-receipt-{host}-g{gpu}.json'
  assert not receipt.exists(), 'A durable host receipt already exists; do not relaunch'
  Q.write(receipt,dict(status='launching',launcher=identity(os.getpid()),host=host,gpu=gpu,utc=Q.now()))
  with (EXTRA/f'worker-{host}-g{gpu}.log').open('ab',buffering=0) as log:
   child=subprocess.Popen(['python3','-u',str(Path(__file__).resolve()),'worker','--host',host,'--gpu',str(gpu)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  record=dict(status='launched',identity=identity(child.pid),host=host,gpu=gpu,utc=Q.now())
  Q.write(receipt,record)
  print(json.dumps(record),flush=True)


def host_status(host,gpu):
 assert socket.gethostname().split('.')[0]==host
 manifest=json.loads(MANIFEST.read_text());group=next(g for g in manifest['groups'] if g['host']==host and g['gpu']==gpu)
 with (EXTRA/f'host-launch-{host}-g{gpu}.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  receipt=EXTRA/f'host-receipt-{host}-g{gpu}.json'
  if not receipt.exists():
   # The coordinator's SSH request failed before a durable host launch. Block
   # any delayed request by installing this receipt under the same host lock.
   Q.write(receipt,dict(status='cancelled_before_launch',host=host,gpu=gpu,utc=Q.now()))
  rec=json.loads(receipt.read_text());state_path=EXTRA/f'worker-{host}-g{gpu}.json'
  state=json.loads(state_path.read_text()) if state_path.exists() else None
  matches=[]
  for path in Path('/proc').glob('[0-9]*'):
   try:
    info=identity(int(path.name));argv=info['argv']
    if info['state']=='Z':continue
    worker=any(x.endswith('data_efficiency_extra.py') for x in argv) and 'worker' in argv and '--gpu' in argv and argv[argv.index('--gpu')+1]==str(gpu)
    job=any(x.endswith('data_efficiency.py') for x in argv) and '--k' in argv and '--seed' in argv and any(argv[argv.index('--k')+1]==str(j['k']) and argv[argv.index('--seed')+1]==str(j['seed']) for j in group['jobs'])
    if worker or job:matches.append(info)
   except (OSError,ValueError,IndexError,TypeError):continue
  locks_free=True;locks=[]
  for path in [ROOT/f'artifacts/gpu{gpu}.lock',ROOT/f'artifacts/objectives/gpu-{gpu}.lock']:
   path.parent.mkdir(parents=True,exist_ok=True);lock=path.open('a')
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
   except BlockingIOError:locks_free=False;lock.close()
  for lock in locks:lock.close()
  result=dict(host=host,gpu=gpu,receipt=rec,worker=state,matching_processes=matches,physical_locks_free=locks_free,safe_to_import=not matches and locks_free)
  print(json.dumps(result),flush=True)


def launch(host,gpu):
 assert socket.gethostname().split('.')[0]=='foureyes'
 with transaction():
  m=json.loads(MANIFEST.read_text());assert m['status'] in ('paused_preparing','running') and not m.get('launches_closed')
  assert all(same(r['parent']) and identity(r['parent']['pid'])['state']=='T' for r in m['parents'])
  target=EXTRA/f'launch-{host}-g{gpu}.json';assert not target.exists()
  Q.write(target,dict(status='intent',host=host,gpu=gpu,utc=Q.now()))
  argv=['python3',str(Path(__file__).resolve()),'hostlaunch','--host',host,'--gpu',str(gpu)]
  if host!='foureyes':argv=SSH+[host,shlex.join(argv)]
  try:
   rec=json.loads(subprocess.check_output(argv,text=True));Q.write(target,rec)
  except BaseException:
   Q.write(target,dict(status='unknown_launch_requires_host_resolution',host=host,gpu=gpu,utc=Q.now(),error=traceback.format_exc()));raise
  m.update(status='running',last_launch_utc=Q.now());Q.write(MANIFEST,m)
  print(json.dumps({'launched':f'{host}:{gpu}','receipt':rec}),flush=True)

def remote_state(host,gpu):
 p=EXTRA/f'worker-{host}-g{gpu}.json'
 if host=='foureyes':return json.loads(p.read_text()) if p.exists() else None
 code='from pathlib import Path;import json;p=Path('+repr(str(p))+');print(p.read_text() if p.exists() else "null")'
 return json.loads(subprocess.check_output(SSH+[host,shlex.join(['python3','-c',code])],text=True))

def pull(group):
 host=group['host'];config=json.loads(Q.CONFIG.read_text())
 if host=='foureyes':return
 for job in group['jobs']:
  run=Q.SOURCE/'experiments'/config['run_root']/f"k{job['k']}-s{job['seed']}";run.mkdir(parents=True,exist_ok=True)
  base=['rsync','-a','--partial','-e','ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10']
  # A failed, never-started job has no remote directory. Probe before copying.
  cmd=['test','-d',str(run)]
  if subprocess.run(SSH+[host,shlex.join(cmd)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
   subprocess.run(base+[f'{host}:{run}/',str(run)+'/'],check=True)
  mp=OUT/'jobs'/f"{job['id']}.json";mp.parent.mkdir(parents=True,exist_ok=True)
  if subprocess.run(SSH+[host,shlex.join(['test','-f',str(mp)])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
   subprocess.run(base+[f'{host}:{mp}',str(mp)],check=True)
  else:
   if run.exists() and not any(run.iterdir()):run.rmdir()


def watch():
 assert socket.gethostname().split('.')[0]=='foureyes'
 while True:
  try:
   # Serialize the whole eligibility snapshot, import, validation and resume
   # with launch and close. No new launch can race the final SIGCONT.
   with transaction():
    m=json.loads(MANIFEST.read_text())
    if m['status'] in ('freeze_failed_rolled_back','resumed'):return
    state={'utc':Q.now(),'groups':[],'watcher':identity(os.getpid())};safe=True;all_done=True
    for group in m['groups']:
     host,gpu=group['host'],group['gpu'];receipt=EXTRA/f'launch-{host}-g{gpu}.json'
     if not receipt.exists():
      if not m.get('launches_closed') and time.time()<m['recovery_deadline_unix']:safe=False
      all_done=False;state['groups'].append({'host':host,'gpu':gpu,'status':'not_launched'});continue
     argv=['python3',str(Path(__file__).resolve()),'hoststatus','--host',host,'--gpu',str(gpu)]
     if host!='foureyes':argv=SSH+[host,shlex.join(argv)]
     observed=json.loads(subprocess.check_output(argv,text=True));worker_state=observed['worker'];row={k:observed[k] for k in ('host','gpu','safe_to_import','matching_processes','physical_locks_free')}
     row['status']=worker_state.get('status') if worker_state else observed['receipt']['status']
     if not observed['safe_to_import']:safe=False
     else:
      pull(group)
      for job in group['jobs']:
       mp=OUT/'jobs'/f"{job['id']}.json";run=Q.SOURCE/'experiments'/json.loads(Q.CONFIG.read_text())['run_root']/f"k{job['k']}-s{job['seed']}"
       if mp.exists():
        meta=json.loads(mp.read_text());directory=Q.DATA/f"sudoku-smalldata/s{job['seed']}-n{job['k']}"
        Q.validate_saved_outputs(run,job,directory)
        assert meta['config_sha256']==m['config_sha256']
        assert meta['data_manifest_sha256']==Q.sha(directory/'subset.json')
        assert meta['data_sha256']=={name:Q.sha(directory/'train'/f'all__{name}.npy') for name in ('inputs','labels','group_indices')}
        if 'training_archive_sha256' in meta:assert Q.sha(run/'train.npz')==meta['training_archive_sha256']
        if meta['status']=='complete':
         assert set(meta['output_sha256'])=={'model.pt','fit.json','test10k.json'}
         for name,h in meta['output_sha256'].items():assert Q.sha(run/name)==h
        else:all_done=False
       else:all_done=False
     if row['status']!='complete':all_done=False
     state['groups'].append(row)
    Q.write(EXTRA/'watchdog-state.json',state|{'safe_to_resume':safe,'all_delegated_complete':all_done})
    if safe and (m.get('launches_closed') or time.time()>=m['recovery_deadline_unix']):
     assert Q.sha(Q.CONFIG)==m['config_sha256']
     subprocess.run([str(Q.UP/'.venv/bin/python'),str(ROOT/'scripts/collect_data_efficiency.py')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
     c=json.loads((OUT/'collection.json').read_text());index={r['job_id']:r for r in c['rows']}
     for group in m['groups']:
      for job in group['jobs']:
       row=index[job['id']];assert row['status']!='invalid',(job['id'],row)
       if all_done:assert row['status'] in ('complete','degenerate'),(job['id'],row['status'])
     for r in m['parents']:
      if same(r['parent']):os.kill(r['parent']['pid'],signal.SIGCONT)
     m.update(status='resumed',resumed_utc=Q.now(),all_delegated_complete=all_done,recovery_mode='success' if all_done else 'safe_partial_handoff');Q.write(MANIFEST,m)
     print(json.dumps({'resumed_parents':[r['parent']['pid'] for r in m['parents']],'all_delegated_complete':all_done}),flush=True);return
  except BaseException as e:
   Q.write(EXTRA/'watchdog-error.json',{'utc':Q.now(),'error':repr(e),'traceback':traceback.format_exc(),'policy':'Do not resume until active delegated processes are absent and outputs validate; retry recovery.'})
  time.sleep(15)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','worker','launch','watch','close','hostlaunch','hoststatus']);p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args();EXTRA.mkdir(parents=True,exist_ok=True)
 if a.action in ('worker','launch'):globals()[a.action](a.host,a.gpu)
 elif a.action in ('hostlaunch','hoststatus'):globals()[{'hostlaunch':'host_launch','hoststatus':'host_status'}[a.action]](a.host,a.gpu)
 elif a.action=='close':
  with transaction():
   m=json.loads(MANIFEST.read_text());m['launches_closed']=True;Q.write(MANIFEST,m)
 else:globals()[a.action]()
