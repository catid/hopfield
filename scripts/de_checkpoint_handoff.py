"""Auditable checkpoint handoff primitives; production actions require explicit CLI.

Plan is read-only. Capture pauses exact original PIDs, archives atomic checkpoints,
stages isolated runs, then retires parents before search children. A detached guard
restores paused originals if capture fails. No GPU work is launched by capture.
"""
from pathlib import Path
import argparse,contextlib,datetime,errno,fcntl,hashlib,json,os,shlex,signal,socket,subprocess,sys,tempfile,time,traceback
import data_efficiency_queue as Q
from data_efficiency_extra import identity as raw_identity, SSH
ROOT=Q.ROOT;BASE=Q.OUT/'migrations';CONFIG=Q.CONFIG
SCRIPTS=('resume_sharded_data_efficiency.py','sharded_sudoku_moves.py','benchmark_sharded_sudoku.py')
NATIVE_PROOFS={4:ROOT/'artifacts/checkpoint_resume_benchmark/20261114T044046Z/foureyes-to-ripper-seed4-proof.json',5:ROOT/'artifacts/checkpoint_resume_benchmark/20261114T044046Z/foureyes-to-work-seed5-proof.json'}
TARGETS={3:('foureyes',[0,1,2,3]),4:('ripper',[0,1,2,3]),5:('work',[0,1])}


def identity(pid):
 try:return raw_identity(pid)
 except OSError as error:
  if error.errno in (errno.ENOENT,errno.ESRCH):return None
  raise


def signal_saved(saved,sig,allow_reparented=False):
 current=identity(saved['pid'])
 if current is None or current['state']=='Z':return False
 assert all(current[k]==saved[k] for k in ('pid','startticks','argv')), 'Refusing changed process identity'
 if current['ppid']!=saved['ppid']:
  old_parent=identity(saved['ppid'])
  assert allow_reparented and (old_parent is None or old_parent['state']=='Z'), 'Refusing unexpected process parent'
 if current['ppid']!=saved['ppid']:print(json.dumps({'event':'verified_orphan_signal','signal':int(sig),'original_ppid':saved['ppid'],'current':current}),flush=True)
 try:os.kill(saved['pid'],sig)
 except ProcessLookupError:return False
 return True


def durable_write(path,record):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
 with tmp.open('w') as handle:
  json.dump(record,handle,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
 os.replace(tmp,path)
 directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(directory)
 finally:os.close(directory)


@contextlib.contextmanager
def journal_lock(journal):
 with journal.with_suffix('.lock').open('a') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX)
  yield


def alive(saved):
 current=identity(saved['pid']) if saved else None
 return bool(current and current['startticks']==saved['startticks'] and current['state']!='Z')


def exact(saved):
 current=identity(saved['pid'])
 return bool(current and all(current[k]==saved[k] for k in ('pid','ppid','startticks','argv')) and current['state']!='Z')


def atomic_copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 assert not dst.exists(),f'Preserve existing snapshot: {dst}'
 h=hashlib.sha256()
 with src.open('rb') as source, tempfile.NamedTemporaryFile(prefix=dst.name+'.tmp-',dir=dst.parent,delete=False) as target:
  temporary=Path(target.name)
  before=os.fstat(source.fileno())
  for block in iter(lambda:source.read(8<<20),b''):h.update(block);target.write(block)
  target.flush();os.fsync(target.fileno());after=os.fstat(source.fileno())
  assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 os.replace(temporary,dst)
 return {'sha256':h.hexdigest(),'bytes':before.st_size,'source_inode':before.st_ino,'source_mtime_ns':before.st_mtime_ns}


def read(path):return json.loads(Path(path).read_text())

def plan(out):
 assert socket.gethostname().split('.')[0]=='foureyes'
 delegation=read(Q.OUT/'extra/delegation.json')
 assert delegation['status']=='resumed' and delegation['all_delegated_complete']
 config=read(CONFIG);collection=read(Q.OUT/'collection.json');rows={r['job_id']:r for r in collection['rows']}
 assert collection['measured_jobs']==99
 for group in delegation['groups']:
  for job in group['jobs']:assert rows[job['id']]['status']=='complete'
 jobs=[]
 for seed,(host,devices) in TARGETS.items():
  lane=seed-3;old=next(r for r in delegation['parents'] if r['lane']==lane)
  parent,child=identity(old['parent']['pid']),identity(old['child']['pid'])
  assert exact(old['parent']) and exact(old['child'])
  assert child['ppid']==parent['pid'] and child['argv'][2:]==['data_efficiency.py','search','--k','1','--seed',str(seed),'--data','test10k']
  assert parent['state'] not in ('T','t') and child['state'] not in ('T','t')
  job=next(j for j in config['jobs'] if j['k']==1 and j['seed']==seed)
  meta_path=Q.OUT/'jobs'/f"{job['id']}.json";meta=read(meta_path)
  run=Q.SOURCE/'experiments'/config['run_root']/f'k1-s{seed}'
  assert meta['status']=='search' and not (run/'test10k.json').exists()
  Q.validate_saved_outputs(run,job,Q.DATA/f'sudoku-smalldata/s{seed}-n1')
  jobs.append(dict(job=job,seed=seed,original_lane=lane,parent=parent,child=child,source_run=str(run),
       original_meta_path=str(meta_path),original_meta_sha256=Q.sha(meta_path),target_host=host,devices=devices,
       candidate_checkpoint_bytes=(run/'test10k-state/ckpt.pt').stat().st_size))
 record=dict(status='planned_no_signals',created_utc=Q.now(),config_sha256=Q.sha(CONFIG),source_sha256=config['source_sha256'],
       test_data_sha256=Q.sha(Q.UP/'data/full-test/full-all.npz'),jobs=jobs,
       source_scripts_sha256={n:Q.sha(ROOT/'scripts'/n) for n in SCRIPTS},
       gates=['Explicit root production go','All target benchmark workers absent and locks released',
              'Ripper 4-GPU and Work 2-GPU numeric proofs plus cross-host parity passed',
              'After retirement, validate same-host Foureyes 4-GPU seed3 proof; use native recovery if gate fails'],
       policy='Only plan snapshots until explicit capture. Initial checkpoint archives remain immutable; output working copies may checkpoint atomically.')
 assert not out.exists();Q.write(out,record);print(json.dumps({'plan':str(out),'seeds':[j['seed'] for j in jobs]}))


def restore_paused(record):
 # Restore children before controllers; children must remain sole writers.
 for item in record['jobs']:
  for key in ('child','parent'):
   saved=item[key]
   current=identity(saved['pid'])
   if current and current['state'] in ('T','t'):signal_saved(saved,signal.SIGCONT)


def guard(journal):
 while True:
  r=read(journal)
  if r['status'] in ('retired_captured','cancelled_restored','complete'):return
  if not alive(r['controller']):
   if not r.get('retirement_committed'):
    restore_paused(r);r.update(status='cancelled_restored',recovery_utc=Q.now(),recovery_reason='Capture controller exited before retirement; original writers resumed');Q.write(journal,r);return
   # Committed retirement is idempotent and never resumes an old controller.
   for key in ('parent','child'):
    for item in r['jobs']:
     saved=item[key]
     if alive(saved):
      signal_saved(saved,signal.SIGTERM,allow_reparented=key=='child')
      signal_saved(saved,signal.SIGCONT,allow_reparented=key=='child')
    deadline=time.monotonic()+30
    while any(alive(j[key]) for j in r['jobs']):
     if time.monotonic()>deadline:raise RuntimeError('Guard could not confirm retired '+key)
     time.sleep(.1)
   r.update(status='retired_native_recovery',recovery_utc=Q.now(),recovery_reason='Controller exited during committed retirement; independent native continuations restore useful work from the preserved checkpoints.');Q.write(journal,r)
   for item in r['jobs']:
    with (journal.parent/f"native-recovery-s{item['seed']}.log").open('ab',buffering=0) as log:
     subprocess.Popen(['python3','-u',str(Path(__file__).resolve()),'native-recover','--journal',str(journal),'--seed',str(item['seed'])],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   return
  time.sleep(2)


def stage_remote(directory,host):
 if host=='foureyes':return
 subprocess.run(SSH+[host,shlex.join(['mkdir','-p',str(directory.parent)])],check=True,timeout=30)
 subprocess.run(['rsync','-a','--partial','--timeout=60','-e','ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10',str(directory)+'/',f'{host}:{directory}/'],check=True,timeout=120)
 files={str(p.relative_to(directory)):Q.sha(p) for p in directory.rglob('*') if p.is_file()}
 code='from pathlib import Path;import hashlib,json;root=Path('+repr(str(directory))+');expected=json.loads('+repr(json.dumps(files))+');assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in expected.items());print(len(expected))'
 subprocess.run(SSH+[host,shlex.join(['python3','-c',code])],check=True,timeout=120)


def capture(plan_path,output,proof4,proof2):
 assert socket.gethostname().split('.')[0]=='foureyes'
 assert not output.exists();output.mkdir(parents=True)
 r=read(plan_path);assert r['status']=='planned_no_signals' and Q.sha(CONFIG)==r['config_sha256']
 assert read(Q.OUT/'extra/delegation.json')['status']=='resumed'
 for name,digest in r['source_scripts_sha256'].items():assert Q.sha(ROOT/'scripts'/name)==digest, 'Execution code changed; regenerate plan'
 for p,host,count in [(proof4,'ripper',4),(proof2,'work',2)]:
  proof=read(p);assert proof['status']=='validated' and proof['exact_state'] and proof['speedup']>1.15 and proof['host']==host and len(proof['devices'])==count
  assert proof['sharding_sha256']==r['source_scripts_sha256']['sharded_sudoku_moves.py']
  assert proof['script_sha256']==r['source_scripts_sha256']['benchmark_sharded_sudoku.py']
 for item in r['jobs']:
  assert exact(item['parent']) and exact(item['child'])
  assert Q.sha(item['original_meta_path'])==item['original_meta_sha256']
  assert not (Path(item['source_run'])/'test10k.json').exists()
 r.update(status='capturing',controller=identity(os.getpid()),capture_started_utc=Q.now(),retirement_committed=False,output=str(output))
 journal=output/'journal.json';Q.write(journal,r)
 with (output/'guard.log').open('ab',buffering=0) as log:
  proc=subprocess.Popen(['python3','-u',str(Path(__file__).resolve()),'guard','--journal',str(journal)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 r['guard']=identity(proc.pid);Q.write(journal,r)
 try:
  for item in r['jobs']:
   assert exact(item['parent']);assert signal_saved(item['parent'],signal.SIGSTOP)
  for item in r['jobs']:
   assert exact(item['child']);assert signal_saved(item['child'],signal.SIGSTOP)
  time.sleep(.1)
  for item in r['jobs']:
   assert identity(item['parent']['pid'])['state']=='T' and identity(item['child']['pid'])['state']=='T'
  import torch
  assert not torch.cuda.is_initialized()
  snapshot_files=[]
  for item in r['jobs']:
   seed=item['seed'];base=output/f'seed{seed}';run=base/f'k1-s{seed}';initial=base/'initial-checkpoint.pt';origin=Path(item['source_run'])
   snapshots={}
   for name in ('model.pt','fit.json','train.npz'):
    snapshots[name]=atomic_copy(origin/name,run/name)
   snapshots['test10k-state/ckpt.pt']=atomic_copy(origin/'test10k-state/ckpt.pt',initial)
   atomic_copy(initial,run/'test10k-state/ckpt.pt')
   atomic_copy(Path(item['original_meta_path']),base/'original-job-meta.json')
   for name in SCRIPTS:atomic_copy(ROOT/'scripts'/name,base/'execution_source'/name)
   saved=torch.load(initial,map_location='cpu',weights_only=False);fit=read(run/'fit.json')
   assert saved['signature']['seed']==seed and saved['signature']['k']==1
   assert saved['signature']['model_sha256']==fit['model_sha256']==snapshots['model.pt']['sha256']
   assert saved['state']['t']%4096==0 and len(saved['state']['act'])>0
   item.update(isolated_run=str(run),archive=str(initial),snapshots=snapshots,
      checkpoint_move=int(saved['state']['t']),checkpoint_elapsed=float(saved['state']['elapsed']),
      checkpoint_active=len(saved['state']['act']),checkpoint_age_seconds_at_capture=time.time()-snapshots['test10k-state/ckpt.pt']['source_mtime_ns']/1e9,
      discarded_work_scope='Original in-memory progress after this atomic checkpoint is not retained; age is a wall-clock proxy, not an exact move count.')
   handoff=dict(seed=seed,k=1,source_host='foureyes',original_writer_stopped=False,config_sha256=r['config_sha256'],
       input_sha256={name:v['sha256'] for name,v in snapshots.items()},test_data_sha256=r['test_data_sha256'],
       checkpoint_archive_path=str(initial.relative_to(ROOT)),original_processes={'parent':item['parent'],'child':item['child']},checkpoint_move=item['checkpoint_move'])
   if seed in (4,5):
    native=NATIVE_PROOFS[seed];assert read(native)['status']=='passed'
    native_copy=base/'native-resume-proof.json';atomic_copy(native,native_copy)
    handoff.update(native_resume_proof_path=str(native_copy.relative_to(ROOT)),native_resume_proof_sha256=Q.sha(native_copy))
   Q.write(base/'handoff.json',handoff)
   if seed in (4,5):atomic_copy(proof4 if seed==4 else proof2,base/'proof.json')
   # A compatible benchmark bundle for the post-retirement same-host seed3 gate.
   snap=output/'benchmark/inputs'/f'seed{seed}'
   for name in ('model.pt','fit.json','train.npz'):
    atomic_copy(run/name,snap/name);snapshot_files.append(dict(seed=seed,snapshot=str(snap/name),sha256=Q.sha(snap/name)))
   atomic_copy(initial,snap/'ckpt.pt');snapshot_files.append(dict(seed=seed,snapshot=str(snap/'ckpt.pt'),sha256=Q.sha(snap/'ckpt.pt')))
   stage_remote(base,item['target_host'])
  Q.write(output/'benchmark/inputs.json',dict(source_commit=read(CONFIG)['source_commit'],source_sha256=r['source_sha256'],data=str(Q.UP/'data'),data_sha256=r['test_data_sha256'],files=snapshot_files))
  # Every snapshot exists and target bytes verify before this durable boundary.
  r.update(status='retirement_committed',retirement_committed=True,retirement_committed_utc=Q.now());durable_write(journal,r)
  for item in r['jobs']:
   saved=item['parent'];signal_saved(saved,signal.SIGTERM);signal_saved(saved,signal.SIGCONT)
  deadline=time.monotonic()+20
  while any(alive(j['parent']) for j in r['jobs']):
   assert time.monotonic()<deadline,'Parent did not retire; children remain paused and guard will recover';time.sleep(.1)
  for item in r['jobs']:
   saved=item['child'];signal_saved(saved,signal.SIGTERM,allow_reparented=True);signal_saved(saved,signal.SIGCONT,allow_reparented=True)
  deadline=time.monotonic()+30
  while any(alive(j['child']) for j in r['jobs']):
   assert time.monotonic()<deadline,'Original writer did not retire';time.sleep(.1)
  for item in r['jobs']:
   base=Path(item['isolated_run']).parent;handoff=read(base/'handoff.json');handoff['original_writer_stopped']=True;handoff['original_writer_retired_utc']=Q.now();Q.write(base/'handoff.json',handoff)
   meta=read(item['original_meta_path']);assert Q.sha(item['original_meta_path'])==item['original_meta_sha256']
   meta['phases'].append(dict(phase='search',argv=item['child']['argv'],exit_code=-signal.SIGTERM,
       interrupted_for_checkpoint_migration=True,original_parent_retired_first=True,completed_utc=Q.now(),
       checkpoint_sha256=handoff['input_sha256']['test10k-state/ckpt.pt'],checkpoint_move=item['checkpoint_move'],
       checkpoint_age_seconds_at_capture=item['checkpoint_age_seconds_at_capture'],migration_journal=str(journal.relative_to(ROOT))))
   meta['status']='checkpoint_migration';Q.write(item['original_meta_path'],meta)
   stage_remote(base,item['target_host'])
  r.update(status='retired_captured',completed_utc=Q.now());Q.write(journal,r)
  print(json.dumps({'journal':str(journal),'status':r['status'],'seeds':[dict(seed=j['seed'],target=j['target_host'],run=j['isolated_run'],checkpoint_move=j['checkpoint_move'],checkpoint_age_seconds=j['checkpoint_age_seconds_at_capture']) for j in r['jobs']]}),flush=True)
 except BaseException:
  r['error']=traceback.format_exc()
  if not r.get('retirement_committed'):
   restore_paused(r);r.update(status='cancelled_restored',recovery_utc=Q.now())
  else:r['status']='retired_recovery_required'
  Q.write(journal,r);raise


def close_sharded_admission(base,seed,journal):
 """Runs on the target host, sharing the resumer's exact admission lock."""
 receipt=base/f'sharded-search-s{seed}.json'
 with (base/f'sharded-search-s{seed}.lock').open('a') as admission:
  fcntl.flock(admission,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if receipt.exists():
   prior=read(receipt)
   assert prior['status']=='native_recovery_committed' and prior['seed']==seed and prior['journal']==str(journal), 'A sharded execution requires explicit recovery'
   return prior
  record=dict(status='native_recovery_committed',seed=seed,journal=str(journal),host=socket.gethostname(),committed_utc=Q.now(),
       reason='Durably close sharded admission before local native fallback; preserve this receipt to reject delayed launches')
  durable_write(receipt,record)
  return record


def native_recover(journal,seed):
 """Fallback only when no sharded resumer was ever admitted for this seed."""
 r=read(journal);item=next(j for j in r['jobs'] if j['seed']==seed);lane=item['original_lane'];base=Path(item['isolated_run']).parent
 assert socket.gethostname().split('.')[0]=='foureyes'
 assert not alive(item['parent']) and not alive(item['child'])
 assert Q.sha(CONFIG)==r['config_sha256']
 locks=[]
 for path in [base/'native-recovery.lock',ROOT/f'artifacts/gpu{lane}.lock',ROOT/f'artifacts/objectives/gpu-{lane}.lock']:
  path.parent.mkdir(parents=True,exist_ok=True);lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
 if item['target_host']=='foureyes':closed=close_sharded_admission(base,seed,journal)
 else:
  # Execute this reviewed helper source remotely without relying on an installed copy.
  code=Path(__file__).read_text().split("if __name__=='__main__':")[0]
  code+='\nprint(json.dumps(close_sharded_admission(Path('+repr(str(base))+'),'+str(seed)+',Path('+repr(str(journal))+'))))\n'
  bootstrap='import sys;sys.path.insert(0,'+repr(str(ROOT/'scripts'))+');exec(compile(sys.stdin.read(),'+repr(str(Path(__file__)))+',"exec"))'
  closed=json.loads(subprocess.check_output(SSH+[item['target_host'],shlex.join(['python3','-c',bootstrap])],input=code,text=True,timeout=60))
 durable_write(base/'native-admission-closed.json',closed)
 assert not Q.gpu(lane)['owners']
 for rel,h in r['source_sha256'].items():assert Q.sha(Q.SOURCE/rel)==h
 run=Path(item['source_run']);assert not (run/'test10k.json').exists()
 assert Q.sha(run/'test10k-state/ckpt.pt')==item['snapshots']['test10k-state/ckpt.pt']['sha256']
 meta=read(item['original_meta_path'])
 if not any(p.get('interrupted_for_checkpoint_migration') for p in meta['phases']):
  meta['phases'].append(dict(phase='search',argv=item['child']['argv'],exit_code=-signal.SIGTERM,interrupted_for_checkpoint_migration=True,completed_utc=Q.now(),checkpoint_move=item['checkpoint_move']))
 argv=[str(Q.UP/'.venv/bin/python'),'-u','data_efficiency.py','search','--k','1','--seed',str(seed),'--data','test10k']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(lane),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CRAM_DATA=str(Q.UP/'data'),V1_DATA=str(Q.UP/'data'),PYTHONPATH=str(Q.SOURCE),TRM_DATA=str(Q.DATA),DE_VARIANTS='99',CRAM_E_ROOT='DE2-v99-recreated42')
 meta.update(status='search',native_checkpoint_recovery={'journal':str(journal.relative_to(ROOT)),'checkpoint_sha256':item['snapshots']['test10k-state/ckpt.pt']['sha256'],'argv':argv,'started_utc':Q.now()});Q.write(item['original_meta_path'],meta)
 target=base/'native-recovery.json';state=dict(status='running',seed=seed,controller=identity(os.getpid()),argv=argv,started_utc=Q.now())
 with (base/'native-recovery-search.log').open('ab',buffering=0) as log:
  child=subprocess.Popen(argv,cwd=Q.SOURCE/'experiments',env=env,stdout=log,stderr=subprocess.STDOUT);state['child']=identity(child.pid);Q.write(target,state)
  child.wait()
 state.update(exit_code=child.returncode,status='complete' if child.returncode==0 else 'failed',completed_utc=Q.now());Q.write(target,state)
 meta['phases'].append(dict(phase='search',argv=argv,exit_code=child.returncode,completed_utc=Q.now(),native_checkpoint_recovery=True))
 if child.returncode:Q.write(item['original_meta_path'],meta);raise RuntimeError('Native recovery failed; checkpoint and failure preserved')
 Q.validate_saved_outputs(run,item['job'],Q.DATA/f'sudoku-smalldata/s{seed}-n1');result=read(run/'test10k.json')
 meta.update(status='complete',completed_utc=Q.now(),summary=result['summary'],output_sha256={name:Q.sha(run/name) for name in ('model.pt','fit.json','test10k.json')});Q.write(item['original_meta_path'],meta)


def import_result(journal,seed):
 with journal_lock(journal):
  return import_result_locked(journal,seed)


def import_result_locked(journal,seed):
 r=read(journal);item=next(j for j in r['jobs'] if j['seed']==seed);base=Path(item['isolated_run']).parent;run=Path(item['isolated_run']);host=item['target_host']
 assert not alive(item['parent']) and not alive(item['child'])
 # A completed receipt is immutable; acquire its writer lock before accepting it.
 # A new resumer cannot start after release because the receipt already exists.
 receipt_path=base/f'sharded-search-s{seed}.json'
 check='import fcntl,json;from pathlib import Path;base=Path('+repr(str(base))+');lock=(base/'+repr(f'sharded-search-s{seed}.lock')+').open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r=json.loads((base/'+repr(receipt_path.name)+').read_text());assert r["status"]=="complete" and r["exit_code"]==0'
 subprocess.run((SSH+[host,shlex.join(['python3','-c',check])]) if host!='foureyes' else ['python3','-c',check],check=True,timeout=60)
 if host!='foureyes':
  subprocess.run(['rsync','-a','--partial','--timeout=60','-e','ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10',f'{host}:{base}/',str(base)+'/'],check=True,timeout=120)
 receipt_path=base/f'sharded-search-s{seed}.json';receipt=read(receipt_path);handoff_path=base/'handoff.json';proof_path=base/'proof.json'
 assert receipt['status']=='complete' and receipt['exit_code']==0 and receipt['handoff_parity']['equal']
 assert receipt['result_sha256']==Q.sha(run/'test10k.json') and receipt['handoff_sha256']==Q.sha(handoff_path) and receipt['proof_sha256']==Q.sha(proof_path)
 for name,h in receipt['execution_source_sha256'].items():assert Q.sha(base/'execution_source'/name)==h
 result=read(run/'test10k.json');assert result['signature']==receipt['original_signature'] and result['summary']['n']==10000
 assert result['per_puzzle']['puzzle']==list(range(10000,20000))
 canonical=Path(item['source_run']);Q.validate_saved_outputs(run,item['job'],Q.DATA/f'sudoku-smalldata/s{seed}-n1')
 meta=read(item['original_meta_path']);receipt_digest=Q.sha(receipt_path)
 if not any(p.get('migration_receipt_sha256')==receipt_digest for p in meta['phases']):
  meta['phases'].append(dict(phase='search',argv=receipt['argv'],exit_code=0,completed_utc=receipt['completed_utc'],resumed_from_checkpoint=True,actual_host=host,migration_receipt_sha256=receipt_digest))
 meta['search_migration']={f'{key}_path':str(path.relative_to(ROOT)) for key,path in [('receipt',receipt_path),('handoff',handoff_path),('proof',proof_path)]}
 meta['search_migration'].update({f'{key}_sha256':Q.sha(path) for key,path in [('receipt',receipt_path),('handoff',handoff_path),('proof',proof_path)]})
 meta.update(status='complete',completed_utc=Q.now(),summary=result['summary'],output_sha256={name:Q.sha(run/name) for name in ('model.pt','fit.json','test10k.json')})
 from collect_data_efficiency import Hashes,migration_evidence
 migration_evidence(item['job'],meta,run,Hashes())
 for name in ('model.pt','fit.json','train.npz'):assert Q.sha(canonical/name)==Q.sha(run/name)
 if (canonical/'test10k.json').exists():
  assert Q.sha(canonical/'test10k.json')==receipt['result_sha256'],'Never overwrite a different final result'
 else:atomic_copy(run/'test10k.json',canonical/'test10k.json')
 durable_write(item['original_meta_path'],meta);item['imported_result_sha256']=Q.sha(canonical/'test10k.json');item['imported_utc']=Q.now();durable_write(journal,r)
 print(json.dumps({'seed':seed,'status':'imported','summary':result['summary']}))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','capture','guard','import','native-recover']);p.add_argument('--plan',type=Path);p.add_argument('--out',type=Path);p.add_argument('--journal',type=Path);p.add_argument('--seed',type=int);p.add_argument('--proof-four',type=Path);p.add_argument('--proof-two',type=Path);a=p.parse_args()
 if a.action=='plan':plan(a.out)
 elif a.action=='capture':capture(a.plan,a.out,a.proof_four,a.proof_two)
 elif a.action=='guard':guard(a.journal)
 elif a.action=='native-recover':native_recover(a.journal,a.seed)
 else:import_result(a.journal,a.seed)
