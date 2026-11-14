#!/usr/bin/env python3
"""Plan or transact migrations of untouched deferred Figure4 jobs.

No active process is stopped. A root lock serializes coordinators; dispatch
locks on every involved host exclude new provenance claims during the config
transaction. Only a free, eligible target lane is launched. `plan` never writes
dispatch configs or starts a worker. `apply` rechecks every precondition.
"""
import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import socket
import subprocess
import sys
import time

ROOT=Path('/home/catid/hopfield')
CONFIG=ROOT/'configs/figure4_jobs.json'
ART=ROOT/'artifacts/figure4/migrations'
FUTURE={'foureyes:0','foureyes:1','foureyes:2','foureyes:3','ripper:2','ripper:3'}
COMMIT='62bbe830c7d6f87f2ba18f30a28039258b986454'
SSH=['ssh','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=10']


def now():return datetime.now(timezone.utc).isoformat()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def encoded(value):return (json.dumps(value,indent=2)+'\n').encode()


def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.migration-{os.getpid()}.tmp')
    with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def save(path,value):atomic_bytes(path,encoded(value))


def live(pid):return bool(pid) and Path(f'/proc/{pid}').exists()


def processes():
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if not any(Path(a).name in ('figure4_queue.py','d2_queue.py') for a in argv):continue
            lane=argv[argv.index('--lane')+1] if '--lane' in argv else None
            rows.append(dict(pid=int(p.name),argv=argv,lane=lane,
                             start_ticks=int(p.joinpath('stat').read_text().split(') ')[1].split()[19])))
        except (FileNotFoundError,PermissionError,ProcessLookupError,ValueError):pass
    return rows


def prerequisite_check(lane,config,original_lane):
    """No resource can be released by PID disappearance without result evidence."""
    host,g=lane.split(':');gpu=int(g);proof=[]
    def receipt(relative):
        p=ROOT/relative;x=read(p);proof.append(dict(path=relative,sha256=digest(p)));return x
    def absent(*pids):
        assert not any(live(p) for p in pids),f'Prerequisite controller still exists: {pids}'
    if original_lane:
        s=receipt(f'artifacts/figure4/{host}-gpu{gpu}-state.json')
        assert s['status']=='queue_complete','Allocated Figure4 lane has not finished its queue'
        absent(s.get('worker_pid'),s.get('child_pid'))
        for j in config['jobs']:
            if j['lane']!=lane:continue
            p=ROOT/'artifacts/figure4/jobs'/j['id']/'execution.json';r=read(p)
            assert r['status']=='complete',f'Target has incomplete existing job {j["id"]}'
            assert r['job']==j,'Target job identity differs from canonical assignment'
            for name,h in r['outputs'].items():assert digest(Path(config['source'])/'experiments'/j['run_relative']/name)==h
        return proof
    assert lane in FUTURE,'Lane has not been authorized as future Figure4 capacity'
    if host=='foureyes':
        stage=receipt('artifacts/figure4/future_capacity/foureyes-ready.json')
        assert stage['stage_ready'] and stage['hostname']=='foureyes'
        assert stage['source']==config['source'] and stage['python']==config['python']
    if host=='foureyes' and gpu<3:
        main=receipt(f'artifacts/main_followups/seed{gpu}.json')
        assert main['status']=='complete',f'Main followups have not completed: {main["status"]}'
        assert main['data_efficiency_exit_code']==0,'Main followups report a failed data-efficiency lane'
        absent(main.get('worker_pid'))
        de=receipt(f'artifacts/data_efficiency/gpu{gpu}-worker.json')
        assert de['status']=='complete',f'Data-efficiency lane has not completed: {de["status"]}'
        absent(de.get('worker_pid'),de.get('child_pid'))
        assert {p['name'] for p in main['phases']}=={'nqueens-free-8','nqueens-free-10','nqueens-8-precondition-on',
                    'nqueens-8-precondition-off','nqueens-10-precondition-off','sudoku-difficulty-zero'}
        for phase in main['phases']:
            assert phase.get('exit_code')==0 and phase.get('marker')
            assert digest(phase['marker'])==phase['marker_sha256']
        cfg=receipt('configs/data_efficiency_jobs.json');jobs=[j for j in cfg['jobs'] if j['lane']==gpu]
        assert len(jobs)==34
        for j in jobs:
            m=receipt(f'artifacts/data_efficiency/jobs/{j["id"]}.json')
            assert m['status']=='complete' and m['config_sha256']==digest(ROOT/'configs/data_efficiency_jobs.json')
            assert set(m['output_sha256'])=={'model.pt','fit.json','test10k.json'}
            run=Path(cfg['source'])/'experiments'/cfg['run_root']/f'k{j["k"]}-s{j["seed"]}'
            for name,h in m['output_sha256'].items():assert digest(run/name)==h
        return proof
    seed=0 if host=='foureyes' else gpu-1
    b=receipt(f'artifacts/long_bptt/seed{seed}.json')
    assert b['status']=='complete',f'Long BPTT fit and test have not completed: {b["status"]}'
    assert b['test_exit_code']==0 and b['test']['n']==50000,'Long BPTT final test did not complete successfully'
    assert b['source_commit']==COMMIT and b['seed']==seed
    absent(b.get('training_pid'))
    run=Path(b['run']);f=read(run/'final.json');t=read(run/'test.json')
    assert f['updates']==f['config']['updates']==max(x['update'] for x in f['curve'])==3840000
    assert f['config']['method']=='bptt' and f['config']['seed']==seed
    assert t['test']==b['test'] and t['selected_update']==b['selected_update']==f['selected']['update']
    assert t['puzzles']==[10000,59999] and len(t['flops'])==50000
    for name,key in [('final.json','final_sha256'),('test.json','test_sha256'),
                     (f'model-u{b["selected_update"]}.pt','selected_checkpoint_sha256')]:assert digest(run/name)==b[key]
    q=receipt(f'artifacts/maze_hard_ablations/queue-{host}-gpu{gpu}.json')
    assigned=[['token-source',seed]]+([['edge-gates',0]] if host=='foureyes' else [])
    assert q['status']=='complete',f'Assigned Maze-Hard ablations have not completed: {q["status"]}'
    assert q['jobs']==assigned and q['marker_seed']==seed
    assert [[j['variant'],j['seed']] for j in q['completed']]==assigned
    absent(q.get('pid'),q.get('child_pid'))
    for variant,s in assigned:
        base=ROOT/f'artifacts/maze_hard_ablations/{variant}'
        a=receipt(f'artifacts/maze_hard_ablations/{variant}/provenance-s{s}.json')
        d=read(base/f'dev-s{s}.json')
        assert a['status']=='complete' and a['result']['n']==d['dev100']['n']==100
        assert d['seed']==s and d['recipe']==a['recipe'] and d['curve'][-1]['u']==4000
        assert not a['test_data_used'] and not a['tuning']
        assert len(a['fit_rows'])==900 and len(a['development_rows'])==100
        assert not set(a['fit_rows'])&set(a['development_rows'])
        assert digest(base/f'model-s{s}.pt')==a['model_sha256']
    return proof


def source_check(config):
    source=Path(config['source']);data=Path(config['data'])
    for rel,h in config['source_sha256'].items():assert digest(source/rel)==h,rel
    for item in config['data_manifest']:assert digest(data/item['path'])==item['sha256'],item['path']
    for rel,h in config['fresh_hbar_prerequisite_sha256'].items():assert digest(source/rel)==h,rel
    for recipe in config['recipe_materializations']:
        for rel in recipe['to']:assert (source/rel).read_text()==recipe['text'],rel
    return dict(source_files=len(config['source_sha256']),data_files=len(config['data_manifest']),
                fresh_hbar_files=len(config['fresh_hbar_prerequisite_sha256']))


def gpu_owners(gpu):
    uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    all_owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    return dict(uuid=uuid,owners=[line for line in all_owners.splitlines() if uuid in line])


def physical_lock_path(config,lane):
    g=lane.split(':')[1]
    default=(ROOT/f'artifacts/objectives/gpu-{g}.lock' if lane in config['lanes']
             else ROOT/f'artifacts/gpu{g}.lock')
    return Path(config.get('physical_gpu_locks',{}).get(lane,str(default)))


def snapshot(config,target,guard_sha):
    host=socket.gethostname().split('.')[0].lower();procs=processes()
    actual=read(CONFIG) if CONFIG.exists() else None
    if actual is not None:
        # A stale scheduler copy could still dispatch an old ownership map.
        assert actual['jobs']==config['jobs'],'Host assignments differ from canonical config'
        for k in ['source','source_sha256','data_manifest','fresh_hbar_prerequisite_sha256','recipe_materializations']:
            assert actual[k]==config[k],f'Host frozen configuration differs: {k}'
    else:assert target in FUTURE and target.split(':')[0]==host
    assert digest(ROOT/'scripts/figure4_queue.py')==guard_sha,'Reviewed dispatch guard differs on host'
    states={}
    for lane in config['lanes']:
        h,g=lane.split(':')
        if h!=host:continue
        p=ROOT/f'artifacts/figure4/{h}-gpu{g}-state.json';states[lane]=read(p) if p.exists() else None
    exists={j['id']:dict(run=(Path(config['source'])/'experiments'/j['run_relative']).exists(),
                        execution=(ROOT/'artifacts/figure4/jobs'/j['id']).exists()) for j in config['jobs']}
    result=dict(host=host,config_sha256=digest(CONFIG) if CONFIG.exists() else None,
                states=states,processes=procs,job_paths=exists,guard_sha256=guard_sha,utc=now())
    if target.split(':')[0]==host:
        try:
            assert not any(p['lane']==target for p in procs),'Target has a live Figure4/D2 supervisor'
            result['prerequisites']=prerequisite_check(target,config,target in config['lanes'])
            result['source_validation']=source_check(config)
            result['gpu']=gpu_owners(int(target.split(':')[1]));assert not result['gpu']['owners'],'Target GPU has compute owners'
            result['target_ready']=True
        except Exception as exc:result.update(target_ready=False,target_blocker=repr(exc))
    return result


def donor_eligible(lane,snapshot):
    state=snapshot['states'].get(lane);procs=[p for p in snapshot['processes'] if p['lane']==lane]
    if not state:return False
    if state.get('status')=='queued_after_d2':
        return bool(procs) and all(any(Path(a).name=='d2_queue.py' for a in p['argv']) for p in procs)
    # A waiting Figure4 supervisor may have loaded older static-dispatch code.
    # The on-disk script hash cannot certify code already loaded in that PID.
    return (state.get('dynamic_dispatch_v1') is True and len(procs)==1
            and procs[0]['pid']==state.get('worker_pid'))


def candidates(config,snapshots,target):
    rows=[]
    for job in config['jobs']:
        if job['dense'] or job['lane']==target:continue
        donor=snapshots[job['lane'].split(':')[0]]
        if not donor_eligible(job['lane'],donor):continue
        # Absence is checked on every involved host, including dead-worker
        # outputs on a former host. Even an empty run directory is ineligible.
        if any(any(s['job_paths'][job['id']].values()) for s in snapshots.values()):continue
        rows.append(job)
    return rows


def revise(config,ids,target,migration_id):
    new=json.loads(json.dumps(config));chosen=set(ids);changes=[]
    assert len(chosen)==len(ids) and chosen
    for old,j in zip(config['jobs'],new['jobs']):
        if j['id'] in chosen:
            assert not j['dense'] and j['lane']!=target
            changes.append(dict(job=j['id'],from_lane=j['lane'],to_lane=target));j['lane']=target
        assert {k:v for k,v in old.items() if k!='lane'}=={k:v for k,v in j.items() if k!='lane'}
    assert len(changes)==len(ids)
    if target not in new['lanes']:
        assert target in FUTURE
        new['lanes'].append(target)
        new.setdefault('wait_for_table7_by_lane',{})[target]=False
        new.setdefault('physical_gpu_locks',{})[target]=str(ROOT/f'artifacts/gpu{target.split(":")[1]}.lock')
        if target in new.get('mps_future_lanes',[]):
            enabled=new.setdefault('mps_probe_lanes',[])
            if target not in enabled:enabled.append(target)
    new.setdefault('dispatch_migrations',[]).append(dict(id=migration_id,utc=now(),changes=changes))
    return new,changes


def serve():
    """Small lock-holding JSON RPC server; run over SSH by the coordinator."""
    dispatch=None;physical=None;staged=None
    try:
        for line in sys.stdin:
            try:
                req=json.loads(line);op=req['op']
                if op=='lock':
                    path=ROOT/'artifacts/figure4/dispatch.lock';path.parent.mkdir(parents=True,exist_ok=True)
                    dispatch=path.open('a');deadline=time.monotonic()+10
                    while True:
                        try:fcntl.flock(dispatch,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                        except BlockingIOError:
                            if time.monotonic()>deadline:raise TimeoutError('Dispatch lock busy')
                            time.sleep(.05)
                    value=dict(locked=True,host=socket.gethostname())
                elif op=='inspect':
                    assert dispatch is not None
                    value=snapshot(req['config'],req['target'],req['guard_sha256'])
                    if value.get('target_ready'):
                        lane=req['target'];g=lane.split(':')[1]
                        lock=physical_lock_path(req['config'],lane)
                        if physical is None:
                            try:
                                physical=lock.open('a');fcntl.flock(physical,fcntl.LOCK_EX|fcntl.LOCK_NB)
                            except (OSError,BlockingIOError) as exc:
                                if physical is not None:physical.close();physical=None
                                value.update(target_ready=False,target_blocker=f'Physical GPU lock unavailable: {exc}')
                        value['physical_lock']=str(lock)
                elif op=='stage':
                    assert dispatch is not None
                    before=CONFIG.read_bytes() if CONFIG.exists() else None
                    assert (hashlib.sha256(before).hexdigest() if before is not None else None)==req['expected_sha256']
                    after=base64.b64decode(req['after_base64']);folder=ART/req['migration_id'];folder.mkdir(parents=True,exist_ok=True)
                    if before is not None:atomic_bytes(folder/'config-before.json',before)
                    atomic_bytes(folder/'config-after.json',after)
                    staged=dict(before=before,after=after,folder=folder)
                    value=dict(staged=True,after_sha256=hashlib.sha256(after).hexdigest())
                elif op=='commit':
                    assert staged is not None and dispatch is not None
                    assert (digest(CONFIG) if CONFIG.exists() else None)==(hashlib.sha256(staged['before']).hexdigest() if staged['before'] else None)
                    atomic_bytes(CONFIG,staged['after']);save(staged['folder']/'remote-receipt.json',dict(status='committed',utc=now(),config_sha256=digest(CONFIG)))
                    value=dict(committed=True,config_sha256=digest(CONFIG))
                elif op=='rollback':
                    assert staged is not None
                    # Only replace our own after-image; never overwrite a third party edit.
                    if CONFIG.exists() and digest(CONFIG)==hashlib.sha256(staged['after']).hexdigest():
                        if staged['before'] is None:CONFIG.unlink()
                        else:atomic_bytes(CONFIG,staged['before'])
                    value=dict(rolled_back=True)
                elif op=='launch':
                    assert physical is not None and staged is not None
                    lane=req['target'];assert not any(p['lane']==lane for p in processes())
                    assert not gpu_owners(int(lane.split(':')[1]))['owners']
                    argv=['/usr/bin/python3',str(ROOT/'scripts/figure4_queue.py'),'worker','--lane',lane]
                    with (staged['folder']/'worker-launch.log').open('ab',buffering=0) as log:
                        child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                    value=dict(pid=child.pid,argv=argv,utc=now())
                    save(staged['folder']/'worker-launch.json',value)
                else:raise ValueError(op)
                print(json.dumps(dict(ok=True,value=value)),flush=True)
            except Exception as exc:print(json.dumps(dict(ok=False,error=repr(exc))),flush=True)
    finally:
        if physical is not None:physical.close()
        if dispatch is not None:dispatch.close()


class Session:
    def __init__(self,host,folder):
        self.host=host;self.stderr=(folder/f'{host}-rpc.log').open('ab',buffering=0)
        payload=base64.b64encode(Path(__file__).read_bytes()).decode()
        code=f'import base64;exec(compile(base64.b64decode({payload!r}),{str(Path(__file__))!r},"exec"))'
        argv=([sys.executable,'-c',code,'serve'] if host==socket.gethostname().split('.')[0].lower()
              else SSH+[host,shlex.join(['python3','-c',code,'serve'])])
        self.proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1)
        try:self.request(dict(op='lock'))
        except BaseException:
            self.close();raise
    def request(self,value,timeout=120):
        self.proc.stdin.write(json.dumps(value)+'\n');self.proc.stdin.flush()
        ready,_,_=select.select([self.proc.stdout],[],[],timeout)
        if not ready:raise TimeoutError(f'{self.host} RPC {value["op"]}')
        line=self.proc.stdout.readline()
        if not line:raise RuntimeError(f'{self.host} RPC exited: {self.proc.poll()}')
        result=json.loads(line)
        if not result['ok']:raise RuntimeError(f'{self.host}: {result["error"]}')
        return result['value']
    def close(self):
        self.proc.stdin.close()
        try:self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:self.proc.terminate();self.proc.wait(timeout=10)
        self.stderr.close()


def coordinate(target,max_jobs=4,apply_plan=None):
    assert 1<=max_jobs<=100
    ART.mkdir(parents=True,exist_ok=True)
    with (ART/'coordinator.lock').open('a') as global_lock:
        fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=CONFIG.read_bytes();config=json.loads(before);before_sha=hashlib.sha256(before).hexdigest()
        assert target in config['lanes'] or target in FUTURE
        mid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+f'-{target.replace(":","-")}'
        folder=ART/mid;folder.mkdir();atomic_bytes(folder/'config-before.json',before)
        guard=digest(ROOT/'scripts/figure4_queue.py')
        hosts=sorted({l.split(':')[0] for l in config['lanes']}|{target.split(':')[0]})
        record=dict(id=mid,target=target,started_utc=now(),status='inspecting',config_before_sha256=before_sha,
                    helper_sha256=digest(Path(__file__)),guard_sha256=guard)
        sessions={};committed=[];staged_hosts=[];canonical_committed=False
        try:
            for host in hosts:sessions[host]=Session(host,folder)
            snapshots={h:s.request(dict(op='inspect',config=config,target=target,guard_sha256=guard)) for h,s in sessions.items()}
            ready=snapshots[target.split(':')[0]].get('target_ready',False)
            eligible=candidates(config,snapshots,target)
            if apply_plan:
                assert apply_plan['config_before_sha256']==before_sha,'Canonical configuration changed since plan'
                ids=apply_plan['job_ids'];assert set(ids)<=set(j['id'] for j in eligible),'Planned job now has ownership or outputs'
                assert ready,'Target prerequisites/GPU/process/lock are not ready'
            else:ids=[j['id'] for j in eligible[:max_jobs]]
            record.update(status='plan_ready' if ready and ids else 'not_ready',target_ready=ready,
                          target_blocker=snapshots[target.split(':')[0]].get('target_blocker'),job_ids=ids,
                          eligible_pending_jobs=len(eligible),snapshots=snapshots)
            if not ids:
                save(folder/'receipt.json',record);return record
            after,changes=revise(config,ids,target,mid);data=encoded(after);after_sha=hashlib.sha256(data).hexdigest()
            atomic_bytes(folder/'config-after.json',data)
            record.update(changes=changes,config_after_sha256=after_sha,config_after_path=str(folder/'config-after.json'))
            save(folder/'receipt.json',record)
            if not apply_plan:return record
            for h,s in sessions.items():
                s.request(dict(op='stage',expected_sha256=snapshots[h]['config_sha256'],after_base64=base64.b64encode(data).decode(),migration_id=mid))
                staged_hosts.append(h)
            for h,s in sessions.items():
                result=s.request(dict(op='commit'));assert result['config_sha256']==after_sha;committed.append(h)
            local_host=socket.gethostname().split('.')[0].lower()
            if local_host in sessions:
                # A future local target shares this canonical filesystem.
                assert digest(CONFIG)==after_sha
            else:
                assert digest(CONFIG)==before_sha
                atomic_bytes(CONFIG,data)
            canonical_committed=True
            record.update(status='committed',committed_utc=now(),hosts=committed);save(folder/'receipt.json',record)
            launched=sessions[target.split(':')[0]].request(dict(op='launch',target=target))
            record.update(status='dispatched',worker=launched);save(folder/'receipt.json',record)
            return record
        except Exception as exc:
            record.update(status='committed_launch_uncertain' if canonical_committed else 'failed',error=repr(exc))
            if not canonical_committed:
                record['rollback']={}
                for h in staged_hosts:
                    try:record['rollback'][h]=sessions[h].request(dict(op='rollback'))
                    except Exception as err:record['rollback'][h]=dict(error=repr(err))
            save(folder/'receipt.json',record);raise
        finally:
            for session in reversed(list(sessions.values())):session.close()


def auto_once(max_jobs=4):
    """Cheap eligibility prefilter, then full locked plan/apply revalidation.

    Called by the study collector. Stale snapshots may delay work or trigger a
    rejected plan; they can never bypass owning-host prerequisites and locks.
    At most one target is refilled per call.
    """
    config=read(CONFIG);possible=[]
    local=socket.gethostname().split('.')[0].lower()
    for lane in config['lanes']:
        host,g=lane.split(':')
        p=ROOT/'artifacts/figure4'/('' if host==local else host)/f'{host}-gpu{g}-state.json'
        if p.exists() and read(p).get('status')=='queue_complete':possible.append(lane)
    for lane in sorted(FUTURE-set(config['lanes'])):
        host,g=lane.split(':');gpu=int(g)
        if host=='foureyes' and gpu<3:
            paths=[ROOT/f'artifacts/main_followups/seed{gpu}.json',ROOT/f'artifacts/data_efficiency/gpu{gpu}-worker.json']
            if all(p.exists() and read(p).get('status')=='complete' for p in paths):possible.append(lane)
        else:
            p=ROOT/f'artifacts/maze_hard_ablations/queue-{host}-gpu{gpu}.json'
            if p.exists() and read(p).get('status')=='complete':possible.append(lane)
    attempts=[]
    for target in possible:
        try:
            plan=coordinate(target,max_jobs)
            if plan['status']=='plan_ready':return coordinate(target,max_jobs,apply_plan=plan)
            attempts.append(dict(target=target,status=plan['status'],blocker=plan.get('target_blocker')))
        except Exception as exc:attempts.append(dict(target=target,status='recheck_rejected',error=repr(exc)))
    return dict(status='waiting_for_eligible_capacity',utc=now(),attempts=attempts)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('serve')
    q=sub.add_parser('plan');q.add_argument('--target',required=True);q.add_argument('--max-jobs',type=int,default=4)
    q=sub.add_parser('apply');q.add_argument('--plan',type=Path,required=True)
    q=sub.add_parser('auto');q.add_argument('--max-jobs',type=int,default=4)
    a=p.parse_args()
    if a.command=='serve':serve();return
    if a.command=='plan':result=coordinate(a.target,a.max_jobs)
    elif a.command=='auto':result=auto_once(a.max_jobs)
    else:
        plan=read(a.plan);result=coordinate(plan['target'],apply_plan=plan)
    print(json.dumps({k:result.get(k) for k in ['id','status','target','target_ready','target_blocker','job_ids','eligible_pending_jobs','worker']}))


if __name__=='__main__':main()
