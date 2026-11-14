"""Consequential dispatch checks; synthetic files only, never GPU processes."""
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import migrate_pending_figure4 as M
import figure4_queue as F


@pytest.fixture
def environment(tmp_path,monkeypatch):
    root=tmp_path;config_path=root/'configs/figure4_jobs.json'
    config_path.parent.mkdir();(root/'artifacts/figure4').mkdir(parents=True)
    (root/'scripts').mkdir();(root/'scripts/figure4_queue.py').write_text('synthetic reviewed guard')
    cfg=dict(source=str(root/'source'),source_sha256={},data_manifest=[],fresh_hbar_prerequisite_sha256={},
             recipe_materializations=[],lanes=['work:0','ripper:0'],jobs=[])
    for index,name in enumerate(['complete','active','pending','dense']):
        cfg['jobs'].append(dict(id=name,lane='work:0',dense=name=='dense',method='pl',seed=index,
                               run_relative=f'runs/{name}',commands=[['train','--seed',str(index)]]))
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(M,'ROOT',root);monkeypatch.setattr(M,'CONFIG',config_path)
    monkeypatch.setattr(M,'ART',root/'artifacts/figure4/migrations');monkeypatch.setattr(F,'ROOT',root)
    return root,config_path,cfg


def snapshots(cfg):
    base=dict(states={'work:0':dict(status='running',dynamic_dispatch_v1=True,worker_pid=100)},
              processes=[dict(pid=100,lane='work:0',argv=['figure4_queue.py'])],
              job_paths={j['id']:dict(run=False,execution=False) for j in cfg['jobs']})
    return {'work':base,'ripper':dict(states={},processes=[],job_paths=deepcopy(base['job_paths']))}


def test_lane_change_preserves_existing_active_and_complete_receipts(environment):
    root,path,cfg=environment;base=dict(config_sha256=F.digest(path),worker_pid=100)
    original={}
    for job in cfg['jobs'][:2]:
        claim=F.claim_job(path,job,'work:0',base,cfg)
        owner=claim[1]/'figure4_provenance.json';original[job['id']]=(claim[2],owner.read_bytes())
    new,changes=M.revise(cfg,['pending'],'ripper:0','test-migration')
    path.write_text(json.dumps(new))
    assert F.digest(path)!=base['config_sha256']
    for job in cfg['jobs'][:2]:
        claim=F.claim_job(path,job,'work:0',base,cfg);owner=claim[1]/'figure4_provenance.json'
        assert (claim[2],owner.read_bytes())==original[job['id']]
        assert json.loads(owner.read_text())['config_sha256']==base['config_sha256']
    assert cfg['jobs'][3]==new['jobs'][3]  # Dense clock mapping is unchanged.
    assert changes==[dict(job='pending',from_lane='work:0',to_lane='ripper:0')]


def test_old_loaded_worker_rereads_lane_under_dispatch_lock(environment):
    root,path,cfg=environment;job=cfg['jobs'][2];result=[];entered=threading.Event()
    with (root/'artifacts/figure4/dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        def claim():
            entered.set();result.append(F.claim_job(path,job,'work:0',{},cfg))
        worker=threading.Thread(target=claim);worker.start();assert entered.wait(1)
        new,_=M.revise(cfg,['pending'],'ripper:0','test');path.write_text(json.dumps(new))
        fcntl.flock(lock,fcntl.LOCK_UN)
    worker.join(2);assert not worker.is_alive() and result==[None]
    assert not (Path(cfg['source'])/'experiments'/job['run_relative']).exists()


def test_untouched_target_claim_is_unique_and_frozen(environment):
    root,path,cfg=environment;job=cfg['jobs'][2]
    new,_=M.revise(cfg,['pending'],'ripper:0','test');path.write_text(json.dumps(new))
    claim=F.claim_job(path,job,'ripper:0',{},new)
    assert claim[0]['seed']==job['seed'] and claim[0]['commands']==job['commands']
    assert claim[0]['lane']=='ripper:0'
    again=F.claim_job(path,job,'ripper:0',{},new);assert again[2]==claim[2]
    assert F.claim_job(path,job,'work:0',{},cfg) is None


def test_dense_jobs_cannot_migrate(environment):
    with pytest.raises(AssertionError):M.revise(environment[2],['dense'],'ripper:0','test')


@pytest.mark.parametrize('field',['run','execution'])
def test_partial_or_dead_worker_outputs_are_not_never_started(environment,field):
    cfg=environment[2];ss=snapshots(cfg)
    ss['ripper']['job_paths']['pending'][field]=True
    assert 'pending' not in [j['id'] for j in M.candidates(cfg,ss,'ripper:0')]


def test_active_old_worker_without_guard_cannot_donate(environment):
    cfg=environment[2];ss=snapshots(cfg);ss['work']['states']['work:0'].pop('dynamic_dispatch_v1')
    assert M.candidates(cfg,ss,'ripper:0')==[]


def test_numerical_recipe_change_rejected_on_claim(environment):
    root,path,cfg=environment;job=cfg['jobs'][2];new=deepcopy(cfg);new['jobs'][2]['seed']=99
    path.write_text(json.dumps(new))
    with pytest.raises(AssertionError):F.claim_job(path,job,'work:0',{},cfg)


def test_unapproved_future_lane_rejected(environment):
    with pytest.raises(AssertionError):M.revise(environment[2],['pending'],'unknown:0','test')


def test_waiting_legacy_figure_worker_is_not_a_guarded_donor(environment):
    cfg=environment[2];ss=snapshots(cfg)
    ss['work']['states']['work:0']=dict(status='queued_after_table7',worker_pid=100)
    assert M.candidates(cfg,ss,'ripper:0')==[]


def test_failed_remote_commit_rolls_back_all_staged_hosts(environment,monkeypatch):
    root,path,cfg=environment;before=path.read_bytes();instances={};launched=[]
    class FakeSession:
        def __init__(self,host,folder):self.host=host;self.content=before;instances[host]=self
        def request(self,request):
            op=request['op']
            if op=='inspect':
                x=deepcopy(snapshots(cfg)[self.host]);x.update(config_sha256=hashlib.sha256(before).hexdigest(),target_ready=True)
                return x
            if op=='stage':self.after=M.base64.b64decode(request['after_base64']);return {}
            if op=='commit':
                self.content=self.after
                if self.host=='work':raise RuntimeError('simulated loss after remote commit')
                return dict(config_sha256=hashlib.sha256(self.content).hexdigest())
            if op=='rollback':self.content=before;return dict(rolled_back=True)
            if op=='launch':launched.append(True);raise AssertionError('Launch after failed transaction')
            raise AssertionError(op)
        def close(self):pass
    monkeypatch.setattr(M,'Session',FakeSession)
    plan=dict(config_before_sha256=hashlib.sha256(before).hexdigest(),job_ids=['pending'])
    with pytest.raises(RuntimeError):M.coordinate('ripper:0',apply_plan=plan)
    assert path.read_bytes()==before and not launched
    assert all(s.content==before for s in instances.values())


def test_physical_lock_matches_worker_default_and_future_allocation(environment):
    root,_,cfg=environment
    assert M.physical_lock_path(cfg,'work:0')==root/'artifacts/objectives/gpu-0.lock'
    assert M.physical_lock_path(cfg,'foureyes:0')==root/'artifacts/gpu0.lock'
    cfg['physical_gpu_locks']={'ripper:0':str(root/'artifacts/gpu0.lock')}
    assert M.physical_lock_path(cfg,'ripper:0')==root/'artifacts/gpu0.lock'


def test_future_lane_waits_for_full_long_bptt_test(environment):
    root,_,cfg=environment;p=root/'artifacts/long_bptt/seed1.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(status='evaluating_selected_checkpoint',test_exit_code=0,test=dict(n=50000))))
    with pytest.raises(AssertionError,match='Long BPTT fit and test have not completed'):
        M.prerequisite_check('ripper:2',cfg,False)


def test_local_target_commits_shared_canonical_file_once(environment,monkeypatch):
    root,path,cfg=environment;before=path.read_bytes();launched=[]
    class FakeSession:
        def __init__(self,host,folder):self.host=host
        def request(self,request):
            op=request['op']
            if op=='inspect':
                ss=snapshots(cfg)
                x=deepcopy(ss.get(self.host,dict(states={},processes=[],job_paths=ss['ripper']['job_paths'])))
                x.update(config_sha256=hashlib.sha256(before).hexdigest(),target_ready=True);return x
            if op=='stage':self.after=M.base64.b64decode(request['after_base64']);return {}
            if op=='commit':
                if self.host=='foureyes':path.write_bytes(self.after)
                return dict(config_sha256=hashlib.sha256(self.after).hexdigest())
            if op=='launch':
                launched.append(request['target']);return dict(pid=999,synthetic=True)
            raise AssertionError(op)
        def close(self):pass
    monkeypatch.setattr(M,'Session',FakeSession);monkeypatch.setattr(M.socket,'gethostname',lambda:'foureyes')
    result=M.coordinate('foureyes:0',apply_plan=dict(config_before_sha256=hashlib.sha256(before).hexdigest(),job_ids=['pending']))
    after=json.loads(path.read_text())
    assert result['status']=='dispatched' and launched==['foureyes:0']
    assert after['jobs'][2]['lane']=='foureyes:0' and after['jobs'][3]==cfg['jobs'][3]
    assert after['physical_gpu_locks']['foureyes:0']==str(root/'artifacts/gpu0.lock')


def test_future_mps_is_enabled_only_by_explicit_allowlist(environment):
    cfg=environment[2]
    plain,_=M.revise(cfg,['pending'],'foureyes:0','plain')
    assert 'foureyes:0' not in plain.get('mps_probe_lanes',[])
    cfg['mps_future_lanes']=['foureyes:0'];cfg['mps_probe_lanes']=['ripper:0']
    opted,_=M.revise(cfg,['pending'],'foureyes:0','opted')
    assert opted['mps_probe_lanes']==['ripper:0','foureyes:0']
