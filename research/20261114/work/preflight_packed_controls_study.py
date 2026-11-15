"""H2bt synthetic lifecycle and host orchestration; no source outcomes or GPUs."""
import ast
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import numpy as np
import study as S
import packed_controls_pipeline as P
from packed_controls_source_parity import Source


def lifecycle(tmp):
    env={**os.environ,'PYTHONPATH':str(S.HERE),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    code='import sys;from pathlib import Path;import packed_controls_study as M;M.DEST=Path(sys.argv[1]);M.GATE_PP=Path(sys.argv[2]);M.worker("gate",0)'
    out=tmp/'worker';command=[sys.executable,'-c',code,str(out),str(tmp/'missing')]
    first=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10);assert first.returncode!=0
    file=out/'failure-gate-gpu0.json';before=S.sha(file);assert json.loads(file.read_text())['status']=='incomplete'
    second=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10);assert second.returncode!=0 and 'FileExistsError' in second.stderr and S.sha(file)==before
    fifo=tmp/'blocked';os.mkfifo(fifo);out=tmp/'signal';child=subprocess.Popen([sys.executable,'-c',code,str(out),str(fifo)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+5
        while not (out/'gate-worker0.json').exists():assert child.poll() is None and time.monotonic()<deadline;time.sleep(.01)
        child.send_signal(signal.SIGALRM);child.communicate(timeout=5)
        record=json.loads((out/'failure-gate-gpu0.json').read_text());assert record['status']=='incomplete' and 'TimeoutError' in record['error']
    finally:
        if child.poll() is None:child.kill();child.communicate()
    out=tmp/'audit';out.mkdir();command=[sys.executable,str(S.HERE/'audit_packed_controls.py'),str(S.ROOT),str(out),'--workers','16']
    first=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10);assert first.returncode!=0
    file=out/'audit-worker.json';before=S.sha(file);assert json.loads(file.read_text())['status']=='incomplete'
    second=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10);assert second.returncode!=0 and 'FileExistsError' in second.stderr and S.sha(file)==before
    source=tmp/'incomplete-source';source.mkdir();file=source/'audit.json';file.write_text(json.dumps({'status':'incomplete'}))
    try:Source(S.ROOT,{'directory':str(source),'protocol_sha256':'none','audit_sha256':S.sha(file)})
    except AssertionError:pass
    else:raise AssertionError('Source consumed before full audit')


def harness(tmp):
    fake=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None),no_grad=nullcontext)
    originals={key:getattr(P,key) for key in ['load_models','batch','payloads','materialize','make_queue','scatter']};old_imports=S.imports;active=[None];batches=[]
    class Barrier:
        def __init__(self,path):self.directory=path;self.gpu=0;self.events=[]
        def gate(self,name):active[0]=name;self.events.append({'stage':name,'wait_seconds':0.});return 0.
    def load(p,mode):
        assert active[0]=='readiness-begin';required=['lp'] if mode=='LP96' else ['bp','lp'];return {s:{m:None for m in required} for s in range(3)},{s:{m:None for m in required} for s in range(3)},[{'seed':s,'model':m,'seconds':0.} for s in range(3) for m in required]
    def prepare(*args):assert active[0].endswith('preparation-begin');return originals['make_queue'](*args)
    def materialize(*args):assert active[0].endswith('materialize-begin');return originals['materialize'](*args)
    def scatter(*args):assert active[0].endswith('scatter-begin');return originals['scatter'](*args)
    def batch(mode,tokens,models,doubles,watchdog,after_answer):
        assert active[0].endswith('materialize-done');assert set(models)==set(doubles)==({'lp'} if mode=='LP96' else {'bp','lp'});n=len(tokens);assert 1<=n<=32;batches.append(n)
        start=time.perf_counter();answer={'prediction':tokens>=2,'chosen_model':np.ones(n,np.int8)};duration=time.perf_counter()-start
        keys=['input_transfer','hidden_readout_and_flags','LP96_complete_prefix','LP96_answer_return','LP96_original_margin_and_return','final_numerical_acceptance','BP_native_solve','BP_cap_flags','BP_answer_return','BP_original_margin_and_return','geometry_screen_and_indices','LP_subset_gather','subset_scatter_and_provenance','endpoint_extraction_and_dispatch','BP_subset_gather'];components={k:0. for k in keys};components['model_provenance']=0.;components['input_transfer']=duration
        result={'answer':answer,'audit':{'couplings':np.ones((n,1),np.float32)},'timing':{'mode':mode,'components':components,'n':n,'actual_BP_inputs':0,'direct_LP_inputs':n,'rescue_LP_inputs':0,'actual_subset_inputs':0 if mode=='LP96' else n,'answer_seconds':duration,'unattributed_in_segment_seconds':0.}}
        after_answer();return result
    def payload(answer,directory,tag,ph,retain,offset):return ((directory/(tag+'.npz'),{'answer_'+k:v for k,v in answer['answer'].items()}),[])
    try:
        S.imports=lambda:(fake,None,None);P.load_models=load;P.make_queue=prepare;P.materialize=materialize;P.scatter=scatter;P.batch=batch;P.payloads=payload
        tokens=np.ones((65,180,180),np.uint8);tokens[:,0,0]=2
        for i in range(65):tokens[i,179,2+i%3]=3
        records=[]
        for mode in ['LP96','BP_geometry_LP96','packed_dispatch']:
            folder=tmp/mode;barrier=Barrier(tmp/('barrier-'+mode));r=P.cohort({},'synthetic',mode,{0:tokens},np.arange(65),folder,lambda:None,lambda **kw:None,barrier);records.append(r)
            assert len(r['model_readiness'])==(3 if mode=='LP96' else 6)
            assert r['model_inputs']==195 and len(r['rows'])==12 and sum(len(row['indices']) for row in r['rows'])==195
            assert all(row['answer_timing'] is None for row in r['rows'] if row['zero_work'])
            for seed in range(3):
                with np.load(folder/f'seed{seed}-global-answer.npz',allow_pickle=False) as z:assert np.array_equal(z['prediction'],tokens>=2) and np.array_equal(z['global_indices'],np.arange(65))
            try:P.cohort({},'synthetic',mode,{0:tokens},np.arange(65),folder,lambda:None,lambda **kw:None,barrier)
            except FileExistsError:pass
            else:raise AssertionError('Cohort overwritten')
        assert sum(batches)==585 and any(n<32 for n in batches)
        return {'synthetic_actual_batches':len(batches),'synthetic_unpadded_inputs':sum(batches),'charged_preparation_materialization_scatter_order_checked':True,'zero_work_rows':sum(row['zero_work'] for r in records for row in r['rows'])}
    finally:
        S.imports=old_imports
        for key,value in originals.items():setattr(P,key,value)


def main():
    started=time.perf_counter();tree=ast.parse((S.HERE/'packed_controls_study.py').read_text());reads=[n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='archive' and isinstance(n.slice,ast.Constant)];assert reads==['tokens']
    dest=S.OUT/'bp_first_packed_controls';dest.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synthetic-',dir=dest) as tmp:lifecycle(Path(tmp));checks=harness(Path(tmp))
    names=sorted(p for p in S.HERE.glob('*packed_controls*.py'))
    for file in names:compile(file.read_text(),str(file),'exec')
    result={'status':'passed','source_outcomes_read':False,'gpu_executions':0,'worker_archive_reads':reads,'failure_overwrite_owned_alarm_and_audit_retention_checked':True,'incomplete_source_guard_checked':True,**checks,'seconds':time.perf_counter()-started,'source_sha256':{str(p.relative_to(S.ROOT)):S.sha(p) for p in names}}
    S.save(dest/'study-synthetic-gate.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
