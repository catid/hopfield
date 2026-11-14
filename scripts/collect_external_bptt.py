"""Collect completed root-managed 3.84M-update BPTT runs for Table 7.

Reads completion receipts before copying any run outputs. This never launches
training or inference. A completed run's full timing log is transferred once;
subsequent calls validate the cached bundle and completion-receipt identity.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/long_bptt/collected'
COMMIT='62bbe830c7d6f87f2ba18f30a28039258b986454'
SSH=['ssh','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=10']
EXPECTED=dict(task='sudoku',method='bptt',lr=2.0,knob=10.0,reg=0.,l1=.1,updates=3840000,
              n_train=0,aug=True)


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def remote_bytes(host,path):
    if host=='foureyes': return Path(path).read_bytes()
    proc=subprocess.run(SSH+[host,shlex.join(['cat','--',str(path)])],capture_output=True,timeout=30)
    if proc.returncode:
        if b'No such file or directory' in proc.stderr: raise FileNotFoundError(str(path))
        raise RuntimeError(f'{host}: {proc.stderr.decode(errors="replace").strip()}')
    return proc.stdout


def copy_file(host,source,dest):
    dest=Path(dest);temp=dest.with_suffix(dest.suffix+f'.{os.getpid()}.tmp')
    if host=='foureyes': shutil.copyfile(source,temp)
    else:
        subprocess.run(['rsync','--archive','--protect-args','--timeout=120','-e',shlex.join(SSH),
                        f'{host}:{source}',str(temp)],check=True,capture_output=True,timeout=300)
    temp.replace(dest)


def validate_completed(folder,marker,seed):
    """Pure local validation; usable in negative checks with synthetic fixtures."""
    folder=Path(folder)
    assert marker['status']=='complete' and marker['test_exit_code']==0 and marker['seed']==seed
    assert marker['source_commit']==COMMIT and marker['test']['n']==50000
    fit=json.loads((folder/'final.json').read_text()); config=json.loads((folder/'config.json').read_text())
    test=json.loads((folder/'test.json').read_text())
    for key,value in dict(EXPECTED,seed=seed).items():
        assert fit['config'][key]==config[key]==value,(key,fit['config'].get(key),config.get(key))
    assert all(config[key]==value for key,value in fit['config'].items())
    assert config['updates_resolved']==3840000
    assert fit['updates']==max(p['update'] for p in fit['curve'])==3840000
    selected=fit['selected']['update']
    assert marker['selected_update']==selected and 1<=selected<=3840000
    assert any(p['update']==selected for p in fit['curve'])
    from finish_long_bptt import validate_test_record
    validate_test_record(test,fit)
    assert marker['test']==test['test']
    assert all(math.isfinite(x) and x>=0 for x in test['flops'])
    assert sha(folder/'final.json')==marker['final_sha256']
    assert sha(folder/'test.json')==marker['test_sha256']
    assert sha(folder/f'model-u{selected}.pt')==marker['selected_checkpoint_sha256']
    assert sha(folder/'training_methods.py')==marker['training_methods_sha256']
    return fit,test,config


def validate_timing_log(path,fit):
    early={};last=0;selected_s=None;resumes=[]
    with Path(path).open() as stream:
        for line in stream:
            entry=json.loads(line)
            if 'resume' in entry:
                resumes.append(entry['resume']);continue
            if 'update' not in entry: continue
            update=entry['update']
            assert update==last+1,(last,update)
            assert math.isfinite(entry['fit_s']) and entry['fit_s']>=0
            last=update
            if update<=21: early[update]=entry['fit_s']
            if update==fit['selected']['update']:selected_s=entry['fit_s']
    assert last==3840000 and len(early)==21
    assert math.isclose(selected_s,fit['selected']['fit_s'],rel_tol=1e-10,abs_tol=1e-8)
    startup=early[1]-statistics.median(early[k+1]-early[k] for k in range(1,21))
    return dict(startup_s=startup,startup_estimate_later_updates=20,
                timing_log_updates=last,resume_entries=resumes)


def collect_seed(seed):
    host='foureyes' if seed==0 else 'ripper';gpu=3 if seed==0 else seed+1
    relative=f'runs/ALT/longD3/sudoku-bptt-lr2-s{seed}'
    record=dict(id=f'objective-sudoku-bptt-long-s{seed}',family='table7_external_long_bptt',
                lane=f'{host}:{gpu}',host=host,task='sudoku',method='bptt',seed=seed,run_relative=relative,
                status='pending',frozen_choice=dict(lr=2.,knob=10.,l1=.1,updates=3840000),
                externally_managed=True,owner='root')
    marker_path=ROOT/f'artifacts/long_bptt/seed{seed}.json'
    try: marker_raw=remote_bytes(host,marker_path)
    except FileNotFoundError:
        record['reason']='Completion receipt has not been created';return record
    marker=json.loads(marker_raw);marker_hash=hashlib.sha256(marker_raw).hexdigest()
    record.update(completion_receipt_status=marker['status'],completion_receipt_sha256=marker_hash,
                  completion_receipt_host=host,completion_receipt_path=str(marker_path))
    if marker['status']!='complete':
        record.update(status='externally_managed',reason='Waiting for full training and selected-checkpoint test completion')
        return record
    assert marker['host']==host and marker['run']==str(ROOT/'upstream/experiments'/relative)
    folder=OUT/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    cached=folder/'collection.json'
    if cached.exists():
        receipt=json.loads(cached.read_text())
        assert receipt['completion_receipt_sha256']==marker_hash,'Completed source receipt changed; do not silently replace a scored run'
        for name,expected in receipt['files_sha256'].items():
            # The immutable collected timing log can be hundreds of MB. A prior
            # content hash plus unchanged size/mtime avoids rehashing it each minute.
            stat=(folder/name).stat()
            if name=='log.jsonl' and receipt.get('timing_log_stat')==[stat.st_size,stat.st_mtime_ns]:
                continue
            assert sha(folder/name)==expected,name
        fit,test,config=validate_completed(folder,marker,seed)
        timing=receipt['timing']
        collected=receipt['collected_utc']
    else:
        selected=marker['selected_update'];run=Path(marker['run'])
        names=['config.json','final.json','test.json',f'model-u{selected}.pt']
        for name in names:copy_file(host,run/name,folder/name)
        copy_file(host,ROOT/'upstream/experiments/training_methods.py',folder/'training_methods.py')
        (folder/'completion.json').write_bytes(marker_raw)
        fit,test,config=validate_completed(folder,marker,seed)
        # Large log is copied only once, after the small records pass validation.
        copy_file(host,run/'log.jsonl',folder/'log.jsonl')
        timing=validate_timing_log(folder/'log.jsonl',fit)
        assert hashlib.sha256(remote_bytes(host,marker_path)).hexdigest()==marker_hash,'Receipt changed during collection'
        collected=now()
        files=names+['training_methods.py','completion.json','log.jsonl']
        receipt=dict(seed=seed,host=host,collected_utc=collected,completion_receipt_sha256=marker_hash,
                     timing=timing,files_sha256={name:sha(folder/name) for name in files},
                     timing_log_stat=[(folder/'log.jsonl').stat().st_size,(folder/'log.jsonl').stat().st_mtime_ns])
        save(cached,receipt)
    record.update(status='complete',final=fit,result=test,config=config,**timing,
                  execution_path=str(cached.relative_to(ROOT)),collection_directory=str(folder.relative_to(ROOT)),
                  collected_utc=collected,files_sha256=receipt['files_sha256'],
                  validation='Full3.84M schedule,50000 test IDs/settings,selected update,model/source/final/test hashes,and complete timing-log sequence')
    return record


def collect():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'collect.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        jobs=[]
        for seed in range(3):
            try:jobs.append(collect_seed(seed))
            except Exception as exc:
                jobs.append(dict(seed=seed,status='collection_invalid',error=repr(exc)))
        summary=dict(schema_version=1,generated_utc=now(),source_commit=COMMIT,
                     script_sha256=sha(__file__),jobs=jobs,complete=all(j['status']=='complete' for j in jobs),
                     protocol='Root-managed Sudoku BPTT,lr2,knob10,L1.1,3.84M updates,seeds0/1/2;50k held-out test rows10000..59999.',
                     inference_count_note='Recorded legacy inference counts; FC2 training counts supplied separately by objective aggregator.',
                     timing_note='Complete timing log copied and validated once per run; startup estimate follows objective_queue aggregation.')
        save(OUT/'summary.json',summary)
    print(json.dumps(dict(output=str(OUT/'summary.json'),statuses=[j['status'] for j in jobs])),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=float,default=60)
    args=p.parse_args()
    while True:
        result=collect()
        if not args.watch or result['complete']:break
        time.sleep(min(60,max(5,args.interval)))
