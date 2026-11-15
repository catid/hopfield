"""H1ao: change only the retention rule on frozen H1an accepted evidence."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PARENT=ROOT/'artifacts/research/20261114/ripper/balanced_filter'
OUT=ROOT/'artifacts/research/20261114/ripper/positive_witness'
PROTOCOL=HERE/'protocol-positive-witness.json'
PUBLIC=ROOT/'results/research/ripper'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def verify():
    p=json.loads(PROTOCOL.read_text())
    for path,digest in (p['code_sha256']|p['input_sha256']).items():assert sha(ROOT/path)==digest,path
    return p

def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    code=[HERE/'positive_witness_study.py',HERE/'audit_positive_witness.py']
    inputs=[HERE/'protocol-balanced-filter.json',PARENT/'audit.json',PARENT/'input_manifest.json',PARENT/'all_source_manifest.json']
    manifest=json.loads((PARENT/'input_manifest.json').read_text())
    for b in range(32):
        d=PARENT/f's{b}'/'filtered';inputs += [d/n for n in ['fit.npz','observations.npz','receipt.json','audit.json']]
        entry=next(e for e in manifest['inputs']if e['seed']==b);inputs.append(ROOT/entry['training_path'])
    p=dict(id='H1ao-positive-equality-witness-retention',frozen_utc=datetime.now(timezone.utc).isoformat(),banks=32,
        scope='Post-hoc mechanism study on the same32 H1an filtered acquired streams, accepted rows, masks and labels. No adaptive trajectory changes, new queries, source rows or solver inputs.',
        supplied_assumptions='The H1an collapse law plus supplied multiplicity prior proves every accepted label clean. This is explicitly conditional on those assumptions; not a general noisy-label rule.',
        new_rule='Retain pair iff its accepted coqueries never observed equal digits. Pairs with zero coqueries remain retained. No peer geometry used in retention.',
        control='Archived H1an likelihood-ratio support, unchanged. Ordinary BP AdamW outside this source-only study; not run.',
        empirical_primary='At least29/32 positive-witness graphs exactly equal true Sudoku peer graph; supports iff exact_count>=29, otherwise falsifies. No independent-edge sampling interval.',
        theorem_checks='Accepted labels are clean under stated law; a true peer pair cannot have an equality witness. Therefore no true edge can be removed. Check separately from exact-recovery target.',
        audit='Independently reconstruct all3240 joint/equal counts from all1110 observed masks/labels per bank, updating only accepted rows. Compare archived counts; reconstruct both witness and retained LR supports; check zero-coquery retention and complete frozen artifact hashes.',
        budget=dict(aggregate_cpu_seconds=120,wall_seconds=60,cpu_workers=1,gpu_seconds=0),
        stopping_rule='Exactly32 fixed archived streams; retain every bank and outcome. No trajectory or acquisition reruns.',
        code_sha256={str(f.relative_to(ROOT)):sha(f)for f in code},input_sha256={str(f.relative_to(ROOT)):sha(f)for f in inputs})
    save(PROTOCOL,p);print(json.dumps({'protocol_sha256':sha(PROTOCOL)}))

def compute():
    import numpy as np
    p=verify();assert not OUT.exists();OUT.mkdir(parents=True);started=time.monotonic()
    models=[]
    for b in range(32):
        path=PARENT/f's{b}'/'filtered/fit.npz'
        with np.load(path,allow_pickle=False)as z:
            baseline=z['support'];joint=z['joint_count'];equal=z['equal_count'];witness=equal==0
        dest=OUT/f's{b}.npz';np.savez_compressed(dest,baseline_support=baseline,witness_support=witness,joint_count=joint,equal_count=equal)
        models.append(dict(bank=b,path=str(dest.relative_to(ROOT)),sha256=sha(dest)))
        save(OUT/'progress.json',dict(stage='constructing',completed_banks=b+1,elapsed_s=time.monotonic()-started))
    save(OUT/'mapping.json',dict(protocol_sha256=sha(PROTOCOL),models=models))
    from audit_positive_witness import audit
    audit()

def launch():
    p=verify();log=OUT.parent/'positive_witness-execution.log';assert not log.exists()
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    started=time.monotonic();samples=[];ticks=os.sysconf('SC_CLK_TCK')
    with log.open('x')as handle:
        process=subprocess.Popen([sys.executable,str(Path(__file__)), 'compute'],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        try:
            while process.poll()is None:
                elapsed=time.monotonic()-started
                try:
                    fields=Path(f'/proc/{process.pid}/stat').read_text().split();cpu=(int(fields[13])+int(fields[14]))/ticks
                except FileNotFoundError:cpu=0
                samples.append(dict(wall_s=elapsed,cpu_s=cpu,pid=process.pid))
                assert elapsed<60 and cpu<120
                time.sleep(.1)
            assert process.returncode==0,log.read_text()
        finally:
            if process.poll()is None:process.terminate();process.wait(timeout=5)
    import resource
    usage=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=usage.ru_utime+usage.ru_stime
    runtime=dict(status='validated',wall_s=time.monotonic()-started,aggregate_cpu_s=cpu,wall_ceiling_s=60,aggregate_cpu_ceiling_s=120,cpu_workers=1,gpu_seconds=0,samples=samples,wrapper_sha256=sha(__file__))
    assert runtime['wall_s']<60 and cpu<120
    result=json.loads((OUT/'audit.json').read_text());result['runtime']=runtime
    save(OUT/'runtime.json',runtime);save(PUBLIC/'positive_witness_runtime.json',runtime)
    save(OUT/'audit.json',result);save(PUBLIC/'positive_witness.json',result)
    with (HERE/'REPORT-positive-witness.md').open('a')as f:f.write(f'\nExecution used {cpu:.4f} CPU-seconds and {runtime["wall_s"]:.4f} wall seconds, within the frozen 120 CPU-second / 60-second ceilings. No GPU work.\n')
    print(json.dumps(dict(primary=result['primary'],grouped=result['grouped'],runtime={k:v for k,v in runtime.items()if k!='samples'})))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','launch','compute'])
    globals()[parser.parse_args().action]()
