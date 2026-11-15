"""H4y: fresh-bank and fresh-input confirmation of fixed magnitude weighting."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
RIPPER=HERE.parent/'ripper'
OUT=ROOT/'artifacts/research/20261114/foureyes/canonical_confirm'
PUBLIC=ROOT/'results/research/foureyes/canonical_confirm'
PROTOCOL=HERE/'canonical_confirm.json'
ARMS=['adamw','uniform','magnitude','restricted']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def verify(p):
    for name,digest in (p['source_sha256']|p['frozen_inputs']).items():assert sha(ROOT/name)==digest,name


def base():
    import canonical_study as module
    module.OUT=OUT;module.PUBLIC=PUBLIC;module.PROTOCOL=PROTOCOL
    return module


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    p=json.loads((HERE/'canonical_study.json').read_text())
    p.update(id='H4y-fresh-magnitude-confirmation',frozen_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Magnitude accuracy>=60percent AND magnitude-uniform gain>=50pp ANDzero magnitude invalidcertificates.',
        decision='Each endpoint usesits marginal95two-way interval; supported ifbothlower bounds meettargets andzero magnitudewrongstops; falsified ifanyupper misses target orany magnitudewrongstop; otherwiseinconclusive.',
        scope='Fresh64sourcebanks and5000reserved official puzzles; fixedH4x recipe, no new tuning. Same officialcorpus, not externaldistribution.',
        arms=ARMS,primary_arm='magnitude',control_arm='uniform',primary_margin_pp=50,
        accuracy_floor_pct=60,secondary='BPAdamW32 and restrictedcounts are separatelyfitted descriptivecontrols; no noninferiorityclaim to them.',
        bank_rng_base=202611340000,noise_rng_base=202611350000,augmentation_rng_base=202611360000,
        raw_penalties={'canonical_weak':1/30000},rows_half_open=[410000,415000],
        chains_by_arm={a:4 for a in ARMS},
        rule='Magnitude usesJ>0support; threshold median(positiveJ)/4,weight1 ifJ>=threshold else1/16. Otherarms unitweights ontheir own supports. No geometry enters fitting orassignment.',
        uncertainty='5000paired two-waybank/inputdraws; gainseed20261116021, absoluteseed20261116022; marginal95, no simultaneouscoverageclaim.',
        source_design='Unchanged H4x threshold, weights and λ chosenbefore freshbanks; no banks filtered by structuraloutcomes.',
        budget=dict(fit_phase_wall_seconds=180,search_phase_wall_seconds=360,total_gpu_minutes_upper_bound=36,max_fit_bank_seconds=30),
        limits=['64 independentlysampled sourcebanks conditionalonsameeligiblecorpus; withinbank no replacement, crossbank overlap allowed.',
                '5000shared newpuzzles andone fixed searchseed; dedup identicalcompleteweights/settings.',
                'No claims of tunedoptimizer ormatchedtraining compute; no freshfamily/distributionclaim.'])
    for file in [Path(__file__),HERE/'audit_canonical_confirm.py',HERE/'canonical_weights.py',RIPPER/'confidence_weight_search.py',RIPPER/'confidence_weight_common.py',RIPPER/'confidence_integer_audit.py']:
        p['source_sha256'][str(file.relative_to(ROOT))]=sha(file)
    p['frozen_inputs'][str((HERE/'canonical_weights.json').relative_to(ROOT))]=sha(HERE/'canonical_weights.json')
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    p=json.loads(PROTOCOL.read_text());verify(p);base().base_module().prepare(p)


def map_models():
    import numpy as np
    p=json.loads(PROTOCOL.read_text());verify(p);audit=json.loads((OUT/'source-audit.json').read_text());assert audit['status']=='validated'
    rows=[];jobs=[];known={};thresholds=[]
    for bank in range(64):
        path=OUT/'fits'/f'b{bank:02d}'/'models.npz'
        assert sha(path)==audit['input_sha256'][str(path.relative_to(ROOT))]
        with np.load(path,allow_pickle=False)as z:
            j=z['canonical_weak_coefficients'];positive=j>0;threshold=float(np.median(j[positive])/4)
            weights={'adamw':16*z['adamw_support'],'uniform':16*positive,'restricted':16*z['restricted_support'],
                'magnitude':np.where(positive,np.where(j>=threshold,16,1),0)}
        thresholds.append(threshold)
        for arm in ARMS:
            w=weights[arm].astype(np.uint8);key=w.tobytes();job=f'b{bank:02d}-{arm}'
            path=OUT/'models'/f'{job}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,support=w>0,weight16=w)
            if key not in known:
                known[key]=job;jobs.append(dict(job_id=job,seed=0,n=600,arm=arm,status='pending',model=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))))
            rows.append(dict(bank=bank,arm=arm,job=known[key],model_path=str(path.relative_to(ROOT)),model_sha256=sha(path),weight16_sha256=hashlib.sha256(w.tobytes()).hexdigest()))
    assert not(OUT/'queue.json').exists();save(OUT/'mapping.json',dict(protocol_sha256=sha(PROTOCOL),rows=rows,thresholds=thresholds));save(OUT/'queue.json',jobs)
    print(json.dumps(dict(logical_cells=len(rows),unique_searches=len(jobs))))


def search(gpu):
    # Reuse the checked H4x worker with a new frozen protocol and output root.
    import canonical_weights as worker
    worker.OUT=OUT;worker.PROTOCOL=PROTOCOL
    worker.verify=verify
    worker.worker(gpu)


def launch(phase):
    p=json.loads(PROTOCOL.read_text());verify(p);start=time.monotonic();workers=[];logs=[];samples=[]
    try:
        for gpu in range(4):
            log=(OUT/f'{phase}-gpu{gpu}.log').open('x');logs.append(log)
            workers.append(subprocess.Popen([str(ROOT/'upstream/.venv/bin/python'),str(Path(__file__)),phase,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(w.poll()is None for w in workers):
            assert time.monotonic()-start<p['budget'][phase+'_phase_wall_seconds'] and all(w.poll()in[None,0]for w in workers)
            q=json.loads((OUT/('fit_queue.json'if phase=='fit'else'queue.json')).read_text())
            samples.append(dict(elapsed=time.monotonic()-start,complete=sum(r['status']=='complete'for r in q),running=sum(r['status']=='running'for r in q),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)))
            time.sleep(2)
        assert all(w.returncode==0 for w in workers)
        result=dict(status='complete',wall_seconds=time.monotonic()-start,samples=samples,returncodes=[w.returncode for w in workers]);save(OUT/(phase+'-launch.json'),result);print(json.dumps({k:v for k,v in result.items()if k!='samples'}))
    finally:
        for w in workers:
            if w.poll()is None:
                os.killpg(w.pid,signal.SIGTERM)
                try:w.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
        for log in logs:log.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','fit','map','search','launch-fit','launch-search']);ap.add_argument('--gpu',type=int);a=ap.parse_args()
    if a.action=='fit':base().fit_worker(a.gpu)
    elif a.action=='search':search(a.gpu)
    elif a.action=='map':map_models()
    elif a.action.startswith('launch-'):launch(a.action.split('-')[1])
    else:globals()[a.action]()
