"""H3l: fixed line-family boundary on queen-plus-knight constraints."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3l';OUT=ROOT/'artifacts/research/20261114/loki/h3l';UP=ROOT/'upstream'
sys.path[:0]=[str(SRC.parent/'h3i'),str(SRC.parent),str(UP),str(UP/'experiments')]
import study as S


def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):S.save(path,value)
def protocol():return read(SRC/'protocol.json')


def prepare():
    assert not(SRC/'protocol.json').exists();parent=read(SRC.parent/'h3j/protocol.json');dataset=read(OUT/'dataset.json')
    sources=[Path(__file__),SRC/'prepare.py',SRC/'enumerate.py',SRC/'reuse_identities.py',SRC.parent/'h3i/backend.py',SRC.parent/'study.py',*sorted((UP/'cram').glob('*.py'))]
    p=dict(id='H3l-superqueen-prior-boundary',frozen_utc=now(),
        hypothesis='On N13 superqueens, the union of N12-fitted homogeneous lines and every observed-zero finite N12 displacement is noninferior to a supplied target-graph oracle within0.1percentage point certificate-first N20 accuracy, with zero wrong stopped chains.',
        antecedent='The fixed line family succeeded on ordinary queens. Superqueens add finite knight offsets that need not form forbidden homogeneous lines; source-size global impossibility may also create zero offsets.',
        line_family=parent['line_family'],
        training='Exactly128 of156 independently enumerated N12 superqueen patterns, fixed permutation seed2026112501. Only training.npz enters construction; N13 target data remain separate.',
        arms=[dict(arm='line-only'),dict(arm='hybrid'),dict(arm='direct-offset'),dict(arm='true-graph-oracle')],
        construction='Fit the unchanged line rule from source positive co-occurrences. Line-only extrapolates retained lines; hybrid unions those lines with EVERY observed-zero finite source offset; direct-offset uses source zero offsets only and treats unseen offsets allowed. No false exclusion is manually removed. All use J=13*forbidden-1, zero unary/diagonal. The final oracle arm explicitly supplies target queen and knight attacks after the three data-derived models are fixed.',
        calibration=parent['calibration'],search=parent['search'],
        primary=dict(arm='hybrid',control='true-graph-oracle',accuracy_noninferiority_margin_pp=.1,required_wrong_stops=0),
        decision='Support if exact one-sided95% upper probability of an input having ANY paired accuracy regression(oracle correct,hybrid wrong) across3seeds is<=0.001 AND hybrid has zero wrong stopped chains. This conservatively certifies mean accuracy NI within0.1pp without degenerate paired-bootstrap precision. Falsify if paired-input bootstrap one-sided95% upper hybrid-minus-oracle accuracy difference<-.1pp OR any hybrid wrong stop; otherwise inconclusive.2000bootstrap replicates,RNG20261114. Report both selectors and coverage descriptively.',
        diagnostics='Only after construction compare inferred target graphs with queen+knight rules, and reconstruct ALL156 source-pattern co-occurrences to distinguish sample absence from source-size global zero support. Preserve every extra offset. Target labels may diagnose valid certificate capacity but never edit models.',
        identity='Deduplicate only exact S,B,q,delta,tau,unit equality, before target search, keeping all four logical arms and explicit matched-seed mappings. First arm in frozen order is representative.',
        dataset=dataset,budget=dict(logical_model_seed_jobs=12,maximum_unique_search_jobs=12,gpu_minutes_envelope=20,optimizer_fits=0),
        validation='Two independently matched exhaustive superqueen enumerations; literal source pair counts and line decisions; full target coefficient and synthetic single-cell checks; exact-gate backend CPU/GPU checks. Independently validate every returned board against the complete1876-solution N13 superqueen set and givens.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        data_sha256={str(path.relative_to(ROOT)):sha(path) for path in [OUT/'training.npz',OUT/'data.npz',OUT/'dataset.json',OUT/'feasibility.json',SRC.parent/'h3j/protocol.json']},
        limits=['This tests the boundary of a declared geometric hypothesis family, not arbitrary-rule learning.',
                'Line-only target failure can reflect family mismatch, while finite zero counts can reflect source-size global impossibility; neither is automatically an implementation failure.',
                'All extra source zero offsets remain in hybrid/direct models. No target-label repair or hyperparameter tuning.',
                'One128-pattern source bank with3 repeated search seeds, not multiple independent training replications.',
                'Target q=13, inherited ordinary-queen N12 temperature unit and strict stopping choices are supplied priors; no superqueen target calibration.',
                'The oracle has explicitly supplied target rules; it is not learned. Unique-input sampling differs from original GRAM.'])
    save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',protocol_sha256=sha(SRC/'protocol.json'))))


def super_valid(boards,inputs):
    import numpy as np
    valid=S.independent_ok(boards,inputs,13);i,j=np.triu_indices(169,1);dr=np.abs(j//13-i//13);dc=np.abs(j%13-i%13)
    knight=((dr==1)&(dc==2))|((dr==2)&(dc==1))
    return valid&~(boards[...,i[knight]]&boards[...,j[knight]]).any(-1)


def construct():
    import numpy as np
    import torch
    from cram import cram_nqueens as Q
    torch.set_num_threads(1);p=protocol();train=dict(np.load(OUT/'training.npz'));labels=train['training_labels']
    i,j=np.triu_indices(144,1);co=labels.astype(np.int64).T@labels.astype(np.int64)
    offsets=np.stack([j//12-i//12,j%12-i%12],1);unique,lookup=np.unique(offsets,axis=0,return_inverse=True)
    counts=np.bincount(lookup,weights=co[i,j]).astype(np.int64);support={tuple(o):bool(c==0) for o,c in zip(unique,counts)}
    candidates=[];retained=[]
    for a in range(-2,3):
        for b in range(-2,3):
            if (a==b==0) or math.gcd(abs(a),abs(b))!=1 or a<0 or (a==0 and b<0):continue
            covered=a*unique[:,0]+b*unique[:,1]==0;nf=int((covered&(counts==0)).sum());na=int((covered&(counts>0)).sum());keep=nf>=3 and na==0
            candidates.append(dict(a=a,b=b,forbidden_offsets=nf,allowed_offsets=na,retained=keep))
            if keep:retained.append((a,b))
    save(OUT/'line-fit.json',dict(status='fitted_from_N12_superqueen_bank_only',utc=now(),candidates=candidates,retained_lines=retained,
        source_forbidden_offsets=int((counts==0).sum()),training_patterns=128,source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json')))
    np.savez_compressed(OUT/'source-support.npz',offsets=unique,counts=counts,training_indices=train['training_indices'])
    i,j=np.triu_indices(169,1);dr=j//13-i//13;dc=j%13-i%13
    line=np.array([any(a*r+b*c==0 for a,b in retained) for r,c in zip(dr,dc)])
    finite=np.array([support.get((r,c),False) for r,c in zip(dr,dc)]);receipts=[];groups={}
    for cfg in p['arms']:
        arm=cfg['arm']
        if arm=='line-only':mask=line
        elif arm=='hybrid':mask=line|finite
        elif arm=='direct-offset':mask=finite
        else:mask=(dr==0)|(dc==0)|(np.abs(dr)==np.abs(dc))|((dr==1)&(np.abs(dc)==2))|((dr==2)&(np.abs(dc)==1))
        J=np.zeros((169,169),np.float32);J[i,j]=13*mask-1;J[j,i]=J[i,j]
        W=torch.zeros(len(i),2,2);W[:,1,1]=torch.tensor(J[i,j]);B=torch.zeros(169,2);dense=Q.full_S(W,torch.tensor(np.stack([i,j])),169)
        assert np.array_equal(dense[:,:,1,1].numpy(),J)
        model=dict(W=W,B=B,S=dense,A=W[:,1,1].clone(),b=torch.zeros(1),q=13,delta=12.,tau=-78.,unit=p['calibration']['unit'],arm=arm)
        signature=hashlib.sha256(J.astype(np.int16).tobytes()+json.dumps(p['calibration'],sort_keys=True).encode()).hexdigest()
        dest=OUT/f'models/{arm}';dest.mkdir(parents=True,exist_ok=True);torch.save(model,dest/'model.pt')
        np.savez_compressed(dest/'construction.npz',pair_i=i,pair_j=j,forbidden=mask)
        rec=dict(status='complete',utc=now(),arm=arm,target_training_patterns=0,q=13,delta=12.,tau=-78.,unit=model['unit'],energy_signature=signature,
            forbidden_pairs=int(mask.sum()),model_sha256=sha(dest/'model.pt'),construction_sha256=sha(dest/'construction.npz'),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'))
        save(dest/'model.json',rec);receipts.append(rec)
        if signature not in groups:groups[signature]=dict(signature=signature,representative=arm,members=[])
        groups[signature]['members'].append(arm)
    save(OUT/'energy-groups.json',dict(status='grouped_before_target_inference',utc=now(),groups=list(groups.values()),logical_models=4,
        distinct_energies=len(groups),actual_unique_search_jobs=3*len(groups),avoided_identical_jobs=12-3*len(groups)))
    save(OUT/'construction-summary.json',dict(status='complete',models=receipts));print(json.dumps(dict(retained_lines=retained,source_forbidden_offsets=int((counts==0).sum()),distinct_target_energies=len(groups))))


def validate_backend(device):
    import numpy as np
    import torch
    import backend
    from cram import cram_nqueens as Q
    # Synthetic Ising fixtures only; no N13 target solution enters validation.
    c=169;model=dict(S=torch.zeros(c,c,2,2),B=torch.zeros(c,2),q=13,delta=12.,tau=-78.,unit=protocol()['calibration']['unit'])
    rng=np.random.default_rng(83114);inputs=np.zeros((16,c),bool)
    for k,row in enumerate(inputs):row[rng.choice(c,2+k%3,replace=False)]=True
    kwargs=dict(chains=3,cap=64,seed=17,device=device,temperature=1.,stop='local',margin=.001)
    start=time.perf_counter();a=Q.solve(model,inputs,**kwargs);b=backend.solve(model,inputs,**kwargs)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    model['S'][:,:,1,1]=12*(1-torch.eye(c));given=np.zeros((1,c),bool);given[0,:3]=True;kwargs.update(chains=1,cap=1)
    original=Q.solve(model,given,**kwargs);gated=backend.solve(model,given,**kwargs)
    assert original[1].all() and not gated[1].any() and original[3].item()==0 and gated[3].item()==1
    given[0,:13]=True;original=Q.solve(model,given,**kwargs);gated=backend.solve(model,given,**kwargs)
    assert all(np.array_equal(x,y) for x,y in zip(original,gated)) and gated[1].all()
    rec=dict(status='passed',utc=now(),device=device,synthetic_inputs=16,chains=3,cap=64,all_five_backend_outputs_identical=True,
        synthetic_three_clue_false_certificate_rejected=True,synthetic_q_gate_acceptance_checked=True,seconds=time.perf_counter()-start,
        backend_sha256=sha(SRC.parent/'h3i/backend.py'),reference_sha256=sha(UP/'cram/cram_nqueens.py'),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'))
    save(OUT/f'backend-validation-{device}.json',rec);print(json.dumps(rec))


def search(cfg,seed,data,progress):
    import numpy as np
    import torch
    import backend
    arm=cfg['arm'];path=OUT/f'models/{arm}/model.pt';model=torch.load(path,weights_only=False);inputs=data['test_inputs']
    progress(stage='search',arm=arm,seed=seed);S.emit('search_start',arm=arm,seed=seed)
    start=time.perf_counter();boards,stopped,energy,moves,flop=backend.solve(model,inputs,chains=20,cap=20000,seed=seed,device='cuda',temperature=1.,stop='local',margin=.001)
    seconds=time.perf_counter()-start;valid=super_valid(boards,inputs);denom=np.minimum(data['test_solution_counts'],20)
    found=np.array([min(len({b.tobytes() for b in samples[ok]}),d) for samples,ok,d in zip(boards,valid,denom)])
    original=energy.argmin(1);selected=[]
    for e,s in zip(energy,stopped):
        pool=np.flatnonzero(s)
        if not len(pool):pool=np.arange(20)
        selected.append(int(pool[e[pool].argmin()]))
    selected=np.array(selected);ids=np.arange(len(inputs));n20=valid[ids,original];cert=valid[ids,selected]
    dest=OUT/f'search/{arm}/s{seed}';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'records.npz',inputs=inputs,boards=np.packbits(boards,-1),stopped=stopped,energy=energy,moves=moves,flop=flop,
        cardinality_integer_additions=168*(moves+1),valid=valid,n20=n20,n20_certificate=cert,n1=valid[:,0],found=found,denom=denom,
        solution_counts=data['test_solution_counts'],original_index=original,certificate_index=selected,queen_counts=boards.sum(-1))
    rec=dict(status='complete',utc=now(),arm=arm,seed=seed,unique_inputs=len(inputs),
        original_accuracy_pct=100*float(n20.mean()),certificate_accuracy_pct=100*float(cert.mean()),coverage_pct=100*float(found.sum()/denom.sum()),
        wrong_stops=int((stopped&~valid).sum()),capped_chains=int((~stopped).sum()),mean_moves=float(moves.mean()),search_seconds=seconds,
        model_path=str(path.relative_to(ROOT)),model_sha256=sha(path),records_sha256=sha(dest/'records.npz'),
        protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__),backend_sha256=sha(SRC.parent/'h3i/backend.py'))
    save(dest/'result.json',rec);S.emit('search_complete',**{k:rec[k] for k in ['arm','seed','certificate_accuracy_pct','coverage_pct','wrong_stops','search_seconds']})


def worker(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    query=lambda *args:subprocess.check_output(args,text=True,timeout=20).strip()
    assert query('hostname')=='loki';device=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader')
    assert device.split(',')[1].strip() not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;p=protocol()
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    validation=read(OUT/'backend-validation-cuda.json');assert validation['status']=='passed' and validation['backend_sha256']==sha(SRC.parent/'h3i/backend.py')
    for cfg in p['arms']:
        dest=OUT/f"models/{cfg['arm']}";rec=read(dest/'model.json');assert rec['model_sha256']==sha(dest/'model.pt') and rec['source_sha256']==sha(__file__)
    data=dict(np.load(OUT/'data.npz'));started=time.monotonic()
    base=dict(status='running',started_utc=now(),gpu=gpu,pid=os.getpid(),host='loki',device=device,protocol_sha256=sha(SRC/'protocol.json'))
    receipt=OUT/f'worker-g{gpu}.json'
    def progress(**values):save(receipt,dict(base,updated_utc=now(),progress=values))
    progress(stage='starting')
    try:
        for cfg in p['arms']:
            for seed in [0,1,2]:
                dest=OUT/f"search/{cfg['arm']}/s{seed}";dest.mkdir(parents=True,exist_ok=True)
                if (dest/'result.json').exists():continue
                claim=(dest/'job.lock').open('a')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                try:
                    if time.monotonic()-started>600:raise RuntimeError('20GPU-minute campaign envelope reached before newjob; partial evidence retained')
                    if not(dest/'result.json').exists():search(cfg,seed,data,progress)
                finally:claim.close()
        save(receipt,dict(base,status='complete',completed_utc=now()))
    except BaseException as exc:save(receipt,dict(base,status='failed',completed_utc=now(),error=repr(exc)));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='mode',required=True)
    sp.add_parser('prepare');sp.add_parser('construct');v=sp.add_parser('validate');v.add_argument('--device',choices=['cpu','cuda'],required=True)
    w=sp.add_parser('worker');w.add_argument('--gpu',type=int,choices=[0,1],required=True)
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':prepare()
    elif a.mode=='construct':construct()
    elif a.mode=='validate':validate_backend(a.device)
    else:worker(a.gpu)
