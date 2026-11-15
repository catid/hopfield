"""H3n: downward transfer from larger superqueen source boards."""
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

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3n';OUT=ROOT/'artifacts/research/20261114/loki/h3n';UP=ROOT/'upstream'
sys.path[:0]=[str(SRC.parent/'h3i'),str(SRC.parent),str(UP),str(UP/'experiments')]
import study as S


def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):S.save(path,value)
def protocol():return read(SRC/'protocol.json')


def prepare():
    assert not(SRC/'protocol.json').exists();parent=read(SRC.parent/'h3m/protocol.json');dataset=read(OUT/'dataset.json')
    sources=[Path(__file__),SRC/'prepare.py',SRC/'enumerate.py',SRC/'reuse_identities.py',SRC.parent/'h3i/backend.py',SRC.parent/'h3l/enumerate.py',SRC.parent/'study.py',*sorted((UP/'cram').glob('*.py'))]
    p=dict(id='H3n-larger-source-downward-transfer',frozen_utc=now(),
        hypothesis='The hybrid inferred from all5180 N14 superqueen patterns is noninferior to the supplied N13 target-graph oracle within0.1percentage points in certificate-first N20 accuracy, with zero wrong stopped chains.',
        antecedent='H3m smaller-source data changed no base-zero offsets. This separately tests larger-source evidence on a smaller target; it is downward transfer, not solving a larger target board.',
        line_family='Enumerate integer(a,b) in[-2,2]^2 except(0,0), gcd(abs(a),abs(b))=1, canonical sign(first nonzero positive). A line is a*dr+b*dc=0. Retain iff it contains>=3 zero-count offsets in that arms source domain and0 positive-count offsets. Retain all qualifying lines; no target selection.',
        arms=[dict(arm='n12-full',source_n=12,source_patterns=156,archive='n12_all'),dict(arm='n14-matched156',source_n=14,source_patterns=156,archive='n14_matched'),
              dict(arm='n14-full',source_n=14,source_patterns=5180,archive='n14_all'),dict(arm='true-graph-oracle',source_n=None,source_patterns=0,archive=None)],
        construction='For each learned arm sum displacement co-occurrences over the declared source patterns, using its own source board size. Apply the unchanged primitive-line rule within that source domain, then union extrapolated lines with EVERY observed-zero finite source offset. Offsets outside that arms source domain are unobserved, not invented zeros. No radius cutoff or target-based exception pruning. Target J=13*forbidden-1, diagonal/unary0. Final oracle explicitly supplies N13 queen+knight attacks.',
        source_controls=dataset['source_controls'],calibration=parent['calibration'],search=parent['search'],
        primary=dict(arm='n14-full',control='true-graph-oracle',noninferiority_margin_pp=.1,required_wrong_stops=0),
        decision='Support if the exact one-sided95% upper probability of any paired search-seed accuracy regression within an input is<=0.001 and zero full-N14 wrong stopped chains. Falsify if paired-input bootstrap one-sided95% upper full-N14-minus-oracle accuracy difference<-0.1pp OR any full-N14 wrong stop; otherwise inconclusive.2000bootstrap draws,RNG20261114,all3searchseeds clustered within input. This conservative regression safeguard also ensures accuracy noninferiority. Matched156/source-size comparisons and coverage are descriptive.',
        diagnostics='After construction audit source zero offsets, target graph errors and valid-certificate capacity. No target diagnostic changes a model.',
        identity=parent['identity'],dataset=dataset,budget=dict(logical_model_seed_jobs=12,maximum_unique_search_jobs=12,gpu_minutes_envelope=20,optimizer_fits=0),
        validation='Two exhaustive N14 enumerations, literal pairwise validity, matched-source selection, literal source displacement counts within each domain, inferred lines, all finite zeros, target coefficients/energies/flips, exact solver identity groups; exhaustive N13 superqueen membership/givens for outputs.',
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},
        data_sha256={str(path.relative_to(ROOT)):sha(path) for path in [OUT/'training.npz',OUT/'data.npz',OUT/'dataset.json',OUT/'feasibility.json',SRC.parent/'h3m/protocol.json',ROOT/'artifacts/research/20261114/loki/h3m/models/n12-full/model.pt']},
        limits=['N14-to-N13 is downward transfer; no larger-target generalization claim.',
            'N12-full and N14-matched156 share label count but change source size. Full-N14 adds5024 labels; its advantage cannot isolate source size from count.',
            'One matched N14 subset, not independent source draws; three seeds repeat search only.',
            'The line family, displacement pooling, all-zero rule, target q and inherited unit are supplied priors.',
            'Every source-zero offset is retained; no target-label model repair or temperature calibration.',
            'Uniform unique partial-input sampling differs from original GRAM multiplicities.'])
    save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',protocol_sha256=sha(SRC/'protocol.json'))))


def construct():
    import numpy as np
    import torch
    from cram import cram_nqueens as Q
    torch.set_num_threads(1);p=protocol();train=dict(np.load(OUT/'training.npz'))
    fits={};supports={};saved={}
    candidates=[(a,b) for a in range(-2,3) for b in range(-2,3) if (a,b)!=(0,0) and math.gcd(abs(a),abs(b))==1 and (a>0 or (a==0 and b>0))]
    for cfg in p['arms']:
        if cfg['archive'] is None:continue
        arm=cfg['arm'];n=cfg['source_n'];labels=train[cfg['archive']];assert len(labels)==cfg['source_patterns']
        offsets=np.array([(r,c) for r in range(n) for c in range(1-n,n) if r>0 or c>0]);lookup={tuple(o):k for k,o in enumerate(offsets)}
        i,j=np.triu_indices(n*n,1);co=labels.astype(np.int64).T@labels.astype(np.int64)
        index=np.array([lookup[b//n-a//n,b%n-a%n] for a,b in zip(i,j)])
        counts=np.bincount(index,weights=co[i,j],minlength=len(offsets)).astype(np.int64)
        supports[arm]={tuple(o):bool(v==0) for o,v in zip(offsets,counts)};decisions=[];lines=[]
        saved[f'{arm}_offsets']=offsets;saved[f'{arm}_counts']=counts
        for a,b in candidates:
            covered=a*offsets[:,0]+b*offsets[:,1]==0;nf=int((covered&(counts==0)).sum());na=int((covered&(counts>0)).sum());keep=nf>=3 and na==0
            decisions.append(dict(a=a,b=b,forbidden_offsets=nf,allowed_offsets=na,retained=keep))
            if keep:lines.append((a,b))
        fits[arm]=dict(source_n=n,source_patterns=len(labels),retained_lines=lines,candidates=decisions,source_zero_offsets=int((counts==0).sum()))
    save(OUT/'source-fits.json',dict(status='fitted_source_only',utc=now(),arms=fits,source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json')))
    np.savez_compressed(OUT/'source-support.npz',**saved)
    i,j=np.triu_indices(169,1);dr=j//13-i//13;dc=j%13-i%13;groups={};receipts=[]
    for cfg in p['arms']:
        arm=cfg['arm']
        if arm=='true-graph-oracle':mask=(dr==0)|(dc==0)|(np.abs(dr)==np.abs(dc))|((dr==1)&(np.abs(dc)==2))|((dr==2)&(np.abs(dc)==1))
        else:
            support=supports[arm];lines=fits[arm]['retained_lines']
            mask=np.array([support.get((r,c),False) or any(a*r+b*c==0 for a,b in lines) for r,c in zip(dr,dc)])
        J=np.zeros((169,169),np.float32);J[i,j]=13*mask-1;J[j,i]=J[i,j]
        W=torch.zeros(len(i),2,2);W[:,1,1]=torch.tensor(J[i,j]);B=torch.zeros(169,2);dense=Q.full_S(W,torch.tensor(np.stack([i,j])),169)
        assert np.array_equal(dense[:,:,1,1].numpy(),J)
        model=dict(W=W,B=B,S=dense,A=W[:,1,1].clone(),b=torch.zeros(1),q=13,delta=12.,tau=-78.,unit=p['calibration']['unit'],arm=arm)
        signature=hashlib.sha256(J.astype(np.int16).tobytes()+json.dumps(p['calibration'],sort_keys=True).encode()).hexdigest()
        dest=OUT/f'models/{arm}';dest.mkdir(parents=True,exist_ok=True);torch.save(model,dest/'model.pt')
        np.savez_compressed(dest/'construction.npz',pair_i=i,pair_j=j,forbidden=mask)
        rec=dict(status='complete',utc=now(),arm=arm,target_training_patterns=0,source_n=cfg['source_n'],source_patterns=cfg['source_patterns'],
            q=13,delta=12.,tau=-78.,unit=model['unit'],energy_signature=signature,forbidden_pairs=int(mask.sum()),
            model_sha256=sha(dest/'model.pt'),construction_sha256=sha(dest/'construction.npz'),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'))
        save(dest/'model.json',rec);receipts.append(rec)
        if signature not in groups:groups[signature]=dict(signature=signature,representative=arm,members=[])
        groups[signature]['members'].append(arm)
    save(OUT/'energy-groups.json',dict(status='grouped_before_target_inference',utc=now(),groups=list(groups.values()),logical_models=4,
        distinct_energies=len(groups),actual_unique_search_jobs=3*len(groups),avoided_identical_jobs=12-3*len(groups)))
    save(OUT/'construction-summary.json',dict(status='complete',models=receipts));print(json.dumps(dict(source_fits=fits,distinct_energies=len(groups))))


def super_valid(boards,inputs):
    import numpy as np
    valid=S.independent_ok(boards,inputs,13);i,j=np.triu_indices(169,1);dr=np.abs(j//13-i//13);dc=np.abs(j%13-i%13)
    knight=((dr==1)&(dc==2))|((dr==2)&(dc==1))
    return valid&~(boards[...,i[knight]]&boards[...,j[knight]]).any(-1)


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
