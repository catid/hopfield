"""H3j: N12-only primitive-line fitting, then zero-shot N13 count-energy search."""
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

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3j';OUT=ROOT/'artifacts/research/20261114/loki/h3j';UP=ROOT/'upstream'
TRAIN=ROOT/'artifacts/research/20261114/loki/h3i'
sys.path[:0]=[str(SRC.parent/'h3i'),str(SRC.parent),str(UP),str(UP/'experiments')]
import study as S


def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):S.save(path,value)
def protocol():return read(SRC/'protocol.json')


def prepare():
    assert not(SRC/'protocol.json').exists();dataset=read(OUT/'dataset.json')
    unit=read(TRAIN/'models/offset128/model.json')['unit']
    sources=[Path(__file__),SRC/'prepare.py',SRC.parent/'h3i/backend.py',SRC.parent/'study.py',*sorted((UP/'cram').glob('*.py'))]
    p=dict(id='H3j-zero-shot-N13-line-extrapolation',frozen_utc=now(),
        hypothesis='A primitive homogeneous-line hypothesis fitted only to128 N12 positive patterns improves N13 certificate-first N20 accuracy by at least10percentage points over direct offset-table transfer, with zero wrong stopped chains.',
        antecedent='H3h/H3i showed strong effects of geometric parameter sharing. This fresh-size study declares a small geometric hypothesis family before target search outcomes.',
        training='Only frozen H3i raw128 N12 complete patterns. Reconstruct the264 unordered displacement co-occurrence counts. No N13 example enters line selection, coefficients, thresholds or temperature.',
        line_family='Enumerate integer(a,b) in[-2,2]^2 except(0,0), gcd(abs(a),abs(b))=1, canonical sign(first nonzero positive). A line is a*dr+b*dc=0. Retain iff it contains>=3 observed forbidden N12 offsets and0 observed allowed N12 offsets. Retain all qualifying lines; no subset or orientation selection on target.',
        arms=[dict(arm='line-extrapolation'),dict(arm='direct-offset'),dict(arm='true-graph-oracle')],
        construction='Target q=13 is supplied explicitly. J=q*forbidden-1 for each unordered N13 cell pair, diagonal/unary0. Extrapolation applies the retained homogeneous line union to every target offset. Direct transfer uses the N12 zero-count support where an offset was observed, treating unseen offsets as allowed. The oracle arm uses the explicitly supplied N13 row/column/diagonal attack graph.',
        calibration=dict(q=13,delta=12.,tau=-78.,unit=unit,
            explanation='q and its symbolic delta=q-1,tau=-q(q-1)/2 are declared structural priors. Every target arm shares the inherited N12 offset128 mean-absolute-field unit. No target fit or label calibration.',
            unit_source='artifacts/research/20261114/loki/h3i/models/offset128/model.json'),
        search=dict(n=13,unique_inputs=4096,chains=20,cap=20000,seeds=[0,1,2],temperature=1.,stop='strict local margin plus q-cardinality gate',margin=.001,
                    floating_operations='57461+859*moves',cardinality_integer_additions='168*(moves+1)'),
        primary=dict(arm='line-extrapolation',control='direct-offset',minimum_accuracy_gain_pp=10.,required_wrong_stops=0),
        decision='Support if paired-input bootstrap one-sided95% lower extrapolation-minus-direct N20 gain>=10pp and zero wrong stopped chains in extrapolation. Falsify if upper gain<10pp or any extrapolation wrong stop; otherwise inconclusive.2000bootstrap replicates,RNG20261114; all3searchseeds cluster within input. Report exact95% upper probability of any wrong stop per input.',
        descriptive='Extrapolation versus oracle accuracy margin0.1pp and coverage margin2pp are descriptive, with paired bootstrap intervals and no independent confirmatory claim. Report original selection, N1, coverage, wrong stops, capped chains and exact certificate existence.',
        dataset=dataset,budget=dict(new_search_jobs=9,gpu_minutes_envelope=20,optimizer_fits=0),
        validation='Independent literal N12 co-occurrence and candidate-line audit; exact dense energies and all single-cell costs against literal target-pair sums on synthetic boards; compare common gated backend with unchanged solver on synthetic N13 fixtures on CPU/GPU. Exhaustive N13 solution validity and sample counts are evaluation-only. Graph diagnostics occur after each model construction.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        data_sha256={str(path.relative_to(ROOT)):sha(path) for path in [OUT/'data.npz',OUT/'dataset.json',TRAIN/'data.npz',TRAIN/'models/offset128/model.json',TRAIN/'models/offset128/model.pt']},
        limits=['The finite primitive-line family, coordinate displacements and target cardinality are explicit priors, not learned from nothing.',
                'The oracle reference contains supplied target rules and is not a learned baseline.',
                'One frozen N12 training bank; search repeats do not establish training-sample uncertainty.',
                'N13 complete patterns cannot occur in N12 training by dimension, but inference still uses the declared geometric prior.',
                'Uniform unique valid1/2/3clue sampling differs from original GRAM input multiplicities; no claim of original paper reproduction.'])
    save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',protocol_sha256=sha(SRC/'protocol.json'))))


def construct():
    import numpy as np
    import torch
    from cram import cram_nqueens as Q
    torch.set_num_threads(1);p=protocol()
    # Construction reads N12 data only. Target data is never loaded here.
    d=dict(np.load(TRAIN/'data.npz'));labels=d['all_solutions'][d['raw128_indices']]
    si,sj=np.triu_indices(144,1);co=labels.astype(np.int64).T@labels.astype(np.int64)
    offsets=np.stack([sj//12-si//12,sj%12-si%12],1);unique,lookup=np.unique(offsets,axis=0,return_inverse=True)
    counts=np.bincount(lookup,weights=co[si,sj]).astype(np.int64);source_support={tuple(off):bool(count==0) for off,count in zip(unique,counts)}
    candidates=[];retained=[]
    for a in range(-2,3):
        for b in range(-2,3):
            if (a==b==0) or math.gcd(abs(a),abs(b))!=1 or (a<0 or (a==0 and b<0)):continue
            covered=a*unique[:,0]+b*unique[:,1]==0
            nforbid=int((covered&(counts==0)).sum());nallow=int((covered&(counts>0)).sum());keep=nforbid>=3 and nallow==0
            candidates.append(dict(a=a,b=b,forbidden_offsets=nforbid,allowed_offsets=nallow,retained=keep))
            if keep:retained.append((a,b))
    save(OUT/'line-fit.json',dict(status='fitted_from_N12_only',candidates=candidates,retained_lines=retained,
        training_patterns=128,source_offset_terms=len(unique),source_forbidden_offsets=int((counts==0).sum()),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json')))
    np.savez_compressed(OUT/'source-support.npz',offsets=unique,counts=counts,training_indices=d['raw128_indices'])
    i,j=np.triu_indices(169,1);dr=j//13-i//13;dc=j%13-i%13;receipts=[]
    for cfg in p['arms']:
        arm=cfg['arm']
        if arm=='line-extrapolation':mask=np.array([any(a*r+b*c==0 for a,b in retained) for r,c in zip(dr,dc)])
        elif arm=='direct-offset':mask=np.array([source_support.get((r,c),False) for r,c in zip(dr,dc)])
        else:mask=(dr==0)|(dc==0)|(np.abs(dr)==np.abs(dc))
        J=np.zeros((169,169),np.float32);J[i,j]=13*mask-1;J[j,i]=J[i,j]
        W=torch.zeros(len(i),2,2);W[:,1,1]=torch.tensor(J[i,j]);B=torch.zeros(169,2);dense=Q.full_S(W,torch.tensor(np.stack([i,j])),169)
        assert np.array_equal(dense[:,:,1,1].numpy(),J)
        model=dict(W=W,B=B,S=dense,A=W[:,1,1].clone(),b=torch.zeros(1),q=13,delta=12.,tau=-78.,unit=p['calibration']['unit'],arm=arm)
        rng=np.random.default_rng(83113);fixtures=np.stack([np.zeros(169,bool),np.ones(169,bool),*rng.integers(0,2,(8,169)).astype(bool),*np.eye(169,dtype=bool)[:3]])
        literal=(fixtures[:,i]*fixtures[:,j]*J[i,j]).sum(1,dtype=np.float64)
        assert np.array_equal(Q.board_energy(dense.double(),B.double(),torch.tensor(fixtures,dtype=torch.long)).numpy(),literal)
        for row,base in zip(fixtures,literal):
            for cell in range(169):
                other=row.copy();other[cell]=~other[cell]
                assert (other[i]*other[j]*J[i,j]).sum()-base==(1-2*int(row[cell]))*(row@J[:,cell])
        dest=OUT/f'models/{arm}';dest.mkdir(parents=True,exist_ok=True);torch.save(model,dest/'model.pt')
        np.savez_compressed(dest/'construction.npz',pair_i=i,pair_j=j,forbidden=mask)
        # Supplied rules are model construction only for the explicitly named oracle.
        attack=(dr==0)|(dc==0)|(np.abs(dr)==np.abs(dc))
        rec=dict(status='complete',utc=now(),arm=arm,q=13,delta=12.,tau=-78.,unit=model['unit'],target_training_patterns=0,
            forbidden_pairs=int(mask.sum()),false_forbidden=int((mask&~attack).sum()),missed_forbidden=int((~mask&attack).sum()),
            literal_pair_checks=dict(boards=13,single_cell_changes=2197,exact=True),
            model_sha256=sha(dest/'model.pt'),construction_sha256=sha(dest/'construction.npz'),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'))
        save(dest/'model.json',rec);receipts.append(rec)
    save(OUT/'construction-summary.json',dict(status='complete',models=receipts));print(json.dumps(dict(retained_lines=retained,models=receipts)))


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
    seconds=time.perf_counter()-start;valid=S.independent_ok(boards,inputs,13);denom=np.minimum(data['test_solution_counts'],20)
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
