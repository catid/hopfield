"""H3m: vary source board-size breadth without pruning finite zero offsets."""
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

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3m';OUT=ROOT/'artifacts/research/20261114/loki/h3m';UP=ROOT/'upstream'
sys.path[:0]=[str(SRC.parent/'h3i'),str(SRC.parent),str(UP),str(UP/'experiments')]
import study as S


def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):S.save(path,value)
def protocol():return read(SRC/'protocol.json')


def prepare():
    assert not(SRC/'protocol.json').exists();parent=read(SRC.parent/'h3l/protocol.json');dataset=read(OUT/'dataset.json')
    sources=[Path(__file__),SRC/'prepare.py',SRC/'enumerate.py',SRC/'reuse_identities.py',SRC.parent/'h3i/backend.py',SRC.parent/'h3l/enumerate.py',SRC.parent/'study.py',*sorted((UP/'cram').glob('*.py'))]
    p=dict(id='H3m-multisize-superqueen-source',frozen_utc=now(),
        hypothesis='Adding all44 N11 superqueen patterns to the same128 N12 patterns raises N13 certificate-first N20 accuracy by at least10percentage points over N12-only hybrid inference, with zero wrong stopped chains.',
        antecedent='H3l exposed source-size global zero offsets. This test adds positive examples from another source size without selecting a locality cutoff or removing exceptions by target geometry.',
        line_family=parent['line_family'],
        arms=[dict(arm='n12-only',n12_patterns=128,n11_patterns=0),dict(arm='multisize',n12_patterns=128,n11_patterns=44),
              dict(arm='n12-full',n12_patterns=156,n11_patterns=0),dict(arm='true-graph-oracle',n12_patterns=0,n11_patterns=0)],
        construction='For each learned arm sum positive displacement co-occurrences across its declared source patterns, using each boards own coordinates/size. The observed offset domain is the union of source domains (N12 for every arm). Apply the unchanged primitive-line rule, then union extrapolated lines with EVERY observed-zero finite source offset. No radius restriction or graph-based exception pruning. J=13*forbidden-1, diagonal/unary0. Final oracle explicitly supplies target queen+knight attacks.',
        source_controls='N12-only uses exactly H3l128 patterns. Mixed source uses those128 plus all44 N11 patterns (172total). Full-N12 uses all156 patterns. This changes data breadth and count; the descriptive full-N12 control adds28 labels versus44 for mixed source, so total labels are not perfectly matched.',
        calibration=parent['calibration'],search=parent['search'],
        primary=dict(arm='multisize',control='n12-only',minimum_accuracy_gain_pp=10.,required_wrong_stops=0),
        decision='Support if paired-input bootstrap one-sided95% lower mixed-minus-N12-only accuracy gain>=10pp AND zero mixed-source wrong stopped chains. Falsify if upper gain<10pp OR any mixed-source wrong stop; otherwise inconclusive.2000bootstrap replicates,RNG20261114; all3searchseeds remain clustered within each input. Full-N12 and oracle comparisons/coverage are descriptive.',
        diagnostics='After construction identify which N12-zero offsets acquire positive N11 support; all remaining zeros stay in the model. Audit all target graph errors and valid-certificate capacity, with target truth used only diagnostically.',
        identity=parent['identity'],dataset=dataset,budget=dict(logical_model_seed_jobs=12,maximum_unique_search_jobs=12,gpu_minutes_envelope=20,optimizer_fits=0),
        validation='Independent N11 enumeration, source-bank identity to H3l, literal per-size co-occurrences and summed evidence, line decisions, retained finite offsets, target coefficient/flip checks, exact solver identity grouping. Exhaustive N13 superqueen membership and givens for all outputs.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        data_sha256={str(path.relative_to(ROOT)):sha(path) for path in [OUT/'training.npz',OUT/'data.npz',OUT/'dataset.json',OUT/'feasibility.json',SRC.parent/'h3l/protocol.json',ROOT/'artifacts/research/20261114/loki/h3l/models/hybrid/model.pt']},
        limits=['Source data breadth and count change together; full-N12 is a descriptive extra-label control with a slightly smaller total than mixed source.',
                'No source-size interpolation theorem or arbitrary-rule learning claim; geometry/line family/target q remain supplied priors.',
                'One fixed N12 bank plus complete finite source universes; three seeds repeat search, not training.',
                'All remaining source-zero offsets are retained. Target labels do not change the rule, coefficients, thresholds or temperature.',
                'Uniform unique partial-input sampling excludes prior cohorts and differs from original GRAM multiplicities.'])
    save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',protocol_sha256=sha(SRC/'protocol.json'))))


def construct():
    import numpy as np
    import torch
    from cram import cram_nqueens as Q
    torch.set_num_threads(1);p=protocol();train=dict(np.load(OUT/'training.npz'))
    offsets=np.array([(r,c) for r in range(12) for c in range(-11,12) if r>0 or c>0]);lookup={tuple(o):k for k,o in enumerate(offsets)}
    def counts(labels,n):
        i,j=np.triu_indices(n*n,1);co=labels.astype(np.int64).T@labels.astype(np.int64)
        index=np.array([lookup[b//n-a//n,b%n-a%n] for a,b in zip(i,j)])
        return np.bincount(index,weights=co[i,j],minlength=len(offsets)).astype(np.int64)
    blocks=dict(n11_all=counts(train['n11_all'],11),n12_base=counts(train['n12_base'],12),n12_all=counts(train['n12_all'],12))
    evidence={'n12-only':blocks['n12_base'],'multisize':blocks['n12_base']+blocks['n11_all'],'n12-full':blocks['n12_all']}
    candidates=[(a,b) for a in range(-2,3) for b in range(-2,3) if (a,b)!=(0,0) and math.gcd(abs(a),abs(b))==1 and (a>0 or (a==0 and b>0))]
    fits={}
    for arm,co in evidence.items():
        decisions=[];lines=[]
        for a,b in candidates:
            covered=a*offsets[:,0]+b*offsets[:,1]==0;nf=int((covered&(co==0)).sum());na=int((covered&(co>0)).sum());keep=nf>=3 and na==0
            decisions.append(dict(a=a,b=b,forbidden_offsets=nf,allowed_offsets=na,retained=keep))
            if keep:lines.append((a,b))
        fits[arm]=dict(retained_lines=lines,candidates=decisions,source_zero_offsets=int((co==0).sum()))
    save(OUT/'source-fits.json',dict(status='fitted_source_only',utc=now(),arms=fits,source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json')))
    np.savez_compressed(OUT/'source-support.npz',offsets=offsets,**blocks,**evidence)
    i,j=np.triu_indices(169,1);dr=j//13-i//13;dc=j%13-i%13;groups={};receipts=[]
    for cfg in p['arms']:
        arm=cfg['arm']
        if arm=='true-graph-oracle':mask=(dr==0)|(dc==0)|(np.abs(dr)==np.abs(dc))|((dr==1)&(np.abs(dc)==2))|((dr==2)&(np.abs(dc)==1))
        else:
            support={tuple(o):bool(v==0) for o,v in zip(offsets,evidence[arm])};lines=fits[arm]['retained_lines']
            mask=np.array([support.get((r,c),False) or any(a*r+b*c==0 for a,b in lines) for r,c in zip(dr,dc)])
        J=np.zeros((169,169),np.float32);J[i,j]=13*mask-1;J[j,i]=J[i,j]
        W=torch.zeros(len(i),2,2);W[:,1,1]=torch.tensor(J[i,j]);B=torch.zeros(169,2);dense=Q.full_S(W,torch.tensor(np.stack([i,j])),169)
        assert np.array_equal(dense[:,:,1,1].numpy(),J)
        model=dict(W=W,B=B,S=dense,A=W[:,1,1].clone(),b=torch.zeros(1),q=13,delta=12.,tau=-78.,unit=p['calibration']['unit'],arm=arm)
        signature=hashlib.sha256(J.astype(np.int16).tobytes()+json.dumps(p['calibration'],sort_keys=True).encode()).hexdigest()
        dest=OUT/f'models/{arm}';dest.mkdir(parents=True,exist_ok=True);torch.save(model,dest/'model.pt')
        np.savez_compressed(dest/'construction.npz',pair_i=i,pair_j=j,forbidden=mask)
        rec=dict(status='complete',utc=now(),arm=arm,target_training_patterns=0,source_n11_patterns=cfg['n11_patterns'],source_n12_patterns=cfg['n12_patterns'],
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
