"""H3i: fixed dihedral augmentation versus unpooled and displacement counts."""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3i';OUT=ROOT/'artifacts/research/20261114/loki/h3i';UP=ROOT/'upstream'
sys.path[:0]=[str(SRC),str(SRC.parent),str(UP),str(UP/'experiments')]
import study as S


def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):S.save(path,value)
def protocol():return read(SRC/'protocol.json')


def prepare():
    assert not(SRC/'protocol.json').exists();data=read(OUT/'dataset.json')
    assert data['no_model_or_search_outcomes']
    original=(UP/'cram/cram_nqueens.py').read_text()
    original=original[original.index('@torch.no_grad()\ndef solve'):original.index('# ---------------------------------------------------------------------------------------- what the couplings learned')]
    actual=(SRC/'backend.py').read_text();actual=actual[actual.index('@torch.no_grad()\ndef solve'):]
    assert actual==original.replace('return ((other - own >= delta) | ~free).all(1)',"return ((other - own >= delta) | ~free).all(1) & (x.sum(1) == model['q'])")
    sources=[Path(__file__),SRC/'prepare.py',SRC/'backend.py',SRC.parent/'study.py',SRC.parent/'h3g/prepare.py',*sorted((UP/'cram').glob('*.py'))]
    p=dict(id='H3i-dihedral-support-repair',frozen_utc=now(),
        hypothesis='At128 independently sampled training patterns, all eight fixed dihedral views improve certificate-first N20 accuracy by at least10percentage points over unaugmented individual-pair counts, while wrong-stop probability remains at most0.1%.',
        antecedent='H3h found that extra empirical exclusions prevented valid certificates and caused initial false stops. This tests a fixed symmetry prior on fresh inputs and a new raw bank, with a common cardinality gate and otherwise unchanged strict margin.',
        primary=dict(arm='augmented128',control='raw128',minimum_accuracy_gain_pp=10.,maximum_wrong_stop_probability=.001),
        arms=[dict(arm='raw128',indices='raw128_indices',pooling='individual',raw_patterns=128),
              dict(arm='augmented128',indices='augmented128_indices',pooling='individual',raw_patterns=128),
              dict(arm='raw1024',indices='raw1024_indices',pooling='individual',raw_patterns=1024),
              dict(arm='offset128',indices='raw128_indices',pooling='displacement',raw_patterns=128)],
        construction='Infer q from training cardinality. Count each cell pair across the specified positive training patterns. Individual arms set J=q*1[count==0]-1 directly; offset128 pools co-occurrences by displacement before the same rule. Diagonal and unary zero. Augmentation uses every rotation and reflection once, deduplicates complete patterns, and weights unique patterns equally. No puzzle rules select coefficients.',
        calibration='Each unique training pattern has equal weight (equivalent to its298 partial-input masks). Only the current arm bank determines delta, tau and unit.',
        search=dict(n=12,unique_inputs=data['test_unique'],chains=20,cap=20000,seeds=[0,1,2],temperature=1.,stop='local plus q-cardinality gate',margin=.001,
                    gate='Require occupied-cell count == q in addition to the original strict local certificate, for every arm. No row/column/diagonal rule enters search.',
                    operation_count='Original floating operation counts retained; q-gate integer additions reported separately as143*(moves+1).'),
        decision='Primary support requires paired-input bootstrap one-sided95% lower augmented-minus-raw accuracy gain>=10pp AND exact95% upper probability that an input has ANY wrong stopped chain across20chains and3seeds<=0.001. This input-cluster safeguard is conservative for per-chain wrong-stop probability. Falsified if bootstrap upper gain<10pp OR exact95% lower probability of any wrong stop per input>0.001; otherwise inconclusive.2000bootstrap replicates,RNG20261114. No training replication is claimed.',
        controls='raw1024 measures additional independent patterns, offset128 measures displacement pooling; both are descriptive. Raw1024 has1024 unique patterns whereas the deduplicated eight-view arm has the exact count in the reserved dataset. No compute/data equivalence is claimed.',
        validation='Rebuild all literal pair counts; validate every transformed pattern by exhaustive solution membership; verify all test completions absent from union of all training banks; compare dense energies and every single-cell cost to literal sums; test gated backend against unchanged backend on own-training partial inputs and reject a synthetic initial false certificate. Independently audit all returned boards and training-derived gates.',
        dataset=data,budget=dict(new_search_jobs=12,gpu_minutes_envelope=20,optimizer_fits=0),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        data_sha256={str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in ['data.npz','dataset.json']},
        limits=['One new raw bank draw, with three search seeds; no independent training replication.',
                'Fresh inputs exclude H3e/H3g evaluation inputs and all compatible completions in the union of all four training banks; this conditions the distribution.',
                'Dihedral augmentation is an explicit supplied geometry prior and changes the number of unique pattern views.',
                'Allowed-pair reward, inferred q gate and certificate-first selection are derivative design choices; this is not the original paper protocol.',
                'Earlier studies considered the entire mathematical solution universe; fresh means new bank draw and unused partial evaluation inputs, not a never-seen mathematical universe.'])
    save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',protocol_sha256=sha(SRC/'protocol.json'))))


def construct():
    import numpy as np
    import torch
    from cram import cram_nqueens as Q
    torch.set_num_threads(1);data=dict(np.load(OUT/'data.npz'));i,j=np.triu_indices(144,1);receipts=[]
    for cfg in protocol()['arms']:
        arm=cfg['arm'];labels=data['all_solutions'][data[cfg['indices']]];q=int(np.unique(labels.sum(1)).item());assert q==12
        counts=labels.astype(np.int64).T@labels.astype(np.int64)
        if cfg['pooling']=='individual':forbidden=counts[i,j]==0
        else:
            offsets=np.stack([j//12-i//12,j%12-i%12],1);unique,lookup=np.unique(offsets,axis=0,return_inverse=True)
            pooled=np.bincount(lookup,weights=counts[i,j]);forbidden=pooled[lookup]==0
        J=np.zeros((144,144),np.float32);J[i,j]=q*forbidden-1;J[j,i]=J[i,j]
        W=torch.zeros(len(i),2,2);W[:,1,1]=torch.tensor(J[i,j]);B=torch.zeros(144,2);dense=Q.full_S(W,torch.tensor(np.stack([i,j])),144)
        assert np.array_equal(dense[:,:,1,1].numpy(),J)
        fields=labels.astype(np.float64)@J;delta=float(((1-2*labels.astype(int))*fields).min());tau=float((.5*labels*fields).sum(1).max())
        assert delta==11 and tau==-66
        model=dict(W=W,B=B,S=dense,A=W[:,1,1].clone(),b=torch.zeros(1),q=q,delta=delta,tau=tau,unit=float(np.abs(fields).mean()),arm=arm)
        rng=np.random.default_rng(83100+len(labels));fixtures=np.stack([np.zeros(144,bool),np.ones(144,bool),*rng.integers(0,2,(8,144)).astype(bool),*labels[:3]])
        literal=(fixtures[:,i]*fixtures[:,j]*J[i,j]).sum(1,dtype=np.float64)
        assert np.array_equal(Q.board_energy(dense.double(),B.double(),torch.tensor(fixtures,dtype=torch.long)).numpy(),literal)
        for row,base in zip(fixtures,literal):
            for cell in range(144):
                other=row.copy();other[cell]=~other[cell]
                assert (other[i]*other[j]*J[i,j]).sum()-base==(1-2*int(row[cell]))*(row@J[:,cell])
        dest=OUT/f'models/{arm}';dest.mkdir(parents=True,exist_ok=True);torch.save(model,dest/'model.pt')
        np.savez_compressed(dest/'construction.npz',pair_i=i,pair_j=j,raw_cooccurrence=counts[i,j],forbidden=forbidden,pattern_indices=data[cfg['indices']])
        # Only after the training-only model is fixed do rules enter this diagnostic.
        attack=(i//12==j//12)|(i%12==j%12)|(np.abs(i//12-j//12)==np.abs(i%12-j%12))
        rec=dict(status='complete',utc=now(),arm=arm,unique_patterns=len(labels),raw_patterns=cfg['raw_patterns'],pooling=cfg['pooling'],
            q=q,delta=delta,tau=tau,unit=model['unit'],forbidden_pairs=int(forbidden.sum()),false_forbidden=int((forbidden&~attack).sum()),missed_forbidden=int((~forbidden&attack).sum()),
            literal_pair_checks=dict(boards=13,single_cell_changes=1872,exact=True),
            model_sha256=sha(dest/'model.pt'),construction_sha256=sha(dest/'construction.npz'),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'))
        save(dest/'model.json',rec);receipts.append(rec)
    save(OUT/'construction-summary.json',dict(status='complete',models=receipts));print(json.dumps(receipts))


def validate_backend(device):
    import numpy as np
    import torch
    import backend
    from cram import cram_nqueens as Q
    data=dict(np.load(OUT/'data.npz'));model=torch.load(OUT/'models/raw128/model.pt',weights_only=False)
    labels=data['all_solutions'][data['raw128_indices'][:16]];inputs=np.zeros_like(labels)
    for row,label in zip(inputs,labels):row[np.flatnonzero(label)[:3]]=True
    kwargs=dict(chains=3,cap=128,seed=17,device=device,temperature=1.,stop='local',margin=.001)
    start=time.perf_counter();a=Q.solve(model,inputs,**kwargs);b=backend.solve(model,inputs,**kwargs)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    c=144;synthetic=dict(model);synthetic['S']=torch.zeros(c,c,2,2);synthetic['S'][:,:,1,1]=11*(1-torch.eye(c))
    synthetic['unit']=1.;given=np.zeros((1,c),bool);given[0,:3]=True
    kwargs.update(chains=1,cap=1)
    original=Q.solve(synthetic,given,**kwargs);gated=backend.solve(synthetic,given,**kwargs)
    assert original[1].all() and not gated[1].any() and original[3].item()==0 and gated[3].item()==1
    rec=dict(status='passed',utc=now(),device=device,own_training_inputs=16,chains=3,cap=128,
        all_five_backend_outputs_identical=True,synthetic_three_clue_false_certificate_rejected=True,
        seconds=time.perf_counter()-start,backend_sha256=sha(SRC/'backend.py'),reference_sha256=sha(UP/'cram/cram_nqueens.py'),
        source_sha256=sha(__file__),model_sha256=sha(OUT/'models/raw128/model.pt'),protocol_sha256=sha(SRC/'protocol.json'))
    save(OUT/f'backend-validation-{device}.json',rec);print(json.dumps(rec))


def search(cfg,seed,data,progress):
    import numpy as np
    import torch
    import backend
    arm=cfg['arm'];path=OUT/f'models/{arm}/model.pt';model=torch.load(path,weights_only=False);inputs=data['test_inputs']
    progress(stage='search',arm=arm,seed=seed);S.emit('search_start',arm=arm,seed=seed)
    start=time.perf_counter();boards,stopped,energy,moves,flop=backend.solve(model,inputs,chains=20,cap=20000,seed=seed,device='cuda',temperature=1.,stop='local',margin=.001)
    seconds=time.perf_counter()-start;valid=S.independent_ok(boards,inputs,12);denom=np.minimum(data['test_solution_counts'],20)
    found=np.array([min(len({b.tobytes() for b in samples[ok]}),d) for samples,ok,d in zip(boards,valid,denom)])
    original=energy.argmin(1);selected=[]
    for e,s in zip(energy,stopped):
        pool=np.flatnonzero(s)
        if not len(pool):pool=np.arange(20)
        selected.append(int(pool[e[pool].argmin()]))
    selected=np.array(selected);ids=np.arange(len(inputs));n20=valid[ids,original];cert=valid[ids,selected]
    dest=OUT/f'search/{arm}/s{seed}';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'records.npz',inputs=inputs,boards=np.packbits(boards,-1),stopped=stopped,energy=energy,moves=moves,flop=flop,
        cardinality_integer_additions=143*(moves+1),valid=valid,n20=n20,n20_certificate=cert,n1=valid[:,0],found=found,denom=denom,
        solution_counts=data['test_solution_counts'],original_index=original,certificate_index=selected,queen_counts=boards.sum(-1))
    rec=dict(status='complete',utc=now(),arm=arm,seed=seed,unique_inputs=len(inputs),
        original_accuracy_pct=100*float(n20.mean()),certificate_accuracy_pct=100*float(cert.mean()),coverage_pct=100*float(found.sum()/denom.sum()),
        wrong_stops=int((stopped&~valid).sum()),capped_chains=int((~stopped).sum()),mean_moves=float(moves.mean()),search_seconds=seconds,
        model_path=str(path.relative_to(ROOT)),model_sha256=sha(path),records_sha256=sha(dest/'records.npz'),
        protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__),backend_sha256=sha(SRC/'backend.py'))
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
    validation=read(OUT/'backend-validation-cuda.json');assert validation['status']=='passed' and validation['backend_sha256']==sha(SRC/'backend.py')
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
