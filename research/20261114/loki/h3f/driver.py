"""Training-only exclusion energy and an explicit compatible-training-bank control."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/'research/20261114/loki/h3f'
OUT=ROOT/'artifacts/research/20261114/loki/h3f'
PREVIOUS=ROOT/'artifacts/research/20261114/loki/h3e'
UP=ROOT/'upstream'
sys.path[:0]=[str(SRC.parent),str(UP),str(UP/'experiments')]
import study as S


def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):S.save(path,value)
def protocol():return read(SRC/'protocol.json')
def emit(event,**values):S.emit(event,**values)


def prepare():
    assert not(SRC/'protocol.json').exists()
    parent=read(SRC.parent/'h3e/protocol.json')
    cache=[]
    for arm,updates in [('adamw-decay',4000),('adam-nodecay',250),('adam-nodecay',1000),('adam-nodecay',4000),('sgd-pre',4000)]:
        for seed in [0,1,2]:
            path=PREVIOUS/f'search/{arm}/u{updates}/s{seed}/result.json';rec=read(path)
            cache.append(dict(arm=arm,updates=updates,seed=seed,path=str(path.relative_to(ROOT)),sha256=sha(path),
                records_sha256=rec['records_sha256'],model_path=rec['model_path'],model_sha256=rec['model_sha256']))
    p=dict(id='H3f-exclusion-control',frozen_utc=now(),
        hypothesis='A no-optimizer energy based on training-only never-observed offsets, J=q*forbidden-1 with unary0, is noninferior to learned Adam250 in certificate-first N20 accuracy within0.1pp and distinct-completion coverage within2pp.',
        antecedent='H3e confirmed the fixed early-Adam recipe on n12. This adaptive control tests whether training support statistics and a fixed energy design suffice; its outcomes were not inspected before freeze.',
        construction='Take only fit complete labels with positive expanded-row multiplicity. Infer q from their common queen count (require a single value). Count weighted queen-queen co-occurrences per offset using the existing offset architecture. An offset is forbidden iff its count is zero. Set every J_d=q*forbidden_d-1 and unary0. No row/column/diagonal attack rule or held-out label enters construction.',
        added_prior='Rewarding every observed/allowed pair by-1 and penalizing every never-observed pair byq-1 is an explicitly added fixed energy design prior. It is not learned-weight recovery or evidence that support inference alone chooses these coefficients.',
        calibration='As in the learned models: tau=max training solution energy, delta=min training solution single-cell margin, unit=mean absolute field weighted by the exact fit expanded-row multiplicities. Require strictly positive training minima margins. No settings selected from test outcomes.',
        geometry='The fixed offset parameterization shares pairs by 2D displacement. This supplied geometry remains present; attack categories are used only in a separately labeled post hoc support audit.',
        primary=dict(arm='exclusion',control='adam-nodecay',control_updates=250,accuracy_margin_pp=.1,coverage_margin_pp=2.,selection='certificate-first with exact original fallback'),
        search=parent['search'],decision=parent['decision'],
        bank=dict(source='The unique complete boards with positive fit multiplicity, packed bitwise; count and report overlap with the full completion universe.',
            inference='For each held-out partial input, filter training-bank boards that contain every given queen, then draw20 independent uniform samples WITH replacement from those compatible boards. No puzzle rule or held-out solution label is queried.',
            seeds=[0,1,2],rng='NumPy default_rng(seed), shared stream across canonical input order',
            purpose='Transparent memorized-solution/template control, not an optimizer or a solution-pattern-held-out test.',
            timing='CPU bank construction and per-seed matching+sampling recorded separately; no CPU-vs-GPU speedup claim.'),
        audit='Recheck every saved board and chosen identity. Report wrong stops, including invalid occupancies belowq, equalq and aboveq; distinguish all returned/capped/stopped strata. Recompute stopped-chain learned margins in float64. Preserve original-selection scores. Support recovery compared with rule-derived offsets only after construction.',
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),SRC.parent/'study.py',*sorted((UP/'cram').glob('*.py')),*sorted((UP/'experiments').glob('*.py'))]},
        data_sha256={str((PREVIOUS/name).relative_to(ROOT)):sha(PREVIOUS/name) for name in ['data.npz','dataset.json']},
        cached_comparisons=cache,parent_protocol_sha256=sha(SRC.parent/'h3e/protocol.json'),
        budget=dict(new_gpu_search_jobs=3,chains_per_job=163840,cap=20000,gpu_minutes_envelope=10,new_optimizer_fits=0),
        limits=['Adaptive reuse of H3e held-out inputs; not fresh confirmation.',
                'Input groups are held out but all14200complete solution patterns may occur in training; measured overlap is reported prominently.',
                'Fixed energy-design and offset-sharing priors remain explicit even though there is no optimizer.',
                'Bank control stores full complete training boards; its success is retrieval and does not imply learned constraint generalization.',
                'Three repeated search or sampling seeds per fixed construction, not independent training samples.'])
    OUT.mkdir(parents=True,exist_ok=True);save(SRC/'protocol.json',p);emit('protocol_frozen',sha256=sha(SRC/'protocol.json'))


def construct():
    import numpy as np
    import torch
    import nqueens_offset as O
    from cram import cram_nqueens as Q
    p=protocol()
    for path,digest in p['data_sha256'].items():assert sha(ROOT/path)==digest
    data=dict(np.load(PREVIOUS/'data.npz'));start=time.perf_counter()
    keep=data['rows_per_solution']>0;labels=data['solutions'][keep];multiplicity=data['rows_per_solution'][keep]
    sizes=np.unique(labels.sum(1));assert len(sizes)==1;q=int(sizes[0])
    n=int(round(labels.shape[1]**.5));assert n*n==labels.shape[1]
    x=labels.astype(np.float64);pair_counts=x.T@(x*multiplicity[:,None])
    idx,offsets=O.offsets(n,'cpu');i,j=Q.pair_index(n*n,'cpu')
    counts=np.bincount(idx.numpy(),weights=pair_counts[i.numpy(),j.numpy()])
    assert np.array_equal(counts,counts.astype(np.int64))
    forbidden=counts==0;A=torch.tensor(q*forbidden.astype(np.float32)-1);b=torch.zeros(1)
    W,B=O.pair_tables(A,b,'ising',n);energy=Q.full_S(W,Q.pair_index(n*n,'cpu'),n*n)
    J=energy[:,:,1,1].numpy().astype(np.float64);fields=x@J
    margins=(1-2*x)*fields;delta=float(margins.min());assert delta>0
    energies=.5*(x*fields).sum(1);assert np.all(energies==-q*(q-1)/2)
    unit=float((np.abs(fields).mean(1)*multiplicity).sum()/multiplicity.sum());tau=float(energies.max())
    model=dict(A=A,b=b,W=W,B=B,S=energy,tau=tau,unit=unit,delta=delta,arm='ising')
    torch.save(model,OUT/'model.pt')
    np.savez_compressed(OUT/'training-support.npz',offsets=offsets.numpy(),counts=counts.astype(np.int64),forbidden=forbidden,
        positive_label_indices=np.flatnonzero(keep),multiplicity=multiplicity,training_margin_minima=margins.min(1))
    bank=np.packbits(labels,axis=1);np.save(OUT/'training-bank.npy',bank)
    construction_seconds=time.perf_counter()-start
    # Geometry is consulted only here, after every model coefficient has been set.
    off=offsets.numpy();reference=(off[:,0]==0)|(off[:,1]==0)|(np.abs(off[:,0])==np.abs(off[:,1]))
    report=dict(status='complete',utc=now(),q_from_training=q,training_solution_patterns=len(labels),
        full_completion_universe=len(data['solutions']),fit_solution_overlap_fraction=len(labels)/len(data['solutions']),
        forbidden_offsets=int(forbidden.sum()),allowed_offsets=int((~forbidden).sum()),
        posthoc_support_audit=dict(exact_recovery=bool(np.array_equal(forbidden,reference)),false_forbidden=int((forbidden&~reference).sum()),missed_forbidden=int((~forbidden&reference).sum()),
            scope='Only a diagnostic after energy construction; attack categories never enter constructor coefficients or search.'),
        unit=unit,tau=tau,delta=delta,all_training_energies=float(energies[0]),minimum_training_margin=delta,
        bank_packed_bytes=int(bank.nbytes),construction_seconds=construction_seconds,
        model_sha256=sha(OUT/'model.pt'),bank_sha256=sha(OUT/'training-bank.npy'),support_sha256=sha(OUT/'training-support.npz'),
        protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__))
    save(OUT/'construction.json',report);emit('construction_complete',**{key:value for key,value in report.items() if key!='utc'})


def select(energy,stopped):
    import numpy as np
    original=energy.argmin(1);selected=np.where(stopped.any(1),np.where(stopped,energy,np.inf).argmin(1),original)
    return original,selected


def score(arm,seed,inputs,boards,stopped,energy,moves,flop,seconds,data):
    import numpy as np
    valid=S.independent_ok(boards,inputs,12);denom=np.minimum(data['test_solution_counts'],20)
    found=np.array([min(len({b.tobytes() for b in sample[ok]}),d) for sample,ok,d in zip(boards,valid,denom)])
    original,selected=select(energy,stopped);ids=np.arange(len(inputs));n20=valid[ids,original];certificate=valid[ids,selected]
    queens=boards.sum(-1);wrong_stopped=stopped&~valid
    dest=OUT/f'{arm}/s{seed}';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'records.npz',inputs=inputs,boards=np.packbits(boards,-1),stopped=stopped,energy=energy,
        moves=moves,flop=flop,valid=valid,n20=n20,n20_certificate=certificate,n1=valid[:,0],found=found,denom=denom,
        queen_counts=queens,original_index=original,certificate_index=selected,solution_counts=data['test_solution_counts'])
    rec=dict(status='complete',utc=now(),arm=arm,seed=seed,unique_inputs=len(inputs),original_accuracy_pct=100*float(n20.mean()),
        certificate_accuracy_pct=100*float(certificate.mean()),coverage_pct=100*float(found.sum()/denom.sum()),
        wrong_stops=int(wrong_stopped.sum()) if arm=='exclusion' else None,
        wrong_stops_by_occupancy={name:int((wrong_stopped&mask).sum()) for name,mask in [('below_q',queens<12),('equal_q',queens==12),('above_q',queens>12)]} if arm=='exclusion' else None,
        capped_chains=int((~stopped).sum()) if arm=='exclusion' else None,mean_moves=float(moves.mean()) if arm=='exclusion' else None,
        search_seconds=seconds,records_sha256=sha(dest/'records.npz'),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__),
        model_sha256=sha(OUT/'model.pt') if arm=='exclusion' else sha(OUT/'training-bank.npy'),
        flag_scope='For bank outputs stopped is a storage marker for returned candidates, not a learned-certificate claim.' if arm=='training-bank' else 'Original learned-margin stopping flag.')
    save(dest/'result.json',rec);emit('evaluation_complete',**{k:rec[k] for k in ['arm','seed','original_accuracy_pct','certificate_accuracy_pct','coverage_pct','search_seconds']})


def bank():
    import numpy as np
    data=dict(np.load(PREVIOUS/'data.npz'));inputs=data['test_inputs']
    labels=np.unpackbits(np.load(OUT/'training-bank.npy'),axis=1)[:,:144].astype(bool)
    for seed in [0,1,2]:
        if (OUT/f'training-bank/s{seed}/result.json').exists():continue
        rng=np.random.default_rng(seed);start=time.perf_counter();boards=[];counts=[]
        for givens in inputs:
            compatible=np.flatnonzero(labels[:,np.flatnonzero(givens)].all(1));assert len(compatible)
            counts.append(len(compatible));boards.append(labels[rng.choice(compatible,size=20,replace=True)])
        boards=np.asarray(boards);seconds=time.perf_counter()-start
        # Compatibility counts are compared with evaluation metadata only after sampling completes.
        assert np.array_equal(counts,data['test_solution_counts'])
        shape=boards.shape[:2];score('training-bank',seed,inputs,boards,np.ones(shape,bool),np.zeros(shape),np.zeros(shape,np.int64),np.zeros(shape,np.int64),seconds,data)


def worker(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    query=lambda *args:subprocess.check_output(args,text=True,timeout=20).strip()
    assert query('hostname')=='loki'
    device=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader')
    assert device.split(',')[1].strip() not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    import numpy as np
    import torch
    from cram import cram_nqueens as Q
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    p=protocol()
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    c=read(OUT/'construction.json');assert c['status']=='complete' and c['model_sha256']==sha(OUT/'model.pt')
    model=torch.load(OUT/'model.pt',weights_only=False);data=dict(np.load(PREVIOUS/'data.npz'));inputs=data['test_inputs']
    base=dict(status='running',started_utc=now(),host='loki',gpu=gpu,device=device,pid=os.getpid(),protocol_sha256=sha(SRC/'protocol.json'))
    receipt=OUT/f'worker-g{gpu}.json';save(receipt,base)
    try:
        for seed in [0,1,2]:
            dest=OUT/f'exclusion/s{seed}';dest.mkdir(parents=True,exist_ok=True)
            if (dest/'result.json').exists():continue
            claim=(dest/'job.lock').open('a')
            try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:claim.close();continue
            try:
                if (dest/'result.json').exists():continue
                save(receipt,dict(base,updated_utc=now(),progress=dict(stage='search',seed=seed,inputs=len(inputs))))
                emit('search_start',seed=seed,gpu=gpu);start=time.perf_counter()
                values=Q.solve(model,inputs,chains=20,cap=20000,seed=seed,device='cuda',temperature=1.,stop='local',margin=.001)
                seconds=time.perf_counter()-start;score('exclusion',seed,inputs,*values,seconds,data)
            finally:claim.close()
        save(receipt,dict(base,status='complete',completed_utc=now()))
    except BaseException as exc:save(receipt,dict(base,status='failed',completed_utc=now(),error=repr(exc)));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='mode',required=True)
    for mode in ['prepare','construct','bank']:sp.add_parser(mode)
    w=sp.add_parser('worker');w.add_argument('--gpu',type=int,choices=[0,1],required=True)
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='prepare':prepare()
    elif args.mode=='construct':construct()
    elif args.mode=='bank':bank()
    else:worker(args.gpu)
