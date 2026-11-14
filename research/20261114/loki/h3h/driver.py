"""Geometry ablation: unpooled cell-pair counts versus cached offset counts."""
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

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3h';OUT=ROOT/'artifacts/research/20261114/loki/h3h';BASE=ROOT/'artifacts/research/20261114/loki/h3g';UP=ROOT/'upstream'
sys.path[:0]=[str(SRC.parent),str(UP),str(UP/'experiments')]
import study as S
spec=importlib.util.spec_from_file_location('pattern_worker',SRC.parent/'h3g/driver.py');G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)


def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):S.save(path,value)
def protocol():return read(SRC/'protocol.json')


def prepare():
    assert not(SRC/'protocol.json').exists()
    parent=read(SRC.parent/'h3g/protocol.json');cache=[]
    for bank in [32,128,512]:
        for seed in [0,1,2]:
            path=BASE/f'search/b{bank}/exclusion/s{seed}/result.json';rec=read(path)
            cache.append(dict(bank=bank,seed=seed,path=str(path.relative_to(ROOT)),sha256=sha(path),records_sha256=rec['records_sha256'],model_path=rec['model_path'],model_sha256=rec['model_sha256']))
    sources=[Path(__file__),SRC.parent/'h3g/driver.py',SRC.parent/'h3e/driver.py',SRC.parent/'study.py',*sorted((UP/'cram').glob('*.py')),*sorted((UP/'experiments').glob('*.py'))]
    p=dict(id='H3h-offset-sharing-ablation',frozen_utc=now(),
        hypothesis='With32training solution patterns, pooling co-occurrence counts by displacement improves certificate-first N20 accuracy by at least10percentage points relative to counting each individual cell pair.',
        antecedent='H3g found strong transfer of the offset-count energy to bank-incompatible inputs. This follow-up isolates the supplied offset-sharing prior; no unpooled-pair inference was inspected before freeze.',
        primary=dict(bank=32,intervention='cached offset-count energy',control='new unpooled cell-pair energy',minimum_accuracy_gain_pp=10.),
        construction='For each nested bank, use only its fit patterns with positive weights. Count each individual unordered cell pair separately, without sharing, translation or rotation. Infer q from training cardinality. J_ij=q*1[cooccurrence_ij==0]-1 fori!=j, diagonal0, unary0. Use same coefficient rule and training-only unit/tau/delta as H3g.',
        calibration='Only current bank patterns, each appearing298times in expanded fitting rows; no held-out complete pattern enters model or calibration.',
        representation='Store the full individual-pair table and dense symmetricS. The backend only evaluates this table; it must not map it through an offset index. Exact binary energies and every one-cell change are checked against literal unordered-pair sums.',
        search=parent['search'],dataset=parent['dataset'],cached_offset_controls=cache,
        decision='Support primary if one-sided95% paired-input bootstrap LOWER offset-minus-pair accuracy difference>=10pp; falsified if upper<10pp; otherwise inconclusive. 2000resamples ofuniqueinputs keeping3searchseeds clustered, RNG20261114.128/512banks andcoveragearesecondary.',
        diagnostics='Before inference, save every initial clamped-state local certificate decision, with2/3queen occupancy. After inference, explicitly score wrong stops by below/equal/aboveq and initial-state identity. The prior true-graph45cell-cover proof is not assumed when extra empirical exclusions exist.',
        validation='Compare full dense energies and all one-cell flip costs with literal individual-pair sums on synthetic sparse/dense boards, without scoring held-out search accuracy. Independently recompute construction counts from selected patterns.',
        budget=dict(new_gpu_search_jobs=9,gpu_minutes_envelope=20,new_optimizer_fits=0),
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},
        data_sha256={str((BASE/name).relative_to(ROOT)):sha(BASE/name) for name in ['data.npz','dataset.json']},
        limits=['Adaptive reuse of the H3g conditioned pattern-holdout cohort, not fresh confirmation.',
                'The same fixed energy-design prior remains; only parameter sharing/count pooling is removed.',
                'A false certificate can occur at the initial clamped board when extra exclusions invalidate the true-graph coverage bound; wrong stops are failures, never discarded.',
                'Three search repeats for one nested bank draw; no training-sample replication.'])
    OUT.mkdir(parents=True,exist_ok=True);save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',protocol_sha256=sha(SRC/'protocol.json'))))


def construct():
    import numpy as np
    import torch
    from cram import cram_nqueens as Q
    torch.set_num_threads(1);data=dict(np.load(BASE/'data.npz'));i,j=np.triu_indices(144,1);receipts=[]
    for bank in [32,128,512]:
        labels=data['all_solutions'][data[f'bank{bank}_indices']];q_values=np.unique(labels.sum(1));assert len(q_values)==1;q=int(q_values[0])
        counts=labels.astype(np.int64).T@labels.astype(np.int64)
        J=(q*(counts==0).astype(np.float32)-1);np.fill_diagonal(J,0)
        W=torch.zeros(len(i),2,2);W[:,1,1]=torch.tensor(J[i,j]);B=torch.zeros(144,2)
        Sfull=Q.full_S(W,torch.tensor(np.stack([i,j])),144);assert np.array_equal(Sfull[:,:,1,1].numpy(),J)
        x=labels.astype(np.float64);fields=x@J.astype(np.float64);energy=.5*(x*fields).sum(1)
        delta=float(((1-2*x)*fields).min());assert delta==q-1 and np.all(energy==-q*(q-1)/2)
        model=dict(W=W,B=B,S=Sfull,A=W[:,1,1].clone(),b=torch.zeros(1),tau=float(energy.max()),unit=float(np.abs(fields).mean()),delta=delta,arm='individual-pair-ising')
        rng=np.random.default_rng(83000+bank);fixtures=np.stack([np.zeros(144,bool),np.ones(144,bool),*rng.integers(0,2,(8,144)).astype(bool),*labels[:3]])
        dense=Q.board_energy(Sfull.double(),B.double(),torch.tensor(fixtures,dtype=torch.long)).numpy()
        literal=(fixtures[:,i]*fixtures[:,j]*J[i,j]).sum(1,dtype=np.float64);assert np.array_equal(dense,literal)
        dense_fields=Q.fields(Sfull.double(),B.double(),torch.tensor(fixtures,dtype=torch.long)).numpy()
        local=(1-2*fixtures.astype(int))*(dense_fields[:,:,1]-dense_fields[:,:,0])
        for board,base,expected in zip(fixtures,literal,local):
            changes=[]
            for cell in range(144):
                changed=board.copy();changed[cell]=~changed[cell]
                changes.append(float((changed[i]*changed[j]*J[i,j]).sum(dtype=np.float64)-base))
            assert np.array_equal(changes,expected)
        initial=data['test_inputs'];initial_fields=initial.astype(np.float64)@J.astype(np.float64)
        margins=(1-2*initial.astype(int))*initial_fields;margins[initial]=np.inf
        initial_stop=(margins>=delta-.001).all(1)
        dest=OUT/f'models/b{bank}/pair-count';dest.mkdir(parents=True,exist_ok=True);torch.save(model,dest/'model.pt')
        np.savez_compressed(dest/'construction.npz',pair_i=i,pair_j=j,cooccurrence=counts[i,j],forbidden=counts[i,j]==0,
            initial_stopped=initial_stop,initial_min_free_margin=margins.min(1))
        # Rule-derived attack categories are diagnostic only, after model construction.
        attack=(i//12==j//12)|(i%12==j%12)|(np.abs(i//12-j//12)==np.abs(i%12-j%12))
        forbidden=counts[i,j]==0
        rec=dict(status='complete',utc=now(),bank=bank,arm='pair-count',q=q,individual_pair_terms=len(i),
            offset_pooling=False,forbidden_pairs=int(forbidden.sum()),posthoc_false_forbidden=int((forbidden&~attack).sum()),posthoc_missed_forbidden=int((~forbidden&attack).sum()),
            training_delta=delta,unit=model['unit'],tau=model['tau'],initial_wrong_certificates=int(initial_stop.sum()),
            initial_wrong_by_clues={str(k):int((initial_stop&(initial.sum(1)==k)).sum()) for k in [2,3]},
            validation=dict(literal_pair_energy_exact=True,all_single_cell_costs_exact=True,synthetic_boards=len(fixtures)),
            model_sha256=sha(dest/'model.pt'),construction_sha256=sha(dest/'construction.npz'),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__))
        save(dest/'model.json',rec);receipts.append(rec)
    save(OUT/'construction-summary.json',dict(status='complete',models=receipts));print(json.dumps(receipts))


def worker(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    query=lambda *args:subprocess.check_output(args,text=True,timeout=20).strip()
    assert query('hostname')=='loki';device=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader')
    assert device.split(',')[1].strip() not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    p=protocol()
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    for bank in [32,128,512]:
        dest=OUT/f'models/b{bank}/pair-count';rec=read(dest/'model.json')
        assert rec['model_sha256']==sha(dest/'model.pt') and rec['source_sha256']==sha(__file__)
    G.OUT=OUT;G.SRC=SRC;data=dict(np.load(BASE/'data.npz'));started=time.monotonic()
    base=dict(status='running',started_utc=now(),gpu=gpu,pid=os.getpid(),host='loki',device=device,protocol_sha256=sha(SRC/'protocol.json'))
    receipt=OUT/f'worker-g{gpu}.json'
    def progress(**values):save(receipt,dict(base,updated_utc=now(),progress=values))
    progress(stage='starting')
    try:
        for bank in [32,128,512]:
            for seed in [0,1,2]:
                dest=OUT/f'search/b{bank}/pair-count/s{seed}';dest.mkdir(parents=True,exist_ok=True)
                if (dest/'result.json').exists():continue
                claim=(dest/'job.lock').open('a')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                try:
                    if time.monotonic()-started>600:raise RuntimeError('20GPU-minute envelope reached before newjob; evidence retained')
                    if not(dest/'result.json').exists():G.search(dict(bank=bank,arm='pair-count',updates=0),seed,data,progress)
                finally:claim.close()
        save(receipt,dict(base,status='complete',completed_utc=now()))
    except BaseException as exc:save(receipt,dict(base,status='failed',completed_utc=now(),error=repr(exc)));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='mode',required=True)
    sp.add_parser('prepare');sp.add_parser('construct');w=sp.add_parser('worker');w.add_argument('--gpu',type=int,choices=[0,1],required=True)
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':prepare()
    elif a.mode=='construct':construct()
    else:worker(a.gpu)
