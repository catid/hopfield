"""H3k: source-bank robustness of the frozen primitive-line inference rule."""
import argparse
import datetime as dt
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3k';OUT=ROOT/'artifacts/research/20261114/loki/h3k';UP=ROOT/'upstream'
sys.path[:0]=[str(SRC.parent/'h3i'),str(SRC.parent),str(UP),str(UP/'experiments')]
import study as S


def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):S.save(path,value)
def protocol():return read(SRC/'protocol.json')


def prepare():
    assert not(SRC/'protocol.json').exists();dataset=read(OUT/'dataset.json');parent=read(SRC.parent/'h3j/protocol.json')
    arms=[dict(arm=f'draw{draw:03d}-n{n}',draw=draw,source_patterns=n) for draw in range(6) for n in [1,8]]
    sources=[Path(__file__),SRC/'prepare.py',SRC.parent/'h3i/backend.py',SRC.parent/'h3j/driver.py',SRC.parent/'study.py',*sorted((UP/'cram').glob('*.py'))]
    p=dict(id='H3k-small-source-bank-robustness',frozen_utc=now(),
        hypothesis='With8 N12 positive training patterns, the previously frozen primitive-line rule recovers exactly the four true homogeneous attack lines in at least99% of independent eligible source-bank draws.',
        primary=dict(source_patterns=8,independent_draws=512,required_recovery_probability=.99),
        decision='Support if the one-sided95% exact Clopper-Pearson lower bound on exact-four-line recovery at8patterns is>=0.99; falsify if the exact upper bound<0.99; otherwise inconclusive. Sizes1/2/4/16 are descriptive. The512 draws are independent samples from one fixed eligible source pool; nested sizes within each draw are paired.',
        line_family=parent['line_family'],source_sizes=[1,2,4,8,16],source_draws=512,
        construction='Compute source displacement co-occurrences only from each draw prefix, apply the unchanged line rule, and retain every fitted line set including failures/spurious lines. Truth is audit-only. The source-bank study does not calibrate thresholds or choose line families.',
        behavior_sampling='Exactly first6 prespecified draw seeds at sizes1 and8, giving12 logical models. No model is replaced based on recovered lines or performance. Three search seeds per distinct solver energy; identical models share explicitly mapped outputs.',
        arms=arms,calibration=parent['calibration'],search=dict(parent['search'],unique_inputs=2048),
        behavior_metrics='N13 N20 both selectors, N1, coverage, wrong stops and capped chains. Behavior is descriptive for the first6 draws, not the primary512-draw inference. Report each logical source draw separately even when energies duplicate.',
        identity='Group by exact solver-used S,B,q,delta,tau,unit identity before target inference; choose the first logical model in frozen arm order as representative. Three new searches per unique group; retain every logical arm through explicit output mappings, never count mappings as independent search replication.',
        dataset=dataset,budget=dict(logical_model_seed_jobs=36,maximum_unique_search_jobs=36,gpu_minutes_envelope=20,optimizer_fits=0),
        validation='Independent reconstruction of all512 nested source fits from literal occupied cell pairs; source draw seeds and exclusions; preserve every source line set and exact recovery outcome. Verify target model coefficients and identity groups, source/target separation, fresh cohort and exact target counts; audit every unique saved search and all mappings.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        data_sha256={str(path.relative_to(ROOT)):sha(path) for path in [OUT/'data.npz',OUT/'dataset.json',SRC.parent/'h3j/protocol.json']},
        limits=['The fixed primitive-line family and coordinate geometry are strong supplied priors. This is not geometry-free discovery.',
                'Source-bank variation is the unit of the primary512-draw inference; search seeds are separate repeated inference draws.',
                'The source population excludes previous restricted H3g/H3i banks, but older full-universe studies prevent never-seen mathematical pattern claims.',
                'Behavior uses the first6 draws only at sizes1/8; no population-wide N13 accuracy estimate is claimed from this subset.',
                'Target cardinality, temperature unit, count-energy design, q gate and certificate-first selection remain fixed from H3j. No N13 training or recalibration.'])
    save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',protocol_sha256=sha(SRC/'protocol.json'))))


def construct():
    import numpy as np
    import torch
    from cram import cram_nqueens as Q
    torch.set_num_threads(1);p=protocol();d=dict(np.load(OUT/'data.npz'));draws=d['source_draw_indices'];solutions=d['source_solutions']
    offsets=np.array([(r,c) for r in range(12) for c in range(-11,12) if r>0 or c>0]);lookup={tuple(o):i for i,o in enumerate(offsets)}
    candidates=np.array([(a,b) for a in range(-2,3) for b in range(-2,3) if (a,b)!=(0,0) and math.gcd(abs(a),abs(b))==1 and (a>0 or (a==0 and b>0))])
    membership=(candidates@offsets.T)==0;features={}
    for index in np.unique(draws):
        cells=np.flatnonzero(solutions[index]);vec=np.zeros(264,np.int16)
        for a,b in itertools.combinations(cells,2):vec[lookup[b//12-a//12,b%12-a%12]]+=1
        features[int(index)]=vec
    count_records=[];retained_records=[];rows=[]
    for draw,indices in enumerate(draws):
        prefix=np.stack([features[int(index)] for index in indices]).cumsum(0)
        for n in p['source_sizes']:
            counts=prefix[n-1];nforbid=(membership&(counts[None,:]==0)).sum(1);nallow=(membership&(counts[None,:]>0)).sum(1)
            retained=(nforbid>=3)&(nallow==0);count_records.append(counts);retained_records.append(retained)
            rows.append(dict(draw=draw,source_seed=int(d['source_draw_seeds'][draw]),source_patterns=n,
                retained_lines=candidates[retained].tolist(),retained_line_count=int(retained.sum()),
                line_signature=';'.join(f'{a},{b}' for a,b in candidates[retained])))
    np.savez_compressed(OUT/'source-fits.npz',offsets=offsets,candidates=candidates,counts=np.array(count_records),retained=np.array(retained_records),
        source_draw_indices=draws,source_draw_seeds=d['source_draw_seeds'])
    save(OUT/'source-fits.json',dict(status='fitted',utc=now(),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'),
        array_sha256=sha(OUT/'source-fits.npz'),rows=rows))
    i,j=np.triu_indices(169,1);dr=j//13-i//13;dc=j%13-i%13;groups={};receipts=[]
    for cfg in p['arms']:
        arm=cfg['arm'];row=rows[cfg['draw']*5+p['source_sizes'].index(cfg['source_patterns'])];lines=row['retained_lines']
        mask=np.array([any(a*r+b*c==0 for a,b in lines) for r,c in zip(dr,dc)])
        J=np.zeros((169,169),np.float32);J[i,j]=13*mask-1;J[j,i]=J[i,j]
        W=torch.zeros(len(i),2,2);W[:,1,1]=torch.tensor(J[i,j]);B=torch.zeros(169,2);dense=Q.full_S(W,torch.tensor(np.stack([i,j])),169)
        assert np.array_equal(dense[:,:,1,1].numpy(),J)
        model=dict(W=W,B=B,S=dense,A=W[:,1,1].clone(),b=torch.zeros(1),q=13,delta=12.,tau=-78.,unit=p['calibration']['unit'],arm=arm)
        signature=hashlib.sha256(J.astype(np.int16).tobytes()+json.dumps(p['calibration'],sort_keys=True).encode()).hexdigest()
        dest=OUT/f'models/{arm}';dest.mkdir(parents=True,exist_ok=True);torch.save(model,dest/'model.pt')
        np.savez_compressed(dest/'construction.npz',pair_i=i,pair_j=j,forbidden=mask)
        rec=dict(status='complete',utc=now(),arm=arm,draw=cfg['draw'],source_seed=row['source_seed'],source_patterns=cfg['source_patterns'],
            retained_lines=lines,energy_signature=signature,q=13,delta=12.,tau=-78.,unit=model['unit'],target_training_patterns=0,
            model_sha256=sha(dest/'model.pt'),construction_sha256=sha(dest/'construction.npz'),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'))
        save(dest/'model.json',rec);receipts.append(rec)
        if signature not in groups:groups[signature]=dict(signature=signature,representative=arm,members=[])
        groups[signature]['members'].append(arm)
    save(OUT/'energy-groups.json',dict(status='grouped_before_target_inference',utc=now(),groups=list(groups.values()),
        logical_models=12,distinct_energies=len(groups),actual_unique_search_jobs=3*len(groups),avoided_identical_jobs=36-3*len(groups)))
    save(OUT/'construction-summary.json',dict(status='complete',models=receipts))
    print(json.dumps(dict(source_models=len(rows),distinct_behavior_energies=len(groups),groups=list(groups.values()))))


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
