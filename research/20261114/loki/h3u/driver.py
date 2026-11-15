"""H3u: matched grouped BP AdamW and fixed count energies, source-only transfer."""
import argparse,datetime as dt,fcntl,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3u';OUT=ROOT/'artifacts/research/20261114/loki/h3u';UP=ROOT/'upstream'
sys.path[:0]=[str(SRC.parent/'h3i'),str(SRC.parent),str(UP),str(UP/'experiments')]
import study as S

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):S.save(p,v)
def protocol():return read(SRC/'protocol.json')
def frozen():
    p=protocol()
    for group in ['source_sha256','data_sha256']:
        for path,h in p[group].items():assert sha(ROOT/path)==h,path
    return p

def prepare():
    assert not(SRC/'protocol.json').exists()
    sources=[Path(__file__),SRC/'prepare.py',SRC/'preflight.py',SRC/'reuse_identities.py',SRC.parent/'h3i/backend.py',SRC.parent/'study.py',UP/'cram/cram_nqueens.py',UP/'cram/nqueens_data.py',SRC.parent/'h3t/protocol.json']
    p=dict(id='H3u-grouped-soft-weight-transfer',frozen_utc=now(),
      hypothesis='Matched grouped BP AdamW improves certificate-first N20 accuracy over the exposure-filtered count control by at least5percentage points, with zero wrong stopped chains.',
      antecedent='Adaptive follow-up to H3t camel-family filter falsification. No H3u optimizer fit or target search has been run. Target universe and prior cohorts are known; this reserves new search keys.',
      arms=[dict(arm=a) for a in ['bp-adamw-grouped','count-hybrid','exposure-filtered','supplied-rule-oracle']],
      architecture='Use all176 unique N12 camel-family source patterns. Infer primitive homogeneous lines by unchanged H3t rule; four disjoint lines share one coefficient each. Each of220 off-line finite offsets gets its own coefficient. One shared unary. Total224couplings+1unary. All arms use this SAME parameterization; old264+1untied architecture is outside this study.',
      line_family=read(SRC.parent/'h3t/protocol.json')['line_family'],
      controls='Source J=12*forbidden−1,b=0. Count retains every observed-zero finite offset; filter retains iff E=sum(m_i*m_j)/176 >=ln(264/.01). Inferred lines are retained unfiltered. Privileged oracle supplies queen+camel attacks under the same grouping, scale and source calibration.',
      fit=dict(optimizer='torch.optim.AdamW',lr=.3,weight_decay=.01,betas=[.9,.999],eps=1e-8,foreach=False,preconditioner=False,updates=4000,cosine_denominator=4000,initialization='allcouplings and unary zero',seed=0,dtype='float32',tf32=False,
        objective='Exact full PL over all176 solutions and each of their298 one/two/three-clue subsets. Occupied free-cell multiplicity231, empty298; normalize by sum of free-cell counts. No target label or partial-input multiplicity used.',
        calibration='Independent float64 reconstruction from final stored float32 source parameters: delta=min allsource/cell flip margin; tau=max source energy; unit=mean absolute source field. Equal complete-source multiplicity298 cancels in unit. Transfer all three scalars unchanged.',
        eligibility='Require finite parameters, finite delta/tau/unit, unit>0 and delta>0. Any failed source-fit eligibility criterion falsifies the primary as source-fit failure: save fit evidence, do not retune, do not run optimizer target searches. All fixed controls still run.'),
      transfer='Copy every source coupling and shared unary unchanged to N13. Extend each inferred line coefficient to all target offsets on that line. For newly representable off-line offsets use fixed−1background. Same convention for all arms. q13 is supplied only as common cardinality stop gate. No q-based coefficient, delta or temperature rescaling; differs from H3tJ13, so all controls rerun.',
      search=dict(n=13,unique_inputs=4096,chains=20,cap=20000,seeds=[0,1,2],temperature=1.,margin=.001,stop='strict local margin plus q13gate',floating_operations='57461+859*moves',cardinality_integer_additions='168*(moves+1)'),
      primary=dict(arm='bp-adamw-grouped',control='exposure-filtered',accuracy_gain_pp=5.,required_wrong_stops=0),
      decision='If source gate fails: falsified(source-fit eligibility). Else support iff paired-input bootstrap one-sided95% lower accuracy gain>=5pp and wrong stopped chains=0; falsify iff upper gain<5pp OR any wrong stopped chain; otherwise inconclusive. Bootstrap2000draws, seed20261114, three search seeds clustered within each input. All other comparisons and coverage descriptive.',
      numeric_audit='Independently reconstruct source calibration, all target coefficients and unary in float64; recompute every saved board energy and free-cell margin. Report energy error extrema/quantiles, strict negative stopped gaps and counts below−1e−6/−1e−4/−1e−3, near-threshold counts at1e−6/1e−4/1e−3, and exact-energy selector disagreements. Any stopped deficit below−.001 or maximum saved energy discrepancy>.1 prevents a supported conclusion (numerical audit inconclusive) without retuning or rerunning. Wrong-stop falsification takes precedence. Exact certificate capacity includes unary terms.',
      identity='Deduplicate exact S,B,q,delta,tau,unit only, before target inference; retain every logical arm with explicit outcome mappings if any.',
      validation='Before fit compare grouped binary PL to literal two-state full-row source fixture in float64 and CPU/GPU loss/gradients on actual176source patterns; verify two AdamW updates against independent moment equations. Before search independently reconstruct all groups, coefficients, source calibration, transfer and every dataset count; frozen backend synthetic parity on CPU/GPU.',
      dataset=read(OUT/'dataset.json'),budget=dict(fit_gpu_worker_seconds_envelope=120,search_gpu_minutes_envelope=20,logical_search_jobs=12,optimizer_fits=1),
      source_sha256={str(x.relative_to(ROOT)):sha(x) for x in sources},data_sha256={str(x.relative_to(ROOT)):sha(x) for x in [OUT/'training.npz',OUT/'data.npz',OUT/'dataset.json']},
      limits=['One complete source universe, one zero-initialized optimizer trajectory; search seeds are not independent source/fitting replicas.','Shared line architecture, default−1background and q13 are supplied priors. The grouped BP AdamW is a matched control, not the original264+1model.','Native source calibration may transfer poorly. No N13 calibration or target-driven tuning.','New matched energies/settings differ from H3t symbolic q13 construction; do not pool or relabel its results.','Exposure filtering remains an independence-model heuristic.'])
    save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',sha256=sha(SRC/'protocol.json'))))

def source_design():
    import numpy as np
    y=np.load(OUT/'training.npz')['n12_all'];n=12;offsets=np.array([(r,c) for r in range(n) for c in range(1-n,n) if r>0 or c>0]);lookup={tuple(o):k for k,o in enumerate(offsets)}
    i,j=np.triu_indices(144,1);index=np.array([lookup[b//12-a//12,b%12-a%12] for a,b in zip(i,j)])
    co=y.astype(np.int64).T@y.astype(np.int64);counts=np.bincount(index,weights=co[i,j],minlength=264).astype(np.int64);m=y.sum(0).astype(np.int64)
    exposure=np.bincount(index,weights=m[i]*m[j],minlength=264)/len(y);lines=[]
    for a in range(-2,3):
        for b in range(-2,3):
            if (a==b==0) or math.gcd(abs(a),abs(b))!=1 or (a<0 or (a==0 and b<0)):continue
            covered=a*offsets[:,0]+b*offsets[:,1]==0
            if ((counts==0)&covered).sum()>=3 and not ((counts>0)&covered).any():lines.append((a,b))
    group=np.full(264,-1,np.int64)
    for k,(a,b) in enumerate(lines):
        take=a*offsets[:,0]+b*offsets[:,1]==0;assert (group[take]==-1).all();group[take]=k
    off=np.flatnonzero(group<0);group[off]=np.arange(len(off))+len(lines)
    assert len(lines)==4 and group.max()+1==224
    return dict(labels=y,offsets=offsets,counts=counts,exposure=exposure,lines=np.array(lines),groups=group,pair_i=i,pair_j=j,pair_groups=group[index])

def matrix(A,b,d,n):
    import numpy as np
    import torch
    i,j=np.triu_indices(n*n,1);lookup={tuple(o):int(g) for o,g in zip(d['offsets'],d['groups'])};idx=[]
    for u,v in zip(i,j):
        dr=v//n-u//n;dc=v%n-u%n
        line=next((k for k,(a,z) in enumerate(d['lines']) if a*dr+z*dc==0),None)
        idx.append(line if line is not None else lookup.get((dr,dc),-1))
    index=torch.tensor(idx,device=A.device);pair=torch.where(index>=0,A[index.clamp_min(0)],A.new_tensor(-1.))
    ti=torch.tensor(i,device=A.device);tj=torch.tensor(j,device=A.device);J=A.new_zeros(n*n,n*n).index_put((ti,tj),pair).index_put((tj,ti),pair)
    return J,b.expand(n*n)

def source_loss(A,b,target,weights,pair_groups):
    import torch
    import torch.nn.functional as F
    i,j=torch.triu_indices(144,144,1,device=A.device)
    J=A.new_zeros(144,144).index_put((i,j),A[pair_groups]).index_put((j,i),A[pair_groups])
    field=target@J+b
    return (F.binary_cross_entropy_with_logits(-field,target,reduction='none')*weights).sum()

def model_from(A,b,d,arm):
    import numpy as np
    import torch
    J,h=matrix(A,b,d,12);x=d['labels'].astype(np.float64);field=x@J.double().numpy()+h.double().numpy()
    calibration=dict(delta=float(((1-2*x)*field).min()),tau=float((.5*x*(field+h.double().numpy())).sum(1).max()),unit=float(np.abs(field).mean()))
    eligible=all(np.isfinite(v) for v in calibration.values()) and calibration['delta']>0 and calibration['unit']>0 and bool(torch.isfinite(A).all() and torch.isfinite(b).all())
    J,h=matrix(A,b,d,13);i,j=torch.triu_indices(169,169,1);W=torch.zeros(len(i),2,2);W[:,1,1]=J[i,j];B=torch.zeros(169,2);B[:,1]=h
    dense=torch.zeros(169,169,2,2);dense[:,:,1,1]=J
    return dict(A=A,b=b,S=dense,B=B,W=W,q=13,arm=arm,**calibration),eligible

def save_model(A,b,d,arm,extra):
    import numpy as np
    import torch
    model,eligible=model_from(A.detach().cpu(),b.detach().cpu(),d,arm);dest=OUT/f'models/{arm}';dest.mkdir(parents=True,exist_ok=True);torch.save(model,dest/'model.pt')
    np.savez_compressed(dest/'source-parameters.npz',A=model['A'].numpy(),b=model['b'].numpy(),offsets=d['offsets'],groups=d['groups'],lines=d['lines'])
    rec=dict(status='complete' if eligible else 'source_fit_ineligible',utc=now(),arm=arm,source_patterns=176,target_training_patterns=0,source_n=12,target_n=13,source_eligible=eligible,
      **{k:model[k] for k in ['delta','tau','unit']},**extra,model_sha256=sha(dest/'model.pt'),source_parameters_sha256=sha(dest/'source-parameters.npz'),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'))
    save(dest/'model.json',rec);print(json.dumps(rec),flush=True)

def construct():
    import numpy as np
    import torch
    frozen();torch.set_num_threads(1);d=source_design();np.savez_compressed(OUT/'source-design.npz',**d)
    for arm in ['count-hybrid','exposure-filtered','supplied-rule-oracle']:
        if arm=='supplied-rule-oracle':
            r,c=d['offsets'].T;mask=(r==0)|(c==0)|(np.abs(r)==np.abs(c))|((r==1)&(np.abs(c)==3))|((r==3)&(np.abs(c)==1))
        else:mask=(d['counts']==0)&((d['exposure']>=math.log(264/.01)) if arm=='exposure-filtered' else True)
        A=torch.full((224,),-1.);A[:len(d['lines'])]=11.
        for k in range(len(d['lines']),224):A[k]=12*bool(mask[d['groups']==k].item())-1
        save_model(A,torch.zeros(1),d,arm,dict(updates=0,optimizer=None))

def fit(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    import torch
    p=frozen();gate=read(OUT/'loss-validation.json');assert gate['status']=='passed'
    lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.check_output(['hostname'],text=True).strip()=='loki'
    dev=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert dev not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(0);d=source_design();target=torch.tensor(d['labels'],device='cuda',dtype=torch.float32)
    weights=torch.where(target.bool(),231.,298.);weights/=weights.sum();groups=torch.tensor(d['pair_groups'],device='cuda');A=torch.zeros(224,device='cuda',requires_grad=True);b=torch.zeros(1,device='cuda',requires_grad=True)
    opt=torch.optim.AdamW([A,b],lr=.3,weight_decay=.01,betas=(.9,.999),eps=1e-8,foreach=False);curve=[];torch.cuda.synchronize();start=time.perf_counter();began=now()
    for k in range(4000):
        if k%100==0 and time.perf_counter()-start>120:raise RuntimeError('Frozen120secondfitbudget exhausted; do not retune')
        opt.zero_grad(set_to_none=True);loss=source_loss(A,b,target,weights,groups);loss.backward();opt.param_groups[0]['lr']=.3*.5*(1+math.cos(math.pi*k/4000));opt.step()
        if (k+1)%250==0:
            value=float(loss.detach());assert math.isfinite(value);row=dict(update=k+1,loss=value,elapsed_seconds=time.perf_counter()-start);curve.append(row);save(OUT/'fit-progress.json',dict(status='running',utc=now(),gpu=gpu,pid=os.getpid(),**row));print(json.dumps(row),flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    save_model(A,b,d,'bp-adamw-grouped',dict(updates=4000,optimizer=p['fit'],fit_seconds=seconds,started_utc=began,completed_utc=now(),gpu=gpu,gpu_uuid=dev,loss=float(loss.detach()),curve=curve,loss_validation_sha256=sha(OUT/'loss-validation.json')))
    save(OUT/'fit-progress.json',dict(status='complete',utc=now(),gpu=gpu,pid=os.getpid(),seconds=seconds))

def camel_valid(boards,inputs):
    import numpy as np
    valid=S.independent_ok(boards,inputs,13);i,j=np.triu_indices(169,1);dr=np.abs(j//13-i//13);dc=np.abs(j%13-i%13)
    camel=((dr==1)&(dc==3))|((dr==3)&(dc==1))
    return valid&~(boards[...,i[camel]]&boards[...,j[camel]]).any(-1)


def validate_backend(device):
    import numpy as np
    import torch
    import backend
    from cram import cram_nqueens as Q
    # Synthetic Ising fixtures only; no N13 target solution enters validation.
    c=169;model=dict(S=torch.zeros(c,c,2,2),B=torch.zeros(c,2),q=13,delta=12.,tau=-78.,unit=25.909505208333332)
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
    seconds=time.perf_counter()-start;valid=camel_valid(boards,inputs);denom=np.minimum(data['test_solution_counts'],20)
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
    assert read(OUT/'independent-preflight.json')['status']=='passed'
    data=dict(np.load(OUT/'data.npz'));started=time.monotonic()
    base=dict(status='running',started_utc=now(),gpu=gpu,pid=os.getpid(),host='loki',device=device,protocol_sha256=sha(SRC/'protocol.json'))
    receipt=OUT/f'worker-g{gpu}.json'
    def progress(**values):save(receipt,dict(base,updated_utc=now(),progress=values))
    progress(stage='starting')
    try:
        for cfg in p['arms']:
            if not read(OUT/f"models/{cfg['arm']}/model.json")['source_eligible']:continue
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
    sp.add_parser('prepare');sp.add_parser('construct')
    v=sp.add_parser('validate');v.add_argument('--device',choices=['cpu','cuda'],required=True)
    f=sp.add_parser('fit');f.add_argument('--gpu',type=int,choices=[0,1],required=True)
    w=sp.add_parser('worker');w.add_argument('--gpu',type=int,choices=[0,1],required=True)
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':prepare()
    elif a.mode=='construct':construct()
    elif a.mode=='validate':validate_backend(a.device)
    elif a.mode=='fit':fit(a.gpu)
    else:worker(a.gpu)
