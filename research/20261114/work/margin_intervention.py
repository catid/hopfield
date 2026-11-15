"""Upward/downward spatial perturbations with matched expected-mean controls."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import numpy as np
import study as S

DEST=S.OUT/"margin_intervention"
BASE=2026111427
NOISE_BASE=2026111428
N=4096
BATCH=64
ARMS=("native","full_gaussian","upward_only","downward_only","constant_up","constant_down")
MEAN_SHIFT=1./math.sqrt(2.*math.pi)


def feasible(tok,pred):
    """Only supplied graph/endpoints and predicted cells; no shortest-path data."""
    pred=np.asarray(pred,dtype=bool);h,w=tok.shape
    if np.any(pred&(tok==0)):return False
    start=tuple(np.argwhere(tok==2)[0]);goal=tuple(np.argwhere(tok==3)[0])
    if not pred[start] or not pred[goal]:return False
    degree=np.zeros_like(tok,dtype=np.int8)
    degree[1:]+=pred[:-1];degree[:-1]+=pred[1:];degree[:,1:]+=pred[:,:-1];degree[:,:-1]+=pred[:,1:]
    if degree[start]!=1 or degree[goal]!=1 or not np.all(degree[pred&(tok==1)]==2):return False
    seen={start};queue=deque([start])
    while queue:
        r,c=queue.popleft()
        for rr,cc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if 0<=rr<h and 0<=cc<w and pred[rr,cc] and (rr,cc) not in seen:
                seen.add((rr,cc));queue.append((rr,cc))
    return len(seen)==int(pred.sum())


def gaussian(indices):
    return np.stack([3.+np.random.default_rng(np.random.SeedSequence([NOISE_BASE,int(i)])).standard_normal((90,90),dtype=np.float32) for i in indices])


def initial_arms(full):
    return [np.full_like(full,3.),full,np.maximum(full,np.float32(3.)),np.minimum(full,np.float32(3.)),np.full_like(full,3.+MEAN_SHIFT),np.full_like(full,3.-MEAN_SHIFT)]


def make_one(index):
    tok,label,distance,opened,eligible=S.maze(90,.20,np.random.SeedSequence([BASE,90,20,index]))
    return tok,label,distance,opened,eligible


def freeze():
    DEST.mkdir(parents=True,exist_ok=True);parent=json.loads((S.HERE/"sign_intervention-protocol.json").read_text());pilot_path=S.OUT/"sign_intervention/batch-preflight.json";pilot=json.loads(pilot_path.read_text());assert pilot["chosen_batch"]==BATCH and pilot["native_answer_outputs_exact"]
    protocol=dict(parent,id="paired-native-margin-interventions",frozen_utc=S.utc(),status="frozen_before_data_and_outcomes",execution_script_sha256=S.sha(__file__),core_script_sha256=S.sha(S.__file__))
    protocol["cohort"]=dict(parent["cohort"],base_seed=BASE,n_unique=N,shards=16)
    protocol["initialization"]={"arms":list(ARMS),"source":"Gaussian float32 u=3+default_rng(SeedSequence([noise_base,input_index])).standard_normal((90,90),dtype=float32). Exact full Gaussian arrays retained; same draw across models and paired spatial interventions.","noise_base":NOISE_BASE,"standard_deviation":1.,"native":"every cell3","full_gaussian":"unchanged saved Gaussian array","upward_only":"elementwise max(3,full Gaussian)","downward_only":"elementwise min(3,full Gaussian)","constant_up":"every cell float32(3+1/sqrt(2*pi))","constant_down":"every cell float32(3-1/sqrt(2*pi))","mean_shift_exact_expression":"1/sqrt(2*pi)","mean_shift_float64":MEAN_SHIFT,"constant_up_float32":float(np.float32(3.+MEAN_SHIFT)),"constant_down_float32":float(np.float32(3.-MEAN_SHIFT)),"mean_matching":"Constants match population expected means of rectified standard-Gaussian perturbations, not each finite input's realized spatial mean. Float32 storage rounding explicit.","all_cells":"Initial transforms apply to all cells; native wall/endpoint clamping unchanged. Separate changed/above/below/negative counts and mean logits retained on free cells.","numpy_version":np.__version__,"development_tuning":False}
    protocol["primary"]={"id":"H2p","hypothesis":"Upward-only initialization is noninferior within0.1percentage point of native joint-three-model valid-shortest success.","loss_margin_pp":.1,"unit":"independent input; fixed-model repeats are not independent trials","decision":"Loss event nativejointsuccess AND NOT upwardjointsuccess. Exact one-sided95% Clopper-Pearson bounds. Support if upper<=0.1pp; falsify if lower>0.1pp; otherwise inconclusive. Gains reported separately, never offset losses."}
    protocol["secondary"]="All comparisons involving full Gaussian, downward-only and constant3±1/sqrt(2*pi), plus upward-only gains, descriptive. Controls separate expected mean shifts from spatial heterogeneity. No additional threshold, amplitude or selector tuning. Every arm returns a single output; shortestness is evaluation-only."
    protocol["budget"]={"total_gpu_minute_envelope":20,"per_worker_wall_seconds":600,"rationale":"Preceding2048-input four-arm batch64 study took1.284summedGPU-minutes;4096inputs and6arms imply ~3.85minutes before overhead. Retain exact final logits.","watchdog":"Check every64-input batch and before eachjob; preserve partial output; no cap/sample extensions."}
    protocol["timing"]="All6arms cap8192 eta.5; rotate arm order by job/batch and CUDA synchronize. Same cached native hidden readout. Batch64 retains exact reserved-input native parity and preceding measured1.914x throughput validation."
    protocol["batch_preflight"]={"path":str(pilot_path.relative_to(S.ROOT)),"sha256":S.sha(pilot_path),"ratio_32_over_64":pilot["ratio_32_over_64"]}
    protocol["preflight"]="Native8192 exact parity on64reservedinputs; deterministic fixture checks max/min around3 and float32 expected-mean constants. No selected-case outcomes used."
    protocol["retained_evidence"]="Exact initial Gaussian arrays and final logits for every arm/model/input; changed/free-negative/above/below/zero counts and realized mean free-cell initial logits. All predicted boards/energies/steps/certificate/finite flags retained."
    protocol["jobs"]=[{"id":f"shard{shard:02d}-s{seed}","shard":shard,"seed":seed,"gpu":(shard*3+seed)%2} for shard in range(16) for seed in range(3)]
    pp=S.HERE/"margin_intervention-protocol.json"
    if pp.exists():raise FileExistsError(pp)
    S.save(pp,protocol);S.emit("margin_protocol_frozen",sha256=S.sha(pp),unique_inputs=N,jobs=len(protocol["jobs"]))


def prepare():
    pp=S.HERE/"margin_intervention-protocol.json";protocol=json.loads(pp.read_text());assert protocol["execution_script_sha256"]==S.sha(__file__)
    datasets=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(16):
            path=DEST/f"shard{shard:02d}.npz"
            if path.exists():raise FileExistsError(path)
            indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8))
            np.savez_compressed(path,initial_gaussian=gaussian(indices),tokens=np.stack([x[0] for x in items]),labels=np.stack([x[1] for x in items]),shortest_edges=np.array([x[2] for x in items]),opened_loops=np.array([x[3] for x in items]),eligible_walls=np.array([x[4] for x in items]),indices=indices)
            datasets.append({"shard":shard,"n":256,"path":str(path.relative_to(S.ROOT)),"sha256":S.sha(path)})
            S.emit("restart_data_prepared",shard=shard,inputs_done=(shard+1)*256,seconds=time.perf_counter()-started)
    S.save(DEST/"datasets.json",{"protocol_sha256":S.sha(pp),"numpy":np.__version__,"datasets":datasets,"seconds":time.perf_counter()-started})


def solve(model,t,a,u0,cap):
    torch,_,_=S.imports();u=u0.clone();n=len(t);done=torch.zeros(n,dtype=torch.bool,device=t.device);finite=torch.ones_like(done)
    used=torch.full((n,),cap,dtype=torch.long,device=t.device)
    with torch.no_grad():
        for k in range(1,cap+1):
            new=u+.5*(model.field(t,torch.sigmoid(u),a)-u)
            u=torch.where((done|~finite)[:,None,None],u,new)
            if k%16==0:
                finite&=torch.isfinite(u).flatten(1).all(1)
                ok=(model.flip_delta(t,(u>0).to(a.dtype),a)>=-1e-6).flatten(1).all(1)&finite
                used=torch.where(ok&~done,torch.full_like(used,k),used);done|=ok
                if (done|~finite).all():break
    return (u>0).to(a.dtype),used,done,finite,u


def preflight(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,M=S.imports();model=S.model_for(0,90,"original","cuda")
    items=[S.maze(90,.2,np.random.SeedSequence([2026111492,i])) for i in range(64)]
    t=torch.tensor(np.stack([x[0] for x in items]).astype(np.int64),device="cuda");checks=[]
    with torch.no_grad():
        psi,_=model.settle(t);a=model.couplings(t,psi)
        for cap in (8192,):
            native=M.descend(model,t,a,steps=cap,eta=.5,margin=3.,certify_every=16)
            fresh=solve(model,t,a,torch.full_like(a[:,0],3.),cap)
            assert all(torch.equal(native[i],fresh[i]) for i in range(3)) and fresh[3].all()
            checks.append({"cap":cap,"native_parity":True,"steps":fresh[1].tolist()})
    fixture=np.array([-1.,0.,1.,3.],np.float32);arms=initial_arms(fixture)
    assert np.array_equal(arms[2],np.array([3.,3.,3.,3.],np.float32)) and np.array_equal(arms[3],fixture)
    assert np.all(arms[4]==np.float32(3.+MEAN_SHIFT)) and np.all(arms[5]==np.float32(3.-MEAN_SHIFT))
    tok=np.zeros((9,9),np.uint8);tok[0:5,0]=1;tok[4,0:5]=1;tok[0:5,4]=1;tok[0,0]=2;tok[0,4]=3
    pred=tok>0;assert feasible(tok,pred)
    fixtures=["valid non-shortestness-agnostic detour"]
    for name,coordinate in [("missing endpoint",(0,0)),("wall selection",(8,8)),("disconnected extra cell",(7,7))]:
        p=pred.copy();p[coordinate]=not p[coordinate];assert not feasible(tok,p);fixtures.append(name)
    branch=tok.copy();branch[4,5]=1;p=branch>0;assert not feasible(branch,p);fixtures.append("branch")
    cycle=tok.copy();cycle[6,6:9]=1;cycle[8,6:9]=1;cycle[6:9,6]=1;cycle[6:9,8]=1;assert not feasible(cycle,cycle>0);fixtures.append("disconnected degree-two cycle")
    S.save(DEST/"preflight.json",{"status":"passed","utc":S.utc(),"protocol_sha256":S.sha(S.HERE/"margin_intervention-protocol.json"),"execution_sha256":S.sha(__file__),"native_checks":checks,"feasibility_fixtures":fixtures,"rectified_and_mean_fixture":True})
    S.emit("restart_preflight_passed",checks=checks)


def worker(gpu):
    lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True);assert inventory.split(",")[1].strip() not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,_=S.imports();from cram.maze_learned import gather
    pp=S.HERE/"margin_intervention-protocol.json";protocol=json.loads(pp.read_text());assert protocol["execution_script_sha256"]==S.sha(__file__) and protocol["core_script_sha256"]==S.sha(S.__file__)
    assert protocol["initialization"]["numpy_version"]==np.__version__
    pre=json.loads((DEST/"preflight.json").read_text());assert pre["status"]=="passed" and pre["protocol_sha256"]==S.sha(pp)
    datasets=json.loads((DEST/"datasets.json").read_text());assert datasets["protocol_sha256"]==S.sha(pp)
    for path,h in protocol["source_sha256"].items():assert S.sha(S.ROOT/path)==h
    state={"status":"running","pid":os.getpid(),"host":platform.node(),"gpu":gpu,"started_utc":S.utc(),"jobs_complete":[]};worker_started=time.perf_counter()
    for ji,job in enumerate(protocol["jobs"]):
        if job["gpu"]!=gpu:continue
        if time.perf_counter()-worker_started>600:raise TimeoutError("Frozen per-worker 600s envelope reached")
        run=DEST/"runs"/job["id"];run.mkdir(parents=True,exist_ok=True)
        if (run/"result.json").exists():raise FileExistsError(run)
        dataset=datasets["datasets"][job["shard"]];assert S.sha(S.ROOT/dataset["path"])==dataset["sha256"]
        source=np.load(S.ROOT/dataset["path"]);tokens=source["tokens"];indices=source["indices"];full_gaussian=source["initial_gaussian"]
        assert S.sha(S.OUT/f"models/seed{job['seed']}.pt")==protocol["models"][str(job["seed"])]["sha256"]
        model=S.model_for(job["seed"],90,"original","cuda");saved={k:[] for k in ["predictions","answer_steps","stopped","finite","energy","feasible","hidden_steps","hidden_good","final_logits","changed_cells","changed_free_cells","negative_cells","negative_free_cells","zero_cells","above_native_free_cells","below_native_free_cells","mean_initial_free_logit"]}
        answer_times=np.zeros(6);hidden_time=0.;selector_time=0.;initial_hash=hashlib.sha256();started=time.perf_counter()
        state.update(current_job=job,completed_inputs=0,updated_utc=S.utc());S.save(DEST/f"worker{gpu}.json",state)
        for lo in range(0,256,BATCH):
            token=tokens[lo:lo+BATCH];t=torch.tensor(token.astype(np.int64),device="cuda")
            with torch.no_grad():
                torch.cuda.synchronize();begin=time.perf_counter();psi,ps=model.settle(t,tol=1e-6,max_steps=4000,check=8)
                lg,la,lc=model.parts(t,model.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,model.zoffs,math.inf),lc[:,None]],1),1)
                hidden_good=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1))
                a=model.couplings(t,psi);torch.cuda.synchronize();hidden_time+=time.perf_counter()-begin
                init=initial_arms(full_gaussian[lo:lo+BATCH])
                for value in init:initial_hash.update(value.tobytes())
                u0=[torch.tensor(value,device="cuda") for value in init]
                if lo==0:
                    for u in u0:solve(model,t,a,u,32)
                    torch.cuda.synchronize()
                outputs={};energies={};offset=(ji+lo//BATCH)%6
                for variant in list(range(6))[offset:]+list(range(6))[:offset]:
                    torch.cuda.synchronize();begin=time.perf_counter();outputs[variant]=solve(model,t,a,u0[variant],8192)
                    energies[variant]=model.energy(t,outputs[variant][0],a)
                    torch.cuda.synchronize();answer_times[variant]+=time.perf_counter()-begin
                values={key:np.stack([outputs[v][k].cpu().numpy() for v in range(6)]) for k,key in enumerate(["predictions","answer_steps","stopped","finite","final_logits"])}
                values["predictions"]=values["predictions"].astype(bool);values["energy"]=np.stack([energies[v].cpu().numpy() for v in range(6)])
                hg=hidden_good.cpu().numpy();begin=time.perf_counter()
                values["feasible"]=np.array([[feasible(tok,pred) for tok,pred in zip(token,values["predictions"][v])] for v in range(6)])
                values["changed_cells"]=np.array([(value!=3.).sum((1,2)) for value in init])
                values["changed_free_cells"]=np.array([((value!=3.)&(token==1)).sum((1,2)) for value in init])
                values["negative_cells"]=np.array([(value<0).sum((1,2)) for value in init])
                values["negative_free_cells"]=np.array([((value<0)&(token==1)).sum((1,2)) for value in init])
                values["zero_cells"]=np.array([(value==0).sum((1,2)) for value in init])
                values["above_native_free_cells"]=np.array([((value>3.)&(token==1)).sum((1,2)) for value in init])
                values["below_native_free_cells"]=np.array([((value<3.)&(token==1)).sum((1,2)) for value in init])
                values["mean_initial_free_logit"]=np.array([(value*(token==1)).sum((1,2),dtype=np.float64)/(token==1).sum((1,2)) for value in init])
                selector_time+=time.perf_counter()-begin
                values["hidden_steps"]=ps.cpu().numpy();values["hidden_good"]=hg
                for key,value in values.items():saved[key].append(value)
            state.update(completed_inputs=lo+BATCH,updated_utc=S.utc(),elapsed_s=time.perf_counter()-started,worker_elapsed_s=time.perf_counter()-worker_started)
            S.save(DEST/f"worker{gpu}.json",state);S.emit("restart_batch_done",gpu=gpu,job=job["id"],inputs=lo+BATCH)
            if time.perf_counter()-worker_started>600:
                np.savez_compressed(run/"partial.npz",**{k:np.concatenate(v,axis=1 if v[0].ndim>=2 else 0) for k,v in saved.items()});raise TimeoutError("Frozen per-worker 600s envelope reached; partial preserved")
        arrays={k:np.concatenate(v,axis=1 if v[0].ndim>=2 else 0) for k,v in saved.items()};np.savez_compressed(run/"outputs.npz",**arrays)
        record={"status":"complete","utc":S.utc(),"job":job,"n":256,"protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"dataset_sha256":dataset["sha256"],"checkpoint_sha256":protocol["models"][str(job["seed"])]["sha256"],
            "outputs_sha256":S.sha(run/"outputs.npz"),"initialization_sha256":initial_hash.hexdigest(),"hidden_seconds_shared":hidden_time,"answer_seconds_by_variant":answer_times.tolist(),"diagnostic_cpu_seconds":selector_time,"wall_seconds":time.perf_counter()-started,"gpu":gpu,"host":platform.node(),"inventory":inventory,"torch":torch.__version__}
        S.save(run/"result.json",record);state["jobs_complete"].append(job["id"]);S.emit("restart_job_complete",**record)
    state.update(status="complete",completed_utc=S.utc(),worker_elapsed_s=time.perf_counter()-worker_started);S.save(DEST/f"worker{gpu}.json",state);S.emit("restart_worker_complete",gpu=gpu,seconds=state["worker_elapsed_s"])


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["freeze","prepare","preflight","worker"]);parser.add_argument("--gpu",type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=="freeze":freeze()
    elif args.command=="prepare":prepare()
    elif args.command=="preflight":preflight(args.gpu)
    else:worker(args.gpu)
