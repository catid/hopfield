"""Four fixed sign interventions on paired initial Gaussian logits."""
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

DEST=S.OUT/"sign_intervention"
BASE=2026111425
NOISE_BASE=2026111426
N=2048
BATCH=64
ARMS=("native","full_gaussian","positive_only","negative_only")


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
    return [np.full_like(full,3.),full,np.where(full<0,np.float32(3.),full),np.where(full>0,np.float32(3.),full)]


def make_one(index):
    tok,label,distance,opened,eligible=S.maze(90,.20,np.random.SeedSequence([BASE,90,20,index]))
    return tok,label,distance,opened,eligible


def freeze():
    DEST.mkdir(parents=True,exist_ok=True);parent=json.loads((S.HERE/"restart-protocol.json").read_text());pilot=json.loads((DEST/"batch-preflight.json").read_text());assert pilot["chosen_batch"]==BATCH and pilot["native_answer_outputs_exact"]
    protocol=dict(parent,id="paired-initial-logit-sign-interventions",frozen_utc=S.utc(),status="frozen_before_data_and_outcomes",execution_script_sha256=S.sha(__file__),core_script_sha256=S.sha(S.__file__))
    protocol["cohort"]=dict(parent["cohort"],base_seed=BASE,n_unique=N,shards=8)
    protocol["initialization"]={"arms":list(ARMS),"source":"Gaussian float32 u=3+default_rng(SeedSequence([noise_base,input_index])).standard_normal((90,90),dtype=float32). Exact full Gaussian arrays retained with each dataset; same draw across all models and all intervention arms.","noise_base":NOISE_BASE,"standard_deviation":1.,"native":"every cell3","full_gaussian":"unchanged saved Gaussian array","positive_only":"where full<0 replace with3; otherwise retain exact full value","negative_only":"where full>0 replace with3; otherwise retain exact full value","zero_treatment":"Negative means strictly<0 and positive means strictly>0. Exactly zero entries remain zero in both sign interventions. Native remains3.","all_cells":"Operations apply to all cells; separately retain counts on free cells. Model's normal clamping unchanged.","numpy_version":np.__version__,"development_tuning":False}
    protocol["inference"]={"eta":.5,"answer_cap_each_arm":8192,"check_every":16,"margin_tolerance":-1e-6,"hidden_tol":1e-6,"hidden_cap":4000,"hidden_check":8,"float":"float32","batch_size":BATCH,"shared_hidden_readout":True,"selector":"None; single output per arm"}
    protocol.pop("selectors",None)
    protocol["primary"]={"id":"H2o","hypothesis":"Positive-only minus full-Gaussian joint-three-model valid-shortest success gain is at least10percentage points.","target_pp":10.,"unit":"independent input; fixed-model repeats are not independent trials","decision":"Paired gains and losses. Form one-sided95% net-gain lower bound as CP gain-lower(alpha=.025) minus CP loss-upper(alpha=.025); upper analogously. Support if lower>=10pp; falsify if upper<10pp; otherwise inconclusive. No conditioned successful-only denominator."}
    protocol["secondary"]="Negative-only/full-Gaussian joint failure overlap, intersection/union, conditional overlap, and all arms versus native descriptive. All finite/certificate/hidden failures retained. No ground-truth selector or adaptive rescue."
    protocol["budget"]={"total_gpu_minute_envelope":20,"per_worker_wall_seconds":600,"rationale":"Previous4096-input native+three study completed in4.05summedGPU-minutes. This study uses2048inputs and4arms; retains exact final logits, which adds serialization cost.","watchdog":"Check after every64-input batch and before eachjob; preserve partial outputs, no cap/sample extensions."}
    protocol["timing"]="Every arm cap8192; rotate four-arm order by job and batch, CUDA-synchronized answer times. Identical shared hidden state. Batch64 chosen from reserved native throughput test: same64inputs, exact output/step/flag parity,5timing repetitions; no primary outcomes used."
    protocol["batch_preflight"]={"path":str((DEST/"batch-preflight.json").relative_to(S.ROOT)),"sha256":S.sha(DEST/"batch-preflight.json"),"ratio_32_over_64":pilot["ratio_32_over_64"]}
    protocol["retained_evidence"]="Exact float32 full Gaussian initial arrays; exact final logits for every arm/model/input; all predicted boards, steps, finite/certificate flags, energies and changed/negative/zero cell counts, including free-cell counts. Sign-arm initialization exactly reconstructible with saved np.where conditions."
    protocol["preflight"]="Exact native8192 parity on reserved64inputs; pure initialization fixture verifies strict negative/positive/zero transforms. No selected-case tuning."
    protocol["jobs"]=[{"id":f"shard{shard:02d}-s{seed}","shard":shard,"seed":seed,"gpu":(shard*3+seed)%2} for shard in range(8) for seed in range(3)]
    pp=S.HERE/"sign_intervention-protocol.json"
    if pp.exists():raise FileExistsError(pp)
    S.save(pp,protocol);S.emit("sign_protocol_frozen",sha256=S.sha(pp),unique_inputs=N,jobs=len(protocol["jobs"]))


def prepare():
    pp=S.HERE/"sign_intervention-protocol.json";protocol=json.loads(pp.read_text());assert protocol["execution_script_sha256"]==S.sha(__file__)
    datasets=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(8):
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
    items=[S.maze(90,.2,np.random.SeedSequence([2026111493,i])) for i in range(64)]
    t=torch.tensor(np.stack([x[0] for x in items]).astype(np.int64),device="cuda");checks=[]
    with torch.no_grad():
        psi,_=model.settle(t);a=model.couplings(t,psi)
        for cap in (8192,):
            native=M.descend(model,t,a,steps=cap,eta=.5,margin=3.,certify_every=16)
            fresh=solve(model,t,a,torch.full_like(a[:,0],3.),cap)
            assert all(torch.equal(native[i],fresh[i]) for i in range(3)) and fresh[3].all()
            checks.append({"cap":cap,"native_parity":True,"steps":fresh[1].tolist()})
    fixture=np.array([-1.,0.,1.,3.],np.float32);arms=initial_arms(fixture)
    assert np.array_equal(arms[2],np.array([3.,0.,1.,3.],np.float32)) and np.array_equal(arms[3],np.array([-1.,0.,3.,3.],np.float32))
    tok=np.zeros((9,9),np.uint8);tok[0:5,0]=1;tok[4,0:5]=1;tok[0:5,4]=1;tok[0,0]=2;tok[0,4]=3
    pred=tok>0;assert feasible(tok,pred)
    fixtures=["valid non-shortestness-agnostic detour"]
    for name,coordinate in [("missing endpoint",(0,0)),("wall selection",(8,8)),("disconnected extra cell",(7,7))]:
        p=pred.copy();p[coordinate]=not p[coordinate];assert not feasible(tok,p);fixtures.append(name)
    branch=tok.copy();branch[4,5]=1;p=branch>0;assert not feasible(branch,p);fixtures.append("branch")
    cycle=tok.copy();cycle[6,6:9]=1;cycle[8,6:9]=1;cycle[6:9,6]=1;cycle[6:9,8]=1;assert not feasible(cycle,cycle>0);fixtures.append("disconnected degree-two cycle")
    S.save(DEST/"preflight.json",{"status":"passed","utc":S.utc(),"protocol_sha256":S.sha(S.HERE/"sign_intervention-protocol.json"),"execution_sha256":S.sha(__file__),"native_checks":checks,"feasibility_fixtures":fixtures,"strict_sign_and_zero_fixture":True})
    S.emit("restart_preflight_passed",checks=checks)


def worker(gpu):
    lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True);assert inventory.split(",")[1].strip() not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,_=S.imports();from cram.maze_learned import gather
    pp=S.HERE/"sign_intervention-protocol.json";protocol=json.loads(pp.read_text());assert protocol["execution_script_sha256"]==S.sha(__file__) and protocol["core_script_sha256"]==S.sha(S.__file__)
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
        model=S.model_for(job["seed"],90,"original","cuda");saved={k:[] for k in ["predictions","answer_steps","stopped","finite","energy","feasible","hidden_steps","hidden_good","final_logits","changed_cells","changed_free_cells","negative_cells","negative_free_cells","zero_cells"]}
        answer_times=np.zeros(4);hidden_time=0.;selector_time=0.;initial_hash=hashlib.sha256();started=time.perf_counter()
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
                outputs={};energies={};offset=(ji+lo//BATCH)%4
                for variant in list(range(4))[offset:]+list(range(4))[:offset]:
                    torch.cuda.synchronize();begin=time.perf_counter();outputs[variant]=solve(model,t,a,u0[variant],8192)
                    energies[variant]=model.energy(t,outputs[variant][0],a)
                    torch.cuda.synchronize();answer_times[variant]+=time.perf_counter()-begin
                values={key:np.stack([outputs[v][k].cpu().numpy() for v in range(4)]) for k,key in enumerate(["predictions","answer_steps","stopped","finite","final_logits"])}
                values["predictions"]=values["predictions"].astype(bool);values["energy"]=np.stack([energies[v].cpu().numpy() for v in range(4)])
                hg=hidden_good.cpu().numpy();begin=time.perf_counter()
                values["feasible"]=np.array([[feasible(tok,pred) for tok,pred in zip(token,values["predictions"][v])] for v in range(4)])
                values["changed_cells"]=np.array([(value!=3.).sum((1,2)) for value in init])
                values["changed_free_cells"]=np.array([((value!=3.)&(token==1)).sum((1,2)) for value in init])
                values["negative_cells"]=np.array([(value<0).sum((1,2)) for value in init])
                values["negative_free_cells"]=np.array([((value<0)&(token==1)).sum((1,2)) for value in init])
                values["zero_cells"]=np.array([(value==0).sum((1,2)) for value in init])
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
