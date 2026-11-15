"""Large fresh-cohort comparison of native initialization with lambda0/+1."""
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
import binary_preserving as B

DEST=S.OUT/"native_relaxation"
BASE=2026111504
N=16384
BATCH=64
ARMS=("native_lambda0","native_lambda1")
STRENGTHS=(0.,1.)


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


def native_initial(indices):
    return np.full((len(indices),90,90),3.,np.float32)


def initial_arms(full):
    return [full,full]


def make_one(index):
    tok,label,distance,opened,eligible=S.maze(90,.20,np.random.SeedSequence([BASE,90,20,index]))
    return tok,label,distance,opened,eligible


def freeze():
    DEST.mkdir(parents=True,exist_ok=True);parent=json.loads((S.HERE/"binary_relaxation-protocol.json").read_text());gate_path=S.OUT/"binary_preserving/validation.json";gate=json.loads(gate_path.read_text());assert gate["status"]=="passed" and gate["helper_sha256"]==S.sha(B.__file__)
    protocol=dict(parent,id="native-initialization-binary-preserving-relaxation",frozen_utc=S.utc(),status="frozen_before_data_and_outcomes",execution_script_sha256=S.sha(__file__),core_script_sha256=S.sha(S.__file__))
    protocol["cohort"]=dict(parent["cohort"],base_seed=BASE,n_unique=N,shards=64)
    protocol["initialization"]={"arms":list(ARMS),"source":"Every cell exactly float32(3), both arms, all inputs/models. Exact compressed constant initial arrays retained; no Gaussian arm.","numpy_version":np.__version__,"development_tuning":False}
    protocol["continuous_intervention"]=dict(parent["continuous_intervention"],strengths=list(STRENGTHS))
    protocol["primary"]={"id":"H2r","hypothesis":"Native u3 with compensated lambda+1 improves joint-three-model valid-shortest success by at least0.05percentage point over native u3/lambda0.","target_pp":.05,"unit":"independent input; fixed-model repeats are not independent trials","decision":"Paired gain/loss events, fixed N=16384. One-sided95% net-gain lower is CP gain-lower(alpha=.025) minus CP loss-upper(alpha=.025), upper analogously. Support if lower>=0.05pp; falsify if upper<0.05pp; otherwise inconclusive. Report exact losses and gains; no adjustment, stopping or cap extensions after outcomes."}
    protocol["secondary"]="Per-model success, native certificate margins, endpoint/branched/non-feasible outputs, step/runtime changes and original/modified continuous trajectories descriptive. No lambda sweep or Gaussian arm. No oracle selection or adaptive rescue."
    protocol["budget"]={"total_gpu_minute_envelope":20,"per_worker_wall_seconds":600,"rationale":"Previous4096-input four-arm compensated study took2.72summedGPU-minutes.4xinputs andhalf thearms imply~5.44minutes plusextra dataset/model loading/serialization.","watchdog":"Check every64-input batch and before eachjob; preserve partial output; no sample/cap changes."}
    protocol["timing"]="Both arms eta.5 cap8192; alternate arm order by job/batch, CUDA-synchronized answer timings. Shared native hidden readout. Final energy/margin/residual recording separately in diagnostic_seconds; firstbatch of shard0 includes prespecified trace diagnostics in answer timing."
    protocol["retained_evidence"]="Exact constant initial arrays, all final logits/predictions/flags/steps, original binary energy/minimum free flip margins, original/modified continuous energy and residual. First8inputs of first64batch in shard0 for eachmodel/arm traced every16updates."
    protocol["preflight"]="Native u3/lambda0 exact original-solver and full-logit parity; trace/nontrace parity at lambda0/+1; existing exhaustive binary-energy/flip/finite-difference invariance gate retained."
    protocol["jobs"]=[{"id":f"shard{shard:02d}-s{seed}","shard":shard,"seed":seed,"gpu":(shard*3+seed)%2} for shard in range(64) for seed in range(3)]
    pp=S.HERE/"native_relaxation-protocol.json"
    if pp.exists():raise FileExistsError(pp)
    S.save(pp,protocol);S.emit("native_relaxation_protocol_frozen",sha256=S.sha(pp),unique_inputs=N,jobs=len(protocol["jobs"]))


def prepare():
    pp=S.HERE/"native_relaxation-protocol.json";protocol=json.loads(pp.read_text());assert protocol["execution_script_sha256"]==S.sha(__file__)
    datasets=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(64):
            path=DEST/f"shard{shard:02d}.npz"
            if path.exists():raise FileExistsError(path)
            indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8))
            np.savez_compressed(path,initial_native=native_initial(indices),tokens=np.stack([x[0] for x in items]),labels=np.stack([x[1] for x in items]),shortest_edges=np.array([x[2] for x in items]),opened_loops=np.array([x[3] for x in items]),eligible_walls=np.array([x[4] for x in items]),indices=indices)
            datasets.append({"shard":shard,"n":256,"path":str(path.relative_to(S.ROOT)),"sha256":S.sha(path)})
            S.emit("restart_data_prepared",shard=shard,inputs_done=(shard+1)*256,seconds=time.perf_counter()-started)
    S.save(DEST/"datasets.json",{"protocol_sha256":S.sha(pp),"numpy":np.__version__,"datasets":datasets,"seconds":time.perf_counter()-started})


def solve(model,t,a,u0,cap,strength=0.,trace=False):
    torch,_,_=S.imports();u=u0.clone();n=len(t);done=torch.zeros(n,dtype=torch.bool,device=t.device);finite=torch.ones_like(done)
    used=torch.full((n,),cap,dtype=torch.long,device=t.device);history=[]
    def safe(values):return [float(v) if math.isfinite(float(v)) else None for v in values.detach().cpu().tolist()]
    with torch.no_grad():
        for k in range(1,cap+1):
            new=u+.5*(B.field(model,t,torch.sigmoid(u),a,strength)-u)
            u=torch.where((done|~finite)[:,None,None],u,new)
            if k%16==0:
                finite&=torch.isfinite(u).flatten(1).all(1)
                ok=(model.flip_delta(t,(u>0).to(a.dtype),a)>=-1e-6).flatten(1).all(1)&finite
                used=torch.where(ok&~done,torch.full_like(used,k),used);done|=ok
                if trace:
                    p=torch.sigmoid(u);base=model.energy(t,p,a);modified=B.energy(model,t,p,a,strength);margin=model.flip_delta(t,(u>0).to(a.dtype),a).masked_fill(~model.free(t),float("inf")).flatten(1).amin(1)
                    residual=(B.field(model,t,p,a,strength)-u).abs().masked_fill(~model.free(t),0).flatten(1).amax(1)
                    history.append({"update":k,"original_continuous_energy":safe(base[:8]),"modified_continuous_energy":safe(modified[:8]),"minimum_original_binary_flip_margin":safe(margin[:8]),"modified_field_residual":safe(residual[:8]),"stopped":done[:8].tolist(),"finite":finite[:8].tolist()})
                if (done|~finite).all():break
    return (u>0).to(a.dtype),used,done,finite,u,history


def preflight(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,M=S.imports();model=S.model_for(0,90,"original","cuda")
    items=[S.maze(90,.2,np.random.SeedSequence([2026111592,i])) for i in range(64)]
    t=torch.tensor(np.stack([x[0] for x in items]).astype(np.int64),device="cuda");checks=[]
    with torch.no_grad():
        psi,_=model.settle(t);a=model.couplings(t,psi)
        for cap in (8192,):
            native=M.descend(model,t,a,steps=cap,eta=.5,margin=3.,certify_every=16)
            fresh=solve(model,t,a,torch.full_like(a[:,0],3.),cap)
            assert all(torch.equal(native[i],fresh[i]) for i in range(3)) and fresh[3].all()
            checks.append({"cap":cap,"native_parity":True,"steps":fresh[1].tolist()})
    import sign_intervention as previous
    with torch.no_grad():
        u=torch.tensor(native_initial(np.arange(64)+1000000),device="cuda")
        reference=previous.solve(model,t,a,u,8192);zero=solve(model,t,a,u,8192,0.)
        assert all(torch.equal(reference[i],zero[i]) for i in range(5))
        for strength in [0.,1.]:
            plain=solve(model,t,a,u,8192,strength);traced=solve(model,t,a,u,8192,strength,True)
            assert all(torch.equal(plain[i],traced[i]) for i in range(5))
            checks.append({"lambda":strength,"trace_parity":True,"zero_native_full_logit_parity":True if strength==0 else None})
    tok=np.zeros((9,9),np.uint8);tok[0:5,0]=1;tok[4,0:5]=1;tok[0:5,4]=1;tok[0,0]=2;tok[0,4]=3
    pred=tok>0;assert feasible(tok,pred)
    fixtures=["valid non-shortestness-agnostic detour"]
    for name,coordinate in [("missing endpoint",(0,0)),("wall selection",(8,8)),("disconnected extra cell",(7,7))]:
        p=pred.copy();p[coordinate]=not p[coordinate];assert not feasible(tok,p);fixtures.append(name)
    branch=tok.copy();branch[4,5]=1;p=branch>0;assert not feasible(branch,p);fixtures.append("branch")
    cycle=tok.copy();cycle[6,6:9]=1;cycle[8,6:9]=1;cycle[6:9,6]=1;cycle[6:9,8]=1;assert not feasible(cycle,cycle>0);fixtures.append("disconnected degree-two cycle")
    S.save(DEST/"preflight.json",{"status":"passed","utc":S.utc(),"protocol_sha256":S.sha(S.HERE/"native_relaxation-protocol.json"),"execution_sha256":S.sha(__file__),"native_checks":checks,"feasibility_fixtures":fixtures,"continuous_helper_gate_sha256":S.sha(S.OUT/"binary_preserving/validation.json")})
    S.emit("restart_preflight_passed",checks=checks)


def worker(gpu):
    lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True);assert inventory.split(",")[1].strip() not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,_=S.imports();from cram.maze_learned import gather
    pp=S.HERE/"native_relaxation-protocol.json";protocol=json.loads(pp.read_text());assert protocol["execution_script_sha256"]==S.sha(__file__) and protocol["core_script_sha256"]==S.sha(S.__file__)
    assert protocol["initialization"]["numpy_version"]==np.__version__
    assert protocol["continuous_intervention"]["helper_sha256"]==S.sha(B.__file__)
    assert S.sha(S.ROOT/protocol["continuous_intervention"]["gate_path"])==protocol["continuous_intervention"]["gate_sha256"]
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
        source=np.load(S.ROOT/dataset["path"]);tokens=source["tokens"];indices=source["indices"];full_initial=source["initial_native"]
        assert S.sha(S.OUT/f"models/seed{job['seed']}.pt")==protocol["models"][str(job["seed"])]["sha256"]
        model=S.model_for(job["seed"],90,"original","cuda");saved={k:[] for k in ["predictions","answer_steps","stopped","finite","energy","feasible","hidden_steps","hidden_good","final_logits","changed_cells","changed_free_cells","negative_cells","negative_free_cells","zero_cells","minimum_original_flip_margin","original_continuous_energy","modified_continuous_energy","continuous_regularizer","modified_field_residual","binary_regularizer"]}
        answer_times=np.zeros(2);hidden_time=0.;selector_time=0.;initial_hash=hashlib.sha256();diagnostics={};started=time.perf_counter()
        state.update(current_job=job,completed_inputs=0,updated_utc=S.utc());S.save(DEST/f"worker{gpu}.json",state)
        for lo in range(0,256,BATCH):
            token=tokens[lo:lo+BATCH];t=torch.tensor(token.astype(np.int64),device="cuda")
            with torch.no_grad():
                torch.cuda.synchronize();begin=time.perf_counter();psi,ps=model.settle(t,tol=1e-6,max_steps=4000,check=8)
                lg,la,lc=model.parts(t,model.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,model.zoffs,math.inf),lc[:,None]],1),1)
                hidden_good=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1))
                a=model.couplings(t,psi);torch.cuda.synchronize();hidden_time+=time.perf_counter()-begin
                init=initial_arms(full_initial[lo:lo+BATCH])
                for value in init:initial_hash.update(value.tobytes())
                u0=[torch.tensor(value,device="cuda") for value in init]
                if lo==0:
                    for v,u in enumerate(u0):solve(model,t,a,u,32,STRENGTHS[v])
                    torch.cuda.synchronize()
                outputs={};energies={};offset=(ji+lo//BATCH)%2
                for variant in list(range(2))[offset:]+list(range(2))[:offset]:
                    torch.cuda.synchronize();begin=time.perf_counter();outputs[variant]=solve(model,t,a,u0[variant],8192,STRENGTHS[variant],trace=(job["shard"]==0 and lo==0))
                    energies[variant]=model.energy(t,outputs[variant][0],a)
                    torch.cuda.synchronize();answer_times[variant]+=time.perf_counter()-begin
                    if job["shard"]==0 and lo==0:diagnostics[ARMS[variant]]=outputs[variant][5]
                values={key:np.stack([outputs[v][k].cpu().numpy() for v in range(2)]) for k,key in enumerate(["predictions","answer_steps","stopped","finite","final_logits"])}
                values["predictions"]=values["predictions"].astype(bool);values["energy"]=np.stack([energies[v].cpu().numpy() for v in range(2)])
                hg=hidden_good.cpu().numpy();begin=time.perf_counter()
                values["feasible"]=np.array([[feasible(tok,pred) for tok,pred in zip(token,values["predictions"][v])] for v in range(2)])
                values["changed_cells"]=np.array([(value!=3.).sum((1,2)) for value in init])
                values["changed_free_cells"]=np.array([((value!=3.)&(token==1)).sum((1,2)) for value in init])
                values["negative_cells"]=np.array([(value<0).sum((1,2)) for value in init])
                values["negative_free_cells"]=np.array([((value<0)&(token==1)).sum((1,2)) for value in init])
                values["zero_cells"]=np.array([(value==0).sum((1,2)) for value in init])
                extra={key:[] for key in ["minimum_original_flip_margin","original_continuous_energy","modified_continuous_energy","continuous_regularizer","modified_field_residual","binary_regularizer"]}
                for v in range(2):
                    p=torch.sigmoid(outputs[v][4]);marg=model.flip_delta(t,outputs[v][0],a).masked_fill(~model.free(t),float("inf")).flatten(1).amin(1)
                    base=model.energy(t,p,a);reg=B.regularizer(model,t,p,STRENGTHS[v]);res=(B.field(model,t,p,a,STRENGTHS[v])-outputs[v][4]).abs().masked_fill(~model.free(t),0).flatten(1).amax(1)
                    binary_reg=B.regularizer(model,t,outputs[v][0],STRENGTHS[v]);assert (binary_reg==0).all()
                    for key,value in zip(extra,[marg,base,base+reg,reg,res,binary_reg]):extra[key].append(value.cpu().numpy())
                values.update({key:np.stack(value) for key,value in extra.items()})
                selector_time+=time.perf_counter()-begin
                values["hidden_steps"]=ps.cpu().numpy();values["hidden_good"]=hg
                for key,value in values.items():saved[key].append(value)
            state.update(completed_inputs=lo+BATCH,updated_utc=S.utc(),elapsed_s=time.perf_counter()-started,worker_elapsed_s=time.perf_counter()-worker_started)
            S.save(DEST/f"worker{gpu}.json",state);S.emit("restart_batch_done",gpu=gpu,job=job["id"],inputs=lo+BATCH)
            if time.perf_counter()-worker_started>600:
                np.savez_compressed(run/"partial.npz",**{k:np.concatenate(v,axis=1 if v[0].ndim>=2 else 0) for k,v in saved.items()});raise TimeoutError("Frozen per-worker 600s envelope reached; partial preserved")
        arrays={k:np.concatenate(v,axis=1 if v[0].ndim>=2 else 0) for k,v in saved.items()};np.savez_compressed(run/"outputs.npz",**arrays)
        if diagnostics:S.save(run/"diagnostics.json",diagnostics)
        record={"diagnostics_sha256":S.sha(run/"diagnostics.json") if diagnostics else None,"status":"complete","utc":S.utc(),"job":job,"n":256,"protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"dataset_sha256":dataset["sha256"],"checkpoint_sha256":protocol["models"][str(job["seed"])]["sha256"],
            "outputs_sha256":S.sha(run/"outputs.npz"),"initialization_sha256":initial_hash.hexdigest(),"hidden_seconds_shared":hidden_time,"answer_seconds_by_variant":answer_times.tolist(),"diagnostic_seconds":selector_time,"wall_seconds":time.perf_counter()-started,"gpu":gpu,"host":platform.node(),"inventory":inventory,"torch":torch.__version__}
        S.save(run/"result.json",record);state["jobs_complete"].append(job["id"]);S.emit("restart_job_complete",**record)
    state.update(status="complete",completed_utc=S.utc(),worker_elapsed_s=time.perf_counter()-worker_started);S.save(DEST/f"worker{gpu}.json",state);S.emit("restart_worker_complete",gpu=gpu,seconds=state["worker_elapsed_s"])


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["freeze","prepare","preflight","worker"]);parser.add_argument("--gpu",type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=="freeze":freeze()
    elif args.command=="prepare":prepare()
    elif args.command=="preflight":preflight(args.gpu)
    else:worker(args.gpu)
