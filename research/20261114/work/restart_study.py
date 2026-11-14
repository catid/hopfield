"""Fixed-budget random restarts with energy-only and feasibility-screen selectors."""
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

DEST=S.OUT/"restart"
BASE=2026111421
NOISE_BASE=2026111422
N=4096


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


def initial(indices,chain):
    return np.stack([3.+np.random.default_rng(np.random.SeedSequence([NOISE_BASE,int(i),chain])).standard_normal((90,90),dtype=np.float32) for i in indices])


def make_one(index):
    tok,label,distance,opened,eligible=S.maze(90,.20,np.random.SeedSequence([BASE,90,20,index]))
    return tok,label,distance,opened,eligible


def freeze():
    DEST.mkdir(parents=True,exist_ok=True);parent=json.loads((S.HERE/"protocol.json").read_text())
    protocol={"id":"fresh-cyclic-fixed-budget-restarts","frozen_utc":S.utc(),"status":"frozen_before_data_and_outcomes",
        "execution_script_sha256":S.sha(__file__),"core_script_sha256":S.sha(S.__file__),"models":parent["models"],"source_sha256":parent["source_sha256"],
        "cohort":{"base_seed":BASE,"n_unique":N,"length":90,"loop_percent":20,"generator":"study.maze; SeedSequence([base,90,20,index]); index 0..4095","shards":16,"per_shard":256},
        "initialization":{"native":"u=3 for every cell","perturbed":"u=3+Z, Z iid standard_normal float32 on all cells; four independent chains indexed 0..3","standard_deviation":1.,"source":"NumPy default_rng(SeedSequence([noise_base,input_index,chain])).standard_normal((90,90),dtype=float32)","noise_base":NOISE_BASE,"shared_across_models":True,"numpy_version":np.__version__,"selection_or_tuning_on_development_graph":False},
        "inference":{"eta":.5,"baseline_cap":8192,"restart_chains":4,"per_restart_cap":2048,"check_every":16,"margin_tolerance":-1e-6,"hidden_tol":1e-6,"hidden_cap":4000,"hidden_check":8,"float":"float32","batch_size":32,"hidden_state":"identical cached native readout shared by baseline and four candidates"},
        "selectors":{"eligibility":"hidden converged, all answer logits finite, finite learned energy, and native single-flip certificate passed before cap; otherwise ineligible","energy_only":"Lowest native discrete learned energy among eligible candidates, ties first chain index; none -> explicit no-eligible failure","feasibility_screen":"Same candidates and eligibility; additionally require selected nodes form one endpoint-connected nonbranching simple S-G path in supplied graph. Lowest learned energy among feasible candidates, ties first chain index; none -> explicit no-feasible failure","no_oracle":"Feasibility has only token graph and candidate board; never labels, true shortest distance, BFS from original graph, or ground-truth best-of-N. Exact shortestness evaluated only after outputs are retained."},
        "primary":{"id":"H2m","hypothesis":"Feasibility screening the same four candidates raises joint-three-model shortest-path success by at least 0.1 percentage point over learned-energy selection on fresh loop20 inputs.","threshold_pp":.1,"unit":"independent input; fixed-model repeats are not independent trials","monotonicity":"If energy-only selects a successful shortest path it is feasible and remains the lowest-energy feasible candidate, with identical tie breaking. Thus feasibility screening cannot lose a successful case. Verify no losses in audit.","decision":"Use exact one-sided 95% Clopper-Pearson bounds for joint-success gain events. Supported if lower >= 0.1pp; falsified if upper < 0.1pp; otherwise inconclusive. No gain yields upper 0.073111pp at N=4096."},
        "secondary":"Joint and per-model success versus native single initialization at same maximum 8192 answer updates per input; report paired gains/losses and uncertainty descriptively. Both initialization distribution and per-chain depth change. Also all candidate outcomes, no-eligible/no-feasible, finite/hidden/certificate failures, energy ranking, summed updates and matched wall time.",
        "budget":{"total_gpu_minute_envelope":40,"per_worker_wall_seconds":1200,"rationale":"Prior 1536-input four-arm study used ~226 total checked-answer seconds across all fixed models/etas; this study has 3.33x as many answer candidates but same 90-grid and eta .5. 40 GPU-minutes allows preparation/serialization overhead without adaptive extra compute.","watchdog":"Check after every 32-input batch and before each job; retain partial artifacts; no cap extensions or extra inputs after outcomes."},
        "timing":"CUDA-synchronized answer timings, 32-step warmup each variant; cyclically rotate five-variant order by job and batch. Hidden preparation separately shared. CPU selectors/serialization separately recorded. No claim this implements paper-native restart selection.",
        "preflight":"Separate reserved inputs check constant-initialization solver exact native parity at both caps and finite checks. Feasibility fixtures include valid detour, branches, walls, missing endpoints, and disconnected cycle. No developmental amplitude tuning.",
        "control":"BP AdamW: outside scope of frozen-model inference study; no new training baseline. Native single initialization is the inference control.",
        "jobs":[{"id":f"shard{shard:02d}-s{seed}","shard":shard,"seed":seed,"gpu":(shard*3+seed)%2} for shard in range(16) for seed in range(3)]}
    pp=S.HERE/"restart-protocol.json"
    if pp.exists():raise FileExistsError(pp)
    S.save(pp,protocol);S.emit("restart_protocol_frozen",sha256=S.sha(pp),unique_inputs=N,jobs=len(protocol["jobs"]))


def prepare():
    pp=S.HERE/"restart-protocol.json";protocol=json.loads(pp.read_text());assert protocol["execution_script_sha256"]==S.sha(__file__)
    datasets=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(16):
            path=DEST/f"shard{shard:02d}.npz"
            if path.exists():raise FileExistsError(path)
            indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8))
            np.savez_compressed(path,tokens=np.stack([x[0] for x in items]),labels=np.stack([x[1] for x in items]),shortest_edges=np.array([x[2] for x in items]),opened_loops=np.array([x[3] for x in items]),eligible_walls=np.array([x[4] for x in items]),indices=indices)
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
    return (u>0).to(a.dtype),used,done,finite


def preflight(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,M=S.imports();model=S.model_for(0,90,"original","cuda")
    items=[S.maze(90,.2,np.random.SeedSequence([2026111496,i])) for i in range(8)]
    t=torch.tensor(np.stack([x[0] for x in items]).astype(np.int64),device="cuda");checks=[]
    with torch.no_grad():
        psi,_=model.settle(t);a=model.couplings(t,psi)
        for cap in (2048,8192):
            native=M.descend(model,t,a,steps=cap,eta=.5,margin=3.,certify_every=16)
            fresh=solve(model,t,a,torch.full_like(a[:,0],3.),cap)
            assert all(torch.equal(native[i],fresh[i]) for i in range(3)) and fresh[3].all()
            checks.append({"cap":cap,"native_parity":True,"steps":fresh[1].tolist()})
    tok=np.zeros((9,9),np.uint8);tok[0:5,0]=1;tok[4,0:5]=1;tok[0:5,4]=1;tok[0,0]=2;tok[0,4]=3
    pred=tok>0;assert feasible(tok,pred)
    fixtures=["valid non-shortestness-agnostic detour"]
    for name,coordinate in [("missing endpoint",(0,0)),("wall selection",(8,8)),("disconnected extra cell",(7,7))]:
        p=pred.copy();p[coordinate]=not p[coordinate];assert not feasible(tok,p);fixtures.append(name)
    branch=tok.copy();branch[4,5]=1;p=branch>0;assert not feasible(branch,p);fixtures.append("branch")
    cycle=tok.copy();cycle[6,6:9]=1;cycle[8,6:9]=1;cycle[6:9,6]=1;cycle[6:9,8]=1;assert not feasible(cycle,cycle>0);fixtures.append("disconnected degree-two cycle")
    S.save(DEST/"preflight.json",{"status":"passed","utc":S.utc(),"protocol_sha256":S.sha(S.HERE/"restart-protocol.json"),"execution_sha256":S.sha(__file__),"native_checks":checks,"feasibility_fixtures":fixtures})
    S.emit("restart_preflight_passed",checks=checks)


def worker(gpu):
    lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True);assert inventory.split(",")[1].strip() not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,_=S.imports();from cram.maze_learned import gather
    pp=S.HERE/"restart-protocol.json";protocol=json.loads(pp.read_text());assert protocol["execution_script_sha256"]==S.sha(__file__) and protocol["core_script_sha256"]==S.sha(S.__file__)
    assert protocol["initialization"]["numpy_version"]==np.__version__
    pre=json.loads((DEST/"preflight.json").read_text());assert pre["status"]=="passed" and pre["protocol_sha256"]==S.sha(pp)
    datasets=json.loads((DEST/"datasets.json").read_text());assert datasets["protocol_sha256"]==S.sha(pp)
    for path,h in protocol["source_sha256"].items():assert S.sha(S.ROOT/path)==h
    state={"status":"running","pid":os.getpid(),"host":platform.node(),"gpu":gpu,"started_utc":S.utc(),"jobs_complete":[]};worker_started=time.perf_counter()
    for ji,job in enumerate(protocol["jobs"]):
        if job["gpu"]!=gpu:continue
        if time.perf_counter()-worker_started>1200:raise TimeoutError("Frozen per-worker 1200s envelope reached")
        run=DEST/"runs"/job["id"];run.mkdir(parents=True,exist_ok=True)
        if (run/"result.json").exists():raise FileExistsError(run)
        dataset=datasets["datasets"][job["shard"]];assert S.sha(S.ROOT/dataset["path"])==dataset["sha256"]
        source=np.load(S.ROOT/dataset["path"]);tokens=source["tokens"];indices=source["indices"]
        assert S.sha(S.OUT/f"models/seed{job['seed']}.pt")==protocol["models"][str(job["seed"])]["sha256"]
        model=S.model_for(job["seed"],90,"original","cuda");saved={k:[] for k in ["predictions","answer_steps","stopped","finite","energy","feasible","hidden_steps","hidden_good","selected_energy","selected_screen"]}
        answer_times=np.zeros(5);hidden_time=0.;selector_time=0.;initial_hash=hashlib.sha256();started=time.perf_counter()
        state.update(current_job=job,completed_inputs=0,updated_utc=S.utc());S.save(DEST/f"worker{gpu}.json",state)
        for lo in range(0,256,32):
            token=tokens[lo:lo+32];t=torch.tensor(token.astype(np.int64),device="cuda")
            with torch.no_grad():
                torch.cuda.synchronize();begin=time.perf_counter();psi,ps=model.settle(t,tol=1e-6,max_steps=4000,check=8)
                lg,la,lc=model.parts(t,model.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,model.zoffs,math.inf),lc[:,None]],1),1)
                hidden_good=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1))
                a=model.couplings(t,psi);torch.cuda.synchronize();hidden_time+=time.perf_counter()-begin
                init=[np.full(token.shape,3.,np.float32)]+[initial(indices[lo:lo+32],chain) for chain in range(4)]
                for value in init[1:]:initial_hash.update(value.tobytes())
                u0=[torch.tensor(value,device="cuda") for value in init]
                if lo==0:
                    for u in u0:solve(model,t,a,u,32)
                    torch.cuda.synchronize()
                outputs={};energies={};offset=(ji+lo//32)%5
                for variant in list(range(5))[offset:]+list(range(5))[:offset]:
                    torch.cuda.synchronize();begin=time.perf_counter();outputs[variant]=solve(model,t,a,u0[variant],8192 if variant==0 else 2048)
                    energies[variant]=model.energy(t,outputs[variant][0],a)
                    torch.cuda.synchronize();answer_times[variant]+=time.perf_counter()-begin
                values={key:np.stack([outputs[v][k].cpu().numpy() for v in range(5)]) for k,key in enumerate(["predictions","answer_steps","stopped","finite"])}
                values["predictions"]=values["predictions"].astype(bool);values["energy"]=np.stack([energies[v].cpu().numpy() for v in range(5)])
                hg=hidden_good.cpu().numpy();begin=time.perf_counter()
                values["feasible"]=np.array([[feasible(tok,pred) for tok,pred in zip(token,values["predictions"][v])] for v in range(5)])
                eligible=values["stopped"][1:]&values["finite"][1:]&np.isfinite(values["energy"][1:])&hg[None]
                for key,mask in [("selected_energy",eligible),("selected_screen",eligible&values["feasible"][1:])]:
                    ranked=np.where(mask,values["energy"][1:],np.inf);selection=ranked.argmin(0)+1;selection[~mask.any(0)]=-1;values[key]=selection
                selector_time+=time.perf_counter()-begin
                values["hidden_steps"]=ps.cpu().numpy();values["hidden_good"]=hg
                for key,value in values.items():saved[key].append(value)
            state.update(completed_inputs=lo+32,updated_utc=S.utc(),elapsed_s=time.perf_counter()-started,worker_elapsed_s=time.perf_counter()-worker_started)
            S.save(DEST/f"worker{gpu}.json",state);S.emit("restart_batch_done",gpu=gpu,job=job["id"],inputs=lo+32)
            if time.perf_counter()-worker_started>1200:
                np.savez_compressed(run/"partial.npz",**{k:np.concatenate(v,axis=1 if v[0].ndim>=2 else 0) for k,v in saved.items()});raise TimeoutError("Frozen per-worker 1200s envelope reached; partial preserved")
        arrays={k:np.concatenate(v,axis=1 if v[0].ndim>=2 else 0) for k,v in saved.items()};np.savez_compressed(run/"outputs.npz",**arrays)
        record={"status":"complete","utc":S.utc(),"job":job,"n":256,"protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"dataset_sha256":dataset["sha256"],"checkpoint_sha256":protocol["models"][str(job["seed"])]["sha256"],
            "outputs_sha256":S.sha(run/"outputs.npz"),"initialization_sha256":initial_hash.hexdigest(),"hidden_seconds_shared":hidden_time,"answer_seconds_by_variant":answer_times.tolist(),"selector_cpu_seconds":selector_time,"wall_seconds":time.perf_counter()-started,"gpu":gpu,"host":platform.node(),"inventory":inventory,"torch":torch.__version__}
        S.save(run/"result.json",record);state["jobs_complete"].append(job["id"]);S.emit("restart_job_complete",**record)
    state.update(status="complete",completed_utc=S.utc(),worker_elapsed_s=time.perf_counter()-worker_started);S.save(DEST/f"worker{gpu}.json",state);S.emit("restart_worker_complete",gpu=gpu,seconds=state["worker_elapsed_s"])


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["freeze","prepare","preflight","worker"]);parser.add_argument("--gpu",type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=="freeze":freeze()
    elif args.command=="prepare":prepare()
    elif args.command=="preflight":preflight(args.gpu)
    else:worker(args.gpu)
