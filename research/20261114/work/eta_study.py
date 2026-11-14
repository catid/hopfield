"""Frozen eta0.2/0.5 paired inference and separate trajectory diagnostics."""
import argparse
import fcntl
import json
import math
import os
import platform
import subprocess
import time
import numpy as np
import study as S


def prepare():
    original=json.loads((S.HERE/"protocol.json").read_text());dest=S.OUT/"eta";dest.mkdir(parents=True,exist_ok=True)
    datasets=[];jobs=[];base=2026111407
    for length in [60,90]:
        for rate in [0,5,20]:
            items=[S.maze(length,rate/100,np.random.SeedSequence([base,length,rate,i])) for i in range(128)]
            path=dest/f"L{length}-loop{rate}.npz"
            if path.exists():raise FileExistsError(path)
            np.savez_compressed(path,tokens=np.stack([x[0] for x in items]),labels=np.stack([x[1] for x in items]),shortest_edges=np.asarray([x[2] for x in items]))
            datasets.append({"length":length,"loop_percent":rate,"n":128,"base_seed":base,"path":str(path.relative_to(S.ROOT)),"sha256":S.sha(path)})
            for seed in range(3):jobs.append({"id":f"L{length}-loop{rate}-s{seed}","length":length,"loop_percent":rate,"seed":seed})
    protocol=dict(original,id="eta-paired-inference",frozen_utc=S.utc(),status="frozen_before_outcomes",datasets=datasets,jobs=jobs,
        execution_script_sha256=S.sha(__file__),core_script_sha256=S.sha(S.__file__),etas=[.2,.5],H2=None,H2b=None,
        H2g="Eta0.5 reduces the median answer updates by at least2x versus eta0.2 at each60/90size, with no observed joint-model success loss and a one-sided95% loss-probability upper bound<=1percentage point.",
        decision="384unique mazes per size, stratified equally across tree/5%/20%loops. Average three fixed-model steps within each maze, then ratio of medians.10000paired stratified-maze bootstrap replicates: support speed threshold if95%lower>=2; falsify if upper<2; otherwise inconclusive. Accuracy support requires zero observed losses and exact one-sided95%loss upper<=1pp; any observed loss contradicts no-observed-loss clause. Model repeats are not independent samples.",
        power_note="With384unique mazes per size, zero lost solves gives exact one-sided95%upper0.7771pp, below1pp margin. This is a generator-distribution bound, not arbitrary-model generalization.",
        inference=dict(original["inference"],answer_max_steps=8192,answer_eta=[.2,.5]),
        timing="Both arms on sameGPU per job, same cached hiddenstate/couplings, warmup32updates for each arm; arm order alternates deterministically with jobindex+batchindex. CUDA-synchronized native answer calls timed; hidden preparation reported separately and shared. No H200 comparison.",
        diagnostic_subset="First8maze indices of each size/rate cell, modelseed0. Separate instrumented replay of the original32-maze batch retains energies every16updates, free-logitresidual, and per-step two-cycle detector. Replay boards/steps/stops must match native primary run. Replays excluded from accuracy sample count and native timing.",
        prefix_check="Bounded reserved-generator cases compare independent2048cap to8192firststop<=2048. Main2048success derived from saved stopped-valid flags andfirststopstep; no main rerun.",
        confirmation="New generator2026111407; no additional eta values or adaptation if0.5 fails.")
    path=S.HERE/"eta-protocol.json"
    if path.exists():raise FileExistsError(path)
    S.save(path,protocol);S.emit("eta_frozen",sha256=S.sha(path),jobs=len(jobs))


def trace(model,t,a,eta):
    torch,_,_=S.imports();n=len(t);u=torch.full(t.shape,3.,device=t.device,dtype=a.dtype)
    stopped=torch.zeros(n,dtype=torch.bool,device=t.device);used=torch.full((n,),8192,dtype=torch.long,device=t.device)
    free=t==1;history=[];old1=None;old2=None;cycle=torch.zeros(n,dtype=torch.bool,device=t.device)
    with torch.no_grad():
        history.append({"update":0,"energy":model.energy(t,torch.sigmoid(u),a)[:8].tolist()})
        for k in range(1,8193):
            field=model.field(t,torch.sigmoid(u),a);new=u+eta*(field-u);u=torch.where(stopped[:,None,None],u,new)
            if old2 is not None:
                one=torch.where(free,(u-old1).abs(),0).flatten(1).amax(1)
                two=torch.where(free,(u-old2).abs(),0).flatten(1).amax(1)
                cycle|=(two<1e-5)&(one>1e-3)&~stopped
            old2=old1;old1=u.clone()
            if k%16==0:
                h=(u>0).to(a.dtype);ok=(model.flip_delta(t,h,a)>=-1e-6).flatten(1).all(1)
                used=torch.where(ok&~stopped,torch.full_like(used,k),used);stopped|=ok
                residual=torch.where(free,(model.field(t,torch.sigmoid(u),a)-u).abs(),0).flatten(1).amax(1)
                history.append({"update":k,"energy":model.energy(t,torch.sigmoid(u),a)[:8].tolist(),"residual":residual[:8].tolist(),"stopped":stopped[:8].tolist()})
                if stopped.all():break
    return (u>0).to(a.dtype),used,stopped,{"history":history,"two_cycle_detected":cycle[:8].tolist()}


def preflight(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,M=S.imports();model=S.model_for(0,60,"original","cuda")
    items=[S.maze(60,0,np.random.SeedSequence([2026111499,i])) for i in range(4)]
    t=torch.as_tensor(np.stack([x[0] for x in items]).astype(np.int64),device="cuda");checks=[]
    with torch.no_grad():
        psi,_=model.settle(t);a=model.couplings(t,psi)
        for eta in [.2,.5]:
            pred,used,done=M.descend(model,t,a,steps=8192,eta=eta,margin=3.,certify_every=16)
            q,v,d,detail=trace(model,t,a,eta)
            assert torch.equal(pred,q) and torch.equal(used,v) and torch.equal(done,d)
            cp,cu,cd=M.descend(model,t,a,steps=2048,eta=eta,margin=3.,certify_every=16)
            assert torch.equal(cd,done&(used<=2048)) and torch.equal(cu,used.clamp_max(2048))
            assert torch.equal(cp[cd],pred[cd])
            checks.append({"eta":eta,"recording_parity":True,"prefix_status_steps_parity":True,"full_steps":used.tolist(),"prefix_stops":cd.tolist()})
    S.save(S.OUT/"eta/preflight.json",{"status":"passed","utc":S.utc(),"checks":checks,"execution_sha256":S.sha(__file__),"protocol_sha256":S.sha(S.HERE/"eta-protocol.json")})
    S.emit("eta_preflight_passed",checks=checks)


def worker(gpu):
    dest=S.OUT/"eta";lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inv=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader"],text=True);assert inv.split(",")[1].strip() not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,M=S.imports();from cram.maze_learned import gather
    pp=S.HERE/"eta-protocol.json";protocol=json.loads(pp.read_text());assert protocol["execution_script_sha256"]==S.sha(__file__)
    assert json.loads((dest/"preflight.json").read_text())["status"]=="passed"
    for path,h in protocol["source_sha256"].items():assert S.sha(S.ROOT/path)==h
    state={"status":"running","pid":os.getpid(),"host":platform.node(),"gpu":gpu,"started_utc":S.utc(),"jobs_complete":[]}
    for ji,job in enumerate(protocol["jobs"]):
        if ji%2!=gpu:continue
        run=dest/"runs"/job["id"];run.mkdir(parents=True,exist_ok=True)
        if (run/"result.json").exists():raise FileExistsError(run)
        d=next(d for d in protocol["datasets"] if d["length"]==job["length"] and d["loop_percent"]==job["loop_percent"])
        assert S.sha(S.ROOT/d["path"])==d["sha256"];data=np.load(S.ROOT/d["path"])
        assert S.sha(S.OUT/"models"/f"seed{job['seed']}.pt")==protocol["models"][str(job["seed"])]["sha256"]
        model=S.model_for(job["seed"],job["length"],"original","cuda");out={str(e):{k:[] for k in ["predictions","answer_steps","stopped","hidden_steps","hidden_good"]} for e in [.2,.5]}
        timing={"0.2":0.,"0.5":0.};hidden_time=0.;diagnostics={};started=time.perf_counter()
        state.update(current_job=job,completed_mazes=0,updated_utc=S.utc());S.save(dest/f"worker{gpu}.json",state)
        for lo in range(0,len(data["tokens"]),32):
            t=torch.as_tensor(data["tokens"][lo:lo+32].astype(np.int64),device="cuda")
            with torch.no_grad():
                torch.cuda.synchronize();p=time.perf_counter();psi,ps=model.settle(t,tol=1e-6,max_steps=4000,check=8)
                lg,la,lc=model.parts(t,model.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,model.zoffs,math.inf),lc[:,None]],1),1)
                good=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1))
                a=model.couplings(t,psi);torch.cuda.synchronize();hidden_time+=time.perf_counter()-p
                if lo==0:
                    for e in [.2,.5]:M.descend(model,t,a,steps=32,eta=e,margin=3.,certify_every=16)
                    torch.cuda.synchronize()
                order=[.2,.5] if (ji+lo//32)%2==0 else [.5,.2]
                for eta in order:
                    torch.cuda.synchronize();p=time.perf_counter();pred,used,done=M.descend(model,t,a,steps=8192,eta=eta,margin=3.,certify_every=16)
                    torch.cuda.synchronize();timing[str(eta)]+=time.perf_counter()-p
                    for key,value in [("predictions",pred.bool()),("answer_steps",used),("stopped",done),("hidden_steps",ps),("hidden_good",good)]:out[str(eta)][key].append(value.cpu().numpy())
                    if job["seed"]==0 and lo==0:
                        q,v,sd,detail=trace(model,t,a,eta)
                        assert torch.equal(q,pred) and torch.equal(v,used) and torch.equal(sd,done)
                        diagnostics[str(eta)]=dict(detail,replay_native_parity=True,retained_maze_indices=list(range(8)))
            state.update(completed_mazes=min(lo+32,len(data["tokens"])),updated_utc=S.utc(),elapsed_s=time.perf_counter()-started)
            S.save(dest/f"worker{gpu}.json",state);S.emit("eta_batch_done",gpu=gpu,job=job["id"],n=state["completed_mazes"])
            if time.perf_counter()-started>600:raise TimeoutError("600s paired-job watchdog")
        arms={}
        for eta in [.2,.5]:
            arrays={k:np.concatenate(v) for k,v in out[str(eta)].items()};arrays.update(S.board_metrics(data["tokens"],data["labels"],arrays["predictions"],arrays["stopped"],arrays["hidden_good"]))
            arrays["success2048"]=arrays["success"]&(arrays["answer_steps"]<=2048)
            path=run/f"eta{eta}.npz";np.savez_compressed(path,**arrays)
            arms[str(eta)]={"solved":int(arrays["success"].sum()),"solved2048":int(arrays["success2048"].sum()),"wrong_stops":int((arrays["stopped"]&~arrays["shortest_path"]).sum()),"answer_budget_failures":int((~arrays["stopped"]).sum()),"hidden_failures":int((~arrays["hidden_good"]).sum()),"answer_steps_mean":float(arrays["answer_steps"].mean()),"answer_seconds":timing[str(eta)],"hidden_plus_answer_seconds":hidden_time+timing[str(eta)],"output_sha256":S.sha(path)}
        if diagnostics:S.save(run/"diagnostics.json",diagnostics)
        record={"status":"complete","utc":S.utc(),"job":job,"n":len(data["tokens"]),"arms":arms,"hidden_seconds_shared":hidden_time,"wall_seconds":time.perf_counter()-started,"protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"core_sha256":S.sha(S.__file__),"dataset_sha256":d["sha256"],"checkpoint_sha256":protocol["models"][str(job["seed"])]["sha256"],"gpu":gpu,"host":platform.node(),"inventory":inv,"torch":torch.__version__,"diagnostics_sha256":S.sha(run/"diagnostics.json") if diagnostics else None}
        S.save(run/"result.json",record);S.emit("eta_job_complete",**record);state["jobs_complete"].append(job["id"])
    state.update(status="complete",completed_utc=S.utc());S.save(dest/f"worker{gpu}.json",state);S.emit("eta_worker_complete",gpu=gpu)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=["prepare","preflight","worker"]);p.add_argument("--gpu",type=int,choices=[0,1]);a=p.parse_args()
    if a.command=="prepare":prepare()
    elif a.command=="preflight":preflight(a.gpu)
    else:worker(a.gpu)
