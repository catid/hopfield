"""Frozen four-arm test of a finite-step stability boundary, with no rescue."""
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
import eta_comb as C

ETAS=(.2,.5,.8,1.)


def prepare():
    parent=json.loads((S.HERE/"eta-protocol.json").read_text());base=2026111409
    dest=S.OUT/"eta_boundary";dest.mkdir(parents=True,exist_ok=True);datasets=[];jobs=[]
    layouts=np.random.default_rng(base).choice(32*32*8,512,replace=False)
    for stratum,length,rate in [("tree",90,0),("loops20",90,20),("comb",160,0)]:
        if stratum=="comb":items=[C.make_comb(64,128,c,base,i) for i,c in enumerate(layouts)]
        else:items=[S.maze(length,rate/100,np.random.SeedSequence([base,length,rate,i])) for i in range(512)]
        path=dest/f"{stratum}.npz"
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,tokens=np.stack([x[0] for x in items]),labels=np.stack([x[1] for x in items]),shortest_edges=np.asarray([x[2] for x in items]))
        datasets.append({"stratum":stratum,"length":length,"loop_percent":rate,"n":512,"base_seed":base,"path":str(path.relative_to(S.ROOT)),"sha256":S.sha(path)})
        for seed in range(3):jobs.append({"id":f"{stratum}-s{seed}","stratum":stratum,"length":length,"loop_percent":rate,"seed":seed})
    protocol=dict(parent,id="eta-stability-boundary",frozen_utc=S.utc(),status="frozen_before_outcomes",datasets=datasets,jobs=jobs,
        execution_script_sha256=S.sha(__file__),core_script_sha256=S.sha(S.__file__),comb_generator_sha256=S.sha(C.__file__),etas=list(ETAS),H2g=None,
        H2i="Eta1.0 loses joint-three-model solves on at least1percentage point of new equally weighted tree/20%-loop/comb inputs that eta0.5 solves at8192steps.",
        decision="Loss event is eta0.5jointsuccess AND NOT eta1.0jointsuccess. Equal weight per3strata. Conservative one-sided95% mixture bounds: average the3stratum Clopper-Pearson bounds at alpha=.05/3. Support1pp loss target if lower>=1pp; falsify if upper<1pp; otherwise inconclusive. Also10000paired-stratified bootstrap95% intervals as descriptive uncertainty. Count eta1gains separately; loss probability is not net accuracy change.",
        sample_size="512fresh inputs per stratum;1536unique inputs total. Zero observed losses yields conservative upper<0.8pp under the declared Bonferroni-stratum construction, enough to test1pp. Fixed-model repeats are not independent inputs.",
        secondary="Report all four fixedetas, per-stratum and overall success, lost/gained solves, median steps and matched checked-answer runtime. Eta0.8exploratory; no selected winning eta and no adaptive rescue. Capped/nonfinite cases remain failures; conditional successful-only speed is descriptive and labeled.",
        inference=dict(parent["inference"],answer_eta=list(ETAS),answer_max_steps=8192),
        implementation="Same native updates and check16, with additional per-row finite-logit check every16updates. Nonfinite rows terminate as failed, never certify. Finite-aware solver checked exactly against native on reserved inputs at everyeta. Main timing includes finite checks for allarms; no native-runtime claim.",
        timing="SameGPU and cached hiddenstate per job; warmup32updates eacharm. Rotate fixedetaorder by jobindex+batchindex. CUDA-synchronized finite-aware answer timing excludes instrumented replay.",
        diagnostic_subset="First8inputs of first32inputbatch perstratum, modelseed0 only, everyeta. Separate replay tracks per-step logit2cycles and energies/residual every16; must exactly match main outputs, usedsteps, stopflags and finiteflags. Sampled energy need not prove monotonicity. Nonfinite diagnostic scalars serialized as null with separate flags.",
        confirmation="New generator2026111409. No eta changes, new seeds, added cases, extended caps, or adaptive rescue after outcomes. Per-job watchdog1200s; partial files/failures preserved.")
    pp=S.HERE/"eta_boundary-protocol.json"
    if pp.exists():raise FileExistsError(pp)
    S.save(pp,protocol);S.emit("eta_boundary_frozen",sha256=S.sha(pp),jobs=len(jobs),unique_inputs=1536)


def checked(model,t,a,eta,steps=8192,diagnostic=False):
    torch,_,_=S.imports();n=len(t);u=torch.full(t.shape,3.,device=t.device,dtype=a.dtype)
    stopped=torch.zeros(n,dtype=torch.bool,device=t.device);finite=torch.ones_like(stopped)
    used=torch.full((n,),steps,dtype=torch.long,device=t.device);first_bad=torch.full_like(used,-1)
    old1=None;old2=None;cycle=torch.zeros_like(stopped);free=t==1;history=[]
    def safe(values):return [v if math.isfinite(v) else None for v in values]
    with torch.no_grad():
        if diagnostic:history.append({"update":0,"energy":safe(model.energy(t,torch.sigmoid(u),a)[:8].tolist())})
        for k in range(1,steps+1):
            new=u+eta*(model.field(t,torch.sigmoid(u),a)-u)
            u=torch.where((stopped|~finite)[:,None,None],u,new)
            if diagnostic:
                if old2 is not None:
                    one=torch.where(free,(u-old1).abs(),0).flatten(1).amax(1)
                    two=torch.where(free,(u-old2).abs(),0).flatten(1).amax(1)
                    cycle|=(two<1e-5)&(one>1e-3)&~stopped&finite
                old2=old1;old1=u.clone()
            if k%16==0:
                now=torch.isfinite(u).flatten(1).all(1)
                first_bad=torch.where(~now&(first_bad<0),torch.full_like(used,k),first_bad);finite&=now
                h=(u>0).to(a.dtype);ok=(model.flip_delta(t,h,a)>=-1e-6).flatten(1).all(1)&finite
                used=torch.where(ok&~stopped,torch.full_like(used,k),used);stopped|=ok
                if diagnostic:
                    residual=torch.where(free,(model.field(t,torch.sigmoid(u),a)-u).abs(),0).flatten(1).amax(1)
                    history.append({"update":k,"energy":safe(model.energy(t,torch.sigmoid(u),a)[:8].tolist()),"residual":safe(residual[:8].tolist()),"stopped":stopped[:8].tolist(),"finite":finite[:8].tolist()})
                if (stopped|~finite).all():break
    return (u>0).to(a.dtype),used,stopped,finite,first_bad,{"history":history,"two_cycle_detected":cycle[:8].tolist()}


def preflight(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,M=S.imports();model=S.model_for(0,90,"original","cuda")
    items=[S.maze(90,0,np.random.SeedSequence([2026111497,i])) for i in range(4)]
    t=torch.as_tensor(np.stack([x[0] for x in items]).astype(np.int64),device="cuda");checks=[]
    with torch.no_grad():
        psi,_=model.settle(t);a=model.couplings(t,psi)
        for eta in ETAS:
            native=M.descend(model,t,a,steps=8192,eta=eta,margin=3.,certify_every=16)
            main=checked(model,t,a,eta);replay=checked(model,t,a,eta,diagnostic=True)
            assert all(torch.equal(native[i],main[i]) for i in range(3))
            assert all(torch.equal(main[i],replay[i]) for i in range(5))
            checks.append({"eta":eta,"native_parity":True,"replay_parity":True,"steps":main[1].tolist(),"stopped":main[2].tolist(),"finite":main[3].tolist()})
    pp=S.HERE/"eta_boundary-protocol.json"
    S.save(S.OUT/"eta_boundary/preflight.json",{"status":"passed","utc":S.utc(),"checks":checks,"execution_sha256":S.sha(__file__),"protocol_sha256":S.sha(pp)})
    S.emit("eta_boundary_preflight_passed",checks=checks)


def worker(gpu):
    dest=S.OUT/"eta_boundary";lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader"],text=True);assert inventory.split(",")[1].strip() not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,_=S.imports();from cram.maze_learned import gather
    pp=S.HERE/"eta_boundary-protocol.json";protocol=json.loads(pp.read_text())
    assert protocol["execution_script_sha256"]==S.sha(__file__) and protocol["core_script_sha256"]==S.sha(S.__file__)
    assert protocol["comb_generator_sha256"]==S.sha(C.__file__)
    preflight=json.loads((dest/"preflight.json").read_text());assert preflight["status"]=="passed" and preflight["protocol_sha256"]==S.sha(pp)
    for path,h in protocol["source_sha256"].items():assert S.sha(S.ROOT/path)==h
    state={"status":"running","pid":os.getpid(),"host":platform.node(),"gpu":gpu,"started_utc":S.utc(),"jobs_complete":[]}
    for ji,job in enumerate(protocol["jobs"]):
        if ji%2!=gpu:continue
        run=dest/"runs"/job["id"];run.mkdir(parents=True,exist_ok=True)
        if (run/"result.json").exists():raise FileExistsError(run)
        dataset=next(d for d in protocol["datasets"] if d["stratum"]==job["stratum"])
        assert S.sha(S.ROOT/dataset["path"])==dataset["sha256"];data=np.load(S.ROOT/dataset["path"])
        assert S.sha(S.OUT/"models"/f"seed{job['seed']}.pt")==protocol["models"][str(job["seed"])]["sha256"]
        model=S.model_for(job["seed"],job["length"],"original","cuda")
        outputs={str(e):{k:[] for k in ["predictions","answer_steps","stopped","answer_finite","first_nonfinite_check","hidden_steps","hidden_good"]} for e in ETAS}
        times={str(e):0. for e in ETAS};hidden_time=0.;diagnostics={};started=time.perf_counter()
        state.update(current_job=job,completed_mazes=0,updated_utc=S.utc());S.save(dest/f"worker{gpu}.json",state)
        for lo in range(0,len(data["tokens"]),32):
            t=torch.as_tensor(data["tokens"][lo:lo+32].astype(np.int64),device="cuda")
            with torch.no_grad():
                torch.cuda.synchronize();start=time.perf_counter();psi,ps=model.settle(t,tol=1e-6,max_steps=4000,check=8)
                lg,la,lc=model.parts(t,model.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,model.zoffs,math.inf),lc[:,None]],1),1)
                good=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1))
                a=model.couplings(t,psi);torch.cuda.synchronize();hidden_time+=time.perf_counter()-start
                if lo==0:
                    for eta in ETAS:checked(model,t,a,eta,steps=32)
                    torch.cuda.synchronize()
                offset=(ji+lo//32)%4;order=ETAS[offset:]+ETAS[:offset]
                for eta in order:
                    torch.cuda.synchronize();start=time.perf_counter();result=checked(model,t,a,eta)
                    torch.cuda.synchronize();times[str(eta)]+=time.perf_counter()-start
                    pred,used,done,finite,first_bad,_=result
                    values=[pred.bool(),used,done,finite,first_bad,ps,good]
                    for key,value in zip(outputs[str(eta)],values):outputs[str(eta)][key].append(value.cpu().numpy())
                    if job["seed"]==0 and lo==0:
                        replay=checked(model,t,a,eta,diagnostic=True)
                        assert all(torch.equal(result[i],replay[i]) for i in range(5))
                        diagnostics[str(eta)]=dict(replay[5],replay_main_parity=True,retained_maze_indices=list(range(8)))
            state.update(completed_mazes=min(lo+32,len(data["tokens"])),updated_utc=S.utc(),elapsed_s=time.perf_counter()-started)
            S.save(dest/f"worker{gpu}.json",state);S.emit("eta_boundary_batch_done",gpu=gpu,job=job["id"],n=state["completed_mazes"])
            if time.perf_counter()-started>1200:raise TimeoutError("1200s boundary job watchdog")
        arms={}
        for eta in ETAS:
            arrays={k:np.concatenate(v) for k,v in outputs[str(eta)].items()}
            arrays.update(S.board_metrics(data["tokens"],data["labels"],arrays["predictions"],arrays["stopped"],arrays["hidden_good"]))
            arrays["success"]&=arrays["answer_finite"]
            path=run/f"eta{eta}.npz";np.savez_compressed(path,**arrays)
            arms[str(eta)]={"solved":int(arrays["success"].sum()),"wrong_stops":int((arrays["stopped"]&~arrays["shortest_path"]).sum()),
                "answer_budget_failures":int((~arrays["stopped"]&arrays["answer_finite"]).sum()),"nonfinite_failures":int((~arrays["answer_finite"]).sum()),
                "hidden_failures":int((~arrays["hidden_good"]).sum()),"answer_steps_mean":float(arrays["answer_steps"].mean()),
                "checked_answer_seconds":times[str(eta)],"hidden_plus_checked_answer_seconds":hidden_time+times[str(eta)],"output_sha256":S.sha(path)}
        if diagnostics:S.save(run/"diagnostics.json",diagnostics)
        record={"status":"complete","utc":S.utc(),"job":job,"n":len(data["tokens"]),"arms":arms,"hidden_seconds_shared":hidden_time,
            "wall_seconds":time.perf_counter()-started,"protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"core_sha256":S.sha(S.__file__),
            "dataset_sha256":dataset["sha256"],"checkpoint_sha256":protocol["models"][str(job["seed"])]["sha256"],"gpu":gpu,"host":platform.node(),
            "inventory":inventory,"torch":torch.__version__,"diagnostics_sha256":S.sha(run/"diagnostics.json") if diagnostics else None}
        S.save(run/"result.json",record);S.emit("eta_boundary_job_complete",**record);state["jobs_complete"].append(job["id"])
    state.update(status="complete",completed_utc=S.utc());S.save(dest/f"worker{gpu}.json",state);S.emit("eta_boundary_worker_complete",gpu=gpu)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["prepare","preflight","worker"]);parser.add_argument("--gpu",type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=="prepare":prepare()
    elif args.command=="preflight":preflight(args.gpu)
    else:worker(args.gpu)
