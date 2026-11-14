"""Fresh causal-comb test of eta scaling, using the audited paired worker."""
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
import eta_study as E
trace=E.trace


def make_comb(spine,depth,layout,base,index):
    shift,dihedral=divmod(int(layout),8);row,col=divmod(shift,32);rotation,flip=divmod(dihedral,2)
    roots=np.random.default_rng(np.random.SeedSequence([base,index,spine])).choice(np.arange(1,spine,2),8,replace=False)
    tok=np.zeros((160,160),np.uint8);tok[row:row+spine+1,col]=1
    for r in roots:tok[row+r,col:col+depth+1]=1
    tok[row,col]=2;tok[row+spine,col]=3;tok=np.rot90(tok,rotation)
    if flip:tok=np.fliplr(tok)
    tok=tok.copy();label,distance,reachable=S.bfs(tok)
    assert distance==spine and int(label.sum())==spine+1 and reachable==int((tok>0).sum())
    op=tok>0;assert int((op[1:]&op[:-1]).sum()+(op[:,1:]&op[:,:-1]).sum())==int(op.sum())-1
    return tok,label,distance


def prepare():
    parent=json.loads((S.HERE/"eta-protocol.json").read_text());base=2026111408
    layout_codes=np.random.default_rng(base).choice(32*32*8,64,replace=False)
    dest=S.OUT/"eta_comb";dest.mkdir(parents=True,exist_ok=True);datasets=[];jobs=[]
    for spine,depth in [(64,32),(64,64),(64,128),(16,64),(128,64)]:
        items=[make_comb(spine,depth,code,base,i) for i,code in enumerate(layout_codes)]
        name=f"spine{spine}-depth{depth}";path=dest/f"{name}.npz"
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,tokens=np.stack([x[0] for x in items]),labels=np.stack([x[1] for x in items]),shortest_edges=np.asarray([x[2] for x in items]),layout_codes=layout_codes)
        datasets.append({"cohort":name,"length":160,"loop_percent":0,"spine_length":spine,"branch_depth":depth,"n":64,"base_seed":base,"path":str(path.relative_to(S.ROOT)),"sha256":S.sha(path)})
        for seed in range(3):jobs.append({"id":f"{name}-s{seed}","length":160,"loop_percent":0,"spine_length":spine,"branch_depth":depth,"seed":seed})
    protocol=dict(parent,id="eta-causal-comb",frozen_utc=S.utc(),status="frozen_before_outcomes",datasets=datasets,jobs=jobs,
        execution_script_sha256=S.sha(__file__),trace_source_sha256=S.sha(E.__file__),core_script_sha256=S.sha(S.__file__),H2g=None,
        H2h="At spine64, the ratio of eta0.2 to eta0.5 answer-update slopes over branch depths32/64/128 lies in [2,3], within20% of inverse-eta prediction2.5, while all stopped outputs remain shortest paths.",
        decision="Fit OLS slope across depths32/64/128 separately for each paired layout and eta after averaging three fixed models. Ratio of mean slopes;10000paired-layout bootstrap95% interval. Support equivalence if whole interval within[2,3], falsify if wholly outside, otherwise inconclusive. Any invalid stopped path or budget exhaustion precludes supported mechanistic claim. No post-outcome tuning.",
        secondary="At fixed depth64, increasing S-Gspine16to128 changes mean answerupdates by within[-32,32], assessed separately for each eta using paired-layout bootstrap95% interval. These are separate hypotheses without multiplicity-adjusted combined claim.",
        generator="160x160grid,8nonadjacent perpendicular equal-depth branches;64fresh paired translations and D4orientations, seed2026111408. Roots fixed per layout/spine across depths. Slopes use sufficiently long branches to reduce certify-every16 quantization.",
        power_note="64paired layouts are the uncertainty units; three model repeats and depth/eta conditions are not independent samples. No99%population success claim.",
        diagnostic_subset="First8layouts per condition, seed0, from separate instrumented replay; exact native parity; both fixedetas. Sampled energy and cycle flags are finite diagnostics only.",
        confirmation="Fresh comb layouts, fixed eta0.2/0.5,8192answer cap,4000hidden cap; no weight or energy change.")
    pp=S.HERE/"eta_comb-protocol.json"
    if pp.exists():raise FileExistsError(pp)
    S.save(pp,protocol);S.emit("eta_comb_frozen",sha256=S.sha(pp),jobs=len(jobs))


def preflight(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,M=S.imports();model=S.model_for(0,160,"original","cuda")
    items=[make_comb(64,d,0,2026111498,i) for i,d in enumerate([32,64,128])]
    t=torch.as_tensor(np.stack([x[0] for x in items]).astype(np.int64),device="cuda");checks=[]
    with torch.no_grad():
        psi,_=model.settle(t);a=model.couplings(t,psi)
        for eta in [.2,.5]:
            pred,used,done=M.descend(model,t,a,steps=8192,eta=eta,margin=3.,certify_every=16)
            q,v,d,detail=trace(model,t,a,eta)
            assert torch.equal(pred,q) and torch.equal(used,v) and torch.equal(done,d)
            checks.append({"eta":eta,"recording_parity":True,"full_steps":used.tolist(),"all_stopped":bool(done.all())})
    S.save(S.OUT/"eta_comb/preflight.json",{"status":"passed","utc":S.utc(),"checks":checks,"execution_sha256":S.sha(__file__),"protocol_sha256":S.sha(S.HERE/"eta_comb-protocol.json")})
    S.emit("eta_comb_preflight_passed",checks=checks)


def worker(gpu):
    dest=S.OUT/"eta_comb";lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inv=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader"],text=True);assert inv.split(",")[1].strip() not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,M=S.imports();from cram.maze_learned import gather
    pp=S.HERE/"eta_comb-protocol.json";protocol=json.loads(pp.read_text());assert protocol["execution_script_sha256"]==S.sha(__file__);assert protocol["trace_source_sha256"]==S.sha(E.__file__)
    assert json.loads((dest/"preflight.json").read_text())["status"]=="passed"
    for path,h in protocol["source_sha256"].items():assert S.sha(S.ROOT/path)==h
    state={"status":"running","pid":os.getpid(),"host":platform.node(),"gpu":gpu,"started_utc":S.utc(),"jobs_complete":[]}
    for ji,job in enumerate(protocol["jobs"]):
        if ji%2!=gpu:continue
        run=dest/"runs"/job["id"];run.mkdir(parents=True,exist_ok=True)
        if (run/"result.json").exists():raise FileExistsError(run)
        d=next(d for d in protocol["datasets"] if d["spine_length"]==job["spine_length"] and d["branch_depth"]==job["branch_depth"])
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
