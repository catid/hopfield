"""Frozen free-fork angle test with matched cycle and tail lengths."""
import argparse
import fcntl
import json
import math
import os
import platform
import subprocess
import sys
import time
import numpy as np
import study as S


def make_graph(route,condition,geometry):
    width,row,col,rotation,flip=map(int,geometry);height=route-width;tail=16
    tok=np.zeros((160,160),np.uint8);tok[row:row+height+1,col]=1;tok[row:row+height+1,col+width]=1
    tok[row,col:col+width+1]=1;tok[row+height,col:col+width+1]=1
    if condition=="corner":
        tok[row-tail:row+1,col]=1;tok[row+height:row+height+tail+1,col+width]=1
        tok[row-tail,col]=2;tok[row+height+tail,col+width]=3
    else:
        mid=row+height//2;tok[mid,col-tail:col+1]=1;tok[mid,col+width:col+width+tail+1]=1
        tok[mid,col-tail]=2;tok[mid,col+width+tail]=3
    tok=np.rot90(tok,rotation)
    if flip:tok=np.fliplr(tok)
    tok=tok.copy();label,distance,reachable=S.bfs(tok);assert distance==route+32 and reachable==int((tok>0).sum())
    op=tok>0;degree=np.zeros_like(tok,int);degree[1:]+=op[:-1];degree[:-1]+=op[1:];degree[:,1:]+=op[:,:-1];degree[:,:-1]+=op[:,1:]
    forks=np.argwhere(op&(degree==3));assert len(forks)==2 and np.all(tok[tuple(forks.T)]==1)
    return tok,label,distance,forks


def prepare():
    parent=json.loads((S.HERE/"long_ties-protocol.json").read_text());base=2026111413
    codes=np.random.default_rng(base).choice(3*16*16*8,128,replace=False);geometry=[]
    for code in codes:
        code,dihedral=divmod(int(code),8);code,col=divmod(code,16);width_index,row=divmod(code,16);rotation,flip=divmod(dihedral,2)
        geometry.append([[4,6,8][width_index],row+16,col+16,rotation,flip])
    dest=S.OUT/"fork_angle";dest.mkdir(parents=True,exist_ok=True);datasets=[];jobs=[]
    for route in [28,42,64]:
        for condition in ["opposite","corner"]:
            items=[make_graph(route,condition,g) for g in geometry];name=f"r{route}-{condition}";path=dest/f"{name}.npz"
            if path.exists():raise FileExistsError(path)
            tokens=np.stack([x[0] for x in items]);assert len({t.tobytes() for t in tokens})==128
            np.savez_compressed(path,tokens=tokens,labels=np.stack([x[1] for x in items]),shortest_edges=np.asarray([x[2] for x in items]),fork_coords=np.stack([x[3] for x in items]),geometries=np.asarray(geometry))
            datasets.append({"cohort":name,"length":160,"loop_percent":0,"route_length":route,"tail_length":16,"condition":condition,"n":128,
                "expected_shortest_path_count":2,"base_seed":base,"path":str(path.relative_to(S.ROOT)),"sha256":S.sha(path)})
            for seed in range(3):jobs.append({"id":f"{name}-s{seed}","cohort":name,"length":160,"loop_percent":0,"route_length":route,"condition":condition,"seed":seed,"arm":"original","answer_steps":8192})
    protocol=dict(parent,id="independent-free-fork-angle-controls",frozen_utc=S.utc(),status="frozen_before_outcomes",datasets=datasets,jobs=jobs,
        execution_script_sha256=S.sha(__file__),core_script_sha256=S.sha(S.__file__),H2k=None,
        H2l="Right-angle alternative routes at free degree3forks lose at least10percentage points joint-three-model success compared with opposite-direction alternatives, at fixedeta0.5.",
        decision="Average route28/42/64 conditions equally within128paired geometry/layoutclusters. Primary difference=mean(jointopposite-jointcorner).10000paired-cluster bootstrap95% interval; support if lower>=10pp, falsify if upper<10pp, otherwise inconclusive. Report route-specific effects separately; zero-discordance cluster upper bound accompanies bootstrap[0,0]. No hard-layout selection.",
        rationale="Selected failedcase alternatives leave freejunctions at rightangles, so their neighboring answer cells interact through the learned diagonal pair tables. Prior rectangular freejunctions split into opposite directions, with neighboring cells2apart outside the3x3pairwindow. This tests that local motif independently of total pathlength and multiplicity.",
        generator="One rectangular cycle with equalroute28/42/64 and twolength16tails ending at S/G. Opposite condition attaches tails at left/rightside midpoints, creating freeforks with opposite alternatives. Corner condition attaches tails vertically above top-left and below bottom-right corners, creating right-angle alternatives.128fresh width4/6/8, translation and D4clusters paired across everyroute/condition. Nodecount,edgecount,degreehistogram,cycleperimeter,tail lengths,S-Gdistance(route+32),and2shortestpaths match exactly; spatial port placement is the intervention.",
        primary_units="128paired geometry/layoutclusters;768inputconditions,3fixedmodels,2304model-input evaluations. Three model repeats and three route lengths are not independent clusters.",
        diagnostic="Independent audit verifies degree3forks free, alternative-neighbor dotproduct0 versus-1 and diagonalinteraction motif. Save predictedboards, energies, minimumflipmargins, attentionreadout at bothforks and oracle-path field there; retain actual learned pair tables and offset ordering. Oracle metrics computed after inference only.",
        confirmation="New generator2026111413; fixedmodels,eta0.5,hidden4000,answer8192. No outcome-dependent cases, modelparameter changes or rescue.",
        limitation="Tests this matched singlecycle/twotail family; negative results would not dismiss heterogeneous-cycle or broader hiddenflow interactions in the retained failure.")
    pp=S.HERE/"fork_angle-protocol.json"
    if pp.exists():raise FileExistsError(pp)
    S.save(pp,protocol);S.emit("fork_angle_frozen",sha256=S.sha(pp),jobs=len(jobs),unique_inputs=768)


def worker(campaign, gpu):
    dest = S.OUT / campaign
    lock = (S.ROOT / f"artifacts/gpu{gpu}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], text=True).strip()
    uuid = inventory.split(",")[1].strip()
    owners = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
    assert uuid not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    torch, _, M = S.imports()
    from cram.maze_learned import gather
    protocol_path = S.HERE / f"{campaign}-protocol.json"; protocol = json.loads(protocol_path.read_text())
    assert S.sha(__file__) == protocol["execution_script_sha256"] and S.sha(S.__file__) == protocol["core_script_sha256"]
    for path, expected in protocol["source_sha256"].items(): assert S.sha(S.ROOT / path) == expected
    state = {"status": "running", "host": platform.node(), "pid": os.getpid(), "gpu": gpu,
             "started_utc": S.utc(), "campaign": campaign, "jobs_complete": [], "protocol_sha256": S.sha(protocol_path)}
    heartbeat = dest / f"worker{gpu}.json"; S.save(heartbeat, state)
    for index, job in enumerate(protocol["jobs"]):
        if index % 2 != gpu: continue
        run = dest / "runs" / job["id"]
        if (run / "result.json").exists(): raise FileExistsError(run)
        run.mkdir(parents=True, exist_ok=True)
        data = next(d for d in protocol["datasets"] if d["cohort"] == job["cohort"] and d["length"] == job["length"] and d["loop_percent"] == job["loop_percent"])
        assert S.sha(S.ROOT / data["path"]) == data["sha256"]
        source = np.load(S.ROOT / data["path"]); tokens, labels = source["tokens"], source["labels"]
        assert S.sha(S.OUT / "models" / f"seed{job['seed']}.pt") == protocol["models"][str(job["seed"])]["sha256"]
        model = S.model_for(job["seed"], job["length"], job["arm"], "cuda")
        output = {k: [] for k in ("predictions", "answer_steps", "stopped", "hidden_steps", "hidden_good", "prediction_energy", "oracle_energy", "minimum_flip_delta", "fork_couplings", "fork_oracle_field")}
        start = time.perf_counter(); hidden_seconds = answer_seconds = 0.
        state.update(current_job=job, completed_mazes=0, updated_utc=S.utc()); S.save(heartbeat, state)
        S.emit("job_start", campaign=campaign, gpu=gpu, job=job)
        for lo in range(0, len(tokens), 32):
            t = torch.as_tensor(tokens[lo:lo + 32].astype(np.int64), device="cuda")
            with torch.no_grad():
                torch.cuda.synchronize(); part = time.perf_counter()
                psi, ps = model.settle(t, max_steps=4000, check=8, tol=1e-6)
                lg, la, lc = model.parts(t, model.mu)
                new = la - torch.logsumexp(torch.cat([lg - gather(psi, model.zoffs, float("inf")), lc[:, None]], 1), 1)
                good = (((new == psi) | ((new - psi).abs() <= 1e-6 * (1 + new.abs()))).flatten(1).all(1)
                        & (torch.isfinite(psi) | (t == 0)).flatten(1).all(1))
                a = model.couplings(t, psi)
                torch.cuda.synchronize(); hidden_seconds += time.perf_counter() - part; part = time.perf_counter()
                pred, used, stopped = M.descend(model, t, a, steps=job["answer_steps"], eta=.5, margin=3., certify_every=16)
                prediction_energy = model.energy(t, pred, a)
                oracle_energy = model.energy(t, torch.as_tensor(labels[lo:lo+32], device="cuda", dtype=a.dtype), a)
                minimum_flip_delta = model.flip_delta(t, pred, a).flatten(1).amin(1)
                coords = torch.as_tensor(source["fork_coords"][lo:lo+32], device="cuda", dtype=torch.long)
                batch_index = torch.arange(len(t), device="cuda")[:, None]
                fork_couplings = a.permute(0,2,3,1)[batch_index,coords[:,:,0],coords[:,:,1]]
                oracle_field = model.field(t, torch.as_tensor(labels[lo:lo+32],device="cuda",dtype=a.dtype),a)
                fork_oracle_field = oracle_field[batch_index,coords[:,:,0],coords[:,:,1]]
                torch.cuda.synchronize(); answer_seconds += time.perf_counter() - part
            for key, value in (("predictions", pred.bool()), ("answer_steps", used), ("stopped", stopped), ("hidden_steps", ps), ("hidden_good", good), ("prediction_energy", prediction_energy), ("oracle_energy", oracle_energy), ("minimum_flip_delta", minimum_flip_delta), ("fork_couplings", fork_couplings), ("fork_oracle_field", fork_oracle_field)):
                output[key].append(value.cpu().numpy())
            state.update(completed_mazes=min(lo + 32, len(tokens)), updated_utc=S.utc(), elapsed_s=time.perf_counter() - start)
            S.save(heartbeat, state); S.emit("batch_done", campaign=campaign, gpu=gpu, job=job["id"], n=state["completed_mazes"])
            if time.perf_counter() - start > 600: raise TimeoutError("Predeclared600s job watchdog")
        output = {k: np.concatenate(v) for k, v in output.items()}
        output.update(S.board_metrics(tokens, labels, output["predictions"], output["stopped"], output["hidden_good"]))
        np.savez_compressed(run / "predictions.npz", **output)
        result = {"status": "complete", "job": job, "utc": S.utc(), "n": len(tokens), "solved": int(output["success"].sum()),
                  "valid_paths": int(output["valid_path"].sum()), "shortest_paths": int(output["shortest_path"].sum()),
                  "canonical_exact": int(output["canonical_exact"].sum()), "wrong_stops": int((output["stopped"] & ~output["shortest_path"]).sum()),
                  "answer_budget_failures": int((~output["stopped"]).sum()), "hidden_failures": int((~output["hidden_good"]).sum()),
                  "hidden_steps_mean": float(output["hidden_steps"].mean()), "answer_steps_mean": float(output["answer_steps"].mean()),
                  "hidden_seconds": hidden_seconds, "answer_seconds": answer_seconds, "wall_seconds": time.perf_counter()-start,
                  "answer_eta": .5, "mass": float(model.masses()[0]), "host": platform.node(), "gpu": gpu, "inventory": inventory,
                  "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version,
                  "script_sha256": S.sha(__file__), "core_script_sha256": S.sha(S.__file__), "protocol_sha256": S.sha(protocol_path),
                  "checkpoint_sha256": protocol["models"][str(job["seed"])]["sha256"], "dataset_sha256": data["sha256"],
                  "predictions_sha256": S.sha(run / "predictions.npz"), "timing_scope": "Instrumented PyTorch eval; hidden includes residual/attention; no H200 comparison"}
        S.save(run / "result.json", result); S.emit("job_complete", **result)
        state["jobs_complete"].append(job["id"]); S.save(heartbeat, state)
    state.update(status="complete", completed_utc=S.utc()); S.save(heartbeat, state); S.emit("worker_complete", campaign=campaign, gpu=gpu)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["prepare","worker"]);parser.add_argument("--gpu",type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=="prepare":prepare()
    else:worker("fork_angle",args.gpu)
