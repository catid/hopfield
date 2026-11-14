"""Frozen equal/unequal-route causal controls, independent of the selected failure."""
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


def make_graph(cycles,condition,geometry):
    width,height,row,col,rotation,flip=map(int,geometry);gap=4;offset={"tied":0,"plus":2,"minus":-2}[condition]
    port=row+height//2+offset;tok=np.zeros((160,160),np.uint8)
    for i in range(cycles):
        left=col+i*(width+gap);right=left+width
        tok[row:row+height+1,left]=1;tok[row:row+height+1,right]=1
        tok[row,left:right+1]=1;tok[row+height,left:right+1]=1
        if i<cycles-1:tok[port,right:right+gap+1]=1
    tok[port,col]=2;tok[port,col+(cycles-1)*(width+gap)+width]=3
    tok=np.rot90(tok,rotation)
    if flip:tok=np.fliplr(tok)
    tok=tok.copy();label,distance,reachable=S.bfs(tok)
    expected=cycles*(height+width-2*abs(offset))+(cycles-1)*gap
    assert distance==expected and reachable==int((tok>0).sum())
    return tok,label,distance


def prepare():
    parent=json.loads((S.HERE/"protocol.json").read_text());base=2026111410
    geometries=[]
    for i in range(64):
        rng=np.random.default_rng(np.random.SeedSequence([base,i]))
        geometries.append([int(rng.choice([4,6,8])),int(rng.choice([12,16,20])),int(rng.integers(32)),int(rng.integers(16)),int(rng.integers(4)),int(rng.integers(2))])
    dest=S.OUT/"tie_controls";dest.mkdir(parents=True,exist_ok=True);datasets=[];jobs=[]
    for cycles in [1,4,8,12]:
        signatures=[]
        for condition in ["tied","plus","minus"]:
            items=[make_graph(cycles,condition,g) for g in geometries];name=f"k{cycles}-{condition}";path=dest/f"{name}.npz"
            if path.exists():raise FileExistsError(path)
            tokens=np.stack([x[0] for x in items]);assert len({t.tobytes() for t in tokens})==64
            np.savez_compressed(path,tokens=tokens,labels=np.stack([x[1] for x in items]),shortest_edges=np.asarray([x[2] for x in items]),geometries=np.asarray(geometries))
            signature=[]
            for tok in tokens:
                op=tok>0;degrees=np.zeros_like(tok,int);degrees[1:]+=op[:-1];degrees[:-1]+=op[1:];degrees[:,1:]+=op[:,:-1];degrees[:,:-1]+=op[:,1:]
                signature.append((int(op.sum()),int((op[1:]&op[:-1]).sum()+(op[:,1:]&op[:,:-1]).sum()),tuple(np.bincount(degrees[op],minlength=5))))
            signatures.append(signature)
            datasets.append({"cohort":name,"length":160,"loop_percent":0,"cycles":cycles,"condition":condition,"n":64,"expected_shortest_path_count":2**cycles if condition=="tied" else 1,"base_seed":base,"path":str(path.relative_to(S.ROOT)),"sha256":S.sha(path)})
            for seed in range(3):jobs.append({"id":f"{name}-s{seed}","cohort":name,"length":160,"loop_percent":0,"cycles":cycles,"condition":condition,"seed":seed,"arm":"original","answer_steps":8192})
        assert signatures[0]==signatures[1]==signatures[2]
    protocol=dict(parent,id="causal-geodesic-tie-controls",frozen_utc=S.utc(),status="frozen_before_outcomes",datasets=datasets,jobs=jobs,
        execution_script_sha256=S.sha(__file__),core_script_sha256=S.sha(S.__file__),H2=None,H2b=None,
        H2j="At eta0.5, tied serial-cycle graphs lose at least10percentage points joint-three-model success compared with the average of paired plus/minus unequal-route controls.",
        decision="Within each of64layout/geometry clusters, average the4cycle-count conditions equally, preserving all3models and3paired arms. Joint success requires all3models. Primary difference=mean(0.5*(jointplus+jointminus)-jointtied).10000paired-cluster bootstrap95% interval: support if lower>=10pp, falsify if upper<10pp, otherwise inconclusive. If no discordance in any condition/arm, also report exact one-sided95% upper bound for unseen discordant-cluster probability; bootstrap[0,0] is not a population guarantee.",
        generator="160grid. Series of1/4/8/12rectangular cycles joined by length4straight bridges.64fresh geometry/layout clusters (width4/6/8,height12/16/20,translations,D4orientation), paired across conditions andcyclecount. Tied ports at opposite side midpoints; unequal controls shift both ports and bridges by+2or-2cells. Each pair matches node count, edge count, degree histogram and cycle sizes. Tied has2^kshortest paths, unequal has1; unequal shortest distance is4kshorter, an explicit remaining difference.",
        limitation="Tests this controlled family only. Degeneracy alone may be insufficient to explain the selected random-maze failure. Shorter unequal geodesic lengths prevent interpreting a difference as solely tie count. Fixed models/layoutclusters are not independent training draws. No selected hard instances or model/eta rescue.",
        inference=dict(parent["inference"],answer_max_steps=8192,answer_eta=.5),
        primary_units="64paired randomized geometry/layout clusters, preserving4cyclecounts and all3models;768input conditions and2304model-input evaluations.",
        diagnostic="Save every predictedboard plus final discrete energy, canonical valid-path energy and minimum discrete flip margin. Oracle label energy is computed only after inference. Check topology and shortest-path counts independently before interpreting outcomes.",
        confirmation="Independent generator2026111410; selected-maze38 forensic interventions excluded entirely. No additional eta arms or outcome-dependent data changes.")
    pp=S.HERE/"tie_controls-protocol.json"
    if pp.exists():raise FileExistsError(pp)
    S.save(pp,protocol);S.emit("tie_controls_frozen",sha256=S.sha(pp),jobs=len(jobs),unique_inputs=768)


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
        output = {k: [] for k in ("predictions", "answer_steps", "stopped", "hidden_steps", "hidden_good", "prediction_energy", "oracle_energy", "minimum_flip_delta")}
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
                torch.cuda.synchronize(); answer_seconds += time.perf_counter() - part
            for key, value in (("predictions", pred.bool()), ("answer_steps", used), ("stopped", stopped), ("hidden_steps", ps), ("hidden_good", good), ("prediction_energy", prediction_energy), ("oracle_energy", oracle_energy), ("minimum_flip_delta", minimum_flip_delta)):
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
    else:worker("tie_controls",args.gpu)
