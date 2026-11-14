"""Prespecified descriptive interventions on the selected cyclic failure."""
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


def prepare():
    parent=json.loads((S.HERE/"tie_controls-protocol.json").read_text());base=2026111411
    source=S.OUT/"eta_boundary/failure-forensics";failure=np.load(source/"s2-eta0.5-states.npz");tok=failure["tokens"];pred=failure["prediction"].astype(bool)
    union=np.load(source/"shortest-path-union.npz")["shortest_union"];op=tok>0
    degree=np.zeros_like(tok,int);degree[1:]+=op[:-1];degree[:-1]+=op[1:];degree[:,1:]+=op[:,:-1];degree[:,:-1]+=op[:,1:]
    selected={tuple(map(int,x)) for x in np.argwhere(pred)}
    def neighbors(node):return sorted((node[0]+dr,node[1]+dc) for dr,dc in S.SIDES if (node[0]+dr,node[1]+dc) in selected)
    junctions=sorted(node for node in selected if len(neighbors(node))>2);assert len(junctions)==2
    paths=[]
    for node in neighbors(junctions[0]):
        prev=junctions[0];current=node;path=[prev,current]
        while current!=junctions[1] and len(neighbors(current))==2:
            nxt=next(x for x in neighbors(current) if x!=prev);prev,current=current,nxt;path.append(current)
        if current==junctions[1]:paths.append(path)
    assert len(paths)==2 and all(len(p)==43 for p in paths)
    def eligible(node):
        if tok[node]!=1 or degree[node]!=2:return False
        adjacent=[(node[0]+dr,node[1]+dc) for dr,dc in S.SIDES if 0<=node[0]+dr<90 and 0<=node[1]+dc<90 and tok[node[0]+dr,node[1]+dc]]
        if any(degree[n]!=2 for n in adjacent):return False
        edited=tok.copy();edited[node]=0
        label,distance,reachable=S.bfs(edited)
        return distance==204 and reachable==int((edited>0).sum())
    targets=[]
    for path in paths:
        ordered=sorted(enumerate(path[1:-1],start=1),key=lambda z:(abs(z[0]-(len(path)-1)/2),z[1]))
        targets.append(next(node for _,node in ordered if eligible(node)))
    candidates=[tuple(map(int,p)) for p in np.argwhere((tok==1)&~union) if eligible(tuple(map(int,p)))]
    assert len(candidates)>=32
    order=np.random.default_rng(base).choice(len(candidates),32,replace=False);controls=[candidates[int(i)] for i in order]
    cases=[{"kind":"targeted_branch_A","removed":list(targets[0])},{"kind":"targeted_branch_B","removed":list(targets[1])}]+[{"kind":"matched_noncritical_control","removed":list(node)} for node in controls]
    items=[]
    for case in cases:
        edited=tok.copy();edited[tuple(case["removed"])]=0;label,distance,reachable=S.bfs(edited);assert distance==204 and reachable==int((edited>0).sum());items.append((edited,label,distance))
    dest=S.OUT/"forensic_edits";dest.mkdir(parents=True,exist_ok=True);path=dest/"data.npz"
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,tokens=np.stack([x[0] for x in items]),labels=np.stack([x[1] for x in items]),shortest_edges=np.asarray([x[2] for x in items]))
    datasets=[{"cohort":"selected-maze38-edits","length":90,"loop_percent":20,"n":len(cases),"path":str(path.relative_to(S.ROOT)),"sha256":S.sha(path)}]
    jobs=[{"id":f"edits-s{s}","cohort":"selected-maze38-edits","length":90,"loop_percent":20,"seed":s,"arm":"original","answer_steps":8192} for s in range(3)]
    protocol=dict(parent,id="selected-failure-graph-interventions",frozen_utc=S.utc(),status="post_outcome_design_frozen_before_intervention_outcomes",datasets=datasets,jobs=jobs,cases=cases,
        execution_script_sha256=S.sha(__file__),core_script_sha256=S.sha(S.__file__),H2j=None,
        question="Descriptive: does deleting one central interior node from either42edge retained cycle branch restore fixedeta0.5 shortest-path solving in models1/2, while matched noncritical cycle-node deletions do not?",
        selection="Selected original loop20maze38 and seed2eta0.5branched answer after observing failure. Find its two42edge paths between2branchjunctions. Each target is nearest path midpoint among degree2nodes with degree2neighbors, whose deletion preserves fullgraph connectedness and S-Gdistance204. Controls:32seeded random eligible degree2nodes outside original shortest-path union. No outcome-driven selection or rescue.",
        matching="All edits remove exactly1node and2edges with equal original neighborhood degree2-2-2; connectedness and S-Gdistance204 retained. Targeted nodes lie on retained geodesic branches; controls outside every shortest path. Same node/edge counts and degree histogram. Independent audit verifies these properties.",
        primary_units="One selected graph,2targeted edits and32matched noncritical edits,3fixedmodels. All outcomes descriptive; no confirmatory population CI or incorporation into H2j.",
        diagnostic="Original unedited outcomes reused without rerunning. Everyedited prediction, discrete energy, valid oracle-path energy and minimumflipmargin saved. No parameter or inference changes.",
        confirmation="Not a confirmation cohort. Independent constructed tie-controls remain a separate frozen study. This targeted evidence cannot establish general failure prevalence.",
        original_graph_sha256=S.sha(source/"shortest-path-union.npz"),selection_state_sha256=S.sha(source/"s2-eta0.5-states.npz"),control_candidates=len(candidates),junctions=[list(x) for x in junctions])
    pp=S.HERE/"forensic_edits-protocol.json"
    if pp.exists():raise FileExistsError(pp)
    S.save(pp,protocol);S.emit("forensic_edits_frozen",sha256=S.sha(pp),targets=targets,control_candidates=len(candidates),jobs=len(jobs))


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
    else:worker("forensic_edits",args.gpu)
