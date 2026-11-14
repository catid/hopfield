"""Execute frozen confirmation/budget protocols using the unchanged core routines."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import study as S


def prepare():
    initial = json.loads((S.HERE / "protocol.json").read_text())
    for campaign in ("confirmation", "budget_extension"):
        destination = S.OUT / campaign
        destination.mkdir(parents=True, exist_ok=True)
        protocol = dict(initial)
        protocol.update(id=f"maze-size-transfer-{campaign}", frozen_utc=S.utc(), parent_protocol_sha256=S.sha(S.HERE / "protocol.json"),
                        execution_script_sha256=S.sha(__file__), core_script_sha256=S.sha(S.__file__))
        if campaign == "confirmation":
            protocol["jobs"] = [dict(j, cohort="confirmation", answer_steps=2048) for j in initial["jobs"]]
            protocol["status"] = "unchanged_frozen_hypotheses_confirmation"
        else:
            n = 256; length = 60; base = 2026111403
            items = [S.maze(length, 0, np.random.SeedSequence([base, length, 0, i])) for i in range(n)]
            path = destination / "data.npz"
            if path.exists(): raise FileExistsError(path)
            np.savez_compressed(path, tokens=np.stack([x[0] for x in items]), labels=np.stack([x[1] for x in items]),
                                shortest_edges=np.asarray([x[2] for x in items]))
            protocol["datasets"] = [{"cohort": campaign, "length": 60, "loop_percent": 0, "n": n,
                                     "path": str(path.relative_to(S.ROOT)), "sha256": S.sha(path), "base_seed": base}]
            protocol["jobs"] = [{"id": f"L60-tree-s{s}-budget{budget}", "length": 60, "loop_percent": 0, "seed": s,
                                 "arm": "original", "cohort": campaign, "answer_steps": budget}
                                for s in range(3) for budget in (2048, 8192)]
            protocol.update(H2=None, H2b=None, H2c="At least99% of new generated60x60 trees are solved by all3 frozen models with8192 answersteps and originalmass.",
                H2c_decision="One-sided95% Clopper-Pearson interval across256 distinct trees: support if lower>=.99; falsify if upper<.99; otherwise inconclusive. Even256/256 gives lower98.836%, so report perfect observed success with inferential inconclusiveness rather than overclaim99% population accuracy.",
                H2d="8192 answersteps improve paired mean solve rate by at least5pp relative to2048 on the same new trees.",
                H2d_decision="10000 paired maze bootstrap replicates; models averaged within maze; support if95% lower>=5pp; falsify if upper<5pp; otherwise inconclusive.",
                exploratory_diagnostics="Before outcomes, compute shortest S-G path length, exact tree diameter via double BFS, and max distance of any open node to the unique S-G path. Explore association with baseline failure and extended answer steps; no topology diagnostic enters inference.",
                unique_primary_mazes=n, inference=dict(initial["inference"], answer_max_steps=[2048,8192]),
                confirmation="New generator seed2026111403; independent of both previous cohorts. No further budget increases if8192 fails.")
        path = S.HERE / f"{campaign}-protocol.json"
        if path.exists(): raise FileExistsError(path)
        S.save(path, protocol)
        S.emit("followup_frozen", campaign=campaign, sha256=S.sha(path), jobs=len(protocol["jobs"]))


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
        model = S.model_for(job["seed"], job["length"], job["arm"], "cuda")
        output = {k: [] for k in ("predictions", "answer_steps", "stopped", "hidden_steps", "hidden_good")}
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
                pred, used, stopped = M.descend(model, t, a, steps=job["answer_steps"], eta=.2, margin=3., certify_every=16)
                torch.cuda.synchronize(); answer_seconds += time.perf_counter() - part
            for key, value in (("predictions", pred.bool()), ("answer_steps", used), ("stopped", stopped), ("hidden_steps", ps), ("hidden_good", good)):
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
                  "mass": float(model.masses()[0]), "host": platform.node(), "gpu": gpu, "inventory": inventory,
                  "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version,
                  "script_sha256": S.sha(__file__), "core_script_sha256": S.sha(S.__file__), "protocol_sha256": S.sha(protocol_path),
                  "checkpoint_sha256": protocol["models"][str(job["seed"])]["sha256"], "dataset_sha256": data["sha256"],
                  "predictions_sha256": S.sha(run / "predictions.npz"), "timing_scope": "Instrumented PyTorch eval; hidden includes residual/attention; no H200 comparison"}
        S.save(run / "result.json", result); S.emit("job_complete", **result)
        state["jobs_complete"].append(job["id"]); S.save(heartbeat, state)
    state.update(status="complete", completed_utc=S.utc()); S.save(heartbeat, state); S.emit("worker_complete", campaign=campaign, gpu=gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["prepare", "worker"])
    parser.add_argument("--campaign", choices=["confirmation", "budget_extension"]); parser.add_argument("--gpu", type=int, choices=[0,1])
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    else: worker(args.campaign, args.gpu)
