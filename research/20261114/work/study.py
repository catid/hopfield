"""Frozen-model Maze-Hard size transfer. All outputs belong to the new campaign."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/research/20261114/work"
UP = ROOT / "upstream"
SIDES = ((1, 0), (0, -1), (0, 1), (-1, 0))


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def emit(event, **kwargs):
    print(json.dumps({"utc": utc(), "event": event, **kwargs}), flush=True)


def bfs(tok):
    import numpy as np
    start = tuple(np.argwhere(tok == 2)[0]); goal = tuple(np.argwhere(tok == 3)[0])
    q = deque([start]); parent = {start: None}; distances = {start: 0}
    h, w = tok.shape
    while q:
        cur = q.popleft()
        for dr, dc in SIDES:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and tok[nxt] and nxt not in parent:
                parent[nxt] = cur; distances[nxt] = distances[cur] + 1; q.append(nxt)
    assert goal in parent
    label = np.zeros_like(tok, dtype=bool); cur = goal
    while cur is not None:
        label[cur] = True; cur = parent[cur]
    return label, distances[goal], len(parent)


def maze(length, rate, seed):
    """Randomized DFS tree on even-coordinate sites, then open closed connectors."""
    import numpy as np
    rng = np.random.default_rng(seed); n = length // 2
    tok = np.zeros((length, length), dtype=np.uint8)
    seen = {(0, 0)}; stack = [(0, 0)]; tok[0, 0] = 1
    while stack:
        r, c = stack[-1]
        options = [(r + dr, c + dc) for dr, dc in SIDES
                   if 0 <= r + dr < n and 0 <= c + dc < n and (r + dr, c + dc) not in seen]
        if not options:
            stack.pop(); continue
        nr, nc = options[int(rng.integers(len(options)))]
        tok[2 * nr, 2 * nc] = 1; tok[r + nr, c + nc] = 1
        seen.add((nr, nc)); stack.append((nr, nc))
    candidates = []
    for r in range(n):
        for c in range(n):
            if r + 1 < n and not tok[2 * r + 1, 2 * c]: candidates.append((2 * r + 1, 2 * c))
            if c + 1 < n and not tok[2 * r, 2 * c + 1]: candidates.append((2 * r, 2 * c + 1))
    count = int(round(rate * len(candidates)))
    for k in rng.permutation(len(candidates))[:count]: tok[candidates[int(k)]] = 1
    tok[0, 0] = 2; tok[2 * n - 2, 2 * n - 2] = 3
    tok = np.rot90(tok, int(rng.integers(4)))
    if rng.integers(2): tok = np.fliplr(tok)
    tok = tok.copy(); label, distance, reachable = bfs(tok)
    assert reachable == int((tok > 0).sum())
    return tok, label, distance, count, len(candidates)


def imports():
    sys.path[:0] = [str(UP), str(UP / "experiments"), str(HERE / "reference_src")]
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import maze_hard_latent as H
    from cram import maze_h3 as M
    return torch, H, M


def model_for(seed, length, arm, device="cpu", dtype=None):
    torch, H, _ = imports()
    recipe = H.Recipe(**json.loads((UP / "experiments/recipes/maze-hard-latent.json").read_text())["recipe"])
    model = H.build(recipe)
    checkpoint = OUT / "models" / f"seed{seed}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    if arm == "mass_scaled":
        with torch.no_grad(): model.mu.add_(2 * math.log(30 / length))
    model = model.to(device=device, dtype=dtype)
    model.triton_settle = False
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def prepare():
    import numpy as np
    protocol_path = HERE / "protocol.json"
    if protocol_path.exists(): raise FileExistsError(protocol_path)
    source = {str(p.relative_to(ROOT)): sha(p) for d in (UP / "cram", UP / "experiments") for p in sorted(d.glob("*.py"))}
    source.update({str(p.relative_to(ROOT)): sha(p) for p in sorted((HERE / "reference_src/hopfield_cram").glob("*.py"))})
    models = {str(s): {"path": str((OUT / "models" / f"seed{s}.pt").relative_to(ROOT)),
                       "sha256": sha(OUT / "models" / f"seed{s}.pt")} for s in range(3)}
    datasets = []
    for cohort, base in (("discovery", 2026111401), ("confirmation", 2026111402)):
        for length in (30, 60):
            for percent in (0, 5, 20):
                items = [maze(length, percent / 100, np.random.SeedSequence([base, length, percent, i])) for i in range(128)]
                arrays = {"tokens": np.stack([x[0] for x in items]), "labels": np.stack([x[1] for x in items]),
                          "shortest_edges": np.asarray([x[2] for x in items]), "opened_loops": np.asarray([x[3] for x in items]),
                          "eligible_walls": np.asarray([x[4] for x in items])}
                path = OUT / "data" / f"{cohort}-L{length}-loop{percent}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists(): raise FileExistsError(path)
                np.savez_compressed(path, **arrays)
                datasets.append({"cohort": cohort, "length": length, "loop_percent": percent, "n": 128,
                                 "path": str(path.relative_to(ROOT)), "sha256": sha(path), "base_seed": base,
                                 "distance_min": int(arrays["shortest_edges"].min()), "distance_max": int(arrays["shortest_edges"].max())})
    jobs = []
    for length in (30, 60):
        for percent in (0, 5, 20):
            for seed in range(3):
                for arm in (("original",) if length == 30 else ("original", "mass_scaled")):
                    jobs.append({"id": f"L{length}-loop{percent}-s{seed}-{arm}", "length": length,
                                 "loop_percent": percent, "seed": seed, "arm": arm, "cohort": "discovery"})
    protocol = {
        "id": "maze-size-transfer-wave1", "frozen_utc": utc(), "status": "frozen_before_outcomes",
        "H2": "At least99% of generated60x60 mazes are solved by all three frozen models, requiring valid shortest simple paths, hidden convergence and answer stopping.",
        "H2b": "Mass scaling(30/L)^2 improves60x60 per-maze mean solve rate across the three models by at least5 percentage points.",
        "primary_length": 60, "rates_equal_weighted": [0, 5, 20], "unique_primary_mazes": 384,
        "model_seeds": [0, 1, 2], "training": "None; existing fresh30x30 trained models reused by immutable hash",
        "generator": "Randomized DFS spanning tree on even grid coordinates with n=floor(L/2), endpoints opposite coarse-grid corners; open exactly round(rate*eligible_closed_connectors) independently selected connectors; rotate/reflect uniformly. This is a new generated distribution, not the original Maze-Hard benchmark.",
        "primary_metric": "All-three-model joint success per unique maze for H2; mean of three success flags per unique maze for paired H2b. No treating repeated models as independent puzzles.",
        "H2_decision": "One-sided95% Clopper-Pearson bound across384 unique mazes: support if lower>=.99, falsify if upper<.99, otherwise inconclusive. Conservative joint-success claim, not average-model accuracy.",
        "H2b_decision": "Paired bootstrap10000 replicates over unique mazes stratified by loop rate, three model seeds averaged within maze. Support if95% interval lower>=5pp, falsify if upper<5pp, otherwise inconclusive.",
        "uncertainty_scope": "Generator-distribution uncertainty; three frozen training seeds are not a random population of arbitrary models.",
        "confirmation": "Separate pre-generated base seed2026111402; no confirmation outcomes until root authorizes the frozen follow-up.",
        "inference": {"hidden_backend": "unmodified PyTorch", "hidden_max_steps": 4000, "hidden_check": 8,
                      "hidden_tolerance": 1e-6, "answer_max_steps": 2048, "answer_eta": .2,
                      "answer_margin": 3., "answer_check": 16, "batch_size": 32, "dtype": "float32"},
        "budget": "Bounded first wave27jobs x128mazes; nominal <=20 GPU-minutes per arm; each job watchdog600s. A timeout remains incomplete, never zero or successful.",
        "arm_30_alias": "Mass factor is exactly1 at30, so scaled30 outputs alias original rather than repeat identical inference.",
        "baselines": "BP AdamW outside post-training transfer protocol; BFS is correctness oracle, not a learned comparison.",
        "models": models, "datasets": datasets, "jobs": jobs, "source_sha256": source,
        "script_sha256": sha(__file__), "upstream_commit": "62bbe830c7d6f87f2ba18f30a28039258b986454",
    }
    save(protocol_path, protocol)
    emit("protocol_frozen", path=str(protocol_path), sha256=sha(protocol_path), jobs=len(jobs))


def preflight():
    import numpy as np
    torch, _, M = imports()
    from hopfield_cram import MazeHardEnergy
    results = []
    for length in (4, 8, 30, 60):
        tok, _, _, _, _ = maze(length, .2, np.random.SeedSequence([77123, length]))
        t = torch.as_tensor(tok[None].astype(np.int64))
        model = model_for(0, length, "original", dtype=torch.float64)
        psi, _ = model.settle(t, tol=1e-10)
        ref = MazeHardEnergy(length, length, tolerance=1e-10).double()
        perm = [model.aoffs.index(d) for d in ref.attention_offsets]
        with torch.no_grad():
            ref.maze.tables.copy_(model.T); ref.maze.unary_cost.copy_(-model.b)
            ref.log_excess_mass.copy_(model.mu.squeeze()); ref.attention_bias.copy_(model.beta[perm]); ref.link_strength.copy_(model.kappa[perm])
            ctx = ref.prepare(t); live = t.flatten(1) > 0
            error = float((-ctx.log_hidden[live] - psi.flatten(1)[live]).abs().max())
            assert error < 2e-7, (length, error)
            p = torch.rand((1, length * length), dtype=torch.float64, generator=torch.Generator().manual_seed(7))
            states = torch.stack([1 - p, p], -1); fields = ref.fields(states, ctx)
            native = model.field(t, p.reshape_as(t), model.couplings(t, psi)).flatten(1)
            free = t.flatten(1) == 1
            field_error = float((-(fields[..., 1] - fields[..., 0])[free] - native[free]).abs().max())
            assert field_error < 2e-7, (length, field_error)
        record = {"length": length, "independent_hidden_max_abs": error, "independent_field_max_abs": field_error}
        if length == 4:
            coords = np.argwhere(tok > 0); lookup = {tuple(c): i for i, c in enumerate(coords)}
            matrix = np.eye(len(coords)) * (4 + float(model.masses()[0])); source = np.zeros(len(coords))
            for i, c in enumerate(coords):
                if tok[tuple(c)] == 3: source[i] = 1
                for dr, dc in SIDES:
                    q = (int(c[0] + dr), int(c[1] + dc))
                    if q in lookup: matrix[i, lookup[q]] = -1
            z = np.linalg.solve(matrix, source)
            assert np.all(z > 0)
            exact = -np.log(z); predicted = np.asarray([float(psi[0, int(r), int(c)]) for r, c in coords])
            dense_error = float(np.max(np.abs(exact - predicted))); assert dense_error < 2e-7
            record["dense_linear_solve_log_error"] = dense_error
        results.append(record)
    result = {"status": "passed", "utc": utc(), "checks": results, "script_sha256": sha(__file__),
              "protocol_sha256": sha(HERE / "protocol.json"), "new_evaluation_outcomes": False}
    save(OUT / "preflight.json", result); emit("preflight_passed", checks=results)


def board_metrics(tokens, labels, predictions, stopped, hidden_good):
    import numpy as np
    valid, shortest, distances = [], [], []
    for tok, label, pred in zip(tokens, labels, predictions.astype(bool)):
        expected, distance, reachable = bfs(tok)
        assert np.array_equal(expected, label) and reachable == int((tok > 0).sum())
        s = tuple(np.argwhere(tok == 2)[0]); g = tuple(np.argwhere(tok == 3)[0])
        degree = np.zeros_like(tok, dtype=np.int16)
        degree[1:] += pred[:-1]; degree[:-1] += pred[1:]; degree[:, 1:] += pred[:, :-1]; degree[:, :-1] += pred[:, 1:]
        ok = bool(pred[s] and pred[g] and not np.any(pred & (tok == 0)) and degree[s] == degree[g] == 1
                  and np.all(degree[pred & (tok == 1)] == 2))
        reached = {s}; queue = deque([s])
        while queue:
            c = queue.popleft()
            for dr, dc in SIDES:
                q = (c[0] + dr, c[1] + dc)
                if 0 <= q[0] < tok.shape[0] and 0 <= q[1] < tok.shape[1] and pred[q] and q not in reached:
                    reached.add(q); queue.append(q)
        ok = ok and len(reached) == int(pred.sum())
        valid.append(ok); shortest.append(ok and int(pred.sum()) == distance + 1); distances.append(distance)
    valid = np.asarray(valid); shortest = np.asarray(shortest)
    exact = np.all(predictions == labels, axis=(1, 2))
    success = shortest & stopped & hidden_good
    return {"success": success, "valid_path": valid, "shortest_path": shortest,
            "canonical_exact": exact, "shortest_edges": np.asarray(distances)}


def worker(gpu):
    import numpy as np
    lockpath = ROOT / f"artifacts/gpu{gpu}.lock"; lockpath.parent.mkdir(parents=True, exist_ok=True)
    lock = lockpath.open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], text=True).strip()
    uuid = inventory.split(",")[1].strip()
    owners = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
    assert uuid not in owners, "GPU has existing owner"
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    torch, _, M = imports()
    from cram.maze_learned import gather
    protocol = json.loads((HERE / "protocol.json").read_text())
    assert protocol["script_sha256"] == sha(__file__)
    assert json.loads((OUT / "preflight.json").read_text())["status"] == "passed"
    for path, expected in protocol["source_sha256"].items(): assert sha(ROOT / path) == expected, path
    heartbeat = OUT / f"worker{gpu}.json"
    state = {"status": "running", "gpu": gpu, "host": platform.node(), "pid": os.getpid(),
             "started_utc": utc(), "inventory": inventory, "protocol_sha256": sha(HERE / "protocol.json"), "jobs_complete": []}
    save(heartbeat, state)
    for index, job in enumerate(protocol["jobs"]):
        if index % 2 != gpu: continue
        directory = OUT / "runs" / job["id"]
        if (directory / "result.json").exists():
            assert json.loads((directory / "result.json").read_text())["status"] == "complete"
            state["jobs_complete"].append(job["id"]); continue
        directory.mkdir(parents=True, exist_ok=True)
        data = next(d for d in protocol["datasets"] if d["cohort"] == job["cohort"] and d["length"] == job["length"] and d["loop_percent"] == job["loop_percent"])
        assert sha(ROOT / data["path"]) == data["sha256"]
        assert sha(OUT / "models" / f"seed{job['seed']}.pt") == protocol["models"][str(job["seed"])]["sha256"]
        arrays = np.load(ROOT / data["path"]); tokens = arrays["tokens"]; labels = arrays["labels"]
        model = model_for(job["seed"], job["length"], job["arm"], "cuda")
        output = {k: [] for k in ("predictions", "answer_steps", "stopped", "hidden_steps", "hidden_good")}
        started = time.perf_counter(); hidden_s = answer_s = 0.
        state.update(current_job=job, completed_mazes=0, updated_utc=utc()); save(heartbeat, state)
        emit("job_start", gpu=gpu, **job)
        for lo in range(0, len(tokens), protocol["inference"]["batch_size"]):
            t = torch.as_tensor(tokens[lo:lo + 32].astype(np.int64), device="cuda")
            with torch.no_grad():
                torch.cuda.synchronize(); a0 = time.perf_counter()
                psi, hidden_steps = model.settle(t, max_steps=4000, check=8, tol=1e-6)
                lg, la, lc = model.parts(t, model.mu)
                new = la - torch.logsumexp(torch.cat([lg - gather(psi, model.zoffs, math.inf), lc[:, None]], 1), 1)
                stable = ((new == psi) | ((new - psi).abs() <= 1e-6 * (1 + new.abs()))).flatten(1).all(1)
                finite = (torch.isfinite(psi) | (t == 0)).flatten(1).all(1)
                couplings = model.couplings(t, psi)
                torch.cuda.synchronize(); hidden_s += time.perf_counter() - a0; a0 = time.perf_counter()
                pred, used, stopped = M.descend(model, t, couplings, steps=2048, eta=.2, margin=3., certify_every=16)
                torch.cuda.synchronize(); answer_s += time.perf_counter() - a0
            for name, tensor in (("predictions", pred.bool()), ("answer_steps", used), ("stopped", stopped),
                                 ("hidden_steps", hidden_steps), ("hidden_good", stable & finite)):
                output[name].append(tensor.cpu().numpy())
            state.update(completed_mazes=min(lo + 32, len(tokens)), updated_utc=utc(), elapsed_s=time.perf_counter()-started)
            save(heartbeat, state); emit("batch_done", gpu=gpu, job=job["id"], n=state["completed_mazes"], elapsed_s=state["elapsed_s"])
            if time.perf_counter() - started > 600: raise TimeoutError("Predeclared600s job budget exceeded; incomplete retained")
        output = {k: np.concatenate(v) for k, v in output.items()}
        metrics = board_metrics(tokens, labels, output["predictions"], output["stopped"], output["hidden_good"])
        output.update(metrics)
        np.savez_compressed(directory / "predictions.npz", **output)
        result = {"status": "complete", "job": job, "utc": utc(), "n": len(tokens), "solved": int(output["success"].sum()),
                  "valid_paths": int(output["valid_path"].sum()), "shortest_paths": int(output["shortest_path"].sum()),
                  "canonical_exact": int(output["canonical_exact"].sum()), "wrong_stops": int((output["stopped"] & ~output["shortest_path"]).sum()),
                  "answer_budget_failures": int((~output["stopped"]).sum()), "hidden_failures": int((~output["hidden_good"]).sum()),
                  "hidden_steps_mean": float(output["hidden_steps"].mean()), "answer_steps_mean": float(output["answer_steps"].mean()),
                  "hidden_seconds": hidden_s, "answer_seconds": answer_s, "wall_seconds": time.perf_counter()-started,
                  "mass": float(model.masses()[0]), "mu": float(model.mu[0]), "host": platform.node(), "gpu": gpu,
                  "inventory": inventory, "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                  "script_sha256": sha(__file__), "protocol_sha256": sha(HERE / "protocol.json"),
                  "checkpoint_sha256": protocol["models"][str(job["seed"])]["sha256"], "dataset_sha256": data["sha256"],
                  "predictions_sha256": sha(directory / "predictions.npz"),
                  "timing_scope": "Instrumented forward evaluation on Blackwell; hidden timing includes residual check and attention. No matched H200 claim."}
        save(directory / "result.json", result); emit("job_complete", **result)
        state["jobs_complete"].append(job["id"]); save(heartbeat, state)
        del model, t, psi, couplings
    state.update(status="complete", completed_utc=utc()); save(heartbeat, state); emit("worker_complete", gpu=gpu)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("command", choices=("prepare", "preflight", "worker")); p.add_argument("--gpu", type=int, choices=(0, 1))
    a = p.parse_args()
    if a.command == "prepare": prepare()
    elif a.command == "preflight": preflight()
    else: worker(a.gpu)
