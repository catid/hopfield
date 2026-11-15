"""Graph-informed offline decoder: native path acceptance, batched lambda1 fallback."""
import argparse
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
import larger_relaxation as L
import binary_preserving as B
from fast_path_validator import Validator,LIBRARY

DEST=S.OUT/"selective_fallback"
PP=S.HERE/"selective_fallback-protocol.json"
BASE=2026111506
N=16384
LENGTH=120
BATCH=64
REPS=3


def freeze():
    parent=json.loads((S.HERE/"larger_relaxation-protocol.json").read_text());gate=DEST/"validator-gate.json";v=json.loads(gate.read_text());assert v["status"]=="passed"
    assert v["source_sha256"]==S.sha(S.HERE/"fast_path_validator.c") and v["wrapper_sha256"]==S.sha(S.HERE/"fast_path_validator.py")
    p=dict(parent,id="graph-informed-native-acceptance-lambda-fallback",frozen_utc=S.utc(),execution_script_sha256=S.sha(__file__))
    p["cohort"]=dict(parent["cohort"],base_seed=BASE)
    p["inference"]=dict(parent["inference"],selector="Native graph-feasibility acceptance; fixed-lambda fallback only on rejected inputs")
    p["initialization"]={"arms":["native","always_lambda1","selective_fallback"],"source":"All answer calls initialize every logit at exactly float32(3). No noise or adaptive initial logits.","numpy_version":np.__version__}
    p["retained_evidence"]="All native/always-lambda repetition0 full logits, boards, steps, flags and diagnostics; actual queued fallback logits and diagnostics; exact native acceptance mask; per-repetition board/step/flag parity hashes and complete component timings. Native accepted outputs are reused exactly in the selective decoder. Diagnostic recomputations are separate."
    p["primary"]={"id":"H2t","hypothesis":"On fresh120x120 loop20, graph-informed native acceptance with batched lambda1 fallback improves joint-three-model valid-shortest success by at least0.05pp and median measured answer overhead is at most2% including validator and fallback preparation.","accuracy_target_pp":.05,"overhead_target_percent":2.,"unit":"16384 independent inputs; three fixed models are not independent trials","accuracy_decision":"Paired CP marginal tail alpha=.025 gives each separately one-sided95% net bound. Support accuracy if lower>=0.05pp, falsify if upper<0.05pp, otherwise inconclusive.","runtime_decision":"Three complete fresh-cohort timing repetitions, no dropping. Per repetition sum both workers' elapsed common native answer seconds and all incremental validator, queue, fallback setup, solve and output-copy seconds. Answer overhead=100*incremental/native_answer. Runtime passes iff median of the three ratios<=2%. Report all ratios and separately end-to-end overhead with startup, input transfer, hidden settling and common output transfers included. These are measured timing ratios, no statistical runtime CI.","conjunction":"Support only if accuracy supports and runtime passes; falsify if accuracy falsifies or runtime fails; otherwise inconclusive. No cap, batching, sample, timing-repetition or implementation changes after outcomes."}
    p["decoder"]={"native_acceptance":"Native finite/certified output and hidden convergence plus supplied graph endpoints selected, no walls, endpoint degree1, all other selected degree2, endpoint traversal covers every selected cell. Exact compiled checker; no shortest distances/labels/length optimization.","fallback":"Rejects only: same native initial u3, fixed lambda+1, unchanged original binary certificate. Always return this fallback output even if it remains invalid; no extra repair.","batching":"Offline per-worker/per-model queue: retain native hidden couplings, defer rejected inputs until end of the16384-input cohort, concatenate into batches<=64. Valid native outputs available immediately, rejected outputs incur cohort deferral. No online per-input latency claim.","added_task_knowledge":"Explicit graph-feasibility validator is a new task-informed decoder component, not paper-native CRAM and not an energy-only change.","controls":"Native prefix exactly shared. Always-lambda1 computed once in original64-input batches, independently of the fallback. Actual fallback runs only rejected subsets in all three timing repetitions; compare outputs against always-lambda control and retain any batching differences. No assumption of parity."}
    p["timing"]={"repetitions":REPS,"native_scope":"Synchronized input CPU-to-GPU transfer, model construction/load once per repetition, hidden settle/readout and convergence check, native answer solve, and CPU return of board/steps/certificate/finite arrays. Shared native costs serve both native and selective controls.","incremental_scope":"Compiled validator call and conversion, rejected-index extraction, device index creation and tensor gathers, metadata queues, model-group concatenation, fallback initialization, answer solve and CPU board/flag return. All charged to selective answer overhead and end-to-end overhead.","always_lambda_scope":"Paired full-batch solve/CPU-return measured only repetition0. Common hidden/readout reused. Order native/always alternates by job and batch before native validation.","excluded_evidence_costs":"Input generation/loading from archive, audits, model-independent warmup, full-logit diagnostic recording, hashing, serialization and reporting are outside decoder timing but included in worker wall/budget. Warmup done before timed repetitions, not per input.","controls_reuse":"Native scientific arrays from repetition0 reused by selective accepted cases; later native repeats are required fair timing repetitions with exact board/step/flag hash checks. Only rejected subsets rerun lambda for fallback timings; always-lambda scientific outputs run once.","clock":"perf_counter around CUDA-synchronized stages; sum component costs, report startup and all incremental costs separately."}
    p["budget"]={"total_gpu_minute_envelope":30,"per_worker_wall_seconds":900,"watchdog":"Check after every batch; preserve already completed runs and partial state; do not reduce sample/repetitions or extend cap after outcomes."}
    p["implementation_sha256"]={str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in ["larger_relaxation.py","binary_preserving.py","fast_path_validator.c","fast_path_validator.py"]}
    p["validator_gate"]={"path":str(gate.relative_to(S.ROOT)),"sha256":S.sha(gate),"binary_sha256":v["binary_sha256"]}
    p["confirmation"]={"parent_protocol_sha256":S.sha(S.HERE/"larger_relaxation-protocol.json"),"parent_result_sha256":S.sha(S.OUT/"larger_relaxation/audit.json"),"change":"New graph-informed selective decoder and explicit runtime hypothesis; fresh independent inputs. No pooling with prior cohort."}
    p["preflight"]="Existing binary-energy invariance gate, native solver parity, compiled validator gate on4102fixtures/random boards and98304prior outputs; fresh reserved120-grid inputs compare full-batch and subset lambda solve/certificate behavior. Any batch differences are retained, not assumed absent."
    if PP.exists():raise FileExistsError(PP)
    S.save(PP,p);S.emit("selective_protocol_frozen",sha256=S.sha(PP),inputs=N,reps=REPS)


def make_one(index):
    return S.maze(LENGTH,.2,np.random.SeedSequence([BASE,LENGTH,20,index]))


def prepare():
    p=json.loads(PP.read_text());assert p["execution_script_sha256"]==S.sha(__file__);started=time.perf_counter();datasets=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(64):
            path=DEST/f"shard{shard:02d}.npz"
            if path.exists():raise FileExistsError(path)
            indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8))
            np.savez_compressed(path,tokens=np.stack([v[0] for v in items]),labels=np.stack([v[1] for v in items]),shortest_edges=np.array([v[2] for v in items]),indices=indices,opened_loops=np.array([v[3] for v in items]),eligible_walls=np.array([v[4] for v in items]))
            datasets.append({"shard":shard,"path":str(path.relative_to(S.ROOT)),"sha256":S.sha(path),"n":256});S.emit("selective_data",shard=shard,inputs_done=(shard+1)*256)
    S.save(DEST/"datasets.json",{"protocol_sha256":S.sha(PP),"datasets":datasets,"seconds":time.perf_counter()-started})


def acquire(gpu):
    lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).strip()
    owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True);assert inventory.split(',')[1].strip() not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    return lock,inventory


def digest(out):
    h=hashlib.sha256()
    for key in ["predictions","answer_steps","stopped","finite"]:h.update(out[key].tobytes())
    return h.hexdigest()


def cpu_answer(output):
    return {"predictions":output[0].bool().cpu().numpy(),"answer_steps":output[1].cpu().numpy(),"stopped":output[2].cpu().numpy(),"finite":output[3].cpu().numpy()}


def diagnostics(torch,model,t,a,output,strength,cpu):
    with torch.no_grad():
        u=output[4];p=torch.sigmoid(u);free=model.free(t);base=model.energy(t,p,a);reg=B.regularizer(model,t,p,strength)
        cpu.update(final_logits=u.cpu().numpy(),energy=model.energy(t,output[0],a).cpu().numpy(),minimum_original_flip_margin=model.flip_delta(t,output[0],a).masked_fill(~free,float('inf')).flatten(1).amin(1).cpu().numpy(),original_continuous_energy=base.cpu().numpy(),modified_continuous_energy=(base+reg).cpu().numpy(),continuous_regularizer=reg.cpu().numpy(),modified_field_residual=(B.field(model,t,p,a,strength)-u).abs().masked_fill(~free,0).flatten(1).amax(1).cpu().numpy())
    return cpu


def preflight(gpu):
    lock,inventory=acquire(gpu);torch,_,M=S.imports();model=S.model_for(0,LENGTH,"original","cuda")
    items=[S.maze(LENGTH,.2,np.random.SeedSequence([2026111595,i])) for i in range(64)];tokens=np.stack([x[0] for x in items]);t=torch.tensor(tokens.astype(np.int64),device="cuda");checks=[]
    with torch.no_grad():
        psi,_=model.settle(t);a=model.couplings(t,psi);u=torch.full_like(a[:,0],3.)
        reference=M.descend(model,t,a,steps=8192,eta=.5,margin=3.,certify_every=16);native=L.solve(model,t,a,u,8192,0.)
        assert all(torch.equal(reference[i],native[i]) for i in range(3));full=L.solve(model,t,a,u,8192,1.)
        for indices in [[0],[1,4,8],[2,5,9,11,17,31,48]]:
            idx=torch.tensor(indices,device="cuda");subset=L.solve(model,t[idx],a[idx],u[idx],8192,1.)
            checks.append({"indices":indices,"same_boards":bool(torch.equal(subset[0],full[0][idx])),"same_steps":bool(torch.equal(subset[1],full[1][idx])),"same_flags":all(torch.equal(subset[k],full[k][idx]) for k in [2,3]),"max_logit_error":float((subset[4]-full[4][idx]).abs().max())})
    validator=Validator();pred=native[0].bool().cpu().numpy();expected=np.array([L.feasible(t,p) for t,p in zip(tokens,pred)]);assert np.array_equal(validator(tokens,pred),expected)
    S.save(DEST/"preflight.json",{"status":"passed","utc":S.utc(),"protocol_sha256":S.sha(PP),"native_parity":True,"subset_checks":checks,"validator_parity":True,"reserved_inputs":64,"inventory":inventory,"script_sha256":S.sha(__file__)})
    S.emit("selective_preflight_passed",checks=checks)


def worker(gpu):
    lock,inventory=acquire(gpu);started=time.perf_counter();torch,_,_=S.imports();from cram.maze_learned import gather
    p=json.loads(PP.read_text());assert p["execution_script_sha256"]==S.sha(__file__)
    for path,h in p["implementation_sha256"].items():assert S.sha(S.ROOT/path)==h
    for path,h in p["source_sha256"].items():assert S.sha(S.ROOT/path)==h
    assert S.sha(LIBRARY)==p["validator_gate"]["binary_sha256"]
    pre=json.loads((DEST/"preflight.json").read_text());assert pre["status"]=="passed" and pre["protocol_sha256"]==S.sha(PP)
    for model in p["models"].values():assert S.sha(S.ROOT/model["path"])==model["sha256"]
    manifest=json.loads((DEST/"datasets.json").read_text());assert manifest["protocol_sha256"]==S.sha(PP)
    data={}
    for ds in manifest["datasets"]:
        assert S.sha(S.ROOT/ds["path"])==ds["sha256"]
        with np.load(S.ROOT/ds["path"]) as x:data[ds["shard"]]={k:x[k] for k in x.files}
    validator=Validator();jobs=[j for j in p["jobs"] if j["gpu"]==gpu];state={"status":"running","pid":os.getpid(),"host":platform.node(),"gpu":gpu,"started_utc":S.utc(),"jobs_complete":[],"repetition":-1};timing=[];fallback_refs={};native_refs={}
    warm=S.model_for(0,LENGTH,"original","cuda");t=torch.tensor(data[0]["tokens"][:64].astype(np.int64),device="cuda")
    with torch.no_grad():psi,_=warm.settle(t);a=warm.couplings(t,psi);L.solve(warm,t,a,torch.full_like(a[:,0],3.),32,0.);L.solve(warm,t,a,torch.full_like(a[:,0],3.),32,1.)
    del warm,t,a,psi
    for rep in range(REPS):
        torch.cuda.synchronize();begin=time.perf_counter();models={seed:S.model_for(seed,LENGTH,"original","cuda") for seed in range(3)};torch.cuda.synchronize()
        costs={"model_setup":time.perf_counter()-begin,"input_setup":0.,"hidden":0.,"native_answer":0.,"native_output":0.,"validator":0.,"queue":0.,"fallback_setup":0.,"fallback_answer":0.,"fallback_output":0.,"always_lambda_answer":0.,"always_lambda_output":0.};queues={s:[] for s in range(3)};rejected=0
        for ji,job in enumerate(jobs):
            d=data[job["shard"]];tokens=d["tokens"];model=models[job["seed"]];saved=[];accepted=[];native_hash=hashlib.sha256();run=DEST/f"runs/{job['id']}"
            state.update(repetition=rep,current_job=job,completed_inputs=0,updated_utc=S.utc());S.save(DEST/f"worker{gpu}.json",state)
            for lo in range(0,256,BATCH):
                token=tokens[lo:lo+BATCH];torch.cuda.synchronize();begin=time.perf_counter();t=torch.tensor(token.astype(np.int64),device="cuda");torch.cuda.synchronize();costs["input_setup"]+=time.perf_counter()-begin
                with torch.no_grad():
                    torch.cuda.synchronize();begin=time.perf_counter();psi,hs=model.settle(t,tol=1e-6,max_steps=4000,check=8);lg,la,lc=model.parts(t,model.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,model.zoffs,math.inf),lc[:,None]],1),1)
                    hg=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1));a=model.couplings(t,psi);u=torch.full_like(a[:,0],3.);torch.cuda.synchronize();costs["hidden"]+=time.perf_counter()-begin
                    outputs={};cpu={};order=[0,1] if rep==0 else [0]
                    if rep==0 and (ji+lo//64)%2:order.reverse()
                    for v in order:
                        name="native" if v==0 else "always_lambda";torch.cuda.synchronize();begin=time.perf_counter();outputs[v]=L.solve(model,t,a,u,8192,float(v));torch.cuda.synchronize();costs[name+"_answer"]+=time.perf_counter()-begin
                        begin=time.perf_counter();cpu[v]=cpu_answer(outputs[v]);torch.cuda.synchronize();costs[name+"_output"]+=time.perf_counter()-begin
                    begin=time.perf_counter();good=validator(token,cpu[0]["predictions"]);good&=cpu[0]["finite"]&cpu[0]["stopped"]&hg.cpu().numpy();reject=np.flatnonzero(~good);costs["validator"]+=time.perf_counter()-begin
                    torch.cuda.synchronize();begin=time.perf_counter()
                    if len(reject):
                        idx=torch.tensor(reject,device="cuda");queues[job["seed"]].append((t[idx],a[idx],d["indices"][lo:lo+64][reject].copy(),hg[idx].cpu().numpy()))
                    torch.cuda.synchronize();costs["queue"]+=time.perf_counter()-begin;rejected+=len(reject)
                    native_hash.update(digest(cpu[0]).encode())
                    if rep==0:
                        for v in [0,1]:diagnostics(torch,model,t,a,outputs[v],float(v),cpu[v])
                        saved.append({k:np.stack([cpu[v][k] for v in [0,1]]) for k in cpu[0]}|{"hidden_good":hg.cpu().numpy(),"hidden_steps":hs.cpu().numpy()});accepted.append(good)
                state.update(completed_inputs=lo+64,updated_utc=S.utc(),worker_elapsed_s=time.perf_counter()-started);S.save(DEST/f"worker{gpu}.json",state)
                if time.perf_counter()-started>900:raise TimeoutError("Frozen900s worker envelope; completed evidence retained")
            if rep==0:
                run.mkdir(parents=True,exist_ok=True)
                if (run/"result.json").exists():raise FileExistsError(run)
                arrays={k:np.concatenate([x[k] for x in saved],axis=0 if k.startswith("hidden_") else 1) for k in saved[0]};arrays["native_accepted"]=np.concatenate(accepted);np.savez_compressed(run/"outputs.npz",**arrays)
                native_refs[job["id"]]=native_hash.hexdigest();S.save(run/"result.json",{"status":"complete","job":job,"protocol_sha256":S.sha(PP),"script_sha256":S.sha(__file__),"outputs_sha256":S.sha(run/"outputs.npz"),"dataset_sha256":manifest["datasets"][job["shard"]]["sha256"],"checkpoint_sha256":p["models"][str(job["seed"])]["sha256"],"native_board_step_flag_digest":native_refs[job["id"]]})
            else:assert native_hash.hexdigest()==native_refs[job["id"]],(rep,job)
            state["jobs_complete"].append(f"rep{rep}/{job['id']}");S.emit("selective_job_complete",gpu=gpu,repetition=rep,job=job["id"],rejected_so_far=rejected)
        fallback_count=0;parity=[]
        for seed in range(3):
            if not queues[seed]:continue
            torch.cuda.synchronize();begin=time.perf_counter();qt=torch.cat([x[0] for x in queues[seed]]);qa=torch.cat([x[1] for x in queues[seed]]);indices=np.concatenate([x[2] for x in queues[seed]]);hidden_good=np.concatenate([x[3] for x in queues[seed]]);torch.cuda.synchronize();costs["fallback_setup"]+=time.perf_counter()-begin
            values=[];hashes=[];model=models[seed]
            for lo in range(0,len(indices),64):
                torch.cuda.synchronize();begin=time.perf_counter();t=qt[lo:lo+64];a=qa[lo:lo+64];u=torch.full_like(a[:,0],3.);torch.cuda.synchronize();costs["fallback_setup"]+=time.perf_counter()-begin
                begin=time.perf_counter();output=L.solve(model,t,a,u,8192,1.);torch.cuda.synchronize();costs["fallback_answer"]+=time.perf_counter()-begin
                begin=time.perf_counter();cpu=cpu_answer(output);torch.cuda.synchronize();costs["fallback_output"]+=time.perf_counter()-begin;hashes.append(digest(cpu))
                if rep==0:values.append(diagnostics(torch,model,t,a,output,1.,cpu))
                fallback_count+=len(t)
            path=DEST/f"fallback-gpu{gpu}-seed{seed}.npz"
            if rep==0:
                arrays={k:np.concatenate([x[k] for x in values]) for k in values[0]};arrays.update(indices=indices,hidden_good=hidden_good);np.savez_compressed(path,**arrays);fallback_refs[seed]=hashes
            else:assert hashes==fallback_refs[seed],(rep,seed)
            parity.append({"seed":seed,"n":len(indices),"board_step_flag_hashes":hashes,"repetition0_parity":True})
        assert fallback_count==rejected
        common=sum(costs[k] for k in ["model_setup","input_setup","hidden","native_answer","native_output"]);extra=sum(costs[k] for k in ["validator","queue","fallback_setup","fallback_answer","fallback_output"])
        timing.append({"repetition":rep,"cost_seconds":costs,"native_common_seconds":common,"incremental_seconds":extra,"answer_overhead_percent":100*extra/costs["native_answer"],"end_to_end_overhead_percent":100*extra/common,"fallback_model_inputs":fallback_count,"exact_native_repeat_parity":True,"fallback_repeat_parity":parity});S.save(DEST/f"timing-gpu{gpu}.json",{"status":"partial" if rep<REPS-1 else "complete","protocol_sha256":S.sha(PP),"gpu":gpu,"inventory":inventory,"repetitions":timing});S.emit("selective_repetition_complete",gpu=gpu,**timing[-1])
    state.update(status="complete",completed_utc=S.utc(),worker_elapsed_s=time.perf_counter()-started);S.save(DEST/f"worker{gpu}.json",state)
    S.emit("selective_worker_complete",gpu=gpu,seconds=state["worker_elapsed_s"])


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["freeze","prepare","preflight","worker"]);parser.add_argument("--gpu",type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=="freeze":freeze()
    elif args.command=="prepare":prepare()
    elif args.command=="preflight":preflight(args.gpu)
    else:worker(args.gpu)
