"""Independent graph/selection, retained-logit and paired timing audit."""
import csv
import hashlib
import json
import time
import numpy as np
import audit as A
from audit_restart import check_path
from diagnose_restart import classify
import study as S

ARMS=("native","always_lambda1","selective_fallback")


def main():
    started=time.perf_counter();dest=S.OUT/"selective_fallback";pp=S.HERE/"selective_fallback-protocol.json";p=json.loads(pp.read_text());evidence={}
    def verify(path,expected=None):
        actual=S.sha(path)
        if expected is not None:assert actual==expected,path
        evidence[str(path.relative_to(S.ROOT))]=actual
    verify(pp);verify(S.HERE/"selective_fallback.py",p["execution_script_sha256"])
    for path,h in p["implementation_sha256"].items():verify(S.ROOT/path,h)
    for path,h in p["source_sha256"].items():verify(S.ROOT/path,h)
    for m in p["models"].values():verify(S.ROOT/m["path"],m["sha256"])
    verify(dest/"validator-gate.json",p["validator_gate"]["sha256"]);verify(dest/"preflight.json");assert json.loads((dest/"preflight.json").read_text())["protocol_sha256"]==S.sha(pp)
    verify(dest/"datasets.json");manifest=json.loads((dest/"datasets.json").read_text());assert manifest["protocol_sha256"]==S.sha(pp)
    verify(dest/"independent-input-audit.json");inputs=json.loads((dest/"independent-input-audit.json").read_text());assert inputs["status"]=="passed" and inputs["protocol_sha256"]==S.sha(pp) and inputs["datasets"]==manifest["datasets"]
    data={};unique=set()
    for ds in manifest["datasets"]:
        verify(S.ROOT/ds["path"],ds["sha256"])
        with np.load(S.ROOT/ds["path"]) as x:d={k:x[k] for k in x.files}
        assert d["tokens"].shape==d["labels"].shape==(256,120,120);assert np.array_equal(d["indices"],np.arange(ds["shard"]*256,(ds["shard"]+1)*256));assert np.array_equal(d["shortest_edges"],inputs["distances"][ds["shard"]])
        unique.update(t.tobytes() for t in d["tokens"]);data[ds["shard"]]=d
    assert len(unique)==16384
    fallback={};fallback_files=[]
    for gpu in [0,1]:
        for seed in range(3):
            path=dest/f"fallback-gpu{gpu}-seed{seed}.npz"
            if not path.exists():continue
            verify(path);fallback_files.append(str(path.relative_to(S.ROOT)))
            with np.load(path) as x:out={k:x[k] for k in x.files}
            for i,index in enumerate(out["indices"]):
                key=(seed,int(index));assert key not in fallback;fallback[key]={k:v[i] for k,v in out.items() if k!="indices"}
    paired={};rows=[];cases=[];wrong=[];cache={};consumed=set();parity={"inputs":0,"different_boards":0,"different_steps":0,"different_flags":0,"maximum_logit_error":0.};native_accepts=0
    for job in p["jobs"]:
        run=dest/f"runs/{job['id']}";verify(run/"result.json");rec=json.loads((run/"result.json").read_text());assert rec["status"]=="complete" and rec["job"]==job and rec["protocol_sha256"]==S.sha(pp)
        assert rec["script_sha256"]==p["execution_script_sha256"] and rec["checkpoint_sha256"]==p["models"][str(job["seed"])]["sha256"] and rec["dataset_sha256"]==manifest["datasets"][job["shard"]]["sha256"]
        verify(run/"outputs.npz",rec["outputs_sha256"])
        with np.load(run/"outputs.npz") as x:out={k:x[k] for k in x.files}
        d=data[job["shard"]];selected={k:np.concatenate([v,v[:1].copy()]) for k,v in out.items() if k not in ["native_accepted","hidden_good","hidden_steps"]};accepted=out["native_accepted"]
        native_accepts+=int(accepted.sum())
        for i,index in enumerate(d["indices"]):
            key=(job["seed"],int(index));pred=out["predictions"][0,i]
            graph_key=(int(index),np.packbits(pred).tobytes())
            if graph_key not in cache:cache[graph_key]=check_path(d["tokens"][i],pred)
            expected=cache[graph_key] and out["stopped"][0,i] and out["finite"][0,i] and out["hidden_good"][i]
            assert bool(accepted[i])==bool(expected)
            if accepted[i]:assert key not in fallback
            else:
                f=fallback[key];consumed.add(key);assert bool(f["hidden_good"])==bool(out["hidden_good"][i])
                for k in selected:selected[k][2,i]=f[k]
                parity["inputs"]+=1;parity["different_boards"]+=int(not np.array_equal(f["predictions"],out["predictions"][1,i]));parity["different_steps"]+=int(f["answer_steps"]!=out["answer_steps"][1,i]);parity["different_flags"]+=int(f["stopped"]!=out["stopped"][1,i] or f["finite"]!=out["finite"][1,i]);parity["maximum_logit_error"]=max(parity["maximum_logit_error"],float(np.abs(f["final_logits"]-out["final_logits"][1,i]).max()))
        pred=selected["predictions"];logits=selected["final_logits"];assert pred.shape==logits.shape==(3,256,120,120);assert logits.dtype==np.float32 and np.array_equal(pred,logits>0)
        assert np.array_equal(selected["finite"],np.isfinite(logits).all((2,3)));assert np.array_equal(selected["stopped"],selected["finite"]&(selected["minimum_original_flip_margin"]>=-1e-6))
        assert np.array_equal(selected["modified_continuous_energy"],selected["original_continuous_energy"]+selected["continuous_regularizer"])
        valid=np.zeros((3,256),bool);shortest=np.zeros_like(valid)
        for v,arm in enumerate(ARMS):
            for i,(tok,board,index) in enumerate(zip(d["tokens"],pred[v],d["indices"])):
                key=(int(index),np.packbits(board).tobytes())
                if key not in cache:cache[key]=check_path(tok,board)
                valid[v,i]=cache[key];shortest[v,i]=valid[v,i] and int(board.sum())==int(d["shortest_edges"][i])+1
            steps=selected["answer_steps"][v];stopped=selected["stopped"][v];finite=selected["finite"][v]
            assert np.all((steps>=16)&(steps<=8192)&(steps%16==0)) and np.all(steps[~stopped]==8192)
            success=shortest[v]&stopped&finite&out["hidden_good"]&np.isfinite(selected["energy"][v]);paired[(job["shard"],job["seed"],v)]=success
            invalid=~shortest[v];endpoints=(pred[v].sum((1,2))==2)&((pred[v]&(d["tokens"]>=2)).sum((1,2))==2);branches=0
            for i in np.flatnonzero(invalid):
                description=classify(d["tokens"][i],pred[v,i]);branches+=int(description["branch_nodes"]>0)
                wrong.append({"input_index":int(d["indices"][i]),"model_seed":job["seed"],"arm":arm,"minimum_original_flip_margin":float(selected["minimum_original_flip_margin"][v,i]),"certified":bool(stopped[i]),"selected_flat_indices":np.flatnonzero(pred[v,i]).tolist(),**description})
            rows.append({"job":job["id"],"model_seed":job["seed"],"shard":job["shard"],"arm":arm,"n":256,"solved":int(success.sum()),"native_accepted":int(accepted.sum()),"wrong_certified":int((invalid&stopped).sum()),"endpoint_only_wrong":int((invalid&endpoints).sum()),"branched_wrong":branches,"valid_longer_path":int((valid[v]&~shortest[v]).sum()),"nonfinite":int((~finite).sum()),"timeouts":int((~stopped&finite).sum()),"hidden_failures":int((~out["hidden_good"]).sum()),"mean_steps":float(steps.mean()),"minimum_wrong_margin":float(selected["minimum_original_flip_margin"][v,invalid&stopped].min()) if np.any(invalid&stopped) else None})
            for i,index in enumerate(d["indices"]):cases.append({"input_index":int(index),"model_seed":job["seed"],"arm":arm,"success":int(success[i]),"feasible":int(valid[v,i]),"shortest":int(shortest[v,i]),"native_accepted":int(accepted[i]),"answer_steps":int(steps[i]),"certified":int(stopped[i]),"finite":int(finite[i]),"minimum_original_flip_margin":float(selected["minimum_original_flip_margin"][v,i])})
    assert consumed==set(fallback)
    joint={arm:np.concatenate([np.stack([paired[(shard,s,v)] for s in range(3)]).all(0) for shard in range(64)]) for v,arm in enumerate(ARMS)}
    gain=int((joint["selective_fallback"]&~joint["native"]).sum());loss=int((~joint["selective_fallback"]&joint["native"]).sum());gl,gu=A.bounds(gain,16384,.025);ll,lu=A.bounds(loss,16384,.025)
    receipts=[]
    for gpu in [0,1]:
        verify(dest/f"timing-gpu{gpu}.json");receipt=json.loads((dest/f"timing-gpu{gpu}.json").read_text());assert receipt["status"]=="complete" and receipt["protocol_sha256"]==S.sha(pp) and len(receipt["repetitions"])==3;receipts.append(receipt)
        verify(dest/f"worker{gpu}.json");state=json.loads((dest/f"worker{gpu}.json").read_text());assert state["status"]=="complete" and len(state["jobs_complete"])==288 and state["worker_elapsed_s"]<=900
    timing=[]
    for rep in range(3):
        parts=[r["repetitions"][rep] for r in receipts];costs={k:sum(x["cost_seconds"][k] for x in parts) for k in parts[0]["cost_seconds"]};common=sum(costs[k] for k in ["model_setup","input_setup","hidden","native_answer","native_output"]);extra=sum(costs[k] for k in ["validator","queue","fallback_setup","fallback_answer","fallback_output"])
        assert sum(x["fallback_model_inputs"] for x in parts)==len(fallback) and all(x["exact_native_repeat_parity"] for x in parts)
        timing.append({"repetition":rep,"cost_seconds":costs,"native_common_seconds":common,"incremental_seconds":extra,"answer_overhead_percent":100*extra/costs["native_answer"],"end_to_end_overhead_percent":100*extra/common})
    overhead=float(np.median([x["answer_overhead_percent"] for x in timing]));accuracy="supported" if gl-lu>=.0005 else "falsified" if gu-ll<.0005 else "inconclusive";runtime="supported" if overhead<=2. else "falsified"
    primary={"hypothesis":"H2t","n_unique":16384,"gains":gain,"losses":loss,"observed_gain_pp":100*(gain-loss)/16384,"one_sided95_net_gain_lower_pp":100*(gl-lu),"one_sided95_net_gain_upper_pp":100*(gu-ll),"target_pp":.05,"accuracy_decision":accuracy,"median_answer_overhead_percent":overhead,"overhead_target_percent":2.,"runtime_decision":runtime,"decision":"falsified" if "falsified" in [accuracy,runtime] else "supported" if accuracy==runtime=="supported" else "inconclusive"}
    secondary={"gain_input_indices":np.flatnonzero(joint["selective_fallback"]&~joint["native"]).tolist(),"loss_input_indices":np.flatnonzero(~joint["selective_fallback"]&joint["native"]).tolist(),"always_lambda_gains_vs_native":int((joint["always_lambda1"]&~joint["native"]).sum()),"always_lambda_losses_vs_native":int((~joint["always_lambda1"]&joint["native"]).sum())}
    perinput=[{"input_index":i,**{f"joint_{arm}_success":int(joint[arm][i]) for arm in ARMS}} for i in range(16384)]
    for name,values in [("runs.csv",rows),("per-model-input.csv",cases),("per-input.csv",perinput)]:
        with (dest/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    S.save(dest/"failure-classification.json",{"post_outcome_descriptive":True,"new_inference":False,"rows":wrong,"protocol_sha256":S.sha(pp)})
    result={"status":"validated","utc":S.utc(),"primary":primary,"secondary":secondary,"joint_success":{k:int(v.sum()) for k,v in joint.items()},"n_unique":16384,"fixed_models":3,"main_jobs":192,"fallback_jobs":len(fallback_files),"new_scientific_trajectories":98304+len(fallback),"timing_repeat_trajectories":98304+2*len(fallback),"logical_scientific_outcomes":147456,"exact_native_output_reuses":native_accepts,"fallback_model_inputs":len(fallback),"timing":timing,"fallback_vs_always_batch_parity":parity,"rows":rows,"evidence_sha256":evidence,"protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"independent_graph_board_checks":len(cache),"input_audit_seconds":inputs["oracle_seconds"],"audit_seconds":time.perf_counter()-started,"scope":"Offline graph-informed decoder with deferred per-model fallback batching; validator uses graph feasibility only, no shortestness. Three fresh-cohort timing repetitions. Added task knowledge and delay are explicit; no online latency claim."}
    S.save(dest/"audit.json",result);S.emit("selective_audit_passed",primary=primary,joint=result["joint_success"],fallback_inputs=len(fallback),seconds=result["audit_seconds"])


if __name__=="__main__":main()
