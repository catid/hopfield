"""Independent candidate/selector audit; shortest labels enter only here."""
import csv
import hashlib
import json
import time
import numpy as np
import audit as A
import study as S


def check_path(tok,pred):
    """Set-based path verifier independent of worker's array degree routine."""
    chosen=set(map(tuple,np.argwhere(pred)));start=tuple(np.argwhere(tok==2)[0]);goal=tuple(np.argwhere(tok==3)[0])
    if start not in chosen or goal not in chosen or any(tok[p]==0 for p in chosen):return False
    def adjacent(p):return [(p[0]-1,p[1]),(p[0],p[1]+1),(p[0]+1,p[1]),(p[0],p[1]-1)]
    if any(sum(q in chosen for q in adjacent(p))!=(1 if p in (start,goal) else 2) for p in chosen):return False
    seen={start};stack=[start]
    while stack:
        for q in adjacent(stack.pop()):
            if q in chosen and q not in seen:seen.add(q);stack.append(q)
    return seen==chosen


def main():
    started=time.perf_counter();dest=S.OUT/"restart";pp=S.HERE/"restart-protocol.json";protocol=json.loads(pp.read_text());evidence={}
    def verify(path,expected=None):
        actual=S.sha(path)
        if expected is not None:assert actual==expected,path
        evidence[str(path.relative_to(S.ROOT))]=actual
    verify(pp);verify(S.HERE/"restart_study.py",protocol["execution_script_sha256"]);verify(S.HERE/"study.py",protocol["core_script_sha256"])
    for path,h in protocol["source_sha256"].items():verify(S.ROOT/path,h)
    for model in protocol["models"].values():verify(S.ROOT/model["path"],model["sha256"])
    verify(dest/"preflight.json");preflight=json.loads((dest/"preflight.json").read_text());assert preflight["status"]=="passed" and preflight["protocol_sha256"]==S.sha(pp)
    verify(dest/"datasets.json");manifest=json.loads((dest/"datasets.json").read_text());assert manifest["protocol_sha256"]==S.sha(pp)
    data={};cache={};init_hashes={};unique_inputs=set();oracle_checks=0;cache_hits=0
    for ds in manifest["datasets"]:
        verify(S.ROOT/ds["path"],ds["sha256"]);loaded=np.load(S.ROOT/ds["path"]);data[ds["shard"]]={k:loaded[k] for k in loaded.files}
        d=data[ds["shard"]];assert len(d["tokens"])==256
        for i,(tok,label,distance) in enumerate(zip(d["tokens"],d["labels"],d["shortest_edges"])):
            ok,independent=A.independent(tok,label);assert ok and independent==distance;oracle_checks+=1
            unique_inputs.add(tok.tobytes())
        digest=hashlib.sha256()
        for lo in range(0,256,32):
            for chain in range(4):
                initial=np.stack([3.+np.random.default_rng(np.random.SeedSequence([protocol["initialization"]["noise_base"],int(i),chain])).standard_normal((90,90),dtype=np.float32) for i in d["indices"][lo:lo+32]])
                digest.update(initial.tobytes())
        init_hashes[ds["shard"]]=digest.hexdigest()
    assert len(unique_inputs)==4096
    saved={};records=[];candidates=[];case_rows=[];graph_checks=0;all_failure_examples=[]
    for job in protocol["jobs"]:
        run=dest/"runs"/job["id"];verify(run/"result.json");r=json.loads((run/"result.json").read_text())
        assert r["status"]=="complete" and r["job"]==job and r["protocol_sha256"]==S.sha(pp) and r["script_sha256"]==protocol["execution_script_sha256"]
        assert r["dataset_sha256"]==manifest["datasets"][job["shard"]]["sha256"] and r["checkpoint_sha256"]==protocol["models"][str(job["seed"])]["sha256"]
        assert r["initialization_sha256"]==init_hashes[job["shard"]]
        verify(run/"outputs.npz",r["outputs_sha256"]);output=np.load(run/"outputs.npz");d=data[job["shard"]];pred=output["predictions"]
        assert pred.shape==(5,256,90,90);feasible=np.zeros((5,256),bool);shortest=np.zeros_like(feasible)
        for v in range(5):
            for i,(tok,p) in enumerate(zip(d["tokens"],pred[v])):
                key=(int(d["indices"][i]),np.packbits(p).tobytes())
                if key not in cache:cache[key]=check_path(tok,p);graph_checks+=1
                else:cache_hits+=1
                feasible[v,i]=cache[key];shortest[v,i]=feasible[v,i] and int(p.sum())==int(d["shortest_edges"][i])+1
        assert np.array_equal(feasible,output["feasible"])
        eligible=output["stopped"]&output["finite"]&np.isfinite(output["energy"])&output["hidden_good"][None]
        success=shortest&eligible
        for v in range(5):
            cap=8192 if v==0 else 2048;steps=output["answer_steps"][v]
            assert np.all((steps>=16)&(steps<=cap)&(steps%16==0)) and np.all(steps[~output["stopped"][v]]==cap)
            assert not np.any(output["stopped"][v]&~output["finite"][v])
            candidates.append({"job":job["id"],"shard":job["shard"],"model_seed":job["seed"],"variant":"native" if v==0 else f"restart{v-1}","n":256,"solved":int(success[v].sum()),"feasible":int(feasible[v].sum()),"wrong_certified":int((output["stopped"][v]&~shortest[v]).sum()),"answer_timeouts":int((~output["stopped"][v]&output["finite"][v]).sum()),"nonfinite":int((~output["finite"][v]).sum()),"hidden_failures":int((~output["hidden_good"]).sum()),"mean_answer_steps":float(steps.mean()),"answer_seconds":r["answer_seconds_by_variant"][v]})
        selected={};selected_steps={};selection_data={}
        for name in ["energy","screen"]:
            selected_indices=[]
            for i in range(256):
                permitted=[v for v in range(1,5) if eligible[v,i] and (name=="energy" or feasible[v,i])]
                chosen=min(permitted,key=lambda v:(float(output["energy"][v,i]),v)) if permitted else -1
                selected_indices.append(chosen)
            chosen=np.array(selected_indices);assert np.array_equal(chosen,output[f"selected_{name}"])
            selected[name]=np.array([v>=0 and success[v,i] for i,v in enumerate(chosen)])
            selection_data[name]=chosen
        assert not np.any(selected["energy"]&~selected["screen"])
        saved[(job["shard"],job["seed"]) ]={"native":success[0],"energy":selected["energy"],"screen":selected["screen"],"native_steps":output["answer_steps"][0],"restart_steps":output["answer_steps"][1:].sum(0)}
        for i,index in enumerate(d["indices"]):
            row={"input_index":int(index),"model_seed":job["seed"],"native_success":bool(success[0,i]),"energy_success":bool(selected["energy"][i]),"screen_success":bool(selected["screen"][i]),"energy_candidate":int(selection_data["energy"][i])-1 if selection_data["energy"][i]>=0 else -1,"screen_candidate":int(selection_data["screen"][i])-1 if selection_data["screen"][i]>=0 else -1,"native_steps":int(output["answer_steps"][0,i]),"restart_total_steps":int(output["answer_steps"][1:,i].sum()),"native_shortest":bool(shortest[0,i]),"feasible_candidates":int((feasible[1:,i]&eligible[1:,i]).sum()),"successful_candidates":int(success[1:,i].sum())}
            case_rows.append(row)
            if not row["native_success"] or not row["energy_success"] or not row["screen_success"]:all_failure_examples.append(row)
        records.append(r|{"native_solved":int(success[0].sum()),"energy_solved":int(selected["energy"].sum()),"screen_solved":int(selected["screen"].sum()),"no_eligible":int((selection_data["energy"]<0).sum()),"no_feasible":int((selection_data["screen"]<0).sum())})
    joint={name:np.concatenate([np.stack([saved[(shard,s)][name] for s in range(3)]).all(0) for shard in range(16)]) for name in ["native","energy","screen"]}
    losses=int((joint["energy"]&~joint["screen"]).sum());gains=int((~joint["energy"]&joint["screen"]).sum());assert losses==0
    lower,upper=A.bounds(gains,4096);primary={"hypothesis":"H2m","n_unique":4096,"joint_gains":gains,"joint_losses":losses,"observed_gain_pp":100*gains/4096,"one_sided95_gain_lower_pp":100*lower,"one_sided95_gain_upper_pp":100*upper,"target_pp":.1,"decision":"supported" if lower>=.001 else "falsified" if upper<.001 else "inconclusive"}
    secondary={}
    for name in ["energy","screen"]:
        gain=int((joint[name]&~joint["native"]).sum());loss=int((joint["native"]&~joint[name]).sum());gl,gu=A.bounds(gain,4096,alpha=.0125);ll,lu=A.bounds(loss,4096,alpha=.0125)
        secondary[name]={"gains_vs_native":gain,"losses_vs_native":loss,"observed_net_gain_pp":100*(gain-loss)/4096,"conservative_two_sided95_net_gain_pp":[100*(gl-lu),100*(gu-ll)],"interval":"Four one-sided Clopper-Pearson bounds with alpha=.0125; Bonferroni difference. Secondary descriptive comparison."}
    for name,rows in [("candidates.csv",candidates),("per-model-input.csv",case_rows)]:
        with (dest/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    input_rows=[{"input_index":i,**{f"joint_{name}_success":int(joint[name][i]) for name in joint}} for i in range(4096)]
    with (dest/"per-input.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(input_rows[0]));writer.writeheader();writer.writerows(input_rows)
    totals={name:int(values.sum()) for name,values in joint.items()}
    result={"status":"validated","utc":S.utc(),"primary":primary,"secondary":secondary,"joint_success":totals,"n_unique":4096,"models":3,"jobs":48,"model_input_rollouts":61440,"selector_evaluations":24576,"selectors_add_no_neural_rollouts":True,"records":records,"candidate_records":candidates,"failure_examples":all_failure_examples,
        "no_eligible_model_inputs":sum(r["no_eligible"] for r in records),"no_feasible_model_inputs":sum(r["no_feasible"] for r in records),"hidden_seconds_shared":sum(r["hidden_seconds_shared"] for r in records),"native_answer_seconds":sum(r["answer_seconds_by_variant"][0] for r in records),"four_restart_answer_seconds":sum(sum(r["answer_seconds_by_variant"][1:]) for r in records),"selector_cpu_seconds":sum(r["selector_cpu_seconds"] for r in records),
        "mean_native_steps":float(np.mean([x["native_steps"] for x in case_rows])),"mean_summed_restart_steps":float(np.mean([x["restart_total_steps"] for x in case_rows])),"mean_candidates_feasible":float(np.mean([x["feasible_candidates"] for x in case_rows])),"unique_inputs_verified":len(unique_inputs),"independent_shortest_oracle_checks":oracle_checks,"independent_candidate_graph_checks":graph_checks,"exact_board_cache_hits":cache_hits,
        "protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"evidence_sha256":evidence,"audit_seconds":time.perf_counter()-started,
        "scope":"Fresh 90-grid loop20 generator and three fixed reproduction models; standard-deviation-one perturbations only. Feasibility is an added graph prior, not native CRAM. No ground-truth labels or distances used in selection. Model repeats are not independent inputs."}
    S.save(dest/"audit.json",result);S.emit("restart_audit_passed",primary=primary,joint_success=totals,seconds=result["audit_seconds"])


if __name__=="__main__":main()
