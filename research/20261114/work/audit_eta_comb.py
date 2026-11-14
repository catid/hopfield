"""Independent topology, shortest-path, and causal slope audit for paired combs."""
import csv
import json
from pathlib import Path
import numpy as np
import audit as A
import study as S
from topology import distances


def main():
    dest=S.OUT/"eta_comb";pp=S.HERE/"eta_comb-protocol.json";protocol=json.loads(pp.read_text());evidence={}
    def verify(path,expected=None):
        path=Path(path);path=path if path.is_absolute() else S.ROOT/path;actual=S.sha(path)
        if expected is not None:assert actual==expected,path
        evidence[str(path.relative_to(S.ROOT))]=actual
    verify(pp);verify(__file__);verify(A.__file__);verify(S.HERE/"topology.py")
    verify(S.HERE/"eta_comb.py",protocol["execution_script_sha256"])
    verify(S.HERE/"eta_study.py",protocol["trace_source_sha256"])
    verify(S.__file__,protocol["core_script_sha256"]);verify(dest/"preflight.json")
    preflight=json.loads((dest/"preflight.json").read_text())
    assert preflight["status"]=="passed" and preflight["protocol_sha256"]==S.sha(pp)
    assert preflight["execution_sha256"]==protocol["execution_script_sha256"]
    for path,h in protocol["source_sha256"].items():verify(path,h)
    for model in protocol["models"].values():verify(model["path"],model["sha256"])
    datasets={};saved={};records=[];diagnostics=[];conditions=[];layout_rows=[]
    for dataset in protocol["datasets"]:
        verify(dataset["path"],dataset["sha256"]);data=np.load(S.ROOT/dataset["path"]);datasets[dataset["cohort"]]=data
        for tok,label,expected_distance in zip(data["tokens"],data["labels"],data["shortest_edges"]):
            valid,distance=A.independent(tok,label);assert valid and distance==expected_distance==dataset["spine_length"]
            op=tok>0;nodes=int(op.sum());edges=int((op[1:]&op[:-1]).sum()+(op[:,1:]&op[:,:-1]).sum());assert edges==nodes-1
            depth=distances(tok,[tuple(x) for x in np.argwhere(label)])
            assert len(depth)==nodes and max(depth.values())==dataset["branch_depth"]
            degrees=np.zeros_like(tok,int);degrees[1:]+=op[:-1];degrees[:-1]+=op[1:];degrees[:,1:]+=op[:,:-1];degrees[:,:-1]+=op[:,1:]
            assert int((degrees[op]==3).sum())==8
    for job in protocol["jobs"]:
        run=dest/"runs"/job["id"];record=json.loads((run/"result.json").read_text());verify(run/"result.json")
        assert record["status"]=="complete" and record["job"]==job and record["protocol_sha256"]==S.sha(pp)
        assert record["script_sha256"]==protocol["execution_script_sha256"] and record["core_sha256"]==protocol["core_script_sha256"]
        assert record["checkpoint_sha256"]==protocol["models"][str(job["seed"])]["sha256"]
        dataset=next(d for d in protocol["datasets"] if d["spine_length"]==job["spine_length"] and d["branch_depth"]==job["branch_depth"])
        assert record["dataset_sha256"]==dataset["sha256"];data=datasets[dataset["cohort"]]
        for eta in ("0.2","0.5"):
            arm=record["arms"][eta];verify(run/f"eta{eta}.npz",arm["output_sha256"]);output=np.load(run/f"eta{eta}.npz")
            valid=[]
            for tok,pred,expected_distance in zip(data["tokens"],output["predictions"],data["shortest_edges"]):
                ok,distance=A.independent(tok,pred);assert distance==expected_distance;valid.append(ok)
            valid=np.asarray(valid);success=valid&output["stopped"]&output["hidden_good"];steps=output["answer_steps"]
            assert np.array_equal(valid,output["shortest_path"]) and np.array_equal(success,output["success"])
            assert np.array_equal(success&(steps<=2048),output["success2048"])
            assert int(success.sum())==arm["solved"] and int((output["stopped"]&~valid).sum())==arm["wrong_stops"]
            assert int((~output["stopped"]).sum())==arm["answer_budget_failures"] and int((~output["hidden_good"]).sum())==arm["hidden_failures"]
            assert np.all((steps>=16)&(steps<=8192)&(steps%16==0)) and np.all(steps[~output["stopped"]]==8192)
            saved[(job["spine_length"],job["branch_depth"],job["seed"],eta)]={"steps":steps,"success":success}
            records.append(dict(job=job,eta=float(eta),n=len(steps),**arm,hidden_seconds_shared=record["hidden_seconds_shared"],gpu=record["gpu"]))
        if record["diagnostics_sha256"]:
            verify(run/"diagnostics.json",record["diagnostics_sha256"])
            for eta,diag in json.loads((run/"diagnostics.json").read_text()).items():
                assert diag["replay_native_parity"]
                energies=np.asarray([h["energy"] for h in diag["history"]]);change=np.diff(energies,axis=0)
                diagnostics.append({"job":job["id"],"eta":float(eta),"n_traces":energies.shape[1],"replay_native_parity":True,
                                    "positive_energy_changes":int((change>0).sum()),"maximum_energy_increase":float(np.maximum(change,0).max()),
                                    "two_cycle_flags":sum(diag["two_cycle_detected"])})
    means={}
    for dataset in protocol["datasets"]:
        spine,depth=dataset["spine_length"],dataset["branch_depth"]
        for eta in ("0.2","0.5"):
            steps=np.stack([saved[(spine,depth,s,eta)]["steps"] for s in range(3)]).mean(0)
            success=np.stack([saved[(spine,depth,s,eta)]["success"] for s in range(3)])
            means[(spine,depth,eta)]=steps
            conditions.append({"spine_length":spine,"branch_depth":depth,"eta":float(eta),"n_layouts":len(steps),
                               "solved_per_seed":success.sum(1).tolist(),"mean_steps":float(steps.mean()),"min_steps":float(steps.min()),"max_steps":float(steps.max())})
            for i in range(len(steps)):
                layout_rows.append({"spine_length":spine,"branch_depth":depth,"eta":float(eta),"layout_index":i,
                                    "layout_code":int(datasets[dataset["cohort"]]["layout_codes"][i]),"mean_model_steps":float(steps[i]),
                                    "joint_success":int(success[:,i].all())})
    depths=np.asarray([32,64,128]);centered=depths-depths.mean();denominator=(centered**2).sum()
    slopes={e:(np.stack([means[(64,d,e)] for d in depths])*centered[:,None]).sum(0)/denominator for e in ("0.2","0.5")}
    rng=np.random.default_rng(2026111422);indices=rng.integers(64,size=(10000,64));ratio=slopes["0.2"].mean()/slopes["0.5"].mean()
    bootstrap=slopes["0.2"][indices].mean(1)/slopes["0.5"][indices].mean(1);ci=np.quantile(bootstrap,[.025,.975]).tolist()
    valid_all=all(r["solved"]==r["n"] for r in records)
    decision="supported" if 2<=ci[0] and ci[1]<=3 else "falsified" if ci[1]<2 or ci[0]>3 else "inconclusive"
    if not valid_all and decision=="supported":decision="inconclusive_due_to_invalid_or_censored_outputs"
    primary={"slope_eta02":float(slopes["0.2"].mean()),"slope_eta05":float(slopes["0.5"].mean()),
             "slope_ratio":float(ratio),"paired_layout_bootstrap95_ratio":ci,"equivalence_band":[2,3],"inverse_eta_prediction":2.5,
             "all_model_maze_arms_valid":valid_all,"decision":decision}
    secondary=[]
    for eta in ("0.2","0.5"):
        delta=means[(128,64,eta)]-means[(16,64,eta)];boot=delta[indices].mean(1);interval=np.quantile(boot,[.025,.975]).tolist()
        decision="supported" if -32<=interval[0] and interval[1]<=32 else "falsified" if interval[1]<-32 or interval[0]>32 else "inconclusive"
        secondary.append({"eta":float(eta),"spine128_minus16_mean_steps":float(delta.mean()),"paired_layout_bootstrap95_steps":interval,
                          "equivalence_band":[-32,32],"decision":decision})
    with (dest/"per-layout.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(layout_rows[0]));writer.writeheader();writer.writerows(layout_rows)
    verify(dest/"per-layout.csv")
    result={"status":"validated","utc":S.utc(),"campaign":"eta_comb","jobs":len(protocol["jobs"]),"evaluated_arms":len(records),
            "saved_board_checks":sum(r["n"] for r in records),"protocol_sha256":S.sha(pp),"primary":primary,"secondary":secondary,
            "conditions":conditions,"diagnostics":diagnostics,"records":records,"evidence_sha256":evidence,"hashed_files":len(evidence),
            "topology_checks":"All320condition-layoutgraphs connected and acyclic,8degree3junctions, exact S-Gspine and maxdistance from oracle path equal declared depth.",
            "scope":"64fresh paired layouts; all3models averaged within layout. This mechanistic equivalence claim concerns fixed trained models and constructed combs, not a general eta stability theorem.","new_inference":False}
    S.save(dest/"audit.json",result);S.emit("eta_comb_audit_complete",jobs=result["jobs"],primary=primary,secondary=secondary)


if __name__=="__main__":main()
