"""Audit horizon/comb outcomes without executing model inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import audit as A
import study as S
from topology import distances


def main(campaign):
    directory=S.OUT/campaign;protocol_path=S.HERE/f"{campaign}-protocol.json";protocol=json.loads(protocol_path.read_text())
    evidence={}
    def verify(path,expected=None):
        path=Path(path);path=path if path.is_absolute() else S.ROOT/path;actual=S.sha(path)
        if expected is not None:assert actual==expected,path
        evidence[str(path.relative_to(S.ROOT))]=actual
    verify(protocol_path);verify(__file__);verify(A.__file__);verify(S.HERE/"topology.py")
    verify(S.HERE/"followup.py",protocol["execution_script_sha256"]);verify(S.__file__,protocol["core_script_sha256"])
    for path,h in protocol["source_sha256"].items():verify(path,h)
    for m in protocol["models"].values():verify(m["path"],m["sha256"])
    rows=[];saved={};dataset_cache={}
    for job in protocol["jobs"]:
        run=directory/"runs"/job["id"];record=json.loads((run/"result.json").read_text())
        assert record["status"]=="complete" and record["job"]==job and record["protocol_sha256"]==S.sha(protocol_path)
        verify(run/"result.json");verify(run/"predictions.npz",record["predictions_sha256"])
        dataset=next(d for d in protocol["datasets"] if d["cohort"]==job["cohort"] and d["length"]==job["length"] and d["loop_percent"]==job["loop_percent"])
        verify(dataset["path"],dataset["sha256"])
        if dataset["path"] not in dataset_cache:dataset_cache[dataset["path"]]=np.load(S.ROOT/dataset["path"])
        data=dataset_cache[dataset["path"]];out=np.load(run/"predictions.npz");valid=[]
        for tok,pred,target_distance in zip(data["tokens"],out["predictions"],data["shortest_edges"]):
            ok,distance=A.independent(tok,pred);assert distance==target_distance;valid.append(ok)
        valid=np.asarray(valid);success=valid&out["stopped"]&out["hidden_good"]
        assert np.array_equal(valid,out["shortest_path"]) and np.array_equal(success,out["success"])
        assert int(success.sum())==record["solved"] and int((out["stopped"]&~valid).sum())==record["wrong_stops"]
        assert int((~out["stopped"]).sum())==record["answer_budget_failures"] and int((~out["hidden_good"]).sum())==record["hidden_failures"]
        assert np.all((out["answer_steps"]>=16)&(out["answer_steps"]<=8192)&(out["answer_steps"]%16==0))
        assert np.all(out["answer_steps"][~out["stopped"]]==8192)
        key=(job["length"],job.get("spine_length"),job.get("branch_depth"),job["seed"])
        saved[key]={k:out[k] for k in ["success","answer_steps","stopped","hidden_good"]}
        rows.append({k:record[k] for k in ["job","n","solved","wrong_stops","answer_budget_failures","hidden_failures","answer_steps_mean","hidden_steps_mean","wall_seconds"]})
    result={"status":"validated","utc":S.utc(),"campaign":campaign,"jobs":len(rows),"saved_board_checks":sum(r["n"] for r in rows),"records":rows}
    if campaign=="horizon_transfer":
        summaries=[]
        for length in [60,90]:
            successful=np.stack([saved[(length,None,None,s)]["success"] for s in range(3)])
            steps=np.stack([saved[(length,None,None,s)]["answer_steps"] for s in range(3)]).mean(0)
            joint=successful.all(0);lower,upper=A.bounds(int(joint.sum()),len(joint));d=next(d for d in protocol["datasets"] if d["length"]==length)
            verify(d["diagnostics_path"],d["diagnostics_sha256"]);diag=np.load(S.ROOT/d["diagnostics_path"])
            relative=np.abs(diag["predicted_answer_steps"]-steps)/steps
            good=(relative<=.1)&joint;pl,pu=A.bounds(int(good.sum()),len(good))
            summaries.append({"length":length,"n_unique":len(joint),"solved_per_seed":successful.sum(1).tolist(),"joint_success":int(joint.sum()),
                              "joint_percent":float(100*joint.mean()),"one_sided95_lower_percent":100*lower,"one_sided95_upper_percent":100*upper,
                              "99percent_decision":"supported" if lower>=.99 else "falsified" if upper<.99 else "inconclusive",
                              "steps_mean":float(steps.mean()),"steps_max":float(steps.max()),"predictor_within10percent":int(good.sum()),
                              "predictor_one_sided95_lower_percent":100*pl,"predictor_one_sided95_upper_percent":100*pu,
                              "predictor95percent_decision":"supported" if pl>=.95 else "falsified" if pu<.95 else "inconclusive",
                              "relative_error_mean":float(relative.mean()),"relative_error_max":float(relative.max())})
        result.update(summaries=summaries,predictor=protocol["predictor"],primary_absolute=summaries[0],primary_predictor=summaries[1])
    else:
        conditions=[];by_condition={};orientations=[]
        for dataset in protocol["datasets"]:
            spine,depth=dataset["spine_length"],dataset["branch_depth"]
            data=dataset_cache[dataset["path"]]
            for tok,label in zip(data["tokens"],data["labels"]):
                op=tok>0;nodes=int(op.sum());edges=int((op[1:]&op[:-1]).sum()+(op[:,1:]&op[:,:-1]).sum());assert edges==nodes-1
                maxdepth=max(distances(tok,[tuple(x) for x in np.argwhere(label)]).values());assert maxdepth==depth
                assert int(label.sum())==spine+1
                degree=np.zeros_like(tok,int);degree[1:]+=op[:-1];degree[:-1]+=op[1:];degree[:,1:]+=op[:,:-1];degree[:,:-1]+=op[:,1:]
                assert int((degree[op]==3).sum())==(8 if depth else 0)
            steps=np.stack([saved[(90,spine,depth,s)]["answer_steps"] for s in range(3)]).mean(0)
            success=np.stack([saved[(90,spine,depth,s)]["success"] for s in range(3)])
            by_condition[(spine,depth)]=steps
            forecast=protocol["predictor"]["slope"]*depth+protocol["predictor"]["intercept"]
            conditions.append({"spine_length":spine,"branch_depth":depth,"n_layouts":len(steps),"solved_per_seed":success.sum(1).tolist(),
                               "steps_mean":float(steps.mean()),"steps_min":float(steps.min()),"steps_max":float(steps.max()),
                               "frozen_prediction":float(forecast),"prediction_mean_residual":float(steps.mean()-forecast)})
            for orientation in sorted(set((data["layout_codes"]%8).tolist())):
                mask=data["layout_codes"]%8==orientation
                orientations.append({"spine_length":spine,"branch_depth":depth,"dihedral_code":orientation,"n_layouts":int(mask.sum()),
                                     "steps_mean":float(steps[mask].mean()),"solved_per_seed":success[:,mask].sum(1).tolist()})
        rng=np.random.default_rng(2026111420);contrasts=[]
        for name,delta,threshold in [("depth64_minus0",by_condition[(64,64)]-by_condition[(64,0)],200),
                                     ("spine64_minus16",by_condition[(64,16)]-by_condition[(16,16)],32)]:
            boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1);ci=np.quantile(boot,[.025,.975]).tolist()
            if name.startswith("depth"):decision="supported" if ci[0]>=threshold else "falsified" if ci[1]<threshold else "inconclusive"
            else:decision="supported" if ci[0]>=-threshold and ci[1]<=threshold else "falsified" if ci[0]>threshold or ci[1]<-threshold else "inconclusive"
            contrasts.append({"contrast":name,"mean_steps":float(delta.mean()),"paired_layout_bootstrap95":ci,"threshold":threshold,"decision":decision})
        result.update(conditions=conditions,contrasts=contrasts,orientations=orientations,topology_verification="Every graph connected and acyclic; exact S-Gdistance/spinelength,8degree3junctions for positive depth, exact maxdistance from targetpath equals specified branchdepth.")
    result.update(evidence_sha256=evidence,hashed_files=len(evidence),new_inference=False,
                  limitation="Oracle paths used only in offline topology diagnostics and correctness checks, never as model input. Inference runs use fixed learned weights and declared budgets.")
    S.save(directory/"audit.json",result);S.emit("followup_audited",campaign=campaign,jobs=len(rows),summary=result.get("summaries",result.get("contrasts")))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--campaign",choices=["horizon_transfer","causal_comb"],required=True);a=p.parse_args();main(a.campaign)
