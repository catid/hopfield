"""Audit four fixed eta arms, including failed and nonfinite outputs."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import audit as A
import study as S

ETAS=("0.2","0.5","0.8","1.0")
STRATA=("tree","loops20","comb")


def main():
    started=time.perf_counter();dest=S.OUT/"eta_boundary";pp=S.HERE/"eta_boundary-protocol.json"
    protocol=json.loads(pp.read_text());evidence={}
    def verify(path,expected=None):
        path=Path(path);path=path if path.is_absolute() else S.ROOT/path;actual=S.sha(path)
        if expected is not None:assert actual==expected,path
        evidence[str(path.relative_to(S.ROOT))]=actual
    verify(pp);verify(__file__);verify(A.__file__)
    verify(S.HERE/"eta_boundary.py",protocol["execution_script_sha256"])
    verify(S.HERE/"eta_comb.py",protocol["comb_generator_sha256"])
    verify(S.__file__,protocol["core_script_sha256"])
    for path,h in protocol["source_sha256"].items():verify(path,h)
    for model in protocol["models"].values():verify(model["path"],model["sha256"])
    verify(dest/"preflight.json");preflight=json.loads((dest/"preflight.json").read_text())
    assert preflight["status"]=="passed" and preflight["protocol_sha256"]==S.sha(pp)
    assert preflight["execution_sha256"]==protocol["execution_script_sha256"]
    records=[];saved={};datasets={};cache={};diagnostics=[];oracle_checks=0;cache_hits=0
    for d in protocol["datasets"]:
        verify(d["path"],d["sha256"]);datasets[d["stratum"]]=np.load(S.ROOT/d["path"])
    for job in protocol["jobs"]:
        run=dest/"runs"/job["id"];record=json.loads((run/"result.json").read_text());verify(run/"result.json")
        assert record["status"]=="complete" and record["job"]==job and record["protocol_sha256"]==S.sha(pp)
        assert record["script_sha256"]==protocol["execution_script_sha256"] and record["core_sha256"]==protocol["core_script_sha256"]
        assert record["checkpoint_sha256"]==protocol["models"][str(job["seed"])]["sha256"]
        dataset=next(d for d in protocol["datasets"] if d["stratum"]==job["stratum"])
        assert record["dataset_sha256"]==dataset["sha256"];data=datasets[job["stratum"]]
        for eta in ETAS:
            arm=record["arms"][eta];verify(run/f"eta{eta}.npz",arm["output_sha256"]);output=np.load(run/f"eta{eta}.npz")
            valid=[]
            for i,(tok,pred,distance) in enumerate(zip(data["tokens"],output["predictions"],data["shortest_edges"])):
                key=(job["stratum"],i,np.packbits(pred).tobytes())
                if key in cache:ok,independent_distance=cache[key];cache_hits+=1
                else:
                    ok,independent_distance=A.independent(tok,pred);cache[key]=(ok,independent_distance);oracle_checks+=1
                assert distance==independent_distance;valid.append(ok)
            valid=np.asarray(valid);finite=output["answer_finite"];steps=output["answer_steps"]
            success=valid&output["stopped"]&output["hidden_good"]&finite
            assert np.array_equal(valid,output["shortest_path"]) and np.array_equal(success,output["success"])
            assert np.all(output["first_nonfinite_check"][finite]==-1)
            assert np.all(output["first_nonfinite_check"][~finite]>=16)
            assert not np.any(output["stopped"]&~finite)
            assert int(success.sum())==arm["solved"] and int((output["stopped"]&~valid).sum())==arm["wrong_stops"]
            assert int((~output["stopped"]&finite).sum())==arm["answer_budget_failures"]
            assert int((~finite).sum())==arm["nonfinite_failures"] and int((~output["hidden_good"]).sum())==arm["hidden_failures"]
            assert np.all((steps>=16)&(steps<=8192)&(steps%16==0)) and np.all(steps[~output["stopped"]]==8192)
            saved[(job["stratum"],job["seed"],eta)]={"steps":steps,"success":success,"finite":finite}
            records.append(dict(job=job,eta=float(eta),n=len(steps),**arm,hidden_seconds_shared=record["hidden_seconds_shared"],gpu=record["gpu"]))
        if record["diagnostics_sha256"]:
            verify(run/"diagnostics.json",record["diagnostics_sha256"])
            for eta,diag in json.loads((run/"diagnostics.json").read_text()).items():
                assert diag["replay_main_parity"]
                energies=np.asarray([h["energy"] for h in diag["history"]],dtype=float);change=np.diff(energies,axis=0)
                finite_pairs=np.isfinite(change);positive=change[finite_pairs]
                diagnostics.append({"stratum":job["stratum"],"eta":float(eta),"n_traces":energies.shape[1],"replay_main_parity":True,
                                    "recorded_energy_transitions":int(finite_pairs.sum()),"positive_energy_changes":int((positive>0).sum()),
                                    "maximum_energy_increase":float(np.maximum(positive,0).max()) if len(positive) else None,
                                    "nonfinite_energy_values":int((~np.isfinite(energies)).sum()),"two_cycle_flags":int(sum(diag["two_cycle_detected"]))})

    joint={};steps={};cells=[];maze_rows=[];losses=[];gains=[];loss_lower=[];loss_upper=[]
    for stratum in STRATA:
        for eta in ETAS:
            successful=np.stack([saved[(stratum,s,eta)]["success"] for s in range(3)])
            joint[(stratum,eta)]=successful.all(0)
            steps[(stratum,eta)]=np.stack([saved[(stratum,s,eta)]["steps"] for s in range(3)]).mean(0)
            selected=[r for r in records if r["job"]["stratum"]==stratum and r["eta"]==float(eta)]
            cells.append({"stratum":stratum,"eta":float(eta),"n_unique":512,"solved_per_seed":successful.sum(1).tolist(),
                          "joint_solved":int(successful.all(0).sum()),"median_steps_including_caps":float(np.median(steps[(stratum,eta)])),
                          "wrong_stops":sum(r["wrong_stops"] for r in selected),"budget_failures":sum(r["answer_budget_failures"] for r in selected),
                          "nonfinite_failures":sum(r["nonfinite_failures"] for r in selected),"hidden_failures":sum(r["hidden_failures"] for r in selected),
                          "checked_answer_seconds":sum(r["checked_answer_seconds"] for r in selected)})
        loss=joint[(stratum,"0.5")]&~joint[(stratum,"1.0")];gain=~joint[(stratum,"0.5")]&joint[(stratum,"1.0")]
        lower,upper=A.bounds(int(loss.sum()),512,alpha=.05/3)
        losses.append(loss);gains.append(gain);loss_lower.append(lower);loss_upper.append(upper)
        for i in range(512):
            row={"stratum":stratum,"maze_index":i,"lost_eta1_vs05":int(loss[i]),"gained_eta1_vs05":int(gain[i])}
            for eta in ETAS:
                suffix=eta.replace(".","");row[f"joint_success_eta{suffix}"]=int(joint[(stratum,eta)][i]);row[f"mean_steps_eta{suffix}"]=float(steps[(stratum,eta)][i])
            maze_rows.append(row)
    lower,upper=float(np.mean(loss_lower)),float(np.mean(loss_upper));rng=np.random.default_rng(2026111423);bootstrap=np.zeros(10000)
    for loss in losses:bootstrap+=loss[rng.integers(512,size=(10000,512))].mean(1)/3
    primary={"n_unique":1536,"losses":int(sum(x.sum() for x in losses)),"gains":int(sum(x.sum() for x in gains)),
             "lost_fraction_pp":float(100*np.mean(losses)),"one_sided95_mixture_lower_pp":100*lower,"one_sided95_mixture_upper_pp":100*upper,
             "bound_method":"Equal average of stratum Clopper-Pearson bounds, alpha=.05/3 each; Bonferroni simultaneous bound.",
             "paired_stratified_bootstrap95_loss_pp":(100*np.quantile(bootstrap,[.025,.975])).tolist(),
             "decision":"supported" if lower>=.01 else "falsified" if upper<.01 else "inconclusive",
             "strata":[{"stratum":stratum,"n_unique":512,"losses":int(losses[i].sum()),"gains":int(gains[i].sum()),
                         "one_sided98_333_lower_pp":100*loss_lower[i],"one_sided98_333_upper_pp":100*loss_upper[i]} for i,stratum in enumerate(STRATA)]}
    summaries=[]
    for eta in ETAS:
        outcome=np.concatenate([joint[(s,eta)] for s in STRATA]);all_steps=np.concatenate([steps[(s,eta)] for s in STRATA]);selected=[r for r in records if r["eta"]==float(eta)]
        summaries.append({"eta":float(eta),"n_unique":len(outcome),"joint_solved":int(outcome.sum()),"joint_accuracy_percent":float(100*outcome.mean()),
                          "median_steps_including_caps":float(np.median(all_steps)),"median_steps_joint_successes_only":float(np.median(all_steps[outcome])) if outcome.any() else None,
                          "checked_answer_seconds":sum(r["checked_answer_seconds"] for r in selected),
                          "hidden_plus_checked_answer_seconds":sum(r["hidden_plus_checked_answer_seconds"] for r in selected),
                          "wrong_stops":sum(r["wrong_stops"] for r in selected),"budget_failures":sum(r["answer_budget_failures"] for r in selected),
                          "nonfinite_failures":sum(r["nonfinite_failures"] for r in selected),"hidden_failures":sum(r["hidden_failures"] for r in selected)})
    with (dest/"per-maze.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(maze_rows[0]));writer.writeheader();writer.writerows(maze_rows)
    verify(dest/"per-maze.csv")
    result={"status":"validated","utc":S.utc(),"campaign":"eta_boundary","jobs":len(protocol["jobs"]),"evaluated_arms":len(records),
            "saved_board_checks":sum(r["n"] for r in records),"protocol_sha256":S.sha(pp),"primary":primary,"summaries":summaries,
            "cells":cells,"diagnostics":diagnostics,"records":records,"evidence_sha256":evidence,"hashed_files":len(evidence),
            "oracle_unique_board_checks":oracle_checks,"oracle_identical_board_cache_hits":cache_hits,"audit_seconds":time.perf_counter()-started,
            "new_inference":False,"limits":["No extrapolation to arbitrary eta, inputs or models; failure to find instability is not a stability theorem.",
                "Joint success counts each input once, requiring all three fixed models to succeed.",
                "Checked timing includes identical finite checks, with sameGPU per job and shared hidden preparation; it is not native solver timing.",
                "Energy/cycle traces cover the prespecified24unique inputs per eta at modelseed0 only; periodic energy samples do not establish monotonicity between samples.",
                "Unchanged weights and energies; no adaptive eta rescue or selected new eta."]}
    S.save(dest/"audit.json",result);S.emit("eta_boundary_audit_complete",primary=primary,summaries=summaries,audit_seconds=result["audit_seconds"])


if __name__=="__main__":main()
