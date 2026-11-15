"""Public practical-default trial with paired rare-event uncertainty."""
import json
import shutil
import numpy as np
import study as S


def main():
    import export_native_relaxation
    export_native_relaxation.main();out=S.ROOT/"results/research/work";source=S.OUT/"larger_relaxation";audit=json.loads((source/"audit.json").read_text());assert audit["status"]=="validated"
    S.save(out/"larger-relaxation-summary.json",{k:v for k,v in audit.items() if k not in ["rows","records","evidence_sha256"]}|{"full_local_audit_sha256":S.sha(source/"audit.json")})
    for name in ["runs","per-model-input","per-input"]:shutil.copyfile(source/f"{name}.csv",out/f"larger-relaxation-{name}.csv")
    rechecks=[json.loads((source/f"diagnostic-recheck-gpu{gpu}.json").read_text()) for gpu in [0,1]];assert all(r["status"]=="passed" for r in rechecks)
    S.save(out/"larger-relaxation-diagnostic-recheck.json",{"status":"passed","model_input_diagnostics":sum(r["model_input_diagnostics"] for r in rechecks),"maximum_error":max(max(row["maximum_errors"].values()) for r in rechecks for row in r["rows"]),"new_answer_rollouts":0,"receipts":rechecks})
    shutil.copyfile(source/"failure-classification.json",out/"larger-relaxation-failure-classification.json")
    shutil.copyfile(source/"scipy-statistics-verification.json",out/"larger-relaxation-statistics-verification.json")
    selected=[]
    for index in audit["secondary"]["loss_input_indices"]:
        shard,i=divmod(index,256);data=np.load(source/f"shard{shard:02d}.npz");models=[]
        for seed in range(3):
            with np.load(source/f"runs/shard{shard:02d}-s{seed}/outputs.npz") as archive:
                models.append({"model_seed":seed,"arms":[{"arm":arm,"selected_flat_indices":np.flatnonzero(archive["predictions"][v,i]).tolist(),"original_binary_energy":float(archive["energy"][v,i]),"minimum_original_flip_margin":float(archive["minimum_original_flip_margin"][v,i]),"answer_steps":int(archive["answer_steps"][v,i]),"feasible":bool(archive["feasible"][v,i]),"certified":bool(archive["stopped"][v,i])} for v,arm in enumerate(["native_lambda0","native_lambda1"])]})
        selected.append({"input_index":index,"tokens":data["tokens"][i].tolist(),"shortest_edges_evaluation_only":int(data["shortest_edges"][i]),"models":models})
    S.save(out/"larger-relaxation-regression-cases.json",{"selection":"Every joint native-to-lambda regression, selected after primary outcomes for explanation; no new inference.","cases":selected,"audit_sha256":S.sha(source/"audit.json")})
    summary=json.loads((out/"summary.json").read_text());summary["jobs"]+=192;summary["primary_jobs"]+=192;summary["model_maze_evaluations"]+=98304;summary["primary_model_maze_evaluations"]+=98304
    summary["run_arm_rows"]+=384;summary["run_tables"].append("larger-relaxation-runs.csv");summary["hypotheses"].append({"campaign":"larger_relaxation","hypothesis":"H2s 120-grid native-u3 lambda+1 improves joint success >=0.05pp over native-u3 lambda0","outcome":audit["primary"]})
    pp=S.HERE/"larger_relaxation-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"]["larger_relaxation-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"]}
    summary["audit_sha256"][str((source/"audit.json").relative_to(S.ROOT))]=S.sha(source/"audit.json");summary["updated_utc"]=S.utc();summary["status"]="eighteen planned studies, two selected-case interventions, and exhaustive shortest-answer energies complete; ongoing research continues separately"
    S.save(out/"summary.json",summary);manifest(out);S.emit("larger_relaxation_public_export",jobs=summary["jobs"],new_rollout_total=summary["model_maze_evaluations"])


def manifest(out):
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"base_export_sha256":S.sha(S.HERE/"export_native_relaxation.py"),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})


if __name__=="__main__":main()
