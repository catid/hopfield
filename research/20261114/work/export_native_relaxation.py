"""Public practical-default trial with paired rare-event uncertainty."""
import json
import shutil
import study as S


def main():
    import export_binary_relaxation
    export_binary_relaxation.main();out=S.ROOT/"results/research/work";source=S.OUT/"native_relaxation";audit=json.loads((source/"audit.json").read_text());assert audit["status"]=="validated"
    S.save(out/"native-relaxation-summary.json",{k:v for k,v in audit.items() if k not in ["rows","records","evidence_sha256"]}|{"full_local_audit_sha256":S.sha(source/"audit.json")})
    for name in ["runs","per-model-input","per-input"]:shutil.copyfile(source/f"{name}.csv",out/f"native-relaxation-{name}.csv")
    rechecks=[json.loads((source/f"diagnostic-recheck-gpu{gpu}.json").read_text()) for gpu in [0,1]];assert all(r["status"]=="passed" for r in rechecks)
    S.save(out/"native-relaxation-diagnostic-recheck.json",{"status":"passed","model_input_diagnostics":sum(r["model_input_diagnostics"] for r in rechecks),"maximum_error":max(max(row["maximum_errors"].values()) for r in rechecks for row in r["rows"]),"new_answer_rollouts":0,"receipts":rechecks})
    shutil.copyfile(source/"failure-classification.json",out/"native-relaxation-failure-classification.json")
    shutil.copyfile(source/"scipy-statistics-verification.json",out/"native-relaxation-statistics-verification.json")
    summary=json.loads((out/"summary.json").read_text());summary["jobs"]+=192;summary["primary_jobs"]+=192;summary["model_maze_evaluations"]+=98304;summary["primary_model_maze_evaluations"]+=98304
    summary["run_arm_rows"]+=384;summary["run_tables"].append("native-relaxation-runs.csv");summary["hypotheses"].append({"campaign":"native_relaxation","hypothesis":"H2r native-u3 lambda+1 improves joint success >=0.05pp over native-u3 lambda0","outcome":audit["primary"]})
    pp=S.HERE/"native_relaxation-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"]["native_relaxation-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"]}
    summary["audit_sha256"][str((source/"audit.json").relative_to(S.ROOT))]=S.sha(source/"audit.json");summary["updated_utc"]=S.utc();summary["status"]="seventeen planned studies, two selected-case interventions, and exhaustive shortest-answer energies complete; ongoing research continues separately"
    S.save(out/"summary.json",summary);manifest(out);S.emit("native_relaxation_public_export",jobs=summary["jobs"],new_rollout_total=summary["model_maze_evaluations"])


def manifest(out):
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"base_export_sha256":S.sha(S.HERE/"export_binary_relaxation.py"),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})


if __name__=="__main__":main()
