"""Public paired sign interventions and exact initialization provenance."""
import json
import shutil
import study as S


def main():
    import export_margin_intervention
    export_margin_intervention.main();out=S.ROOT/"results/research/work";source=S.OUT/"binary_relaxation";audit=json.loads((source/"audit.json").read_text());assert audit["status"]=="validated"
    S.save(out/"binary-relaxation-summary.json",{k:v for k,v in audit.items() if k not in ["rows","records","evidence_sha256"]}|{"full_local_audit_sha256":S.sha(source/"audit.json")})
    for name in ["runs","per-model-input","per-input"]:shutil.copyfile(source/f"{name}.csv",out/f"binary-relaxation-{name}.csv")
    shutil.copyfile(S.OUT/"binary_preserving/validation.json",out/"binary-preserving-validation.json")
    shutil.copyfile(source/"selected-cases.json",out/"binary-relaxation-selected-cases.json")
    rechecks=[json.loads((source/f"diagnostic-recheck-gpu{gpu}.json").read_text()) for gpu in [0,1]]
    assert all(r["status"]=="passed" for r in rechecks)
    S.save(out/"binary-relaxation-diagnostic-recheck.json",{"status":"passed","model_input_diagnostics":sum(r["model_input_diagnostics"] for r in rechecks),"maximum_error":max(max(row["maximum_errors"].values()) for r in rechecks for row in r["rows"]),"new_answer_rollouts":0,"receipts":rechecks})
    traces={f"model{seed}":json.loads((source/f"runs/shard00-s{seed}/diagnostics.json").read_text()) for seed in range(3)}
    S.save(out/"binary-relaxation-trajectories.json",{"scope":"Prespecified first8inputs of first64-input batch of shard0, all3models and4arms; original primary trajectories, no reruns","traces":traces})
    summary=json.loads((out/"summary.json").read_text());summary["jobs"]+=48;summary["primary_jobs"]+=48;summary["model_maze_evaluations"]+=49152;summary["primary_model_maze_evaluations"]+=49152
    summary["run_arm_rows"]+=192;summary["run_tables"].append("binary-relaxation-runs.csv");summary["hypotheses"].append({"campaign":"binary_relaxation","hypothesis":"H2q compensated lambda+1 improves identical-Gaussian joint success >=5pp","outcome":audit["primary"]})
    pp=S.HERE/"binary_relaxation-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"]["binary_relaxation-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"]}
    summary["audit_sha256"][str((source/"audit.json").relative_to(S.ROOT))]=S.sha(source/"audit.json");summary["updated_utc"]=S.utc();summary["status"]="sixteen planned studies, two selected-case interventions, and exhaustive shortest-answer energies complete; ongoing research continues separately"
    S.save(out/"summary.json",summary);manifest(out);S.emit("binary_relaxation_public_export",jobs=summary["jobs"],new_rollout_total=summary["model_maze_evaluations"])


def manifest(out):
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"base_export_sha256":S.sha(S.HERE/"export_margin_intervention.py"),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})


if __name__=="__main__":main()
