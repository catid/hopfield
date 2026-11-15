"""Public paired sign interventions and exact initialization provenance."""
import json
import shutil
import study as S


def main():
    import export_sign_intervention
    export_sign_intervention.main();out=S.ROOT/"results/research/work";source=S.OUT/"margin_intervention";audit=json.loads((source/"audit.json").read_text());assert audit["status"]=="validated"
    S.save(out/"margin-intervention-summary.json",{k:v for k,v in audit.items() if k not in ["rows","records","evidence_sha256"]}|{"full_local_audit_sha256":S.sha(source/"audit.json")})
    for name in ["runs","per-model-input","per-input"]:shutil.copyfile(source/f"{name}.csv",out/f"margin-intervention-{name}.csv")
    shutil.copyfile(source/"audit-cache-validation.json",out/"margin-audit-cache-validation.json")
    shutil.copyfile(source/"output-parity.json",out/"margin-output-parity.json")
    summary=json.loads((out/"summary.json").read_text());summary["jobs"]+=48;summary["primary_jobs"]+=48;summary["model_maze_evaluations"]+=73728;summary["primary_model_maze_evaluations"]+=73728
    summary["run_arm_rows"]+=288;summary["run_tables"].append("margin-intervention-runs.csv");summary["hypotheses"].append({"campaign":"margin_intervention","hypothesis":"H2p upward-only initial logits noninferior within0.1pp to native","outcome":audit["primary"]})
    pp=S.HERE/"margin_intervention-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"]["margin_intervention-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"]}
    summary["audit_sha256"][str((source/"audit.json").relative_to(S.ROOT))]=S.sha(source/"audit.json");summary["updated_utc"]=S.utc();summary["status"]="fifteen planned studies, two selected-case interventions, and exhaustive shortest-answer energies complete; ongoing research continues separately"
    S.save(out/"summary.json",summary);manifest(out);S.emit("margin_intervention_public_export",jobs=summary["jobs"],new_rollout_total=summary["model_maze_evaluations"])


def manifest(out):
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"base_export_sha256":S.sha(S.HERE/"export_sign_intervention.py"),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})


if __name__=="__main__":main()
