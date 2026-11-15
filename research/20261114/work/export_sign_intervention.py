"""Public paired sign interventions and exact initialization provenance."""
import json
import shutil
import study as S


def main():
    import export_native_restarts
    export_native_restarts.main();out=S.ROOT/"results/research/work";source=S.OUT/"sign_intervention";audit=json.loads((source/"audit.json").read_text());assert audit["status"]=="validated"
    S.save(out/"sign-intervention-summary.json",{k:v for k,v in audit.items() if k not in ["rows","records","evidence_sha256"]}|{"full_local_audit_sha256":S.sha(source/"audit.json")})
    for name in ["runs","per-model-input","per-input"]:shutil.copyfile(source/f"{name}.csv",out/f"sign-intervention-{name}.csv")
    summary=json.loads((out/"summary.json").read_text());summary["jobs"]+=24;summary["primary_jobs"]+=24;summary["model_maze_evaluations"]+=24576;summary["primary_model_maze_evaluations"]+=24576
    summary["run_arm_rows"]+=96;summary["run_tables"].append("sign-intervention-runs.csv");summary["hypotheses"].append({"campaign":"sign_intervention","hypothesis":"H2o resetting initially negative logits raises joint success >=10pp over fullGaussian","outcome":audit["primary"]})
    pp=S.HERE/"sign_intervention-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"]["sign_intervention-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"]}
    summary["audit_sha256"][str((source/"audit.json").relative_to(S.ROOT))]=S.sha(source/"audit.json");summary["updated_utc"]=S.utc();summary["status"]="fourteen planned studies, two selected-case interventions, and exhaustive shortest-answer energies complete; ongoing research continues separately"
    S.save(out/"summary.json",summary);manifest(out);S.emit("sign_intervention_public_export",jobs=summary["jobs"],new_rollout_total=summary["model_maze_evaluations"])


def manifest(out):
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"base_export_sha256":S.sha(S.HERE/"export_native_restarts.py"),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})


if __name__=="__main__":main()
