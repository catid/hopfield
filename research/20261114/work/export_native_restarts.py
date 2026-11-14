"""Publish native-plus-three candidate study with exact reuse accounting."""
import csv
import json
import shutil
import study as S


def main():
    import export_restart
    export_restart.main();out=S.ROOT/"results/research/work";source=S.OUT/"native_restart";audit=json.loads((source/"audit.json").read_text());assert audit["status"]=="validated"
    S.save(out/"native-restart-summary.json",{k:v for k,v in audit.items() if k not in ["records","candidate_records","failure_examples","evidence_sha256"]}|{"full_local_audit_sha256":S.sha(source/"audit.json")})
    rows=[]
    for r in audit["records"]:
        rows.append({"job":r["job"]["id"],"shard":r["job"]["shard"],"model_seed":r["job"]["seed"],**{k:r[k] for k in ["n","native_solved","energy_solved","screen_solved","length_solved","no_eligible","no_feasible","hidden_seconds_shared","selector_cpu_seconds","wall_seconds","native_prefix_reused_batches","actual_answer_batch_calls"]},"native_answer_seconds":r["answer_seconds_by_variant"][0],"four_candidate_equivalent_answer_seconds":sum(r["answer_seconds_by_variant"][1:])})
    with (out/"native-restart-runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name in ["candidates","per-model-input","per-input"]:shutil.copyfile(source/f"{name}.csv",out/f"native-restart-{name}.csv")
    summary=json.loads((out/"summary.json").read_text());summary["jobs"]+=48;summary["primary_jobs"]+=48;summary["model_maze_evaluations"]+=audit["model_input_rollouts"];summary["primary_model_maze_evaluations"]+=audit["model_input_rollouts"]
    summary["run_arm_rows"]+=240;summary["run_tables"].append("native-restart-candidates.csv");summary["hypotheses"].append({"campaign":"native_restart","hypothesis":"H2n feasible minimum-node-count selection noninferior within0.1pp to native8192","outcome":audit["primary"]})
    summary["native_prefix_reuse"]={"logical_candidate_reference_outcomes":61440,"new_answer_rollouts":audit["model_input_rollouts"],"exact_reused_native_prefixes":audit["native_prefix_reused_outcomes"],"reused_prefixes_not_added_to_total_evaluations":True}
    pp=S.HERE/"native_restart-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"]["native_restart-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"]}
    summary["audit_sha256"][str((source/"audit.json").relative_to(S.ROOT))]=S.sha(source/"audit.json");summary["updated_utc"]=S.utc();summary["status"]="thirteen planned studies, two selected-case interventions, and exhaustive shortest-answer energies complete; ongoing research continues separately"
    S.save(out/"summary.json",summary);manifest(out);S.emit("native_restart_public_export",jobs=summary["jobs"],new_rollout_total=summary["model_maze_evaluations"])


def manifest(out):
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"base_export_sha256":S.sha(S.HERE/"export_restart.py"),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})


if __name__=="__main__":main()
