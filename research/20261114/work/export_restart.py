"""Public fresh-data evidence for restart selectors, including all failures."""
import csv
import json
import shutil
import study as S


def main():
    import export_landscape
    export_landscape.main();out=S.ROOT/"results/research/work";source=S.OUT/"restart";audit=json.loads((source/"audit.json").read_text());assert audit["status"]=="validated"
    S.save(out/"restart-summary.json",{k:v for k,v in audit.items() if k not in ["records","candidate_records","failure_examples","evidence_sha256"]}|{"full_local_audit_sha256":S.sha(source/"audit.json")})
    rows=[]
    for r in audit["records"]:
        rows.append({"job":r["job"]["id"],"shard":r["job"]["shard"],"model_seed":r["job"]["seed"],**{k:r[k] for k in ["n","native_solved","energy_solved","screen_solved","no_eligible","no_feasible","hidden_seconds_shared","selector_cpu_seconds","wall_seconds"]},"native_answer_seconds":r["answer_seconds_by_variant"][0],"restart_answer_seconds":sum(r["answer_seconds_by_variant"][1:])})
    with (out/"restart-runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name in ["candidates","per-model-input","per-input"]:shutil.copyfile(source/f"{name}.csv",out/f"restart-{name}.csv")
    shutil.copyfile(source/"failure-diagnostics.json",out/"restart-failure-diagnostics.json")
    error_types=json.loads((source/"all-error-types.json").read_text())
    S.save(out/"restart-all-error-types.json",{k:v for k,v in error_types.items() if k!="evidence_sha256"})
    summary=json.loads((out/"summary.json").read_text());summary["jobs"]+=48;summary["primary_jobs"]+=48;summary["model_maze_evaluations"]+=61440;summary["primary_model_maze_evaluations"]+=61440
    summary["run_arm_rows"]+=240;summary["run_tables"].append("restart-candidates.csv");summary["hypotheses"].append({"campaign":"restart","hypothesis":"H2m feasibility screening raises joint success >=0.1pp over energy-only selection of the same four candidates","outcome":audit["primary"]})
    pp=S.HERE/"restart-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"]["restart-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"]}
    summary["audit_sha256"][str((source/"audit.json").relative_to(S.ROOT))]=S.sha(source/"audit.json");summary["updated_utc"]=S.utc();summary["status"]="twelve planned studies, two selected-case interventions, and exhaustive shortest-answer energies complete; ongoing research continues separately"
    S.save(out/"summary.json",summary);manifest(out);S.emit("restart_public_export",jobs=summary["jobs"],evaluations=summary["model_maze_evaluations"])


def manifest(out):
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"base_export_sha256":S.sha(S.HERE/"export_landscape.py"),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})


if __name__=="__main__":main()
