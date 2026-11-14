"""Public results for the independent matched fork-angle test."""
import csv
import json
import shutil
import study as S
import export_coupling_swap


def main():
    export_coupling_swap.main();out=S.ROOT/"results/research/work";source=S.OUT/"fork_angle";audit=json.loads((source/"audit.json").read_text());assert audit["status"]=="validated"
    S.save(out/"fork-angle-summary.json",{k:v for k,v in audit.items() if k not in ["records","evidence_sha256"]}|{"full_local_audit_sha256":S.sha(source/"audit.json")})
    rows=[]
    for r in audit["records"]:
        j=r["job"];rows.append({"job":j["id"],"route_length":j["route_length"],"condition":j["condition"],"model_seed":j["seed"],
            **{k:r[k] for k in ["n","solved","wrong_stops","answer_budget_failures","hidden_failures","answer_steps_mean","hidden_steps_mean","wall_seconds"]}})
    with (out/"fork-angle-runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shutil.copyfile(source/"per-layout.csv",out/"fork-angle-per-layout.csv");shutil.copyfile(source/"fork-readouts.csv",out/"fork-angle-readouts.csv")
    summary=json.loads((out/"summary.json").read_text());summary["jobs"]+=18;summary["primary_jobs"]+=18;summary["model_maze_evaluations"]+=2304;summary["primary_model_maze_evaluations"]+=2304
    summary["run_arm_rows"]+=18;summary["run_tables"].append("fork-angle-runs.csv");summary["hypotheses"].append({"campaign":"fork_angle","hypothesis":"H2l right-angle forks lose >=10pp joint success","outcome":audit["primary"]})
    pp=S.HERE/"fork_angle-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"]["fork_angle-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"]}
    summary["audit_sha256"][str((source/"audit.json").relative_to(S.ROOT))]=S.sha(source/"audit.json");summary["updated_utc"]=S.utc();summary["status"]="eleven planned studies and two selected-case interventions complete; ongoing research continues separately"
    S.save(out/"summary.json",summary)
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"base_export_sha256":S.sha(export_coupling_swap.__file__),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})
    S.emit("fork_angle_public_export",jobs=summary["jobs"],evaluations=summary["model_maze_evaluations"])


if __name__=="__main__":main()
