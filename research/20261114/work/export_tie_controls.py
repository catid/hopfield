"""Publish independent negative controls separately from selected-case edits."""
import csv
import json
import shutil
import study as S
import export_boundary


def main():
    export_boundary.main();out=S.ROOT/"results/research/work";summary=json.loads((out/"summary.json").read_text())
    for campaign,hypothesis in [("tie_controls","H2j tie multiplicity loses >=10pp"),("long_ties","H2k long tied routes lose >=10pp")]:
        source=S.OUT/campaign;audit=json.loads((source/"audit.json").read_text());assert audit["status"]=="validated"
        S.save(out/f"{campaign}-summary.json",{k:v for k,v in audit.items() if k not in ["records","evidence_sha256"]}|{"full_local_audit_sha256":S.sha(source/"audit.json")})
        rows=[]
        for r in audit["records"]:
            j=r["job"];rows.append(dict(job=j["id"],cycles=j.get("cycles",1),route_length=j.get("route_length"),condition=j["condition"],
                model_seed=j["seed"],n=r["n"],solved=r["solved"],wrong_stops=r["wrong_stops"],budget_failures=r["answer_budget_failures"],
                hidden_failures=r["hidden_failures"],mean_answer_steps=r["answer_steps_mean"],wall_seconds=r["wall_seconds"]))
        with (out/f"{campaign}-runs.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        shutil.copyfile(source/"per-layout.csv",out/f"{campaign}-per-layout.csv")
        summary["jobs"]+=audit["jobs"];summary["run_arm_rows"]+=len(rows);summary["model_maze_evaluations"]+=audit["saved_board_checks"]
        summary["hypotheses"].append({"campaign":campaign,"hypothesis":hypothesis,"outcome":audit["primary"]})
        pp=S.HERE/f"{campaign}-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"][f"{campaign}-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"]}
        summary["audit_sha256"][str((source/"audit.json").relative_to(S.ROOT))]=S.sha(source/"audit.json");summary["run_tables"].append(f"{campaign}-runs.csv")
    summary["primary_jobs"]=summary["jobs"];summary["primary_model_maze_evaluations"]=summary["model_maze_evaluations"]
    path=S.OUT/"forensic_edits/audit.json";edits=json.loads(path.read_text());assert edits["status"]=="validated"
    S.save(out/"forensic-edits-summary.json",edits)
    with (out/"forensic-edits-cases.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(edits["rows"][0]));writer.writeheader();writer.writerows(edits["rows"])
    with (out/"forensic-edits-runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(edits["summaries"][0]));writer.writeheader();writer.writerows(edits["summaries"])
    pp=S.HERE/"forensic_edits-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"]["forensic_edits-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"],"post_outcome_descriptive":True}
    summary["descriptive_jobs"]=3;summary["descriptive_model_maze_evaluations"]=102;summary["jobs"]+=3;summary["model_maze_evaluations"]+=102
    summary["run_arm_rows"]+=3;summary["run_tables"].append("forensic-edits-runs.csv")
    summary["selected_case_interventions"]={"dataset_index":38,"summary":"forensic-edits-summary.json","targeted_joint_successes":2,"targeted_edits":2,"matched_joint_successes":0,"matched_edits":32,"scope":"One selected graph; no population inference."}
    summary["updated_utc"]=S.utc();summary["status"]="ten planned studies and one selected-case intervention complete; ongoing research continues separately"
    S.save(out/"summary.json",summary)
    shutil.copyfile(S.OUT/"eta_boundary/failure-forensics/energy-decomposition.json",out/"cyclic-failure-energy-decomposition.json")
    for ext in ("png","pdf","svg"):shutil.copyfile(S.OUT/f"figures/tie-controls-and-selected-edits.{ext}",out/f"tie-controls-and-selected-edits.{ext}")
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"base_export_sha256":S.sha(export_boundary.__file__),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})
    S.emit("tie_controls_public_export",jobs=summary["jobs"],primary_evaluations=summary["primary_model_maze_evaluations"],descriptive_evaluations=102)


if __name__=="__main__":main()
