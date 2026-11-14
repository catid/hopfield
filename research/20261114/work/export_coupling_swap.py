"""Publish corrected counterfactuals and the explicit excluded-attempt receipt."""
import json
import shutil
import study as S
import export_tie_controls


def main():
    export_tie_controls.main();out=S.ROOT/"results/research/work";source=S.OUT/"coupling_swap_corrected";audit=json.loads((source/"audit.json").read_text())
    assert audit["status"]=="validated" and audit["all12_identity_full_state_checks_passed"]
    S.save(out/"coupling-swap-corrected-summary.json",{k:v for k,v in audit.items() if k!="evidence_sha256"}|{"full_local_audit_sha256":S.sha(source/"audit.json")})
    shutil.copyfile(source/"arms.csv",out/"coupling-swap-corrected-arms.csv")
    shutil.copyfile(source/"preflight.json",out/"coupling-swap-identity-preflight.json")
    shutil.copyfile(S.OUT/"coupling_swap/protocol-deviation.json",out/"coupling-swap-excluded-v1.json")
    summary=json.loads((out/"summary.json").read_text());summary["updated_utc"]=S.utc();summary["status"]="ten planned studies and two selected-case interventions complete; ongoing research continues separately"
    summary["jobs"]+=3;summary["model_maze_evaluations"]+=30;summary["descriptive_jobs"]+=3;summary["descriptive_model_maze_evaluations"]+=30;summary["run_arm_rows"]+=30
    summary["descriptive_includes_baseline_arms"]=12;summary["run_tables"].append("coupling-swap-corrected-arms.csv")
    summary["excluded_diagnostic"]={"jobs":3,"arms":30,"reason":"Attention self/support masking confound; no causal conclusion retained.","receipt":"coupling-swap-excluded-v1.json"}
    summary["corrected_coupling_swap"]={"summary":"coupling-swap-corrected-summary.json","identity_checks":12,"arms":30,
        "conclusion":"Readout from either targeted deletion restores the original geometry for all3models; original readout on edited geometry fails. Matchednoncritical readout doesnot restore failedmodels. Selected-case result only."}
    pp=S.HERE/"coupling_swap_corrected-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"]["coupling_swap_corrected-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"],"post_outcome_descriptive":True}
    summary["audit_sha256"][str((source/"audit.json").relative_to(S.ROOT))]=S.sha(source/"audit.json")
    S.save(out/"summary.json",summary)
    for ext in ("png","pdf","svg"):shutil.copyfile(S.OUT/f"figures/coupling-geometry-separation.{ext}",out/f"coupling-geometry-separation.{ext}")
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"base_export_sha256":S.sha(export_tie_controls.__file__),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})
    S.emit("corrected_swap_public_export",jobs=summary["jobs"],evaluations=summary["model_maze_evaluations"],excluded_arms=30)


if __name__=="__main__":main()
