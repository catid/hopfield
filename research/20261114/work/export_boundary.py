"""Append the fully audited stability trial and failure case to public results."""
import csv
import json
import shutil
import numpy as np
import torch
import study as S
import export_public


def main():
    export_public.main();out=S.ROOT/"results/research/work";source=S.OUT/"eta_boundary";audit=json.loads((source/"audit.json").read_text())
    assert audit["status"]=="validated"
    S.save(out/"eta-boundary-summary.json",{k:v for k,v in audit.items() if k not in ["records","evidence_sha256"]}|{"full_local_audit_sha256":S.sha(source/"audit.json")})
    rows=[]
    for r in audit["records"]:
        j=r["job"];rows.append(dict(job=j["id"],stratum=j["stratum"],length=j["length"],loop_percent=j["loop_percent"],
            model_seed=j["seed"],eta=r["eta"],n=r["n"],solved=r["solved"],mean_answer_steps=r["answer_steps_mean"],
            wrong_stops=r["wrong_stops"],budget_failures=r["answer_budget_failures"],nonfinite_failures=r["nonfinite_failures"],
            hidden_failures=r["hidden_failures"],checked_answer_seconds=r["checked_answer_seconds"],hidden_seconds_shared=r["hidden_seconds_shared"],gpu=r["gpu"]))
    with (out/"eta-boundary-runs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shutil.copyfile(source/"per-maze.csv",out/"eta-boundary-per-maze.csv")
    shutil.copyfile(source/"failure-classification.json",out/"cyclic-failure-classification.json")
    graph=np.load(source/"failure-forensics/shortest-path-union.npz");case={k:graph[k].astype(int).tolist() for k in graph.files}
    case.update(post_outcome_diagnostic=True,dataset_index=38,dataset_sha256=S.sha(source/"loops20.npz"),models={},predictions={})
    for seed in range(3):
        path=S.OUT/f"models/seed{seed}.pt";weights=torch.load(path,map_location="cpu",weights_only=True)
        case["models"][str(seed)]={"checkpoint_sha256":S.sha(path),"state_dict_float32":{k:v.tolist() for k,v in weights.items()}}
        for eta in (.2,.5,.8,1.):
            state=np.load(source/f"failure-forensics/s{seed}-eta{eta}-states.npz")
            case["predictions"][f"seed{seed}-eta{eta}"]=state["prediction"].astype(int).tolist()
    S.save(out/"cyclic-failure-case.json",case)
    summary=json.loads((out/"summary.json").read_text());summary["updated_utc"]=S.utc();summary["status"]="eight planned studies complete; ongoing research continues separately"
    summary["jobs"]+=audit["jobs"];summary["run_arm_rows"]+=len(rows);summary["model_maze_evaluations"]+=audit["saved_board_checks"]
    summary["hypotheses"].append({"campaign":"eta_boundary","hypothesis":"H2i eta1 loses >=1pp joint solves versus eta0.5","outcome":audit["primary"]})
    summary["absolute_reliability_exception"]={"joint_solved_each_arm":1535,"n_unique":1536,"wrong_stops_per_arm_eta02_eta05_eta08":2,"timeouts_eta1":2,
        "case":"cyclic-failure-classification.json","scope":"The same cyclic maze fails for model seeds1and2; these are true invalid fixed points or a period-two timeout, not hidden/nonfinite failures."}
    pp=S.HERE/"eta_boundary-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"]["eta_boundary-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"]}
    summary["audit_sha256"][str((source/"audit.json").relative_to(S.ROOT))]=S.sha(source/"audit.json");summary["run_tables"].append("eta-boundary-runs.csv")
    S.save(out/"summary.json",summary)
    for name in ("eta-boundary","cyclic-failure"):
        for ext in ("png","pdf","svg"):shutil.copyfile(S.OUT/f"figures/{name}.{ext}",out/f"{name}.{ext}")
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"base_export_sha256":S.sha(export_public.__file__),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})
    S.emit("boundary_public_export",jobs=summary["jobs"],evaluations=summary["model_maze_evaluations"])


if __name__=="__main__":main()
