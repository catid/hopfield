"""Export compact public metrics, plotted data and provenance for Work studies."""
import csv
import json
from pathlib import Path
import shutil
import study as S


def main():
    out=S.ROOT/"results/research/work";out.mkdir(parents=True,exist_ok=True)
    campaigns=["discovery","confirmation","budget_extension","horizon_transfer","causal_comb"]
    audits={};hashes={};rows=[];hypotheses=[]
    for campaign in campaigns:
        directory=S.OUT if campaign=="discovery" else S.OUT/campaign;p=directory/"audit.json"
        a=json.loads(p.read_text());assert a["status"]=="validated";audits[campaign]=a;hashes[str(p.relative_to(S.ROOT))]=S.sha(p)
        for r in a["records"]:
            j=r["job"];rows.append(dict(campaign=campaign,job=j["id"],length=j["length"],loop_percent=j["loop_percent"],
                spine_length=j.get("spine_length"),branch_depth=j.get("branch_depth"),model_seed=j["seed"],arm=j["arm"],
                cap=j.get("answer_steps",2048),n=r["n"],solved=r["solved"],wrong_stops=r["wrong_stops"],
                answer_budget_failures=r["answer_budget_failures"],hidden_failures=r["hidden_failures"],
                mean_answer_steps=r["answer_steps_mean"],mean_hidden_steps=r["hidden_steps_mean"],wall_seconds=r["wall_seconds"]))
        if campaign in ["discovery","confirmation","budget_extension"]:
            hypotheses.append({"campaign":campaign,"hypothesis":"H2c" if campaign=="budget_extension" else "H2","outcome":a["absolute_hypothesis"]})
            hypotheses.append({"campaign":campaign,"hypothesis":"H2d" if campaign=="budget_extension" else "H2b","outcome":a["paired_hypothesis"]})
        elif campaign=="horizon_transfer":
            hypotheses.extend([{"campaign":campaign,"hypothesis":"H2c independent confirmation","outcome":a["primary_absolute"]},
                               {"campaign":campaign,"hypothesis":"H2f branch-depth forecast90x90","outcome":a["primary_predictor"]}])
        else:
            hypotheses.extend({"campaign":campaign,"hypothesis":"H2e "+c["contrast"],"outcome":c} for c in a["contrasts"])
    original_jobs=len(rows)
    eta_path=S.OUT/"eta/audit.json"
    eta=json.loads(eta_path.read_text());assert eta["status"]=="validated"
    hashes[str(eta_path.relative_to(S.ROOT))]=S.sha(eta_path)
    eta_rows=[]
    for r in eta["records"]:
        j=r["job"]
        row=dict(campaign="eta",job=j["id"],length=j["length"],loop_percent=j["loop_percent"],
            spine_length=None,branch_depth=None,model_seed=j["seed"],arm=f"eta{r['eta']}",
            cap=8192,n=r["n"],solved=r["solved"],wrong_stops=r["wrong_stops"],
            answer_budget_failures=r["answer_budget_failures"],hidden_failures=r["hidden_failures"],
            mean_answer_steps=r["answer_steps_mean"],mean_hidden_steps=None,wall_seconds=None,
            eta=r["eta"],solved2048=r["solved2048"],answer_seconds=r["answer_seconds"],
            hidden_seconds_shared=r["hidden_seconds_shared"],hidden_plus_answer_seconds=r["hidden_plus_answer_seconds"],
            gpu=r["gpu"])
        rows.append(row);eta_rows.append(row)
    fields=list(dict.fromkeys(key for r in rows for key in r))
    with (out/"runs.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    with (out/"eta-runs.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(eta_rows[0]));w.writeheader();w.writerows(eta_rows)
    shutil.copyfile(S.OUT/"eta/per-maze.csv",out/"eta-per-maze.csv")
    S.save(out/"eta-summary.json",{k:v for k,v in eta.items() if k not in ["records","diagnostics","evidence_sha256"]}|{
        "diagnostics":[{k:v for k,v in d.items() if k!="recorded_updates"} for d in eta["diagnostics"]],
        "audit_sha256":S.sha(eta_path)})
    hypotheses.extend({"campaign":"eta","hypothesis":f"H2g eta0.5 speed and accuracy at {s['length']}","outcome":s} for s in eta["summaries"])
    topology=json.loads((S.OUT/"budget_extension/topology.json").read_text())
    with (out/"topology.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(topology["per_maze"][0]));w.writeheader();w.writerows(topology["per_maze"])
    for key in ["conditions","orientations"]:
        values=audits["causal_comb"][key]
        with (out/f"comb-{key}.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    protocols={}
    for name in ["protocol","confirmation-protocol","budget_extension-protocol","horizon_transfer-protocol","causal_comb-protocol","eta-protocol"]:
        path=S.HERE/f"{name}.json";d=json.loads(path.read_text());protocols[name]={"sha256":S.sha(path),"frozen_utc":d["frozen_utc"]}
    public={"updated_utc":S.utc(),"title":"Maze-Hard size transfer, causal branch pruning and eta acceleration",
            "status":"six planned studies complete; ongoing research continues separately", "host":"work",
            "jobs":original_jobs+eta["jobs"],"run_arm_rows":len(rows),"model_maze_evaluations":sum(r["n"] for r in rows),"hypotheses":hypotheses,
            "protocols":protocols,"audit_sha256":hashes,"model_sha256":json.loads((S.HERE/"protocol.json").read_text())["models"],
            "limits":["Generated DFS-tree/loopy/comb distributions differ from official Maze-Hard.",
                      "Training seeds are repeated frozen models, not independent maze samples.",
                      "Oracle S-G paths enter correctness and offline topology diagnostics only, never model inference.",
                      "Instrumented Blackwell wall clocks are not H200 speed comparisons or matched baseline training.",
                      "BP AdamW outside post-training inference protocol; negative and inconclusive outcomes retained."],
            "report":"research/20261114/work/REPORT.md","source":"research/20261114/work/"}
    S.save(out/"summary.json",public)
    for ext in ["png","pdf","svg"]:shutil.copyfile(S.OUT/f"figures/pruning-mechanism.{ext}",out/f"pruning-mechanism.{ext}")
    for ext in ["png","pdf","svg"]:shutil.copyfile(S.OUT/f"figures/eta-acceleration.{ext}",out/f"eta-acceleration.{ext}")
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})
    S.emit("public_export",jobs=public["jobs"],model_maze_evaluations=sum(r["n"] for r in rows),directory=str(out))


if __name__=="__main__":main()
