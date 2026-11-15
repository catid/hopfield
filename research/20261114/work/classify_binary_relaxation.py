"""Selected descriptive recovery and branched-stop cases from frozen trial."""
import csv
import json
import numpy as np
from diagnose_restart import classify
import study as S


def main():
    dest=S.OUT/"binary_relaxation";audit=json.loads((dest/"audit.json").read_text());assert audit["status"]=="validated"
    with (dest/"per-input.csv").open() as f:inputs=list(csv.DictReader(f))
    recovered=[int(r["input_index"]) for r in inputs if r["joint_native_success"]=="0" and r["joint_gaussian_lambda1_success"]=="1"]
    nonendpoint=[]
    for record in audit["records"]:
        job=record["job"];a=np.load(dest/f"runs/{job['id']}/outputs.npz");d=np.load(dest/f"shard{job['shard']:02d}.npz");pred=a["predictions"][2]
        for i in np.flatnonzero((~a["feasible"][2])&(pred.sum((1,2))>2)):
            nonendpoint.append({"input_index":int(d["indices"][i]),"seed":job["seed"],"minimum_original_flip_margin":float(a["minimum_original_flip_margin"][2,i]),**classify(d["tokens"][i],pred[i])})
    selected=sorted(set(recovered)|{r["input_index"] for r in nonendpoint});graphs=[]
    arms=["native","gaussian_lambda0","gaussian_lambda1","gaussian_lambda_minus1"]
    for index in selected:
        shard,i=divmod(index,256);d=np.load(dest/f"shard{shard:02d}.npz");tok=d["tokens"][i];rows=[]
        for seed in range(3):
            out=np.load(dest/f"runs/shard{shard:02d}-s{seed}/outputs.npz")
            for v,arm in enumerate(arms):
                pred=out["predictions"][v,i];valid=bool(out["feasible"][v,i])
                rows.append({"seed":seed,"arm":arm,"feasible":valid,"shortest":valid and int(pred.sum())==int(d["shortest_edges"][i])+1,"original_binary_energy":float(out["energy"][v,i]),"minimum_original_flip_margin":float(out["minimum_original_flip_margin"][v,i]),"selected_flat_indices":np.flatnonzero(pred).tolist(),**classify(tok,pred)})
        graphs.append({"input_index":index,"shape":[90,90],"tokens":tok.tolist(),"shortest_edges_evaluation_only":int(d["shortest_edges"][i]),"rows":rows})
    result={"status":"classified","utc":S.utc(),"post_outcome_descriptive":True,"recovered_native_joint_failures":recovered,"nonendpoint_wrong_lambda1":nonendpoint,"graphs":graphs,"audit_sha256":S.sha(dest/"audit.json"),"script_sha256":S.sha(__file__),"new_inference":False,"scope":"Cases selected after primary outcomes for explanation; no new hypothesis or population recovery claim."}
    S.save(dest/"selected-cases.json",result);S.emit("binary_relaxation_cases_classified",recovered=recovered,nonendpoint=nonendpoint)


if __name__=="__main__":main()
