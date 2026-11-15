"""Classify every wrong native/default-intervention output after the trial."""
import json
import numpy as np
from diagnose_restart import classify
import study as S


def main():
    dest=S.OUT/"larger_relaxation";audit=json.loads((dest/"audit.json").read_text());assert audit["status"]=="validated";rows=[]
    for record in audit["records"]:
        job=record["job"]
        with np.load(dest/f"runs/{job['id']}/outputs.npz") as archive:out={key:archive[key] for key in ["predictions","feasible","stopped","minimum_original_flip_margin","energy"]}
        data=np.load(dest/f"shard{job['shard']:02d}.npz")
        for v,arm in enumerate(["native_lambda0","native_lambda1"]):
            pred=out["predictions"][v];shortest=out["feasible"][v]&(pred.sum((1,2))==data["shortest_edges"]+1)
            for i in np.flatnonzero(~shortest):
                p=pred[i];rows.append({"input_index":int(data["indices"][i]),"seed":job["seed"],"arm":arm,"feasible":bool(out["feasible"][v,i]),"certified":bool(out["stopped"][v,i]),"minimum_original_flip_margin":float(out["minimum_original_flip_margin"][v,i]),"original_binary_energy":float(out["energy"][v,i]),"shortest_edges_evaluation_only":int(data["shortest_edges"][i]),"selected_flat_indices":np.flatnonzero(p).tolist(),**classify(data["tokens"][i],p)})
    S.save(dest/"failure-classification.json",{"status":"classified","utc":S.utc(),"all_wrong_model_input_outputs":len(rows),"rows":rows,"primary":audit["primary"],"paired_joint_events":audit["secondary"],"post_outcome_descriptive":True,"new_inference":False,"audit_sha256":S.sha(dest/"audit.json"),"script_sha256":S.sha(__file__)})
    S.emit("larger_relaxation_failures_classified",wrong_outputs=len(rows))


if __name__=="__main__":main()
