"""Descriptive exact-board and stopping-time comparison to native outputs."""
import json
import numpy as np
import study as S


def main():
    dest=S.OUT/"margin_intervention";audit=json.loads((dest/"audit.json").read_text());assert audit["status"]=="validated"
    names=["full_gaussian","upward_only","downward_only","constant_up","constant_down"]
    counts={name:{"different_boards":0,"different_steps":0} for name in names}
    for r in audit["records"]:
        path=dest/f"runs/{r['job']['id']}/outputs.npz";assert S.sha(path)==r["outputs_sha256"]
        with np.load(path) as a:p=a["predictions"];steps=a["answer_steps"]
        for v,name in enumerate(names,1):
            counts[name]["different_boards"]+=int(np.any(p[v]!=p[0],axis=(1,2)).sum())
            counts[name]["different_steps"]+=int((steps[v]!=steps[0]).sum())
    S.save(dest/"output-parity.json",{"status":"checked","post_outcome_descriptive":True,"model_inputs":12288,"comparisons":counts,"audit_sha256":S.sha(dest/"audit.json"),"script_sha256":S.sha(__file__)})
    S.emit("margin_output_parity_checked",comparisons=counts)


if __name__=="__main__":main()
