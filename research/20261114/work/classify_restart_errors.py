"""Classify every retained wrong candidate without new inference."""
import json
import numpy as np
import study as S


def main():
    dest=S.OUT/"restart";audit=json.loads((dest/"audit.json").read_text());assert audit["status"]=="validated"
    totals={"invalid_candidates":0,"invalid_endpoint_only":0,"energy_invalid_returns":0,"energy_endpoint_returns":0,"screen_no_feasible":0};evidence={}
    for r in audit["records"]:
        job=r["job"];path=dest/f"runs/{job['id']}/outputs.npz";assert S.sha(path)==r["outputs_sha256"];evidence[str(path.relative_to(S.ROOT))]=S.sha(path)
        a=np.load(path);d=np.load(dest/f"shard{job['shard']:02d}.npz");p=a["predictions"];invalid=~a["feasible"]
        ends=(p&(d["tokens"]>=2)[None]).sum((2,3))==2;only=(p.sum((2,3))==2)&ends
        totals["invalid_candidates"]+=int(invalid.sum());totals["invalid_endpoint_only"]+=int((invalid&only).sum())
        for i,j in enumerate(a["selected_energy"]):
            if j>=0 and invalid[j,i]:
                totals["energy_invalid_returns"]+=1;totals["energy_endpoint_returns"]+=int(only[j,i])
        totals["screen_no_feasible"]+=int((a["selected_screen"]<0).sum())
    result={"status":"classified","utc":S.utc(),"post_outcome_descriptive":True,"totals":totals,"audit_sha256":S.sha(dest/"audit.json"),"script_sha256":S.sha(__file__),"evidence_sha256":evidence,"new_inference":False}
    S.save(dest/"all-error-types.json",result);S.emit("restart_all_errors_classified",**totals)


if __name__=="__main__":main()
