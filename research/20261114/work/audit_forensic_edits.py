"""Descriptive independent audit of edits to one selected failure graph."""
import json
import numpy as np
import audit as A
import study as S
from topology import distances


def main():
    dest=S.OUT/"forensic_edits";pp=S.HERE/"forensic_edits-protocol.json";protocol=json.loads(pp.read_text())
    assert S.sha(S.HERE/"forensic_edits.py")==protocol["execution_script_sha256"]
    dataset=protocol["datasets"][0];assert S.sha(S.ROOT/dataset["path"])==dataset["sha256"];data=np.load(S.ROOT/dataset["path"])
    original=np.load(S.OUT/"eta_boundary/failure-forensics/shortest-path-union.npz");tok0=original["tokens"];union=original["shortest_union"]
    signatures=[];cases=[]
    for i,(case,tok,label) in enumerate(zip(protocol["cases"],data["tokens"],data["labels"])):
        expected=tok0.copy();expected[tuple(case["removed"])]=0;assert np.array_equal(tok,expected)
        ok,distance=A.independent(tok,label);assert ok and distance==204
        start=tuple(np.argwhere(tok==2)[0]);goal=tuple(np.argwhere(tok==3)[0]);ds=distances(tok,[start]);dg=distances(tok,[goal]);op=tok>0
        assert len(ds)==int(op.sum())==int((tok0>0).sum())-1
        counts={start:1}
        for node in sorted(ds,key=ds.get):
            if ds[node]+dg[node]!=204:continue
            if node!=start:counts[node]=sum(counts.get((node[0]+dr,node[1]+dc),0) for dr,dc in S.SIDES if ds.get((node[0]+dr,node[1]+dc),-1)==ds[node]-1)
        if case["kind"].startswith("targeted"):assert counts[goal]<352212 and union[tuple(case["removed"])]
        else:assert counts[goal]==352212 and not union[tuple(case["removed"])]
        degrees=np.zeros_like(tok,int);degrees[1:]+=op[:-1];degrees[:-1]+=op[1:];degrees[:,1:]+=op[:,:-1];degrees[:,:-1]+=op[:,1:]
        signatures.append((int(op.sum()),int((op[1:]&op[:-1]).sum()+(op[:,1:]&op[:,:-1]).sum()),tuple(np.bincount(degrees[op],minlength=5))))
        cases.append(dict(case,index=i,shortest_path_count=counts[goal],shortest_edges=distance))
    assert all(x==signatures[0] for x in signatures)
    rows=[];summaries=[];hashes={str(pp.relative_to(S.ROOT)):S.sha(pp),dataset["path"]:dataset["sha256"]}
    for job in protocol["jobs"]:
        run=dest/"runs"/job["id"];record=json.loads((run/"result.json").read_text());assert record["status"]=="complete" and record["protocol_sha256"]==S.sha(pp)
        assert record["checkpoint_sha256"]==S.sha(S.OUT/f"models/seed{job['seed']}.pt")
        path=run/"predictions.npz";assert S.sha(path)==record["predictions_sha256"];out=np.load(path)
        hashes[str(path.relative_to(S.ROOT))]=S.sha(path);hashes[str((run/"result.json").relative_to(S.ROOT))]=S.sha(run/"result.json")
        for i,(tok,pred) in enumerate(zip(data["tokens"],out["predictions"])):
            valid,distance=A.independent(tok,pred);success=valid and bool(out["stopped"][i]) and bool(out["hidden_good"][i]);assert success==bool(out["success"][i])
            rows.append({"seed":job["seed"],"case_index":i,"kind":cases[i]["kind"],"success":success,"stopped":bool(out["stopped"][i]),
                         "answer_steps":int(out["answer_steps"][i]),"hidden_good":bool(out["hidden_good"][i]),"selected_cells":int(pred.sum()),
                         "minimum_flip_delta":float(out["minimum_flip_delta"][i]),"prediction_energy":float(out["prediction_energy"][i]),
                         "oracle_energy":float(out["oracle_energy"][i])})
        summaries.append({"seed":job["seed"],"target_A_success":bool(out["success"][0]),"target_B_success":bool(out["success"][1]),
                          "matched_controls_solved":int(out["success"][2:].sum()),"n_matched_controls":32,"wrong_stops":record["wrong_stops"],"budget_failures":record["answer_budget_failures"]})
    result={"status":"validated","utc":S.utc(),"post_outcome_descriptive":True,"jobs":3,"saved_board_checks":102,"cases":cases,"summaries":summaries,"rows":rows,
            "protocol_sha256":S.sha(pp),"source_sha256":S.sha(__file__),"evidence_sha256":hashes,
            "matching":"All34edits have identical node/edge counts and degree histogram; connectedness and shortest distance204 preserved. Controls retain352212shortest paths; target deletions reduce multiplicity.",
            "scope":"One adaptively selected failure graph. No population confidence interval or claim of confirmatory recovery; unchanged original failures remain in the primary trial."}
    S.save(dest/"audit.json",result);S.emit("forensic_edits_audited",summaries=summaries,target_cases=cases[:2])


if __name__=="__main__":main()
