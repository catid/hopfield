"""Independently verify reciprocal coupling swaps, saved boards and energy terms."""
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
import audit as A
import study as S


def edge_mask(tok,offsets):
    out=np.zeros((len(offsets),*tok.shape),bool);h,w=tok.shape
    for k,(dr,dc) in enumerate(offsets):
        if k==0:continue
        for r in range(h):
            for c in range(w):
                rr,cc=r+dr,c+dc
                if 0<=rr<h and 0<=cc<w:out[k,r,c]=tok[r,c]>0 and tok[rr,cc]>0
    return out


def main():
    dest=S.OUT/"coupling_swap";pp=S.HERE/"coupling_swap-protocol.json";protocol=json.loads(pp.read_text());evidence={}
    def verify(path,expected=None):
        actual=S.sha(path)
        if expected is not None:assert actual==expected,path
        evidence[str(path.relative_to(S.ROOT))]=actual
    verify(pp);verify(S.HERE/"coupling_swap.py",protocol["script_sha256"]);verify(S.HERE/"audit_coupling_swap.py")
    verify(S.HERE/"audit.py");verify(S.HERE/"study.py",protocol["core_sha256"])
    for path,h in protocol["source_sha256"].items():verify(S.ROOT/path,h)
    rows=[];lookup={}
    for seed in range(3):
        model=S.model_for(seed,90,"original","cpu",torch.float64)
        verify(S.OUT/f"models/seed{seed}.pt",protocol["models"][str(seed)]["sha256"])
        records=[];states=[]
        for index in range(10):
            run=dest/f"s{seed}-arm{index}";record=json.loads((run/"result.json").read_text());verify(run/"result.json")
            assert record["status"]=="complete" and record["seed"]==seed and record["protocol_sha256"]==S.sha(pp)
            assert record["script_sha256"]==protocol["script_sha256"] and record["native_capture_parity"] and record["edited_baseline_batch_parity"]
            verify(run/"states.npz",record["states_sha256"]);records.append(record);states.append(np.load(run/"states.npz"))
        original=states[0];orig_mask=edge_mask(original["tokens"],model.aoffs)
        for edit_index in range(3):
            own,transferred,reciprocal=states[1+3*edit_index:4+3*edit_index]
            expected=original["tokens"].copy();expected[tuple(protocol["selected_cases"][edit_index]["removed"])]=0
            assert np.array_equal(own["tokens"],expected) and np.array_equal(reciprocal["tokens"],expected)
            assert np.array_equal(transferred["tokens"],original["tokens"])
            assert np.array_equal(transferred["couplings"],own["couplings"]*orig_mask)
            assert np.array_equal(reciprocal["couplings"],original["couplings"]*edge_mask(expected,model.aoffs))
        for index,(record,state) in enumerate(zip(records,states)):
            valid,distance=A.independent(state["tokens"],state["prediction"]);assert distance==204
            finite=bool(np.isfinite(state["logits"]).all());assert finite==record["finite"]
            success=valid and record["stopped"] and finite
            t=torch.tensor(state["tokens"].astype(np.int64))[None];a=torch.tensor(state["couplings"],dtype=torch.float64)[None]
            energy_rows={}
            for name,key in [("prediction","prediction"),("oracle","oracle_label")]:
                nu=torch.tensor(state[key],dtype=torch.float64)[None];nc=model.clamped(t,nu);free=model.free(t).to(torch.float64)
                st=torch.stack([1-nc,nc],1);out,_,sf=model.support(a,nc)
                terms={"entropy":free*(nc*(nc+1e-12).log()+(1-nc)*(1-nc+1e-12).log()),"latent_gated_links":-nc*out-sf*nc*nc,
                       "maze_unique_pairs":.5*(F.conv2d(st,model.mu_kernel(),padding=1)*st).sum(1),"maze_unique_bias":-free*(st*model.b[None,:,None,None]).sum(1)}
                total=sum(float(v.sum()) for v in terms.values());assert abs(total-float(model.energy(t,nu,a)[0]))<1e-8
                assert abs(total-record[f"{name}_energy_float32"])<.1
                energy_rows[name]={"total":total,**{k:float(v.sum()) for k,v in terms.items()}}
            nu=torch.tensor(state["prediction"],dtype=torch.float64)[None];flip=float(model.flip_delta(t,nu,a).min())
            assert abs(flip-record["minimum_flip_delta"])<1e-3
            if record["stopped"]:assert record["minimum_flip_delta"]>=-1e-6
            row={"seed":seed,"case":record["case"],"arm":record["arm"],"edit_index":record["edit_index"],"success":success,
                 "valid_shortest_path":valid,"stopped":record["stopped"],"finite":finite,"answer_steps":record["answer_steps"],
                 "selected_cells":int(state["prediction"].sum()),"minimum_flip_delta_float64":flip,"coupling_L1_incident":record["coupling_L1_incident"],
                 "coupling_L1_nonincident":record["coupling_L1_nonincident"],"changed_coupling_count":record["changed_coupling_count"],
                 "energy_terms":energy_rows,"states_sha256":record["states_sha256"]}
            rows.append(row);lookup[(seed,record["case"],record["arm"])]=row
    summaries=[]
    for case in ["original"]+[c["kind"] for c in protocol["selected_cases"]]:
        arms=[r["arm"] for r in rows if r["seed"]==0 and r["case"]==case]
        for arm in arms:
            selected=[lookup[(s,case,arm)] for s in range(3)]
            summaries.append({"case":case,"arm":arm,"success_by_seed":[r["success"] for r in selected],"steps_by_seed":[r["answer_steps"] for r in selected],
                              "joint_success":all(r["success"] for r in selected),"minimum_flip_margin_by_seed":[r["minimum_flip_delta_float64"] for r in selected]})
    with (dest/"arms.csv").open("w",newline="") as f:
        flat=[{k:v for k,v in r.items() if k!="energy_terms"}|{f"{state}_{term}":value for state,terms in r["energy_terms"].items() for term,value in terms.items()} for r in rows]
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    verify(dest/"arms.csv")
    result={"status":"validated","utc":S.utc(),"post_outcome_descriptive":True,"jobs":3,"saved_board_checks":30,"baseline_arms":12,"counterfactual_swap_arms":18,
            "protocol_sha256":S.sha(pp),"summaries":summaries,"rows":rows,"evidence_sha256":evidence,
            "checks":"Exact reciprocal coefficient mapping and zero-masking; independent Dijkstra/degree/connectivity; native/captured-logit parity; float64energy terms recomputed from retainedfloat32coefficients.",
            "scope":"One selected failure graph and3preselected edits,3fixedmodels,eta0.5. No population confidence interval or modelparameter tuning; original primary failures remain unchanged."}
    S.save(dest/"audit.json",result);S.emit("coupling_swap_audited",summaries=summaries)


if __name__=="__main__":main()
