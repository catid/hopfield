"""Independent topology and paired-cluster outcome audit for long-cycle tie controls."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import audit as A
import study as S
from topology import distances


def verify_graphs(protocol):
    datasets={};signatures={};counts=[]
    for d in protocol["datasets"]:
        assert S.sha(S.ROOT/d["path"])==d["sha256"];data=np.load(S.ROOT/d["path"]);datasets[d["cohort"]]=data
        signature=[]
        assert len({t.tobytes() for t in data["tokens"]})==128
        for tok,label,expected in zip(data["tokens"],data["labels"],data["shortest_edges"]):
            valid,shortest=A.independent(tok,label);assert valid and shortest==expected
            start=tuple(np.argwhere(tok==2)[0]);goal=tuple(np.argwhere(tok==3)[0]);ds=distances(tok,[start]);dg=distances(tok,[goal])
            op=tok>0;nodes=int(op.sum());edges=int((op[1:]&op[:-1]).sum()+(op[:,1:]&op[:,:-1]).sum());assert len(ds)==nodes and edges-nodes+1==1
            path_counts={start:1}
            for node in sorted(ds,key=ds.get):
                if ds[node]+dg[node]!=shortest:continue
                if node!=start:path_counts[node]=sum(path_counts.get((node[0]+dr,node[1]+dc),0) for dr,dc in S.SIDES if ds.get((node[0]+dr,node[1]+dc),-1)==ds[node]-1)
            assert path_counts[goal]==d["expected_shortest_path_count"]
            degree=np.zeros_like(tok,int);degree[1:]+=op[:-1];degree[:-1]+=op[1:];degree[:,1:]+=op[:,:-1];degree[:,:-1]+=op[:,1:]
            signature.append((nodes,edges,tuple(np.bincount(degree[op],minlength=5))))
        signatures[(d["route_length"],d["condition"])]=signature
        counts.append({"cohort":d["cohort"],"n":len(signature),"verified_shortest_paths":d["expected_shortest_path_count"],"distance_min":int(data["shortest_edges"].min()),"distance_max":int(data["shortest_edges"].max())})
    for k in [28,42,64]:
        assert signatures[(k,"tied")]==signatures[(k,"plus")]==signatures[(k,"minus")]
        tied=datasets[f"r{k}-tied"]["shortest_edges"]
        assert np.all(tied-datasets[f"r{k}-plus"]["shortest_edges"]==4)
        assert np.all(tied-datasets[f"r{k}-minus"]["shortest_edges"]==4)
    return datasets,counts


def main(graphs_only=False):
    dest=S.OUT/"long_ties";pp=S.HERE/"long_ties-protocol.json";protocol=json.loads(pp.read_text());datasets,counts=verify_graphs(protocol)
    if graphs_only:
        S.save(dest/"graph-preflight.json",{"status":"passed","utc":S.utc(),"graphs_checked":1152,"counts":counts,"script_sha256":S.sha(__file__),"protocol_sha256":S.sha(pp)})
        S.emit("tie_graphs_verified",graphs=1152,counts=counts);return
    evidence={}
    def verify(path,expected=None):
        path=Path(path);path=path if path.is_absolute() else S.ROOT/path;actual=S.sha(path)
        if expected is not None:assert actual==expected,path
        evidence[str(path.relative_to(S.ROOT))]=actual
    verify(pp);verify(__file__);verify(A.__file__);verify(S.HERE/"topology.py");verify(S.HERE/"long_ties.py",protocol["execution_script_sha256"]);verify(S.__file__,protocol["core_script_sha256"])
    for path,h in protocol["source_sha256"].items():verify(path,h)
    for model in protocol["models"].values():verify(model["path"],model["sha256"])
    for d in protocol["datasets"]:verify(d["path"],d["sha256"])
    records=[];saved={}
    for job in protocol["jobs"]:
        run=dest/"runs"/job["id"];record=json.loads((run/"result.json").read_text());verify(run/"result.json")
        assert record["status"]=="complete" and record["job"]==job and record["protocol_sha256"]==S.sha(pp)
        assert record["script_sha256"]==protocol["execution_script_sha256"] and record["core_script_sha256"]==protocol["core_script_sha256"]
        assert record["checkpoint_sha256"]==protocol["models"][str(job["seed"])]["sha256"]
        verify(run/"predictions.npz",record["predictions_sha256"]);data=datasets[job["cohort"]];out=np.load(run/"predictions.npz");valid=[]
        for tok,pred,expected in zip(data["tokens"],out["predictions"],data["shortest_edges"]):
            ok,distance=A.independent(tok,pred);assert distance==expected;valid.append(ok)
        valid=np.asarray(valid);success=valid&out["stopped"]&out["hidden_good"]
        assert np.array_equal(success,out["success"]) and np.array_equal(valid,out["shortest_path"])
        assert int(success.sum())==record["solved"] and int((out["stopped"]&~valid).sum())==record["wrong_stops"]
        assert int((~out["stopped"]).sum())==record["answer_budget_failures"] and int((~out["hidden_good"]).sum())==record["hidden_failures"]
        assert np.all(out["minimum_flip_delta"][out["stopped"]]>=-1e-6)
        saved[(job["route_length"],job["condition"],job["seed"])]=dict(success=success,steps=out["answer_steps"],energy=out["prediction_energy"],oracle=out["oracle_energy"],margin=out["minimum_flip_delta"])
        records.append({k:record[k] for k in ["job","n","solved","wrong_stops","answer_budget_failures","hidden_failures","answer_steps_mean","hidden_steps_mean","wall_seconds"]})
    joint={};cells=[];layout_rows=[]
    for k in [28,42,64]:
        for condition in ["tied","plus","minus"]:
            successes=np.stack([saved[(k,condition,s)]["success"] for s in range(3)]);joint[(k,condition)]=successes.all(0)
            means=np.stack([saved[(k,condition,s)]["steps"] for s in range(3)]).mean(0)
            cells.append({"route_length":k,"condition":condition,"n_layouts":128,"solved_per_seed":successes.sum(1).tolist(),"joint_solved":int(successes.all(0).sum()),"mean_steps":float(means.mean())})
            for i in range(128):layout_rows.append({"route_length":k,"condition":condition,"layout_index":i,"joint_success":int(successes[:,i].all()),"mean_model_steps":float(means[i])})
    delta=np.stack([.5*(joint[(k,"plus")].astype(float)+joint[(k,"minus")].astype(float))-joint[(k,"tied")].astype(float) for k in [42,64]]).mean(0)
    rng=np.random.default_rng(2026111425);boot=delta[rng.integers(128,size=(10000,128))].mean(1);ci=(100*np.quantile(boot,[.025,.975])).tolist()
    primary={"n_layout_clusters":128,"equal_weight_long_routes":[42,64],"unequal_minus_tied_joint_success_pp":float(100*delta.mean()),
             "paired_cluster_bootstrap95_pp":ci,"decision":"supported" if ci[0]>=10 else "falsified" if ci[1]<10 else "inconclusive"}
    discordant=np.stack([(joint[(k,"plus")]!=joint[(k,"tied")])|(joint[(k,"minus")]!=joint[(k,"tied")]) for k in [42,64]]).any(0)
    primary["discordant_clusters"]=int(discordant.sum())
    reference=.5*(joint[(28,"plus")].astype(float)+joint[(28,"minus")].astype(float))-joint[(28,"tied")].astype(float)
    primary["route28_control_difference_pp"]=float(100*reference.mean())
    primary["long_minus_route28_difference_pp"]=float(100*(delta-reference).mean())
    if not discordant.any():primary["zero_discordance_one_sided95_cluster_upper_pp"]=100*(1-.05**(1/128))
    with (dest/"per-layout.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(layout_rows[0]));writer.writeheader();writer.writerows(layout_rows)
    verify(dest/"per-layout.csv")
    result={"status":"validated","utc":S.utc(),"campaign":"long_ties","jobs":len(records),"saved_board_checks":sum(r["n"] for r in records),
            "protocol_sha256":S.sha(pp),"primary":primary,"cells":cells,"records":records,"graph_checks":counts,"evidence_sha256":evidence,
            "limits":["These newly constructed controls were fixed before outcomes and exclude selected-maze38 interventions.","Equal node/edge/degree/cycle-size structure is verified; unequal controls have4shorter geodesic length, so a detected difference is not exclusively attributed to ties.","Three fixed models and three route lengths are preserved within128layout/geometry bootstrap clusters.","Negative outcome would show degeneracy alone insufficient in this family, not refute the retained random-maze exception."],"new_inference":False}
    S.save(dest/"audit.json",result);S.emit("tie_audited",primary=primary,cells=cells)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--graphs-only",action="store_true");main(parser.parse_args().graphs_only)
