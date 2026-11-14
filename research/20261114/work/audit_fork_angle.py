"""Independent matched-graph, local-motif and saved-board fork-angle audit."""
import argparse
import csv
import json
import numpy as np
import audit as A
import study as S
from topology import distances


def graphs(protocol):
    datasets={};signatures={};annotations={}
    for dataset in protocol["datasets"]:
        assert S.sha(S.ROOT/dataset["path"])==dataset["sha256"];data=np.load(S.ROOT/dataset["path"]);datasets[dataset["cohort"]]=data;sign=[];meta=[]
        assert len({t.tobytes() for t in data["tokens"]})==128
        for tok,label,expected,recorded_forks in zip(data["tokens"],data["labels"],data["shortest_edges"],data["fork_coords"]):
            valid,distance=A.independent(tok,label);assert valid and distance==expected==dataset["route_length"]+32
            start=tuple(np.argwhere(tok==2)[0]);goal=tuple(np.argwhere(tok==3)[0]);ds=distances(tok,[start]);dg=distances(tok,[goal]);op=tok>0
            nodes=int(op.sum());edges=int((op[1:]&op[:-1]).sum()+(op[:,1:]&op[:,:-1]).sum());assert len(ds)==nodes and edges==nodes
            paths={start:1}
            for node in sorted(ds,key=ds.get):
                if ds[node]+dg[node]!=distance:continue
                if node!=start:paths[node]=sum(paths.get((node[0]+dr,node[1]+dc),0) for dr,dc in S.SIDES if ds.get((node[0]+dr,node[1]+dc),-1)==ds[node]-1)
            assert paths[goal]==2
            degree=np.zeros_like(tok,int);degree[1:]+=op[:-1];degree[:-1]+=op[1:];degree[:,1:]+=op[:,:-1];degree[:,:-1]+=op[:,1:]
            forks=np.argwhere(op&(degree==3));assert np.array_equal(forks,recorded_forks) and len(forks)==2
            local=[]
            for row,col in forks:
                node=(int(row),int(col));assert tok[node]==1
                adjacent=[(dr,dc) for dr,dc in S.SIDES if (node[0]+dr,node[1]+dc) in ds]
                forward=[v for v in adjacent if ds.get((node[0]+v[0],node[1]+v[1]))==ds[node]+1]
                backward=[v for v in adjacent if ds.get((node[0]+v[0],node[1]+v[1]))==ds[node]-1]
                alternatives=forward if len(forward)==2 else backward;assert len(alternatives)==2
                dot=sum(x*y for x,y in zip(*alternatives));assert dot==(0 if dataset["condition"]=="corner" else -1)
                difference=tuple(alternatives[1][i]-alternatives[0][i] for i in range(2));diagonal=tuple(abs(x) for x in difference)==(1,1)
                assert diagonal==(dataset["condition"]=="corner")
                local.append({"coordinates":[int(row),int(col)],"role":"upstream" if len(forward)==2 else "downstream", "alternative_offsets":[list(v) for v in alternatives],"alternative_dot_product":dot,"diagonal_pair_interaction":diagonal})
            sign.append((nodes,edges,tuple(np.bincount(degree[op],minlength=5)),distance));meta.append(local)
        signatures[(dataset["route_length"],dataset["condition"])]=sign;annotations[dataset["cohort"]]=meta
    for route in [28,42,64]:assert signatures[(route,"corner")]==signatures[(route,"opposite")]
    return datasets,annotations


def main(graphs_only=False):
    dest=S.OUT/"fork_angle";pp=S.HERE/"fork_angle-protocol.json";protocol=json.loads(pp.read_text());datasets,annotations=graphs(protocol)
    if graphs_only:
        S.save(dest/"graph-preflight.json",{"status":"passed","utc":S.utc(),"graphs_checked":768,"protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"local_forks_checked":1536,
             "checks":"Allgraphs connected, onecycle, exactly2shortestpaths; within-pair node/edge/degree counts and route+32distance match; twofree degree3forks have dot0+diagonal motif in corner, dot-1+no diagonal motif in opposite."})
        S.emit("fork_angle_graphs_verified",graphs=768,free_forks=1536);return
    evidence={}
    def verify(path,expected=None):
        actual=S.sha(path)
        if expected is not None:assert actual==expected,path
        evidence[str(path.relative_to(S.ROOT))]=actual
    verify(pp);verify(S.HERE/"fork_angle.py",protocol["execution_script_sha256"]);verify(S.HERE/"audit_fork_angle.py");verify(S.HERE/"audit.py");verify(S.HERE/"topology.py")
    for path,h in protocol["source_sha256"].items():verify(S.ROOT/path,h)
    model_terms={}
    for seed in range(3):
        model=S.model_for(seed,160,"original");path=S.OUT/f"models/seed{seed}.pt";verify(path,protocol["models"][str(seed)]["sha256"])
        model_terms[str(seed)]={"checkpoint_sha256":S.sha(path),"pair_offsets":[list(x) for x in model.mu_offsets],"pair_tables":model.T.tolist(),
            "pair_interaction_J":(model.T[:,1,1]-model.T[:,1,0]-model.T[:,0,1]+model.T[:,0,0]).tolist(),"attention_offsets":[list(x) for x in model.aoffs],
            "kappa":model.kappa.tolist(),"bias":model.b.tolist(),"beta":model.beta.tolist(),"coupling_type":model.coupling}
    records=[];saved={};readouts=[]
    for job in protocol["jobs"]:
        run=dest/"runs"/job["id"];record=json.loads((run/"result.json").read_text());verify(run/"result.json");verify(run/"predictions.npz",record["predictions_sha256"])
        assert record["status"]=="complete" and record["job"]==job and record["protocol_sha256"]==S.sha(pp)
        assert record["script_sha256"]==protocol["execution_script_sha256"] and record["checkpoint_sha256"]==protocol["models"][str(job["seed"])]["sha256"]
        data=datasets[job["cohort"]];out=np.load(run/"predictions.npz");valid=[]
        for tok,pred,expected in zip(data["tokens"],out["predictions"],data["shortest_edges"]):
            ok,distance=A.independent(tok,pred);assert distance==expected;valid.append(ok)
        valid=np.asarray(valid);success=valid&out["stopped"]&out["hidden_good"]
        assert np.array_equal(success,out["success"]) and np.array_equal(valid,out["shortest_path"])
        assert int(success.sum())==record["solved"] and int((out["stopped"]&~valid).sum())==record["wrong_stops"]
        assert int((~out["stopped"]).sum())==record["answer_budget_failures"] and int((~out["hidden_good"]).sum())==record["hidden_failures"]
        assert np.allclose(out["fork_couplings"].sum(-1),1,atol=1e-6) and np.isfinite(out["fork_couplings"]).all()
        saved[(job["route_length"],job["condition"],job["seed"])]=dict(success=success,steps=out["answer_steps"])
        terms=model_terms[str(job["seed"])];offsets=[tuple(x) for x in terms["attention_offsets"]];kappa=np.asarray(terms["kappa"])
        for i,local in enumerate(annotations[job["cohort"]]):
            for fi,m in enumerate(local):
                indices=[offsets.index(tuple(v)) for v in m["alternative_offsets"]];a=out["fork_couplings"][i,fi];weighted=a*kappa
                readouts.append({"route_length":job["route_length"],"condition":job["condition"],"model_seed":job["seed"],"layout_index":i,"fork_role":m["role"],
                    "attention_alternative1":float(a[indices[0]]),"attention_alternative2":float(a[indices[1]]),"weighted_alternative1":float(weighted[indices[0]]),
                    "weighted_alternative2":float(weighted[indices[1]]),"self_attention":float(a[0]),"oracle_field":float(out["fork_oracle_field"][i,fi]),
                    "offset_order_attention":json.dumps(a.tolist()),"alternative_offsets":json.dumps(m["alternative_offsets"]),"diagonal_pair_interaction":m["diagonal_pair_interaction"]})
        records.append({k:record[k] for k in ["job","n","solved","wrong_stops","answer_budget_failures","hidden_failures","answer_steps_mean","hidden_steps_mean","wall_seconds"]})
    joint={};cells=[];layout_rows=[]
    for route in [28,42,64]:
        for condition in ["opposite","corner"]:
            success=np.stack([saved[(route,condition,s)]["success"] for s in range(3)]);steps=np.stack([saved[(route,condition,s)]["steps"] for s in range(3)]).mean(0);joint[(route,condition)]=success.all(0)
            cells.append({"route_length":route,"condition":condition,"n_layouts":128,"solved_per_seed":success.sum(1).tolist(),"joint_solved":int(success.all(0).sum()),"mean_steps":float(steps.mean())})
            for i in range(128):layout_rows.append({"route_length":route,"condition":condition,"layout_index":i,"joint_success":int(success[:,i].all()),"mean_model_steps":float(steps[i])})
    delta=np.stack([joint[(r,"opposite")].astype(float)-joint[(r,"corner")].astype(float) for r in [28,42,64]]).mean(0)
    rng=np.random.default_rng(2026111426);boot=delta[rng.integers(128,size=(10000,128))].mean(1);ci=(100*np.quantile(boot,[.025,.975])).tolist()
    discordant=np.stack([joint[(r,"opposite")]!=joint[(r,"corner")] for r in [28,42,64]]).any(0)
    primary={"n_layout_clusters":128,"opposite_minus_corner_joint_success_pp":float(100*delta.mean()),"paired_cluster_bootstrap95_pp":ci,
        "decision":"supported" if ci[0]>=10 else "falsified" if ci[1]<10 else "inconclusive","discordant_clusters":int(discordant.sum()),
        "per_route_difference_pp":{str(r):float(100*(joint[(r,"opposite")].mean()-joint[(r,"corner")].mean())) for r in [28,42,64]}}
    if not discordant.any():primary["zero_discordance_one_sided95_cluster_upper_pp"]=100*(1-.05**(1/128))
    for filename,values in [("per-layout.csv",layout_rows),("fork-readouts.csv",readouts)]:
        with (dest/filename).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        verify(dest/filename)
    result={"status":"validated","utc":S.utc(),"jobs":len(records),"saved_board_checks":sum(r["n"] for r in records),"protocol_sha256":S.sha(pp),
        "primary":primary,"cells":cells,"model_terms":model_terms,"records":records,"evidence_sha256":evidence,"new_inference":False,
        "graph_checks":"768graphs and1536freeforks verified independently: matched connectivity/node-edge-degree counts,cycleperimeter,S-Gdistance and2shortestpaths; diagonal-interaction motif differs asdeclared.",
        "scope":"Independent matched singlecycle/twotail family, fixed3models andeta0.5.128pairedclusters retain allrouteconditions/modelrepeats. No selected failures or outcome-dependent layout changes."}
    S.save(dest/"audit.json",result);S.emit("fork_angle_audited",primary=primary,cells=cells)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--graphs-only",action="store_true");main(parser.parse_args().graphs_only)
