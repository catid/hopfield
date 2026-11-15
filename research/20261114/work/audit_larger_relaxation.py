"""Independent sign transforms, logits, graph outcomes and paired statistics."""
import csv
from concurrent.futures import ProcessPoolExecutor
from diagnose_restart import classify
import hashlib
import inspect
import json
import time
import numpy as np
import audit as A
from audit_restart import check_path
import study as S

ARMS=("native_lambda0","native_lambda1")


def check_dataset(path):
    with np.load(path) as source:
        tokens=source["tokens"];labels=source["labels"];distances=source["shortest_edges"]
    checked=[]
    for tok,label,expected in zip(tokens,labels,distances):
        ok,actual=A.independent(tok,label);assert ok and actual==expected;checked.append(actual)
    return checked


def main():
    started=time.perf_counter();dest=S.OUT/"larger_relaxation";pp=S.HERE/"larger_relaxation-protocol.json";protocol=json.loads(pp.read_text());evidence={}
    def verify(path,expected=None):
        actual=S.sha(path)
        if expected is not None:assert actual==expected,path
        evidence[str(path.relative_to(S.ROOT))]=actual
    verify(pp);verify(S.HERE/"larger_relaxation.py",protocol["execution_script_sha256"]);verify(S.HERE/"study.py",protocol["core_script_sha256"])
    verify(S.HERE/"binary_preserving.py",protocol["continuous_intervention"]["helper_sha256"])
    verify(S.ROOT/protocol["continuous_intervention"]["gate_path"],protocol["continuous_intervention"]["gate_sha256"])
    for path,h in protocol["source_sha256"].items():verify(S.ROOT/path,h)
    for model in protocol["models"].values():verify(S.ROOT/model["path"],model["sha256"])
    verify(dest/"preflight.json");pre=json.loads((dest/"preflight.json").read_text());assert pre["status"]=="passed" and pre["protocol_sha256"]==S.sha(pp)
    pilot=protocol["batch_preflight"];verify(S.ROOT/pilot["path"],pilot["sha256"])
    verify(dest/"datasets.json");manifest=json.loads((dest/"datasets.json").read_text());assert manifest["protocol_sha256"]==S.sha(pp)
    data={};inits={};init_hashes={};unique=set();oracle_checks=0
    input_receipt=dest/"independent-input-audit.json";verify(input_receipt);checked=json.loads(input_receipt.read_text())
    assert checked["status"]=="passed" and checked["protocol_sha256"]==S.sha(pp) and checked["datasets"]==manifest["datasets"]
    assert checked["graph_checker_sha256"]==S.sha(A.__file__) and checked["dataset_checker_source_sha256"]==hashlib.sha256(inspect.getsource(check_dataset).encode()).hexdigest()
    independent_distances=checked["distances"];oracle_seconds=checked["oracle_seconds"]
    for ds in manifest["datasets"]:
        verify(S.ROOT/ds["path"],ds["sha256"]);raw=np.load(S.ROOT/ds["path"]);d={k:raw[k] for k in raw.files};data[ds["shard"]]=d
        full=d["initial_native"];assert full.shape==(256,120,120) and full.dtype==np.float32 and np.all(full==3.)
        assert np.array_equal(d["indices"],np.arange(ds["shard"]*256,(ds["shard"]+1)*256))
        assert d["tokens"].shape==d["labels"].shape==(256,120,120)
        arms=[full,full];inits[ds["shard"]]=arms
        digest=hashlib.sha256()
        for lo in range(0,256,64):
            for a in arms:digest.update(a[lo:lo+64].tobytes())
        init_hashes[ds["shard"]]=digest.hexdigest()
        assert np.array_equal(independent_distances[ds["shard"]],d["shortest_edges"])
        oracle_checks+=len(d["tokens"])
        for tok in d["tokens"]:unique.add(tok.tobytes())
    assert len(unique)==16384
    cache={};graph_checks=0;cache_hits=0;saved={};records=[];rows=[];cases=[]
    for job in protocol["jobs"]:
        run=dest/f"runs/{job['id']}";verify(run/"result.json");r=json.loads((run/"result.json").read_text());assert r["status"]=="complete" and r["job"]==job and r["protocol_sha256"]==S.sha(pp)
        assert r["script_sha256"]==protocol["execution_script_sha256"] and r["checkpoint_sha256"]==protocol["models"][str(job["seed"])]["sha256"]
        assert r["dataset_sha256"]==manifest["datasets"][job["shard"]]["sha256"] and r["initialization_sha256"]==init_hashes[job["shard"]]
        verify(run/"outputs.npz",r["outputs_sha256"])
        with np.load(run/"outputs.npz") as archive:out={key:archive[key] for key in archive.files}
        d=data[job["shard"]];pred=out["predictions"];logits=out["final_logits"]
        assert pred.shape==logits.shape==(2,256,120,120) and logits.dtype==np.float32
        assert np.array_equal(pred,logits>0);assert np.array_equal(out["finite"],np.isfinite(logits).all((2,3)))
        assert np.array_equal(out["stopped"],out["finite"]&(out["minimum_original_flip_margin"]>=-1e-6))
        assert np.all(out["binary_regularizer"]==0)
        assert np.allclose(out["modified_continuous_energy"],out["original_continuous_energy"]+out["continuous_regularizer"],rtol=0,atol=0,equal_nan=True)
        if r["diagnostics_sha256"]:
            verify(run/"diagnostics.json",r["diagnostics_sha256"]);trace=json.loads((run/"diagnostics.json").read_text())
            for v,arm in enumerate(ARMS):
                last=trace[arm][-1]
                for key,stored in [("original_continuous_energy","original_continuous_energy"),("modified_continuous_energy","modified_continuous_energy"),("minimum_original_binary_flip_margin","minimum_original_flip_margin"),("modified_field_residual","modified_field_residual")]:
                    assert np.allclose(np.array(last[key],dtype=float),out[stored][v,:8],rtol=0,atol=0,equal_nan=True)
                assert last["stopped"]==out["stopped"][v,:8].tolist() and last["finite"]==out["finite"][v,:8].tolist()

        valid=np.zeros((2,256),bool);shortest=np.zeros_like(valid)
        for v,arm in enumerate(ARMS):
            initial=inits[job["shard"]][v];free=d["tokens"]==1
            expected={"changed_cells":(initial!=3).sum((1,2)),"changed_free_cells":((initial!=3)&free).sum((1,2)),"negative_cells":(initial<0).sum((1,2)),"negative_free_cells":((initial<0)&free).sum((1,2)),"zero_cells":(initial==0).sum((1,2))}
            for key,value in expected.items():assert np.array_equal(value,out[key][v])
            for i,(tok,p) in enumerate(zip(d["tokens"],pred[v])):
                key=(int(d["indices"][i]),np.packbits(p).tobytes())
                if key in cache:cache_hits+=1
                else:cache[key]=check_path(tok,p);graph_checks+=1
                valid[v,i]=cache[key];shortest[v,i]=valid[v,i] and int(p.sum())==int(d["shortest_edges"][i])+1
        assert np.array_equal(valid,out["feasible"])
        finite=out["finite"];stopped=out["stopped"];steps=out["answer_steps"]
        assert np.all((steps>=16)&(steps<=8192)&(steps%16==0)) and np.all(steps[~stopped]==8192) and not np.any(stopped&~finite)
        success=shortest&finite&stopped&out["hidden_good"][None]&np.isfinite(out["energy"]);saved[(job["shard"],job["seed"])]=success
        for v,arm in enumerate(ARMS):
            invalid=~shortest[v];endpoint=(pred[v].sum((1,2))==2)&((pred[v]&(d["tokens"]>=2)).sum((1,2))==2)
            branched=np.zeros(256,bool)
            for i in np.flatnonzero(invalid&~endpoint):branched[i]=classify(d["tokens"][i],pred[v,i])["branch_nodes"]>0
            rows.append({"job":job["id"],"shard":job["shard"],"model_seed":job["seed"],"arm":arm,"n":256,"solved":int(success[v].sum()),"valid_paths":int(valid[v].sum()),"wrong_certified":int((stopped[v]&invalid).sum()),"endpoint_only_wrong":int((invalid&endpoint).sum()),"branched_wrong":int((invalid&branched).sum()),"no_feasible_output":int((~valid[v]).sum()),"valid_longer_path":int((valid[v]&~shortest[v]).sum()),"timeouts":int((~stopped[v]&finite[v]).sum()),"nonfinite":int((~finite[v]).sum()),"hidden_failures":int((~out["hidden_good"]).sum()),"mean_steps":float(steps[v].mean()),"answer_seconds":r["answer_seconds_by_variant"][v],"mean_changed_free_cells":float(out["changed_free_cells"][v].mean()),"mean_negative_free_cells":float(out["negative_free_cells"][v].mean()),"minimum_wrong_certified_margin":float(out["minimum_original_flip_margin"][v][stopped[v]&invalid].min()) if np.any(stopped[v]&invalid) else None,"strict_wrong_certificates":int((stopped[v]&invalid&(out["minimum_original_flip_margin"][v]>0)).sum())})
            for i,index in enumerate(d["indices"]):
                cases.append({"input_index":int(index),"model_seed":job["seed"],"arm":arm,"success":bool(success[v,i]),"valid_path":bool(valid[v,i]),"shortest_path":bool(shortest[v,i]),"answer_steps":int(steps[v,i]),"certified":bool(stopped[v,i]),"finite":bool(finite[v,i]),"energy":float(out["energy"][v,i]) if np.isfinite(out["energy"][v,i]) else None,**{k:int(out[k][v,i]) for k in ["changed_cells","changed_free_cells","negative_cells","negative_free_cells","zero_cells"]},**{k:float(out[k][v,i]) if np.isfinite(out[k][v,i]) else None for k in ["minimum_original_flip_margin","original_continuous_energy","modified_continuous_energy","continuous_regularizer","modified_field_residual"]}})
        records.append(r)
    joint={arm:np.concatenate([np.stack([saved[(shard,s)][v] for s in range(3)]).all(0) for shard in range(64)]) for v,arm in enumerate(ARMS)}
    gain=int((joint["native_lambda1"]&~joint["native_lambda0"]).sum());loss=int((~joint["native_lambda1"]&joint["native_lambda0"]).sum());gl,gu=A.bounds(gain,16384,alpha=.025);ll,lu=A.bounds(loss,16384,alpha=.025);lower=gl-lu;upper=gu-ll
    primary={"hypothesis":"H2s","n_unique":16384,"gains":gain,"losses":loss,"observed_gain_pp":100*(gain-loss)/16384,"one_sided95_net_gain_lower_pp":100*lower,"one_sided95_net_gain_upper_pp":100*upper,"target_pp":.05,"decision":"supported" if lower>=.0005 else "falsified" if upper<.0005 else "inconclusive"}
    secondary={"gain_input_indices":np.flatnonzero(joint["native_lambda1"]&~joint["native_lambda0"]).tolist(),"loss_input_indices":np.flatnonzero(~joint["native_lambda1"]&joint["native_lambda0"]).tolist(),"common_failure_input_indices":np.flatnonzero(~joint["native_lambda1"]&~joint["native_lambda0"]).tolist()}
    per_input=[{"input_index":i,**{f"joint_{arm}_success":int(joint[arm][i]) for arm in ARMS}} for i in range(16384)]
    for name,values in [("runs.csv",rows),("per-model-input.csv",cases),("per-input.csv",per_input)]:
        with (dest/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    result={"status":"validated","utc":S.utc(),"primary":primary,"secondary":secondary,"joint_success":{k:int(v.sum()) for k,v in joint.items()},"n_unique":16384,"models":3,"jobs":192,"model_input_rollouts":98304,"rows":rows,"records":records,"initial_logit_arrays_verified":16384,"final_logit_arrays_verified":98304,"independent_graph_checks":graph_checks,"independent_distance_checks":oracle_checks,"exact_board_cache_hits":cache_hits,"batch_preflight":json.loads((S.ROOT/pilot["path"]).read_text()),"protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"evidence_sha256":evidence,"parallel_oracle_seconds":oracle_seconds,"oracle_processes":8,"audit_seconds":time.perf_counter()-started,"scope":"Native-u3 compensated diagonal lambda0/+1 comparison with identical binary energies and original certificates on fresh120-grid loop20 inputs, three frozen models, eta.5. No candidate selector or ground-truth intervention. All failures retained."}
    result["size_shift_preflight"]=pre
    S.save(dest/"audit.json",result);S.emit("larger_relaxation_audit_passed",primary=primary,joint_success=result["joint_success"],seconds=result["audit_seconds"])


if __name__=="__main__":main()
