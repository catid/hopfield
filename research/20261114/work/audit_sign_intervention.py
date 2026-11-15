"""Independent sign transforms, logits, graph outcomes and paired statistics."""
import csv
import hashlib
import json
import time
import numpy as np
import audit as A
from audit_restart import check_path
import study as S

ARMS=("native","full_gaussian","positive_only","negative_only")


def main():
    started=time.perf_counter();dest=S.OUT/"sign_intervention";pp=S.HERE/"sign_intervention-protocol.json";protocol=json.loads(pp.read_text());evidence={}
    def verify(path,expected=None):
        actual=S.sha(path)
        if expected is not None:assert actual==expected,path
        evidence[str(path.relative_to(S.ROOT))]=actual
    verify(pp);verify(S.HERE/"sign_intervention.py",protocol["execution_script_sha256"]);verify(S.HERE/"study.py",protocol["core_script_sha256"])
    for path,h in protocol["source_sha256"].items():verify(S.ROOT/path,h)
    for model in protocol["models"].values():verify(S.ROOT/model["path"],model["sha256"])
    verify(dest/"preflight.json");pre=json.loads((dest/"preflight.json").read_text());assert pre["status"]=="passed" and pre["strict_sign_and_zero_fixture"] and pre["protocol_sha256"]==S.sha(pp)
    pilot=protocol["batch_preflight"];verify(S.ROOT/pilot["path"],pilot["sha256"])
    verify(dest/"datasets.json");manifest=json.loads((dest/"datasets.json").read_text());assert manifest["protocol_sha256"]==S.sha(pp)
    data={};inits={};init_hashes={};unique=set();oracle_checks=0
    for ds in manifest["datasets"]:
        verify(S.ROOT/ds["path"],ds["sha256"]);raw=np.load(S.ROOT/ds["path"]);d={k:raw[k] for k in raw.files};data[ds["shard"]]=d
        full=d["initial_gaussian"];regenerated=np.stack([3.+np.random.default_rng(np.random.SeedSequence([protocol["initialization"]["noise_base"],int(i)])).standard_normal((90,90),dtype=np.float32) for i in d["indices"]]);assert np.array_equal(full,regenerated)
        pos=full.copy();pos[pos<0]=3.;neg=full.copy();neg[neg>0]=3.;arms=[np.full_like(full,3.),full,pos,neg];inits[ds["shard"]]=arms
        assert np.array_equal(pos[full==0],full[full==0]) and np.array_equal(neg[full==0],full[full==0])
        digest=hashlib.sha256()
        for lo in range(0,256,64):
            for a in arms:digest.update(a[lo:lo+64].tobytes())
        init_hashes[ds["shard"]]=digest.hexdigest()
        for tok,label,distance in zip(d["tokens"],d["labels"],d["shortest_edges"]):
            ok,actual=A.independent(tok,label);assert ok and actual==distance;oracle_checks+=1;unique.add(tok.tobytes())
    assert len(unique)==2048
    cache={};graph_checks=0;cache_hits=0;saved={};records=[];rows=[];cases=[]
    for job in protocol["jobs"]:
        run=dest/f"runs/{job['id']}";verify(run/"result.json");r=json.loads((run/"result.json").read_text());assert r["status"]=="complete" and r["job"]==job and r["protocol_sha256"]==S.sha(pp)
        assert r["script_sha256"]==protocol["execution_script_sha256"] and r["checkpoint_sha256"]==protocol["models"][str(job["seed"])]["sha256"]
        assert r["dataset_sha256"]==manifest["datasets"][job["shard"]]["sha256"] and r["initialization_sha256"]==init_hashes[job["shard"]]
        verify(run/"outputs.npz",r["outputs_sha256"]);out=np.load(run/"outputs.npz");d=data[job["shard"]];pred=out["predictions"];logits=out["final_logits"]
        assert pred.shape==logits.shape==(4,256,90,90) and logits.dtype==np.float32
        assert np.array_equal(pred,logits>0);assert np.array_equal(out["finite"],np.isfinite(logits).all((2,3)))
        valid=np.zeros((4,256),bool);shortest=np.zeros_like(valid)
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
            rows.append({"job":job["id"],"shard":job["shard"],"model_seed":job["seed"],"arm":arm,"n":256,"solved":int(success[v].sum()),"valid_paths":int(valid[v].sum()),"wrong_certified":int((stopped[v]&invalid).sum()),"endpoint_only_wrong":int((invalid&endpoint).sum()),"timeouts":int((~stopped[v]&finite[v]).sum()),"nonfinite":int((~finite[v]).sum()),"hidden_failures":int((~out["hidden_good"]).sum()),"mean_steps":float(steps[v].mean()),"answer_seconds":r["answer_seconds_by_variant"][v],"mean_changed_free_cells":float(out["changed_free_cells"][v].mean()),"mean_negative_free_cells":float(out["negative_free_cells"][v].mean())})
            for i,index in enumerate(d["indices"]):
                cases.append({"input_index":int(index),"model_seed":job["seed"],"arm":arm,"success":bool(success[v,i]),"valid_path":bool(valid[v,i]),"shortest_path":bool(shortest[v,i]),"answer_steps":int(steps[v,i]),"certified":bool(stopped[v,i]),"finite":bool(finite[v,i]),"energy":float(out["energy"][v,i]) if np.isfinite(out["energy"][v,i]) else None,**{k:int(out[k][v,i]) for k in ["changed_cells","changed_free_cells","negative_cells","negative_free_cells","zero_cells"]}})
        records.append(r)
    joint={arm:np.concatenate([np.stack([saved[(shard,s)][v] for s in range(3)]).all(0) for shard in range(8)]) for v,arm in enumerate(ARMS)}
    gain=int((joint["positive_only"]&~joint["full_gaussian"]).sum());loss=int((~joint["positive_only"]&joint["full_gaussian"]).sum());gl,gu=A.bounds(gain,2048,alpha=.025);ll,lu=A.bounds(loss,2048,alpha=.025);lower=gl-lu;upper=gu-ll
    primary={"hypothesis":"H2o","n_unique":2048,"gains":gain,"losses":loss,"observed_gain_pp":100*(gain-loss)/2048,"one_sided95_net_gain_lower_pp":100*lower,"one_sided95_net_gain_upper_pp":100*upper,"target_pp":10.,"decision":"supported" if lower>=.1 else "falsified" if upper<.1 else "inconclusive","bounds":"For each one-sided net bound, two exact binomial tail bounds at alpha=.025; Bonferroni yields95% coverage. Paired gain/loss events use independent inputs."}
    fullfail=~joint["full_gaussian"];negfail=~joint["negative_only"];intersection=int((fullfail&negfail).sum());union=int((fullfail|negfail).sum())
    secondary={"negative_vs_full_failure_overlap":{"full_failures":int(fullfail.sum()),"negative_only_failures":int(negfail.sum()),"intersection":intersection,"union":union,"jaccard":intersection/union if union else None,"negative_only_fraction_of_full_failures":intersection/int(fullfail.sum()) if fullfail.any() else None},"vs_native":{arm:{"gains":int((joint[arm]&~joint["native"]).sum()),"losses":int((~joint[arm]&joint["native"]).sum())} for arm in ARMS if arm!="native"}}
    per_input=[{"input_index":i,**{f"joint_{arm}_success":int(joint[arm][i]) for arm in ARMS}} for i in range(2048)]
    for name,values in [("runs.csv",rows),("per-model-input.csv",cases),("per-input.csv",per_input)]:
        with (dest/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    result={"status":"validated","utc":S.utc(),"primary":primary,"secondary":secondary,"joint_success":{k:int(v.sum()) for k,v in joint.items()},"n_unique":2048,"models":3,"jobs":24,"model_input_rollouts":24576,"rows":rows,"records":records,"initial_logit_arrays_verified":2048,"final_logit_arrays_verified":24576,"independent_graph_checks":graph_checks,"independent_distance_checks":oracle_checks,"exact_board_cache_hits":cache_hits,"batch_preflight":json.loads((dest/"batch-preflight.json").read_text()),"protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"evidence_sha256":evidence,"audit_seconds":time.perf_counter()-started,"scope":"Fixed sigma1 initial-logit sign decomposition on fresh90-grid loop20 inputs, three frozen models, eta.5. No candidate selector or ground-truth intervention. All failures retained."}
    S.save(dest/"audit.json",result);S.emit("sign_audit_passed",primary=primary,joint_success=result["joint_success"],seconds=result["audit_seconds"])


if __name__=="__main__":main()
