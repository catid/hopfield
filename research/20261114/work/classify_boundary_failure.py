"""Independent graph classification of the retained failure and its exact replays."""
import json
from pathlib import Path
import numpy as np
import study as S
from topology import distances


def main():
    dest=S.OUT/"eta_boundary";source=dest/"loops20.npz";data=np.load(source);index=38;tok=data["tokens"][index]
    start=tuple(np.argwhere(tok==2)[0]);goal=tuple(np.argwhere(tok==3)[0]);ds=distances(tok,[start]);dg=distances(tok,[goal]);length=ds[goal]
    union=np.zeros_like(tok,bool)
    for node,d in ds.items():union[node]=d+dg[node]==length
    counts={start:1}
    for node in sorted(ds,key=ds.get):
        if not union[node]:continue
        if node!=start:counts[node]=sum(counts.get((node[0]+dr,node[1]+dc),0) for dr,dc in S.SIDES if ds.get((node[0]+dr,node[1]+dc),-1)==ds[node]-1)
    rows=[]
    for seed in range(3):
        for eta in (.2,.5,.8,1.):
            path=dest/f"runs/loops20-s{seed}/eta{eta}.npz";output=np.load(path);pred=output["predictions"][index]
            selected={tuple(x) for x in np.argwhere(pred)};remaining=set(selected);components=[]
            while remaining:
                node=next(iter(remaining));stack=[node];component={node};remaining.remove(node)
                while stack:
                    r,c=stack.pop()
                    for dr,dc in S.SIDES:
                        other=(r+dr,c+dc)
                        if other in remaining:remaining.remove(other);component.add(other);stack.append(other)
                components.append({"cells":len(component),"contains_start":start in component,"contains_goal":goal in component})
            degree=np.zeros_like(tok,int);degree[1:]+=pred[:-1];degree[:-1]+=pred[1:];degree[:,1:]+=pred[:,:-1];degree[:,:-1]+=pred[:,1:]
            violations=pred&(degree!=np.where((tok==2)|(tok==3),1,2));branches=pred&(degree>np.where((tok==2)|(tok==3),1,2))
            connected=any(c["contains_start"] and c["contains_goal"] for c in components)
            if not connected:kind="disconnected"
            elif branches.any():kind="branched_connected_subgraph"
            elif len(components)>1:kind="path_plus_disconnected_components"
            elif violations.any():kind="degree_invalid"
            elif int(pred.sum())!=length+1:kind="valid_path_but_longer"
            else:kind="valid_shortest_path"
            replay_path=dest/f"failure-forensics/s{seed}-eta{eta}.json";replay=json.loads(replay_path.read_text())
            assert replay["native_saved_batch_parity"]
            rows.append({"seed":seed,"eta":eta,"kind":kind,"selected_cells":int(pred.sum()),"stopped":bool(output["stopped"][index]),
                         "answer_steps":int(output["answer_steps"][index]),"degree_violations":int(violations.sum()),"branching_cells":int(branches.sum()),
                         "component_count":len(components),"component_summary":sorted(components,key=lambda c:c["cells"],reverse=True),
                         "prediction_equals_shortest_union":bool(np.array_equal(pred,union)),"selected_outside_shortest_union":int((pred&~union).sum()),
                         "unselected_shortest_union":int((~pred&union).sum()),
                         **{k:replay[k] for k in ["hidden_steps","hidden_good","hidden_absolute_residual","two_cycle_detected","minimum_flip_delta","negative_flip_count","discrete_energy","oracle_label_energy"]},
                         "final_continuous_energy":replay["history"][-1]["continuous_energy"],"final_free_logit_residual":replay["history"][-1]["residual_free_max"],
                         "states_sha256":replay["states_sha256"],"forensic_receipt_sha256":S.sha(replay_path),"original_output_sha256":S.sha(path)})
    output={"status":"classified","utc":S.utc(),"post_outcome_diagnostic":True,"dataset_index":index,"shortest_path_edges":length,
            "number_of_shortest_paths":str(counts[goal]),"shortest_path_union_cells":int(union.sum()),"open_cells":int((tok>0).sum()),
            "dataset_sha256":S.sha(source),"script_sha256":S.sha(__file__),"rows":rows,
            "scope":"Selected after observing failure; graph algorithms and oracle energies are forensic diagnostics, not model inputs or a replacement of frozen primary outcomes."}
    S.save(dest/"failure-classification.json",output);np.savez_compressed(dest/"failure-forensics/shortest-path-union.npz",tokens=tok,shortest_union=union)
    S.emit("failure_classified",shortest_path_edges=length,number_of_shortest_paths=str(counts[goal]),shortest_path_union_cells=int(union.sum()),
           rows=[{k:r[k] for k in ["seed","eta","kind","minimum_flip_delta","two_cycle_detected"]} for r in rows])


if __name__=="__main__":main()
