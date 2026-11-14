"""Preregistered exploratory topology/rollout diagnostics; no solver input changes."""
from collections import deque
import json
import numpy as np
import study as S


def distances(tok, sources):
    dist={p:0 for p in sources};queue=deque(sources);h,w=tok.shape
    while queue:
        p=queue.popleft()
        for dr,dc in ((-1,0),(0,1),(1,0),(0,-1)):
            q=(p[0]+dr,p[1]+dc)
            if 0<=q[0]<h and 0<=q[1]<w and tok[q]>0 and q not in dist:
                dist[q]=dist[p]+1;queue.append(q)
    return dist


def ranks(x):
    _,inverse,counts=np.unique(x,return_inverse=True,return_counts=True)
    centers=np.cumsum(counts)-.5*(counts+1)
    return centers[inverse]


def main():
    directory=S.OUT/"budget_extension";data=np.load(directory/"data.npz")
    original=np.stack([np.load(directory/"runs"/f"L60-tree-s{s}-budget2048"/"predictions.npz")["success"] for s in range(3)])
    extended=np.stack([np.load(directory/"runs"/f"L60-tree-s{s}-budget8192"/"predictions.npz")["answer_steps"] for s in range(3)])
    top=[]
    for tok,label in zip(data["tokens"],data["labels"]):
        op=tok>0;nodes=int(op.sum());edges=int((op[1:]&op[:-1]).sum()+(op[:,1:]&op[:,:-1]).sum())
        assert edges==nodes-1
        start=tuple(np.argwhere(tok==2)[0]);goal=tuple(np.argwhere(tok==3)[0])
        d=distances(tok,[start]);assert len(d)==nodes
        farthest=max(d,key=d.get);diameter=max(distances(tok,[farthest]).values())
        branch=max(distances(tok,[tuple(x) for x in np.argwhere(label)]).values())
        top.append(dict(shortest_edges=d[goal],tree_diameter=diameter,max_branch_distance=branch,open_nodes=nodes))
    steps=extended.mean(0);failure=~original.all(0);correlations={}
    for key in top[0]:
        x=np.asarray([r[key] for r in top],float)
        correlations[key]={"spearman_with_extended_steps":None if x.std()==0 else float(np.corrcoef(ranks(x),ranks(steps))[0,1]),
                           "mean_baseline_success":float(x[~failure].mean()),"mean_baseline_failure":float(x[failure].mean()) if failure.any() else None,
                           "min":float(x.min()),"max":float(x.max())}
    result={"status":"complete_exploratory","utc":S.utc(),"n_unique":len(top),"correlations":correlations,
            "per_maze":[dict(t,baseline_any_model_failure=bool(f),extended_answer_steps_mean=float(s)) for t,f,s in zip(top,failure,steps)],
            "protocol_sha256":S.sha(S.HERE/"budget_extension-protocol.json"),"dataset_sha256":S.sha(directory/"data.npz"),
            "script_sha256":S.sha(__file__),"scope":"Preregistered input-topology diagnostics computed after frozen inference; no diagnostic or label entered inference. Correlations are exploratory, not a fitted stopping policy or causal proof."}
    S.save(directory/"topology.json",result);S.emit("topology_complete",correlations=correlations)


if __name__=="__main__":main()
