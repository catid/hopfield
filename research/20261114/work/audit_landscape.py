"""Independent exhaustive path archive and CPU full-grid energy checks."""
import csv
import json
import time
from collections import deque
import numpy as np
import study as S


def main():
    started=time.perf_counter();dest=S.OUT/"path_landscape";pp=S.HERE/"path_landscape-v2-protocol.json"
    protocol=json.loads(pp.read_text());evidence={}
    def verify(path,expected=None):
        actual=S.sha(path)
        if expected is not None:assert actual==expected,path
        evidence[str(path.relative_to(S.ROOT))]=actual
    verify(pp);verify(S.HERE/"enumerate_landscape_v2.py",protocol["script_sha256"])
    verify(S.HERE/"study.py",protocol["core_sha256"])
    for path,h in protocol["source_sha256"].items():verify(S.ROOT/path,h)
    verify(dest/"paths.npy",protocol["paths_sha256"])
    paths=np.load(dest/"paths.npy",mmap_mode="r");state=np.load(S.OUT/"eta_boundary/failure-forensics/s2-eta0.5-states.npz")
    tok=state["tokens"];h,w=tok.shape;flat=tok.ravel();start=int(np.flatnonzero(flat==2)[0]);goal=int(np.flatnonzero(flat==3)[0])
    distance=np.full(h*w,-1,np.int64);distance[start]=0;counts=[0]*(h*w);counts[start]=1;queue=deque([start])
    while queue:
        p=queue.popleft();r,c=divmod(p,w)
        for rr,cc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if not (0<=rr<h and 0<=cc<w):continue
            q=rr*w+cc
            if not flat[q]:continue
            if distance[q]<0:distance[q]=distance[p]+1;queue.append(q)
            if distance[q]==distance[p]+1:counts[q]+=counts[p]
    assert counts[goal]==len(paths)==352212 and distance[goal]==204 and paths.shape==(352212,205)
    for lo in range(0,len(paths),8192):
        ids=paths[lo:lo+8192].astype(np.int64)
        assert (ids[:,0]==start).all() and (ids[:,-1]==goal).all() and (flat[ids]>0).all()
        assert (np.abs(np.diff(ids//w,axis=1))+np.abs(np.diff(ids%w,axis=1))==1).all()
        assert (distance[ids]==np.arange(205)).all()
    packed=np.asarray(paths).view(np.dtype((np.void,paths.dtype.itemsize*paths.shape[1]))).ravel()
    unique=len(np.unique(packed));assert unique==len(paths)
    torch,_,_=S.imports();rows=[];full_grid_checks=[]
    for seed in range(3):
        record=json.loads((dest/f"seed{seed}/result.json").read_text());verify(dest/f"seed{seed}/result.json")
        verify(dest/f"seed{seed}/energies.npy",record["energies_sha256"])
        verify(S.OUT/f"models/seed{seed}.pt",protocol["models"][str(seed)]["sha256"])
        source=S.OUT/f"eta_boundary/failure-forensics/s{seed}-eta0.5-states.npz";verify(source,protocol["state_sha256"][str(seed)])
        assert record["protocol_sha256"]==S.sha(pp) and record["script_sha256"]==protocol["script_sha256"]
        values=np.load(dest/f"seed{seed}/energies.npy");assert values.shape==(len(paths),) and np.isfinite(values).all()
        low=int(values.argmin());high=int(values.argmax());assert (low,high)==(record["min_rank"],record["max_rank"])
        assert float(values[low])==record["min_valid_energy"] and float(values[high])==record["max_valid_energy"]
        verify(dest/f"seed{seed}/extrema-paths.npz");ext=np.load(dest/f"seed{seed}/extrema-paths.npz")
        assert np.array_equal(ext["tokens"],tok) and np.array_equal(ext["min_path"],paths[low]) and np.array_equal(ext["max_path"],paths[high])
        state=np.load(source);model=S.model_for(seed,90,"original","cpu",torch.float64)
        t=torch.tensor(tok.astype(np.int64))[None];a=torch.tensor(state["couplings"],dtype=torch.float64)[None]
        masks=np.zeros((4,h*w),np.float64);masks[0,paths[low]]=1;masks[1,paths[high]]=1
        masks[2,[start,goal]]=1;masks[3]=state["prediction"].ravel()
        with torch.no_grad():cpu=model.energy(t.expand(4,-1,-1),torch.tensor(masks.reshape(4,h,w)),a.expand(4,-1,-1,-1)).numpy()
        expected=np.array([record[k] for k in ["min_valid_energy","max_valid_energy","endpoint_only_energy","retained_eta05_energy"]])
        error=float(np.abs(cpu-expected).max());assert error<1e-7
        assert bool((values>cpu[2]).all())==record["endpoint_below_every_valid_shortest_path"]
        margin=model.flip_delta(t,torch.tensor(state["oracle_label"],dtype=torch.float64)[None],a)[0].numpy()
        free=tok==1;minmargin=float(margin[free].min());assert minmargin>0
        full_grid_checks.append({"seed":seed,"cpu_float64_max_error":error,"canonical_valid_path_minimum_free_flip_margin":minmargin})
        rows.append(record|{"energy_quantiles":np.quantile(values,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),"cpu_full_grid_max_error":error})
    result={"status":"validated","utc":S.utc(),"post_outcome_descriptive":True,"unique_graphs":1,"unique_verified_shortest_paths":unique,
            "model_energy_evaluations":len(paths)*3,"new_neural_rollouts":0,"rows":rows,"cpu_checks":full_grid_checks,
            "path_audit":"Independent forward BFS/counting, all archived rows checked for endpoints, open cells, adjacency and increasing distance; exact full-row byte uniqueness and total count establish exhaustive coverage.",
            "scope":"All shortest-path answers on one selected graph and three fixed models. Does not enumerate longer simple paths or all binary states; no global energy-minimum or population claim.",
            "script_sha256":S.sha(__file__),"protocol_sha256":S.sha(pp),"evidence_sha256":evidence,"audit_seconds":time.perf_counter()-started}
    S.save(dest/"audit.json",result);S.emit("landscape_audit_passed",paths=unique,energy_evaluations=len(paths)*3,seconds=result["audit_seconds"])


if __name__=="__main__":main()
