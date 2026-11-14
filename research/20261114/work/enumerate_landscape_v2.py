"""Exhaustive shortest-path energies for the selected graph; no model tuning."""
import argparse
import fcntl
import json
import os
import subprocess
import time
import numpy as np
import study as S
import audit as A
from topology import distances


def prepare():
    started=time.perf_counter();dest=S.OUT/"path_landscape";dest.mkdir(parents=True,exist_ok=True)
    state=np.load(S.OUT/"eta_boundary/failure-forensics/s2-eta0.5-states.npz");tok=state["tokens"];h,w=tok.shape
    start=tuple(np.argwhere(tok==2)[0]);goal=tuple(np.argwhere(tok==3)[0]);ds=distances(tok,[start]);dg=distances(tok,[goal]);distance=ds[goal]
    valid,independent_distance=A.independent(tok,state["oracle_label"]);assert valid and independent_distance==distance==204
    nodes=[n for n in ds if ds[n]+dg[n]==distance];children=np.zeros((h*w,4),np.int64);cumulative=np.zeros_like(children);counts=np.zeros(h*w,np.int64);counts[goal[0]*w+goal[1]]=1
    for node in sorted(nodes,key=ds.get,reverse=True):
        if node==goal:continue
        nexts=sorted((node[0]+dr,node[1]+dc) for dr,dc in S.SIDES if ds.get((node[0]+dr,node[1]+dc))==ds[node]+1 and dg.get((node[0]+dr,node[1]+dc))==dg[node]-1)
        idx=node[0]*w+node[1];ids=[p[0]*w+p[1] for p in nexts];children[idx,:len(ids)]=ids
        weights=np.zeros(4,np.int64);weights[:len(ids)]=counts[ids];cumulative[idx]=weights.cumsum();counts[idx]=weights.sum()
    total=int(counts[start[0]*w+start[1]]);assert total==352212
    diagonal_checks=0
    for node in ds:
        for dr,dc in [(1,-1),(1,1)]:
            other=(node[0]+dr,node[1]+dc)
            if other in ds:
                assert (node[0]+dr,node[1]) in ds or (node[0],node[1]+dc) in ds
                diagonal_checks+=1
    path=dest/"paths.npy"
    if path.exists():raise FileExistsError(path)
    out=np.lib.format.open_memmap(path,mode="w+",dtype=np.uint16,shape=(total,distance+1))
    distance_lookup=np.full(h*w,-1,np.int64)
    for node,value in ds.items():distance_lookup[node[0]*w+node[1]]=value
    for lo in range(0,total,8192):
        hi=min(total,lo+8192);ranks=np.arange(lo,hi,dtype=np.int64);remaining=ranks.copy();current=np.full(hi-lo,start[0]*w+start[1],np.int64);batch=np.empty((hi-lo,distance+1),np.uint16)
        for step in range(distance+1):
            batch[:,step]=current
            if step==distance:break
            sums=cumulative[current];choice=(remaining[:,None]>=sums).sum(1)
            remaining-=np.where(choice>0,sums[np.arange(len(current)),choice-1],0);current=children[current,choice]
        assert np.all(current==goal[0]*w+goal[1]) and np.all(remaining==0)
        ids=batch.astype(np.int64);assert np.all(tok.reshape(-1)[ids]>0)
        assert np.all(np.abs(np.diff(ids//w,axis=1))+np.abs(np.diff(ids%w,axis=1))==1)
        assert np.all(distance_lookup[ids]==np.arange(distance+1))
        recovered=np.zeros(hi-lo,np.int64)
        for step in range(distance):
            parents=ids[:,step];actual=ids[:,step+1];choices=children[parents]==actual[:,None];assert np.all(choices.sum(1)==1)
            which=choices.argmax(1);recovered+=np.where(which>0,cumulative[parents,np.maximum(which-1,0)],0)
        assert np.array_equal(recovered,ranks);out[lo:hi]=batch
    out.flush();del out
    protocol={"id":"exhaustive-selected-shortest-path-energy","frozen_utc":S.utc(),"post_outcome_descriptive":True,"status":"frozen_before_energy_outcomes",
        "script_sha256":S.sha(__file__),"core_sha256":S.sha(S.__file__),"paths_sha256":S.sha(path),"path_count":total,"path_nodes":distance+1,"graph_shape":[h,w],
        "graph_state_sha256":S.sha(S.OUT/"eta_boundary/failure-forensics/s2-eta0.5-states.npz"),
        "models":json.loads((S.HERE/"protocol.json").read_text())["models"],"source_sha256":json.loads((S.HERE/"protocol.json").read_text())["source_sha256"],
        "state_sha256":{str(s):S.sha(S.OUT/f"eta_boundary/failure-forensics/s{s}-eta0.5-states.npz") for s in range(3)},
        "enumeration":{"method":"DAG suffix counts and lexicographic unranking; everypath checked against endpoints,open cells,orthogonal adjacency,monotone Dijkstra/BFS distance,and reverse-ranking identity. Allranks0..N-1present exactlyonce.","seconds":time.perf_counter()-started,"diagonal_neighbor_checks":diagonal_checks},
        "energy":"Evaluate every shortest-path binary answer using retainedfloat32couplings andfloat32modelweights cast tofloat64. The exact binary quadratic expansion uses endpoint baseline, free-cell flip costs, consecutive orthogonal interactions and diagonal turn interactions. Everyopendiagonal pair has an open orthogonal intermediary in this graph, verified before execution; thus along a shortest path anyselecteddiagonal pair is exactlytwosteps apart.",
        "validation":"Compare sparse expansion with full-grid repository energy on256evenlyspaced prespecified ranks and allreported extrema, atol1e-7. Time bothafterwarmup on same256paths with5CUDA-synchronized repeats. Saveall352212energies permodel and extrema paths. No tuning or inference-rule selection.",
        "jobs":[{"seed":s,"gpu":s%2} for s in range(3)],"batch_size":8192,"scope":"One selected graph; exhaustive shortest answers, not allpossiblylonger simple paths or allbinary answers. No population claim."}
    pp=S.HERE/"path_landscape-v2-protocol.json"
    if pp.exists():raise FileExistsError(pp)
    S.save(pp,protocol);S.emit("landscape_paths_frozen",sha256=S.sha(pp),paths=total,enumeration_seconds=protocol["enumeration"]["seconds"])


def worker(gpu):
    lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True);assert uuid not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,_=S.imports();dest=S.OUT/"path_landscape";pp=S.HERE/"path_landscape-v2-protocol.json";protocol=json.loads(pp.read_text());assert protocol["script_sha256"]==S.sha(__file__)
    assert S.sha(dest/"paths.npy")==protocol["paths_sha256"];paths=np.load(dest/"paths.npy",mmap_mode="r");h,w=protocol["graph_shape"]
    for path,expected in protocol["source_sha256"].items():assert S.sha(S.ROOT/path)==expected
    for job in protocol["jobs"]:
        if job["gpu"]!=gpu:continue
        seed=job["seed"];out=dest/f"seed{seed}";out.mkdir(exist_ok=True)
        if (out/"result.json").exists():raise FileExistsError(out)
        assert S.sha(S.OUT/f"models/seed{seed}.pt")==protocol["models"][str(seed)]["sha256"]
        model=S.model_for(seed,90,"original","cuda",torch.float64);state_path=S.OUT/f"eta_boundary/failure-forensics/s{seed}-eta0.5-states.npz"
        assert S.sha(state_path)==protocol["state_sha256"][str(seed)];state=np.load(state_path)
        t=torch.tensor(state["tokens"].astype(np.int64),device="cuda")[None];a=torch.tensor(state["couplings"],device="cuda",dtype=torch.float64)[None]
        zero=torch.zeros((1,h,w),device="cuda",dtype=torch.float64);base=model.energy(t,zero,a)[0];linear=model.flip_delta(t,zero,a)[0].reshape(-1)
        tables=(model.T[:,1,1]-model.T[:,1,0]-model.T[:,0,1]+model.T[:,0,0]).cpu().numpy();pair={tuple(off):float(v) for off,v in zip(model.mu_offsets,tables)}
        attention=state["couplings"].astype(np.float64);kappa=model.kappa.cpu().numpy();offsets=list(model.aoffs);cardinals=[(-1,0),(0,-1),(0,1),(1,0)];jcard=[]
        for dr,dc in cardinals:
            off=(dr,dc);opposite=(-dr,-dc);mu=pair.get(off,pair.get(opposite));forward=offsets.index(off);reverse=offsets.index(opposite)
            rr=np.arange(h)[:,None]+dr;cc=np.arange(w)[None,:]+dc;valid=(rr>=0)&(rr<h)&(cc>=0)&(cc<w)
            back=np.where(valid,attention[reverse,np.clip(rr,0,h-1),np.clip(cc,0,w-1)],0)
            jcard.append((mu-kappa[forward]*attention[forward]-kappa[reverse]*back).reshape(-1))
        interactions=torch.tensor(np.stack(jcard,axis=1),device="cuda",dtype=torch.float64);jplus=pair[(1,1)];jminus=pair[(1,-1)];jdiagonal=torch.tensor([jminus,jplus],device="cuda",dtype=torch.float64)
        def sparse(ids):
            interior=ids[:,1:-1];delta=interior[:,1:]-interior[:,:-1]
            direction=torch.where(delta==-w,0,torch.where(delta==-1,1,torch.where(delta==1,2,3)))
            edge=interactions[interior[:,:-1],direction].sum(1)
            dr=interior[:,2:]//w-interior[:,:-2]//w;dc=interior[:,2:]%w-interior[:,:-2]%w
            diag=torch.where((dr.abs()==1)&(dc.abs()==1),jdiagonal[(dr*dc>0).long()],0.).sum(1)
            return base+linear[interior].sum(1)+edge+diag
        def full(ids):
            board=torch.zeros((len(ids),h*w),device="cuda",dtype=torch.float64);board.scatter_(1,ids,1.)
            return model.energy(t.expand(len(ids),-1,-1),board.reshape(-1,h,w),a.expand(len(ids),-1,-1,-1))
        ranks=np.linspace(0,len(paths)-1,256,dtype=np.int64);sample=torch.tensor(np.array(paths[ranks],dtype=np.int64),device="cuda")
        reference=full(sample);fast=sparse(sample);parity=float((reference-fast).abs().max());assert parity<1e-7,parity
        timings={}
        for name,function in [("full_grid",full),("sparse_exact",sparse)]:
            function(sample);torch.cuda.synchronize();values=[]
            for _ in range(5):
                begin=time.perf_counter();function(sample);torch.cuda.synchronize();values.append(time.perf_counter()-begin)
            timings[name]={"batch_paths":256,"seconds":values,"median_seconds":float(np.median(values))}
        energies=np.empty(len(paths),np.float64);begin=time.perf_counter()
        for lo in range(0,len(paths),8192):
            ids=torch.tensor(np.array(paths[lo:lo+8192],dtype=np.int64),device="cuda");energies[lo:lo+len(ids)]=sparse(ids).cpu().numpy()
            if lo%(8192*8)==0:S.emit("landscape_energy_progress",seed=seed,paths_done=lo+len(ids),total=len(paths))
        elapsed=time.perf_counter()-begin;assert np.isfinite(energies).all();np.save(out/"energies.npy",energies)
        low=int(energies.argmin());high=int(energies.argmax());extrema=torch.tensor(np.array(paths[[low,high]],dtype=np.int64),device="cuda");direct=full(extrema).cpu().numpy()
        assert np.allclose(direct,energies[[low,high]],rtol=0,atol=1e-7)
        retained=torch.tensor(state["prediction"],device="cuda",dtype=torch.float64)[None];retained_energy=float(model.energy(t,retained,a)[0])
        baseline=float(base);record={"status":"complete","utc":S.utc(),"seed":seed,"gpu":gpu,"path_count":len(paths),"min_valid_energy":float(energies[low]),"max_valid_energy":float(energies[high]),
            "min_rank":low,"max_rank":high,"endpoint_only_energy":baseline,"retained_eta05_energy":retained_energy,
            "endpoint_minus_min_valid":baseline-float(energies[low]),"retained_minus_min_valid":retained_energy-float(energies[low]),
            "all_valid_below_endpoint":bool(np.all(energies<baseline)),"endpoint_below_every_valid_shortest_path":bool(np.all(baseline<energies)),
            "sparse_full_max_error_256":parity,"extrema_full_max_error":float(np.max(np.abs(direct-energies[[low,high]]))),"energy_evaluation_seconds":elapsed,
            "timing":timings,"full_grid_over_sparse_timing_ratio":timings["full_grid"]["median_seconds"]/timings["sparse_exact"]["median_seconds"],
            "protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"checkpoint_sha256":S.sha(S.OUT/f"models/seed{seed}.pt"),"coupling_state_sha256":S.sha(state_path),
            "energies_sha256":S.sha(out/"energies.npy"),"paths_sha256":protocol["paths_sha256"],"precision":"float64 from retainedfloat32weights/readout"}
        np.savez_compressed(out/"extrema-paths.npz",tokens=state["tokens"],min_path=paths[low],max_path=paths[high],min_rank=low,max_rank=high)
        S.save(out/"result.json",record);S.emit("landscape_energy_complete",**record)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["prepare","worker"]);parser.add_argument("--gpu",type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=="prepare":prepare()
    else:worker(args.gpu)
