"""Independent saved-board audit, exact binomial bounds and paired maze analysis."""
import argparse
import heapq
import json
import math
from pathlib import Path
import numpy as np
import study as S


def independent(tok, pred):
    """Dijkstra from goal and degree/connectivity checks, independent of generator BFS."""
    start = tuple(np.argwhere(tok == 2)[0]); goal = tuple(np.argwhere(tok == 3)[0])
    dist = {goal: 0}; queue = [(0, goal)]; sides = ((-1,0),(0,1),(1,0),(0,-1)); h, w = tok.shape
    while queue:
        d, p = heapq.heappop(queue)
        if dist[p] != d: continue
        for dr, dc in sides:
            q = (p[0]+dr, p[1]+dc)
            if 0<=q[0]<h and 0<=q[1]<w and tok[q] and d+1<dist.get(q,10**9):
                dist[q]=d+1;heapq.heappush(queue,(d+1,q))
    chosen = {tuple(x) for x in np.argwhere(pred)}
    if start not in chosen or goal not in chosen or any(tok[p] == 0 for p in chosen): return False, dist[start]
    for p in chosen:
        degree = sum((p[0]+dr,p[1]+dc) in chosen for dr,dc in sides)
        if degree != (1 if p in (start,goal) else 2): return False, dist[start]
    seen={start};stack=[start]
    while stack:
        p=stack.pop()
        for dr,dc in sides:
            q=(p[0]+dr,p[1]+dc)
            if q in chosen and q not in seen:seen.add(q);stack.append(q)
    return seen==chosen and len(chosen)==dist[start]+1, dist[start]


def cdf(k,n,p):
    if k<0:return 0.
    if k>=n:return 1.
    if p==0:return 1.
    if p==1:return 0.
    xs=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)+i*math.log(p)+(n-i)*math.log1p(-p) for i in range(k+1)]
    m=max(xs);return min(1.,math.exp(m)*sum(math.exp(x-m) for x in xs))


def bounds(k,n,alpha=.05):
    if k==0:lower=0.
    else:
        lo,hi=0.,1.
        for _ in range(70):
            p=(lo+hi)/2
            if 1-cdf(k-1,n,p)<alpha:lo=p
            else:hi=p
        lower=(lo+hi)/2
    if k==n:upper=1.
    else:
        lo,hi=0.,1.
        for _ in range(70):
            p=(lo+hi)/2
            if cdf(k,n,p)>alpha:lo=p
            else:hi=p
        upper=(lo+hi)/2
    return lower,upper


def main(campaign):
    directory=S.OUT if campaign=="discovery" else S.OUT/campaign
    protocol_path=S.HERE/("protocol.json" if campaign=="discovery" else f"{campaign}-protocol.json")
    protocol=json.loads(protocol_path.read_text());evidence={str(protocol_path.relative_to(S.ROOT)):S.sha(protocol_path)}
    def verify(path,h=None):
        p=S.ROOT/path;actual=S.sha(p)
        if h is not None:assert actual==h,path
        evidence[str(p.relative_to(S.ROOT))]=actual
    for p,h in protocol["source_sha256"].items():verify(p,h)
    for m in protocol["models"].values():verify(m["path"],m["sha256"])
    verify(str(Path(__file__).relative_to(S.ROOT)))
    records=[];saved={}
    for job in protocol["jobs"]:
        run=directory/"runs"/job["id"];r=json.loads((run/"result.json").read_text())
        assert r["status"]=="complete" and r["job"]==job and r["protocol_sha256"]==S.sha(protocol_path)
        verify(str((run/"result.json").relative_to(S.ROOT)));verify(str((run/"predictions.npz").relative_to(S.ROOT)),r["predictions_sha256"])
        data=next(d for d in protocol["datasets"] if d["cohort"]==job["cohort"] and d["length"]==job["length"] and d["loop_percent"]==job["loop_percent"])
        verify(data["path"],data["sha256"]);source=np.load(S.ROOT/data["path"]);out=np.load(run/"predictions.npz")
        valid=[]
        for t,p,d in zip(source["tokens"],out["predictions"],source["shortest_edges"]):
            ok,distance=independent(t,p);assert d==distance;valid.append(ok)
        valid=np.asarray(valid);success=valid & out["stopped"] & out["hidden_good"]
        assert np.array_equal(valid,out["shortest_path"]) and np.array_equal(success,out["success"])
        assert int(success.sum())==r["solved"] and int((out["stopped"] & ~valid).sum())==r["wrong_stops"]
        assert int((~out["stopped"]).sum())==r["answer_budget_failures"] and int((~out["hidden_good"]).sum())==r["hidden_failures"]
        cap=job.get("answer_steps",2048);assert np.all((out["answer_steps"]>=16)&(out["answer_steps"]<=cap)&(out["answer_steps"]%16==0))
        assert np.all(out["answer_steps"][~out["stopped"]]==cap)
        records.append({k:r[k] for k in ("job","n","solved","wrong_stops","answer_budget_failures","hidden_failures","answer_steps_mean","hidden_steps_mean","wall_seconds")})
        saved[(job["length"],job["loop_percent"],job["seed"],job["arm"],cap)]=dict(success=success,answer_steps=out["answer_steps"],predictions=out["predictions"])
    comparisons=[];rng=np.random.default_rng(2026111419)
    rates=[0] if campaign=="budget_extension" else [0,5,20]
    all_joint=[];differences=[];strata=[]
    for rate in rates:
        original=np.stack([saved[(60,rate,s,"original",2048)]["success"] for s in range(3)])
        other=np.stack([saved[(60,rate,s,"original" if campaign=="budget_extension" else "mass_scaled",8192 if campaign=="budget_extension" else 2048)]["success"] for s in range(3)])
        joint=(other if campaign=="budget_extension" else original).all(0);all_joint.append(joint)
        delta=other.mean(0)-original.mean(0);differences.append(delta);strata.append(delta)
        comparisons.append({"loop_percent":rate,"original_solved_per_seed":original.sum(1).tolist(),"alternative_solved_per_seed":other.sum(1).tolist(),
                            "n_unique":original.shape[1],"joint_success":int(joint.sum()),"paired_difference_pp":float(100*delta.mean()),
                            "paired_discordant_mazes":int(np.any(original!=other,axis=0).sum())})
    joint=np.concatenate(all_joint);delta=np.concatenate(differences);low,high=bounds(int(joint.sum()),len(joint))
    bootstrap=np.zeros(10000)
    for d in strata:bootstrap+=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)/len(strata)
    ci=(100*np.quantile(bootstrap,[.025,.975])).tolist();mean=float(100*delta.mean())
    absolute={"success":int(joint.sum()),"n_unique":len(joint),"accuracy_percent":float(100*joint.mean()),
              "one_sided95_lower_percent":100*low,"one_sided95_upper_percent":100*high,
              "decision":"supported" if low>=.99 else "falsified" if high<.99 else "inconclusive"}
    paired={"difference_pp":mean,"stratified_paired_bootstrap95_pp":ci,"decision":"supported" if ci[0]>=5 else "falsified" if ci[1]<5 else "inconclusive"}
    if np.all(delta==0):paired["zero_discordance_abs_mean_upper95_pp"]=100*(1-.05**(1/len(delta)))
    receipt={"status":"validated","utc":S.utc(),"campaign":campaign,"protocol_sha256":S.sha(protocol_path),"jobs":len(records),
             "saved_board_checks":sum(r["n"] for r in records),"records":records,"comparisons":comparisons,"absolute_hypothesis":absolute,
             "paired_hypothesis":paired,"evidence_sha256":evidence,"hashed_files":len(evidence),"new_inference":False,
             "scope":"Independent Dijkstra from goal plus degree/connectivity/shortest checks on saved boards; train seeds are repeated models, not independent puzzles."}
    S.save(directory/"audit.json",receipt);S.emit("audit_complete",campaign=campaign,jobs=len(records),absolute=absolute,paired=paired,hashed_files=len(evidence))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--campaign",choices=["discovery","confirmation","budget_extension"],default="discovery");a=p.parse_args();main(a.campaign)
