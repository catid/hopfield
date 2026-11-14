"""Frozen H3 N-Queens optimizer study. All new evidence stays in this campaign."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "research/20261114/loki"
OUT = ROOT / "artifacts/research/20261114/loki"
UP = ROOT / "upstream"
sys.path[:0] = [str(UP), str(UP / "experiments")]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def emit(event, **values):
    print(json.dumps(dict(utc=now(), event=event, **values)), flush=True)


def protocol():
    return json.loads((SRC / "protocol.json").read_text())


def prepare():
    import numpy as np
    from cram import nqueens_data as D
    assert not (SRC / "protocol.json").exists(), "Protocol is immutable"
    cohorts = {}
    for n, discovery in [(10, 512), (8, 128)]:
        x, y = D.load_tokens(n, "train")
        fit_rows, held_rows = D.carve_validation(x == 2, fraction=.1, seed=0)
        held_unique, first = np.unique(x[held_rows], axis=0, return_index=True)
        order = np.random.default_rng(20261114 + n).permutation(len(first))
        disc = held_rows[first[order[:discovery]]]
        conf = held_rows[first[order[discovery:]]]
        assert not set(map(bytes, x[fit_rows])) & set(map(bytes, x[held_rows]))
        assert not set(map(bytes, x[disc])) & set(map(bytes, x[conf]))
        data = D.DATA / f"nqueens-{n}/train"
        bundle = OUT / f"cohorts-n{n}.npz"
        np.savez_compressed(bundle, fit_rows=fit_rows, held_rows=held_rows,
                            discovery_rows=disc, confirmation_rows=conf,
                            discovery_inputs=x[disc], confirmation_inputs=x[conf])
        cohorts[str(n)] = dict(fit_rows=len(fit_rows), held_unique=len(first), discovery_unique=len(disc),
            confirmation_unique=len(conf), cohort_file=str(bundle.relative_to(ROOT)), cohort_sha256=sha(bundle),
            data_sha256={str((data / f"all__{key}.npy").relative_to(ROOT)): sha(data / f"all__{key}.npy")
                         for key in ("inputs", "labels", "num_solutions")})
    arms = {f"adamw-lr{lr:g}": dict(optimizer="adamw", lr=lr, precond=False, weight_decay=.01)
            for lr in [.01, .03, .1, .3, 1.]}
    arms.update({"sgd-pre": dict(optimizer="sgd", lr=30., precond=True, weight_decay=0.),
                 "sgd-plain": dict(optimizer="sgd", lr=30., precond=False, weight_decay=0.)})
    p = dict(id="H3-nqueens-adamw", frozen_utc=now(), source_commit="62bbe830c7d6f87f2ba18f30a28039258b986454",
        hypothesis="BP AdamW without geometry-derived preconditioning is noninferior to frozen SGD+preconditioning at n10: N20 accuracy within0.1 percentage point and coverage within2 points, with4000 updates.",
        arms=arms, fit=dict(method="pl", arm="ising", updates=4000, initialization="zero", full_batch=True,
            schedule="lr*0.5*(1+cos(pi*k/4000))", sgd_momentum=.5, adamw_betas=[.9,.999], adamw_eps=1e-8,
            lam=0., tf32=False, training_seed=0,
            repeats="One fit per optimizer setting; PL is deterministic. Seeds below are inference draws, not distinct fitted models."),
        search=dict(chains=20, cap=20000, temperature=1., stop="local", margin=.001, seeds=[0,1,2]),
        cohorts=cohorts, split="Pinned carve_validation:10% distinct train inputs, seed0. Canonical unique held inputs shuffled by20261114+n; first512(n10)/128(n8) discovery, all remainder confirmation.",
        selection="n10 AdamW LR chosen once on discovery mean lowest-energyN20 accuracy across3searchseeds; tie higher paper-style coverage; tie lower LR. No stopping/temperature tuning. n8 transfers this LR without tuning.",
        confirmation="SelectedAdamW + bothSGDcontrols only; n10 primary, n8 secondary. No test files opened.",
        metrics=dict(primary_accuracy="Lowest-energy chain valid and contains givens, equal weight per unique input, then mean over3searchseeds",
            primary_coverage="Number distinct valid completions among20chains capped atmin(solution_count,20); sum over unique inputs / sum capped counts, then mean over3searchseeds",
            secondary=["N1 accuracy", "wrong stops", "budget failures", "moves", "complete fit seconds", "search seconds", "PL and offset contrast"]),
        decision=dict(accuracy_margin_pp=.1, coverage_margin_pp=2., confidence="One-sided95% noninferiority; paired unique-input bootstrap2000 draws, RNG20261114. Search repeats stay within each resampled input.",
            accuracy_safeguard="Support also requires exact one-sided95% Clopper-Pearson upper bound on fraction of inputs with ANY paired accuracy regression across3seeds <0.001. This is a conservative lower bound on mean accuracy difference, preventing a degenerate zero-difference bootstrap from asserting unsupported precision.",
            rule="Support if accuracy safeguard and bootstrap lower coverage difference>-2pp; falsified if bootstrap upper accuracy difference<-.1pp or upper coverage difference<-2pp; otherwise inconclusive. Discovery never establishes H3."),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), *sorted((UP/"experiments").glob("*.py")), *sorted((UP/"cram").glob("*.py"))]},
        host="loki", preserve="Original reproduction files remain unchanged. New source and outputs isolated.",
        limits=["Frozen published SGD LR was selected in the original paper; this is not a new equal-sized optimizer hyperparameter-search comparison.","Unique-input evaluation differs from repeating every duplicate-label row in the original Table3 search, so compare only paired local controls here.","Training labels can repeat completed boards across disjoint partial inputs, as in official GRAM protocol."])
    save(SRC / "protocol.json", p)
    emit("protocol_frozen", cohorts=cohorts, sha256=sha(SRC / "protocol.json"))


def independent_ok(boards, givens, n):
    import numpy as np
    b = np.asarray(boards, bool).reshape(len(givens), -1, n, n)
    ok = (b.sum(-1) == 1).all(-1) & (b.sum(-2) == 1).all(-1)
    rows, cols = np.indices((n,n))
    for d in range(-n+1,n):
        ok &= b[..., rows-cols == d].sum(-1) <= 1
    for d in range(2*n-1):
        ok &= b[..., rows+cols == d].sum(-1) <= 1
    ok &= (np.asarray(boards) >= np.asarray(givens)[:,None,:]).all(-1)
    return ok


def validate_cpu():
    """Independent PL-gradient and optimizer checks, no experimental outcomes."""
    import numpy as np
    import torch
    import nqueens_offset as O
    import training_methods as AT
    torch.set_num_threads(1)
    n=4; rng=np.random.default_rng(93)
    labels=torch.tensor(rng.integers(0,2,(5,n*n)),dtype=torch.long)
    givens=torch.zeros_like(labels,dtype=torch.bool);givens[:,0]=True
    task=O.OffsetTask(givens,labels,n,"ising")
    with torch.no_grad():task.A.copy_(torch.tensor(rng.normal(size=task.A.shape),dtype=torch.float32));task.b.fill_(.13)
    task.begin();loss=AT.loss_fn("pl",None,task,task.ctx,torch.Generator(),{})
    ag=torch.autograd.grad(loss,task.params())
    i,j=O.Q.pair_index(n*n,"cpu");idx,_=O.offsets(n,"cpu")
    aa=task.A.detach().clone().requires_grad_();bb=task.b.detach().clone().requires_grad_()
    # Literal binary local conditional energy differences, summed unordered pairs.
    fields=bb.expand(len(labels),n*n).clone()
    for p in range(len(i)):
        fields[:,i[p]]=fields[:,i[p]]+aa[idx[p]]*labels[:,j[p]]
        fields[:,j[p]]=fields[:,j[p]]+aa[idx[p]]*labels[:,i[p]]
    # P(y=1)=sigmoid(-field): NLL(y=1)=softplus(field).
    nll=torch.nn.functional.softplus(fields)*(labels==1)+torch.nn.functional.softplus(-fields)*(labels==0)
    independent=nll[~givens].mean();bg=torch.autograd.grad(independent,(aa,bb))
    torch.testing.assert_close(loss,independent,atol=1e-6,rtol=1e-6)
    for a,b in zip(ag,bg):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    param=torch.tensor([.7,-.2],dtype=torch.float64,requires_grad=True);manual=param.detach().clone();m=torch.zeros_like(manual);v=m.clone()
    opt=torch.optim.AdamW([param],lr=.03,weight_decay=.01,foreach=False)
    for step,g in enumerate([torch.tensor([.4,-.8],dtype=torch.float64),torch.tensor([-.2,.3],dtype=torch.float64)],1):
        param.grad=g;opt.step();manual.mul_(1-.03*.01);m=.9*m+.1*g;v=.999*v+.001*g*g
        manual-=.03*(m/(1-.9**step))/(torch.sqrt(v/(1-.999**step))+1e-8)
        torch.testing.assert_close(param,manual,rtol=1e-12,atol=1e-12)
    saved_updates=O.UPDATES
    for precondition in (False,True):
        O.UPDATES=3
        control,_=O.fit(dict(method="pl",arm="ising",lr=30.,knob=None,lam=0.,precond=precondition),
                        givens.numpy().astype(np.uint8)+1,labels.numpy().astype(np.uint8)+1,n,0,device="cpu")
        check=O.OffsetTask(givens,labels,n,"ising");bufs=[torch.zeros_like(t) for t in check.params()]
        cnt=torch.bincount(O.offsets(n,"cpu")[0]).float();pre=cnt/cnt.mean()
        for k in range(3):
            check.begin();loss=AT.loss_fn("pl",None,check,check.ctx,torch.Generator(),{})
            grads=torch.autograd.grad(loss,check.params());lr=30*.5*(1+math.cos(math.pi*k/3))
            if precondition:grads=(grads[0]/pre,grads[1])
            with torch.no_grad():
                for par,g,buf in zip(check.params(),grads,bufs):buf.mul_(.5).add_(g);par.sub_(lr*buf)
        assert torch.equal(check.A,control["A"]) and torch.equal(check.b,control["b"])
    O.UPDATES=saved_updates
    save(OUT/"cpu-validation.json",dict(status="passed",utc=now(),source_sha256=sha(Path(__file__)),
        checks=["Literal unordered-pair PL loss and parameter gradients", "Two AdamW updates versus independent bias-corrected moment formula", "Three SGD updates with/without preconditioning bitwise equal to pinned upstream fit"],new_experimental_outcomes=False))
    emit("cpu_validation_passed")


def fit_model(n, arm, progress):
    import numpy as np
    import torch
    import nqueens_offset as O
    import nqueens_free as N
    import training_methods as AT
    p=protocol();cfg=p["arms"][arm];dest=OUT/f"n{n}/fits/{arm}"
    if (dest/"fit.json").exists():
        rec=json.loads((dest/"fit.json").read_text());assert rec["model_sha256"]==sha(dest/"model.pt")
        return torch.load(dest/"model.pt",weights_only=False),rec
    x,y=O.Q.load_tokens(n,"train");rows=np.load(OUT/f"cohorts-n{n}.npz")["fit_rows"]
    torch.cuda.synchronize();start=time.perf_counter()
    xi=torch.as_tensor(x[rows],device="cuda")==2;target=(torch.as_tensor(y[rows],device="cuda")==2).long()
    task=O.OffsetTask(xi,target,n,"ising",dense=False);gen=torch.Generator(device="cuda").manual_seed(0)
    bufs=[torch.zeros_like(t) for t in task.params()]
    counts=torch.bincount(O.offsets(n,"cuda")[0]).float();pre=counts/counts.mean()
    opt=torch.optim.AdamW(task.params(),lr=cfg["lr"],weight_decay=cfg["weight_decay"],foreach=False) if cfg["optimizer"]=="adamw" else None
    curve=[];torch.cuda.synchronize();loopstart=time.perf_counter()
    for k in range(4000):
        task.begin();loss=AT.loss_fn("pl",None,task,task.ctx,gen,{})
        grads=torch.autograd.grad(loss,task.params());lr=cfg["lr"]*.5*(1+math.cos(math.pi*k/4000))
        if opt is not None:
            opt.param_groups[0]["lr"]=lr
            for par,g in zip(task.params(),grads):par.grad=g
            opt.step()
        else:
            if cfg["precond"]:grads=(grads[0]/pre,grads[1])
            with torch.no_grad():
                for par,g,buf in zip(task.params(),grads,bufs):buf.mul_(.5).add_(g);par.sub_(lr*buf)
        if (k+1)%250==0:
            value=float(loss.detach());assert math.isfinite(value)
            row=dict(update=k+1,loss=value);curve.append(row);progress(stage="fit",n=n,arm=arm,**row)
    torch.cuda.synchronize();loop_s=time.perf_counter()-loopstart
    with torch.no_grad():
        A,b=task.A.detach(),task.b.detach();assert torch.isfinite(A).all() and torch.isfinite(b).all()
        W,B=O.pair_tables(A,b,"ising",n);S,tau,unit,delta=N.ready(W,B,target)
    torch.cuda.synchronize();fit_s=time.perf_counter()-start
    model=dict(W=W.cpu(),B=B.cpu(),S=S.cpu(),tau=tau,unit=unit,delta=delta,A=A.cpu(),b=b.cpu(),arm="ising")
    dest.mkdir(parents=True,exist_ok=True);torch.save(model,dest/"model.pt")
    rec=dict(status="complete",utc=now(),n=n,arm=arm,config=cfg,updates=4000,rows=len(rows),training_seed=0,
        loop_seconds=loop_s,complete_fit_seconds=fit_s,final_loss=float(loss.detach()),curve=curve,
        tau=tau,unit=unit,delta=delta,offsets=O.offset_summary(A.cpu(),b.cpu(),"ising",n),
        model_sha256=sha(dest/"model.pt"),protocol_sha256=sha(SRC/"protocol.json"),source_sha256=sha(Path(__file__)),
        timing_scope="One complete fit in dedicated worker; CUDA context preinitialized, no warm timing repeats; fit progress logging included.")
    save(dest/"fit.json",rec);emit("fit_complete",n=n,arm=arm,seconds=fit_s,loss=rec["final_loss"])
    return model,rec


def search(n,arm,cohort,seed,model,progress):
    import numpy as np
    from cram import cram_nqueens as Q
    from cram import nqueens_data as D
    dest=OUT/f"n{n}/{cohort}/{arm}/s{seed}"
    if (dest/"result.json").exists():return
    z=np.load(OUT/f"cohorts-n{n}.npz");inputs=z[f"{cohort}_inputs"]==2
    p=protocol();s=p["search"]
    progress(stage="search",n=n,arm=arm,cohort=cohort,seed=seed,inputs=len(inputs));emit("search_start",n=n,arm=arm,cohort=cohort,seed=seed,inputs=len(inputs))
    started=time.perf_counter()
    boards,stopped,energy,moves,flop=Q.solve(model,inputs,chains=20,temperature=1.,cap=20000,seed=seed,device="cuda",stop="local",margin=.001)
    seconds=time.perf_counter()-started
    ok=independent_ok(boards,inputs,n);assert np.array_equal(ok,Q.completes(boards,inputs,n))
    all_solutions=np.asarray(D.completions(np.zeros(n*n,bool),n),bool)
    solution_counts=np.asarray([all_solutions[:,row].all(1).sum() for row in inputs]);denom=np.minimum(solution_counts,20)
    assert (denom>0).all()
    found=np.asarray([len({b.tobytes() for b in boards[i,ok[i]]}) for i in range(len(inputs))]);found=np.minimum(found,denom)
    selected=np.argmin(energy,axis=1);n20=ok[np.arange(len(inputs)),selected]
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/"records.npz",inputs=inputs,boards=np.packbits(boards,axis=-1),stopped=stopped,
        energy=energy,moves=moves,flop=flop,valid=ok,n20=n20,n1=ok[:,0],found=found,denom=denom,solution_counts=solution_counts)
    rec=dict(status="complete",utc=now(),n=n,arm=arm,cohort=cohort,inference_seed=seed,unique_inputs=len(inputs),
        n20_accuracy_pct=100*float(n20.mean()),n1_accuracy_pct=100*float(ok[:,0].mean()),
        coverage_pct=100*float(found.sum()/denom.sum()),wrong_stops=int((stopped&~ok).sum()),budget_failures=int((~stopped).sum()),
        mean_moves=float(moves.mean()),search_seconds=seconds,records_sha256=sha(dest/"records.npz"),
        model_sha256=sha(OUT/f"n{n}/fits/{arm}/model.pt"),protocol_sha256=sha(SRC/"protocol.json"),source_sha256=sha(Path(__file__)),
        inference_rule="Pinned unchanged focused Gibbs; selected lowest-energy final/best-returned chain, including capped chains exactly as paper convention.",
        independent_validation="NumPy row/col/diagonal/givens checker plus complete92/724solution enumeration; no label matching needed.")
    save(dest/"result.json",rec);emit("search_complete",**{k:rec[k] for k in ("n","arm","cohort","inference_seed","n20_accuracy_pct","coverage_pct","wrong_stops","budget_failures","search_seconds")})


def worker(gpu, phase):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",CRAM_DATA=str(UP/"data"))
    lock=(ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    query=lambda *args:subprocess.check_output(args,text=True,timeout=20).strip()
    device=query("nvidia-smi",f"--id={gpu}","--query-gpu=index,uuid,name,driver_version","--format=csv,noheader")
    uuid=device.split(",")[1].strip();owners=query("nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader")
    assert uuid not in owners,"GPU already owned"
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    assert platform.node()=="loki", "This worker is assigned only Loki"
    p=protocol();assert sha(Path(__file__))==p["source_sha256"][str(Path(__file__).relative_to(ROOT))]
    for path,h in p["source_sha256"].items():assert sha(ROOT/path)==h,path
    for block in p["cohorts"].values():
        assert sha(ROOT/block["cohort_file"])==block["cohort_sha256"]
        for path,h in block["data_sha256"].items():assert sha(ROOT/path)==h,path
    receipt=OUT/f"worker-{phase}-g{gpu}.json"
    base=dict(status="running",started_utc=now(),pid=os.getpid(),host=platform.node(),gpu=gpu,device=device,phase=phase,
        protocol_sha256=sha(SRC/"protocol.json"),source_sha256=sha(Path(__file__)),torch=torch.__version__,cuda=torch.version.cuda)
    def progress(**kw):save(receipt,dict(base,updated_utc=now(),progress=kw))
    progress(stage="starting")
    try:
        if phase=="discovery":
            arms=(["adamw-lr0.01","adamw-lr0.1","adamw-lr1","sgd-plain"] if gpu==0 else ["adamw-lr0.03","adamw-lr0.3","sgd-pre"])
            for arm in arms:
                model,_=fit_model(10,arm,progress)
                for seed in p["search"]["seeds"]:search(10,arm,"discovery",seed,model,progress)
        else:
            selected=json.loads((OUT/"selection.json").read_text())["selected_arm"]
            jobs=[(n,a) for n in (10,8) for a in (selected,"sgd-pre","sgd-plain")]
            for n,arm in jobs[gpu::2]:
                model,_=fit_model(n,arm,progress)
                for seed in p["search"]["seeds"]:search(n,arm,"confirmation",seed,model,progress)
        save(receipt,dict(base,status="complete",completed_utc=now()));emit("worker_complete",phase=phase,gpu=gpu)
    except BaseException as exc:
        save(receipt,dict(base,status="failed",completed_utc=now(),error=repr(exc)));raise


def select():
    import numpy as np
    assert not (OUT/"selection.json").exists(),"Selection already frozen"
    rows=[]
    for arm,cfg in protocol()["arms"].items():
        rs=[json.loads((OUT/f"n10/discovery/{arm}/s{s}/result.json").read_text()) for s in (0,1,2)]
        assert all(r["status"]=="complete" for r in rs)
        rows.append(dict(arm=arm,n20_accuracy_pct=float(np.mean([r["n20_accuracy_pct"] for r in rs])),coverage_pct=float(np.mean([r["coverage_pct"] for r in rs])),lr=cfg["lr"]))
    best=min([r for r in rows if r["arm"].startswith("adamw")],key=lambda r:(-r["n20_accuracy_pct"],-r["coverage_pct"],r["lr"]))
    save(OUT/"selection.json",dict(status="frozen",utc=now(),selected_arm=best["arm"],rows=rows,protocol_sha256=sha(SRC/"protocol.json"),test_used=False,confirmation_used=False))
    emit("selection_frozen",selected=best,rows=rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="mode",required=True)
    sub.add_parser("prepare");sub.add_parser("validate");sub.add_parser("select")
    w=sub.add_parser("worker");w.add_argument("--gpu",type=int,choices=(0,1),required=True);w.add_argument("--phase",choices=("discovery","confirmation"),required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=="prepare":prepare()
    elif args.mode=="validate":validate_cpu()
    elif args.mode=="select":select()
    else:worker(args.gpu,args.phase)
