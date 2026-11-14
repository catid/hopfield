"""Frozen descriptive separation of geometry and latent-edge readout on maze38."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import numpy as np
import study as S


def prepare():
    parent=json.loads((S.HERE/"forensic_edits-protocol.json").read_text());cases=parent["cases"][:3]
    protocol={"id":"selected-case-coupling-geometry-swaps-corrected","frozen_utc":S.utc(),"status":"post_outcome_design_frozen_before_swap_outcomes",
        "excluded_predecessor_sha256":"23d9130d800b08469dfee8be204569c5d20732ea9994e14b698bf30c185d4ac1","script_sha256":S.sha(__file__),"core_sha256":S.sha(S.__file__),"source_sha256":parent["source_sha256"],"models":parent["models"],
        "selected_cases":cases,"selected_indices":[0,1,2],"eta":.5,"answer_cap":8192,"hidden_cap":4000,"certify_every":16,
        "jobs":[{"seed":s,"gpu":(3+s)%2} for s in range(3)],
        "arms":["original_geometry_original_couplings","edited_geometry_edited_couplings","original_geometry_edited_couplings","edited_geometry_original_couplings"],
        "question":"For this selected graph, is removal of a critical cycle node necessary, or is the changed latent-gated edge readout sufficient to restore shortest paths?",
        "masking":"Transfer the donor latent attention readout on the intersection of donor and recipient open-cell supports, including the self offset. All nonself coefficients touching the donor-deleted or recipient-wall node are exactlyzero; valid selfcoefficients copied without change. No renormalization or scaling. The same support masking applied to identity baselines must preserve native full logits, steps and boards exactly. Maze-Unique pair/bias parameters and recipient tokens unchanged.",
        "sampling":"Two successful targeted branch deletions and the first matchednoncritical deletion from the alreadyfrozen edit protocol. No selection among swap outcomes. All3frozenmodels, fixedeta0.5, no rescue. Original/original baseline once permodel; each of3edits gets its own/own baseline and2reciprocal swaps:10arms permodel,30total.",
        "classification":"Descriptive selected-case causal intervention, no populationconfidenceinterval. Compare bothreciprocal swaps to their unchanged baselines; retain wrongstops, timeouts and energy terms.",
        "validation":"Before any corrected swaps, all3models and4sourcegraphs pass identity-transfer equality of native full logits, steps, stopflags and boards. Native logits captured read-only from the return frame without modifying upstream code. Original32-input edited baseline replay also exact. Validate donor/recipient support masks and retained selfcoefficients. Native/captured-logit comparison on everyarm remains an additional solver check, not a substitute for identity transfer.",
        "data_sha256":S.sha(S.OUT/"forensic_edits/data.npz"),"edit_protocol_sha256":S.sha(S.HERE/"forensic_edits-protocol.json"),
        "state_sha256":{f"seed{s}":S.sha(S.OUT/f"eta_boundary/failure-forensics/s{s}-eta0.5-states.npz") for s in range(3)}}
    pp=S.HERE/"coupling_swap_corrected-protocol.json"
    if pp.exists():raise FileExistsError(pp)
    S.save(pp,protocol);S.emit("coupling_swap_corrected_frozen",sha256=S.sha(pp),evaluated_arms=30)


def captured(model,t,a):
    torch,_,_=S.imports();u=torch.full(t.shape,3.,device=t.device,dtype=a.dtype);used=torch.full((len(t),),8192,dtype=torch.long,device=t.device);done=torch.zeros(len(t),dtype=torch.bool,device=t.device)
    with torch.no_grad():
        for k in range(1,8193):
            new=u+.5*(model.field(t,torch.sigmoid(u),a)-u);u=torch.where(done[:,None,None],u,new)
            if k%16==0:
                pred=(u>0).to(a.dtype);ok=(model.flip_delta(t,pred,a)>=-1e-6).flatten(1).all(1)
                used=torch.where(ok&~done,torch.full_like(used,k),used);done|=ok
                if done.all():break
    return (u>0).to(a.dtype),used,done,u


def support_mask(model,t):
    from cram.maze_learned import gather
    op=t>0
    return op[:,None]&gather(op,model.aoffs,False)


def transfer(model,source_t,source_a,recipient_t):
    torch,_,_=S.imports();allowed=support_mask(model,source_t)&support_mask(model,recipient_t)
    out=source_a*allowed
    assert not bool((out[~allowed]!=0).any())
    assert torch.equal(out[:,0][allowed[:,0]],source_a[:,0][allowed[:,0]])
    return out


def native_with_logits(model,t,a):
    _,_,M=S.imports();captured_state={}
    def profile(frame,event,arg):
        if event=="return" and frame.f_code is M.descend.__code__:
            captured_state["logits"]=frame.f_locals["u"].detach().clone()
    sys.setprofile(profile)
    try:result=M.descend(model,t,a,steps=8192,eta=.5,margin=3.,certify_every=16)
    finally:sys.setprofile(None)
    assert "logits" in captured_state
    return (*result,captured_state["logits"])


def preflight(gpu):
    lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True);assert uuid not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,M=S.imports();pp=S.HERE/"coupling_swap_corrected-protocol.json";protocol=json.loads(pp.read_text());data=np.load(S.OUT/"forensic_edits/data.npz");checks=[]
    for seed in range(3):
        model=S.model_for(seed,90,"original","cuda");original=np.load(S.OUT/f"eta_boundary/failure-forensics/s{seed}-eta0.5-states.npz")
        t0=torch.as_tensor(original["tokens"].astype(np.int64),device="cuda")[None];a0=torch.as_tensor(original["couplings"],device="cuda")[None]
        te=torch.as_tensor(data["tokens"][:32].astype(np.int64),device="cuda")
        with torch.no_grad():
            psi,_=model.settle(te,tol=1e-6,max_steps=4000,check=8);ae=model.couplings(te,psi)
            baseline=M.descend(model,te,ae,steps=8192,eta=.5,margin=3.,certify_every=16);expected=np.load(S.OUT/f"forensic_edits/runs/edits-s{seed}/predictions.npz")
            assert np.array_equal(baseline[0].cpu().bool().numpy(),expected["predictions"][:32]) and np.array_equal(baseline[1].cpu().numpy(),expected["answer_steps"][:32])
            for name,t,a in [("original",t0,a0)]+[(f"edit{i}",te[i:i+1],ae[i:i+1]) for i in range(3)]:
                raw=native_with_logits(model,t,a);identity_a=transfer(model,t,a,t);identity=native_with_logits(model,t,identity_a);cap=captured(model,t,identity_a)
                assert all(torch.equal(raw[i],identity[i]) for i in range(4))
                assert all(torch.equal(identity[i],cap[i]) for i in range(4))
                checks.append({"seed":seed,"graph":name,"identity_full_logits_steps_stops_boards":True,"capture_native_full_state":True,"valid_self_coefficients_retained":True,"support_mask_checked":True})
    dest=S.OUT/"coupling_swap_corrected";dest.mkdir(exist_ok=True)
    S.save(dest/"preflight.json",{"status":"passed","utc":S.utc(),"checks":checks,"script_sha256":S.sha(__file__),"protocol_sha256":S.sha(pp)})
    S.emit("corrected_identity_preflight_passed",graphs=len(checks))


def worker(gpu):
    lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True);assert uuid not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,M=S.imports();from cram.maze_learned import gather
    pp=S.HERE/"coupling_swap_corrected-protocol.json";protocol=json.loads(pp.read_text());assert protocol["script_sha256"]==S.sha(__file__)
    for path,h in protocol["source_sha256"].items():assert S.sha(S.ROOT/path)==h
    source=S.OUT/"forensic_edits";assert S.sha(source/"data.npz")==protocol["data_sha256"];data=np.load(source/"data.npz")
    dest=S.OUT/"coupling_swap_corrected";dest.mkdir(exist_ok=True)
    gate=json.loads((dest/"preflight.json").read_text());assert gate["status"]=="passed" and gate["protocol_sha256"]==S.sha(pp) and gate["script_sha256"]==S.sha(__file__)
    for job in protocol["jobs"]:
        if job["gpu"]!=gpu:continue
        seed=job["seed"];assert S.sha(S.OUT/f"models/seed{seed}.pt")==protocol["models"][str(seed)]["sha256"]
        model=S.model_for(seed,90,"original","cuda");state_path=S.OUT/f"eta_boundary/failure-forensics/s{seed}-eta0.5-states.npz"
        assert S.sha(state_path)==protocol["state_sha256"][f"seed{seed}"];original=np.load(state_path)
        original_t=torch.as_tensor(original["tokens"].astype(np.int64),device="cuda")[None]
        original_a=torch.as_tensor(original["couplings"],device="cuda")[None]
        edit_t=torch.as_tensor(data["tokens"][:32].astype(np.int64),device="cuda")
        with torch.no_grad():
            psi,hidden_steps=model.settle(edit_t,tol=1e-6,max_steps=4000,check=8);edit_a=model.couplings(edit_t,psi)
            base=M.descend(model,edit_t,edit_a,steps=8192,eta=.5,margin=3.,certify_every=16)
            expected=np.load(source/f"runs/edits-s{seed}/predictions.npz")
            assert np.array_equal(base[0].cpu().bool().numpy(),expected["predictions"][:32])
            assert np.array_equal(base[1].cpu().numpy(),expected["answer_steps"][:32]) and np.array_equal(base[2].cpu().numpy(),expected["stopped"][:32])
            original_mask=support_mask(model,original_t)
            arms=[("original","original_geometry_original_couplings",original_t,transfer(model,original_t,original_a,original_t),original["oracle_label"],None)]
            for i,case in enumerate(protocol["selected_cases"]):
                t=edit_t[i:i+1];a=edit_a[i:i+1];label=data["labels"][i]
                arms.extend([(case["kind"],"edited_geometry_edited_couplings",t,transfer(model,t,a,t),label,i),
                             (case["kind"],"original_geometry_edited_couplings",original_t,transfer(model,t,a,original_t),original["oracle_label"],i),
                             (case["kind"],"edited_geometry_original_couplings",t,transfer(model,original_t,original_a,t),label,i)])
            records=[]
            for ai,(case,arm,t,a,label,edit_index) in enumerate(arms):
                pred,used,done,u=captured(model,t,a);native=native_with_logits(model,t,a)
                assert all(torch.equal(x,y) for x,y in zip((pred,used,done,u),native))
                if ai==0:
                    assert np.array_equal(pred[0].cpu().numpy(),original["prediction"])
                delta=model.flip_delta(t,pred,a);field=model.field(t,torch.sigmoid(u),a)
                run=dest/f"s{seed}-arm{ai}";run.mkdir(exist_ok=True);path=run/"states.npz"
                if path.exists():raise FileExistsError(path)
                np.savez_compressed(path,tokens=t[0].cpu().numpy(),couplings=a[0].cpu().numpy(),prediction=pred[0].cpu().numpy(),logits=u[0].cpu().numpy(),
                    oracle_label=label,flip_delta=delta[0].cpu().numpy(),continuous_field=field[0].cpu().numpy())
                changed=(a-original_a)*original_mask;removed=protocol["selected_cases"][edit_index]["removed"] if edit_index is not None else None
                local=torch.zeros_like(original_t,dtype=torch.bool)
                if removed is not None:local[0,removed[0],removed[1]]=True
                incident=local[:,None]|gather(local,model.aoffs,False)
                label_tensor=torch.as_tensor(label,device="cuda",dtype=a.dtype)[None]
                record={"status":"complete","case":case,"arm":arm,"seed":seed,"gpu":gpu,"edit_index":edit_index,"removed":removed,
                    "answer_steps":int(used[0]),"stopped":bool(done[0]),"finite":bool(torch.isfinite(u).all()),"native_capture_parity":True,
                    "edited_baseline_batch_parity":True,"identity_preflight_passed":True,"coupling_change_scope":"Original-graph live coefficients only; self retained","minimum_flip_delta":float(delta.min()),"prediction_energy_float32":float(model.energy(t,pred,a)[0]),
                    "oracle_energy_float32":float(model.energy(t,label_tensor,a)[0]),"changed_coupling_count":int((changed!=0).sum()),
                    "coupling_L1_incident":float(changed.abs()[incident].sum()),"coupling_L1_nonincident":float(changed.abs()[~incident].sum()),
                    "states_sha256":S.sha(path),"protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"checkpoint_sha256":S.sha(S.OUT/f"models/seed{seed}.pt")}
                S.save(run/"result.json",record);records.append(record);S.emit("coupling_swap_corrected_arm",seed=seed,case=case,arm=arm,steps=record["answer_steps"],stopped=record["stopped"])
            S.save(dest/f"seed{seed}.json",{"status":"complete","utc":S.utc(),"gpu":gpu,"seed":seed,"arms":len(records),"protocol_sha256":S.sha(pp)})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["prepare","preflight","worker"]);parser.add_argument("--gpu",type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=="prepare":prepare()
    elif args.command=="preflight":preflight(args.gpu)
    else:worker(args.gpu)
