"""Freeze independent horizon confirmation and branch-depth transfer prediction."""
import json
import numpy as np
import followup
import study as S
from topology import distances


def main():
    parent=json.loads((S.HERE/"budget_extension-protocol.json").read_text())
    old=json.loads((S.OUT/"budget_extension/topology.json").read_text())["per_maze"]
    x=np.asarray([r["max_branch_distance"] for r in old]);y=np.asarray([r["extended_answer_steps_mean"] for r in old])
    slope,intercept=np.polyfit(x,y,1)
    directory=S.OUT/"horizon_transfer";directory.mkdir(parents=True,exist_ok=True);datasets=[];jobs=[]
    for length,n,base in [(60,384,2026111404),(90,128,2026111405)]:
        items=[S.maze(length,0,np.random.SeedSequence([base,length,i])) for i in range(n)]
        tokens=np.stack([r[0] for r in items]);labels=np.stack([r[1] for r in items]);shortest=np.asarray([r[2] for r in items])
        branch=np.asarray([max(distances(t,[tuple(p) for p in np.argwhere(y)]).values()) for t,y in zip(tokens,labels)])
        predictions=slope*branch+intercept
        path=directory/f"L{length}.npz"
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,tokens=tokens,labels=labels,shortest_edges=shortest)
        diag=directory/f"L{length}-frozen-diagnostics.npz"
        np.savez_compressed(diag,max_branch_distance=branch,predicted_answer_steps=predictions)
        datasets.append({"cohort":"horizon_transfer","length":length,"loop_percent":0,"n":n,"base_seed":base,
                         "path":str(path.relative_to(S.ROOT)),"sha256":S.sha(path),"diagnostics_path":str(diag.relative_to(S.ROOT)),"diagnostics_sha256":S.sha(diag)})
        for seed in range(3):jobs.append({"id":f"L{length}-tree-s{seed}-budget8192","length":length,"loop_percent":0,"seed":seed,
                                         "arm":"original","cohort":"horizon_transfer","answer_steps":8192})
    protocol=dict(parent,id="horizon_transfer",frozen_utc=S.utc(),status="frozen_before_new_outcomes",datasets=datasets,jobs=jobs,
                  execution_script_sha256=S.sha(followup.__file__),core_script_sha256=S.sha(S.__file__),
                  preparation_script_sha256=S.sha(__file__),parent_protocol_sha256=S.sha(S.HERE/"budget_extension-protocol.json"),
                  H2c="At least99% of new384 generated60x60 trees are solved by all3 frozen models using8192 answersteps.",
                  H2c_decision="One-sided95% exact Clopper-Pearson bounds on384 unique joint-success flags; lower>=.99 supported, upper<.99 falsified, otherwise inconclusive.",
                  H2d=None,H2d_decision=None,
                  H2f="Frozen linear branch-depth predictor from previous256cohort predicts mean answersteps on at least95% of new90x90 trees to within10% relative error.",
                  H2f_decision="Each90x90maze counts once; all3model mean is response. Any hidden/answer cap failure counts prediction failure, never dropped. One-sided95% exact interval for within10% fraction: lower>=.95 supported, upper<.95 falsified, otherwise inconclusive. Point error summaries descriptive.",
                  predictor={"slope":float(slope),"intercept":float(intercept),"training_n_unique":256,
                             "training_topology_sha256":S.sha(S.OUT/"budget_extension/topology.json"),
                             "covariate":"Maximum graph distance to unique S-G path computed with BFS; this uses oracle path information only as an offline mechanistic diagnostic, never a practical model-only budget selector or solver input."},
                  inference=dict(parent["inference"],answer_max_steps=8192),unique_primary_mazes=384,
                  confirmation="Both cohorts independent from all earlier generator seeds;90x90 is a new transfer test. No cap increases after outcomes.")
    path=S.HERE/"horizon_transfer-protocol.json"
    if path.exists():raise FileExistsError(path)
    S.save(path,protocol);S.emit("transfer_frozen",sha256=S.sha(path),predictor=protocol["predictor"],jobs=len(jobs))


if __name__=="__main__":main()
