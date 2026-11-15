"""Independent validator gate on fixtures, random boards and prior held-out outputs."""
import json
import subprocess
import time
import numpy as np
import study as S
from audit_restart import check_path
from fast_path_validator import Validator,LIBRARY


def main():
    LIBRARY.parent.mkdir(parents=True,exist_ok=True)
    command=["cc","-O3","-std=c11","-shared","-fPIC",str(S.HERE/"fast_path_validator.c"),"-o",str(LIBRARY)]
    subprocess.run(command,check=True);validator=Validator();rng=np.random.default_rng(2026111594)
    tokens=[];preds=[]
    # Valid detour, missing endpoints, wall cell, branch, disconnected cycle.
    t=np.zeros((9,9),np.uint8);t[:5,0]=1;t[4,:5]=1;t[:5,4]=1;t[0,0]=2;t[0,4]=3
    tokens.append(t.copy());preds.append(t>0)
    for q in [(0,0),(0,4),(8,8)]:
        p=t>0;p[q]=not p[q];tokens.append(t.copy());preds.append(p)
    for change in ["branch","cycle"]:
        x=t.copy()
        if change=="branch":x[4,5]=1
        else:x[6,6:9]=1;x[8,6:9]=1;x[6:9,6]=1;x[6:9,8]=1
        tokens.append(x);preds.append(x>0)
    for i in range(4096):
        t=(rng.random((9,9))>.3).astype(np.uint8);t[0,0]=2;t[8,8]=3
        tokens.append(t);preds.append(rng.random((9,9))<rng.uniform(.01,.99))
    tok=np.stack(tokens);p=np.stack(preds);expected=np.array([check_path(t,x) for t,x in zip(tok,p)])
    assert np.array_equal(validator(tok,p),expected)
    source=S.OUT/"larger_relaxation";audit=json.loads((source/"audit.json").read_text());checked=0;fast=0.;slow=0.;cache={}
    for rec in audit["records"]:
        job=rec["job"];d=np.load(source/f"shard{job['shard']:02d}.npz")
        with np.load(source/f"runs/{job['id']}/outputs.npz") as out:pred=out["predictions"];feasible=out["feasible"]
        for v in range(2):
            begin=time.perf_counter();actual=validator(d["tokens"],pred[v]);fast+=time.perf_counter()-begin
            assert np.array_equal(actual,feasible[v]);checked+=len(actual)
            for index,t,p,answer in zip(d["indices"],d["tokens"],pred[v],actual):
                key=(int(index),np.packbits(p).tobytes())
                if key not in cache:
                    begin=time.perf_counter();cache[key]=check_path(t,p);slow+=time.perf_counter()-begin
                assert bool(answer)==cache[key]
    result={"status":"passed","utc":S.utc(),"fixtures_and_random_boards":len(tok),"prior_trial_boards_checked":checked,"unique_graph_board_pairs":len(cache),"compiled_total_seconds":fast,"independent_unique_total_seconds":slow,"not_like_for_like_timing":"Compiled timing covers all98304 boards; cached independent comparison only unique graph-board pairs. No speedup ratio inferred.","source_sha256":S.sha(S.HERE/"fast_path_validator.c"),"wrapper_sha256":S.sha(S.HERE/"fast_path_validator.py"),"binary_sha256":S.sha(LIBRARY),"gate_script_sha256":S.sha(__file__),"compiler":subprocess.check_output(["cc","--version"],text=True).splitlines()[0],"compile_command":command,"source_audit_sha256":S.sha(source/"audit.json"),"semantics":"Exactly one supplied start and goal, both selected; no walls; selected endpoints have degree1, all other selected cells degree2; endpoint-to-endpoint traversal covers every selected cell. Rejects extra cycles and branches. No shortestness information."}
    S.save(LIBRARY.parent/"validator-gate.json",result);S.emit("fast_path_gate_passed",boards=checked,fast_seconds=fast,random_and_fixture_count=len(tok))


if __name__=="__main__":main()
