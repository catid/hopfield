"""Freeze causal comb interventions before measuring their predictions."""
import json
import numpy as np
import followup
import study as S


def main():
    parent=json.loads((S.HERE/"horizon_transfer-protocol.json").read_text());base=2026111406
    rng=np.random.default_rng(base);layout_codes=rng.choice(26*26*8,64,replace=False)
    layouts=[]
    for code in layout_codes:
        shift,dihedral=divmod(int(code),8);row,col=divmod(shift,26);rotation,flip=divmod(dihedral,2)
        layouts.append((row,col,rotation,flip))
    conditions=[(64,d) for d in [0,8,16,32,64]]+[(16,16),(32,16)]
    directory=S.OUT/"causal_comb";directory.mkdir(parents=True,exist_ok=True);datasets=[];jobs=[]
    for spine,depth in conditions:
        toks=[];ys=[];ds=[]
        for i,(row,col,rotation,flip) in enumerate(layouts):
            roots=np.random.default_rng(np.random.SeedSequence([base,i,spine])).choice(np.arange(1,spine,2),8,replace=False)
            tok=np.zeros((90,90),np.uint8);tok[row:row+spine+1,col]=1
            for r in roots:tok[row+r,col:col+depth+1]=1
            tok[row,col]=2;tok[row+spine,col]=3
            tok=np.rot90(tok,rotation)
            if flip:tok=np.fliplr(tok)
            tok=tok.copy();label,distance,reachable=S.bfs(tok)
            assert distance==spine and int(label.sum())==spine+1 and reachable==int((tok>0).sum())
            op=tok>0;assert int((op[1:]&op[:-1]).sum()+(op[:,1:]&op[:,:-1]).sum())==int(op.sum())-1
            toks.append(tok);ys.append(label);ds.append(distance)
        name=f"comb-spine{spine}-depth{depth}";path=directory/f"{name}.npz"
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,tokens=np.stack(toks),labels=np.stack(ys),shortest_edges=np.asarray(ds),layout_codes=layout_codes)
        datasets.append({"cohort":name,"length":90,"loop_percent":0,"spine_length":spine,"branch_depth":depth,"n":64,
                         "path":str(path.relative_to(S.ROOT)),"sha256":S.sha(path),"base_seed":base,"distinct_inputs":len({t.tobytes() for t in toks})})
        for seed in range(3):jobs.append({"id":f"{name}-s{seed}","length":90,"loop_percent":0,"seed":seed,"arm":"original", "cohort":name,
                                         "spine_length":spine,"branch_depth":depth,"answer_steps":8192})
    protocol=dict(parent,id="causal_comb",frozen_utc=S.utc(),status="frozen_before_comb_outcomes",datasets=datasets,jobs=jobs,
                  preparation_script_sha256=S.sha(__file__),execution_script_sha256=S.sha(followup.__file__),core_script_sha256=S.sha(S.__file__),
                  H2c=None,H2c_decision=None,H2f=None,H2f_decision=None,
                  H2e="At fixed S-Gspine64, increasing8side-branches from depth0 to64 increases mean answersteps by at least200; at fixed branchdepth16, changing spine16to64 changes mean answersteps by no more than32 in magnitude.",
                  H2e_decision="Two separate preregistered paired64-layout bootstrap10000 contrasts, three models averaged per layout. Depth support if95% lower>=200; falsify if upper<200. Spine equivalence support if entire95% interval within[-32,32], falsify if entirely beyond a boundary; otherwise inconclusive. Each contrast reported separately, no multiplicity-adjusted combined claim.",
                  generator="90x90 grid; vertical straight S-Gspine of length16,32or64; exactly8perpendicular branches at nonadjacent odd spine coordinates; depth0,8,16,32or64.64unique translation/rotation/reflection layouts sampled without replacement and paired across conditions.8branch roots sampled by fixed layout/spine seed; depths do not change roots.",
                  forecast="Compare per-condition steps to previously frozen slope4.352984700053693*depth+11.720719172409504; descriptive residual, no refit.",
                  primary_metric="Per-layout mean actual answersteps across3models. Capped steps retained and failure counts reported; causal interpretation requires valid shortest paths. Allmaze labels/checks are oracle-only, not solver inputs.",
                  unique_primary_layouts=64,confirmation="New comb generator2026111406; no changes to learnedweights, attention, hiddenmax4000 or answercap8192.")
    path=S.HERE/"causal_comb-protocol.json"
    if path.exists():raise FileExistsError(path)
    S.save(path,protocol);S.emit("comb_frozen",sha256=S.sha(path),conditions=conditions,jobs=len(jobs))


if __name__=="__main__":main()
