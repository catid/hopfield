"""Scientific plots from audited saved results only."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as S


def main():
    comb=json.loads((S.OUT/"causal_comb/audit.json").read_text())
    transfer=json.loads((S.OUT/"horizon_transfer/audit.json").read_text())
    rows=[r for r in comb["conditions"] if r["spine_length"]==64]
    x=np.asarray([r["branch_depth"] for r in rows]);y=np.asarray([r["steps_mean"] for r in rows])
    lower=y-np.asarray([r["steps_min"] for r in rows]);upper=np.asarray([r["steps_max"] for r in rows])-y
    diag=np.load(S.OUT/"horizon_transfer/L90-frozen-diagnostics.npz")
    actual=np.stack([np.load(S.OUT/f"horizon_transfer/runs/L90-tree-s{s}-budget8192/predictions.npz")["answer_steps"] for s in range(3)]).mean(0)
    p=transfer["predictor"];fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    ax=axes[0];ax.errorbar(x,y,yerr=np.stack([lower,upper]),fmt="o-",capsize=3,label="Causal comb: mean, layout range")
    xx=np.linspace(0,64,100);ax.plot(xx,p["slope"]*xx+p["intercept"],"--",color="gray",label="Frozen tree-data predictor")
    ax.set(xlabel="Side-branch depth (spine length fixed at 64)",ylabel="Answer updates to stopping",title="Controlled branch depth changes rollout time")
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1];ax.scatter(diag["predicted_answer_steps"],actual,s=20,alpha=.7)
    lim=max(actual.max(),diag["predicted_answer_steps"].max())*1.05;ax.plot([0,lim],[0,lim],"--",color="gray")
    ax.set(xlabel="Predicted updates from frozen 60×60 fit",ylabel="Measured updates on 90×90 trees",title="Independent size-transfer prediction",xlim=(0,lim),ylim=(0,lim));ax.grid(alpha=.2)
    fig.text(.5,.01,"Three fixed trained models averaged per maze; 64 paired comb layouts and 128 independent 90×90 trees.\nBranch depth uses the oracle path only for offline analysis; it never enters inference.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));out=S.OUT/"figures";out.mkdir(exist_ok=True)
    for ext in ["png","pdf","svg"]:fig.savefig(out/f"pruning-mechanism.{ext}",dpi=180)
    plt.close(fig)
    S.save(out/"manifest.json",{"source_script_sha256":S.sha(__file__),"input_sha256":{str(p.relative_to(S.ROOT)):S.sha(p) for p in [S.OUT/"causal_comb/audit.json",S.OUT/"horizon_transfer/audit.json",S.OUT/"horizon_transfer/L90-frozen-diagnostics.npz"]},"outputs_sha256":{f"pruning-mechanism.{ext}":S.sha(out/f"pruning-mechanism.{ext}") for ext in ["png","pdf","svg"]}})


if __name__=="__main__":main()
