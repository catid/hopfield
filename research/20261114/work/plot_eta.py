"""Paired eta effect and budget-success figure from the independent audit."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as S


def main():
    source=S.OUT/"eta/audit.json";audit=json.loads(source.read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    ax=axes[0]
    for position,row in enumerate(audit["summaries"]):
        mean=row["ratio_of_medians"];low,high=row["paired_stratified_bootstrap95_ratio"]
        ax.errorbar(position,mean,yerr=[[mean-low],[high-mean]],fmt="o",capsize=6,color="#0072B2",ms=8)
        ax.annotate(f"{mean:.3f}×",(position,high),xytext=(0,7),textcoords="offset points",ha="center")
    ax.axhline(2,color="gray",ls="--",label="Frozen 2× threshold")
    ax.axhline(2.5,color="#D55E00",ls=":",label="Inverse-step prediction: 2.5×")
    ax.set(xticks=[0,1],xticklabels=["60×60","90×90"],ylabel="Median updates at η=0.2 / η=0.5",
           title="Fewer updates with unchanged learned energy",ylim=(1.9,2.7),xlim=(-.6,1.6))
    ax.legend(fontsize=8,loc="lower center");ax.grid(axis="y",alpha=.2)
    ax=axes[1];x=np.arange(2);width=.34
    for offset,eta,key,color in [(-.5,.2,"joint_solved2048_eta02","#0072B2"),(.5,.5,"joint_solved2048_eta05","#E69F00")]:
        vals=[100*r[key]/r["n_unique"] for r in audit["summaries"]]
        bars=ax.bar(x+offset*width,vals,width,label=f"η={eta}",color=color)
        for bar,r in zip(bars,audit["summaries"]):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{r[key]}/{r['n_unique']}",ha="center",fontsize=9)
    ax.set(xticks=x,xticklabels=["60×60","90×90"],ylim=(0,110),ylabel="Joint three-model success (%)",
           title="More mazes solved within 2,048 updates")
    ax.legend(loc="lower left",fontsize=8);ax.grid(axis="y",alpha=.2)
    fig.text(.5,.01,"384 new mazes per size: 128 each at 0%, 5%, and 20% opened loops; three fixed models averaged per maze.\nLeft: paired stratified bootstrap 95% intervals. Both arms solved every maze at the 8,192 cap; no lost solves.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));out=S.OUT/"figures"
    for ext in ("png","pdf","svg"):fig.savefig(out/f"eta-acceleration.{ext}",dpi=180)
    plt.close(fig)
    S.save(out/"eta-manifest.json",{"script_sha256":S.sha(__file__),"audit_sha256":S.sha(source),
          "outputs_sha256":{f"eta-acceleration.{ext}":S.sha(out/f"eta-acceleration.{ext}") for ext in ("png","pdf","svg")}})


if __name__=="__main__":main()
