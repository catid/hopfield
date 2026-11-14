"""Negative independent controls and explicitly selected-case interventions."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as S


def main():
    long=json.loads((S.OUT/"long_ties/audit.json").read_text());edits=json.loads((S.OUT/"forensic_edits/audit.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for condition,label,color in [("tied","Two equal routes","#0072B2"),("plus","Unequal routes (+2 port shift)","#D55E00"),("minus","Unequal routes (−2 port shift)","#009E73")]:
        rows=[r for r in long["cells"] if r["condition"]==condition]
        axes[0].plot([r["route_length"] for r in rows],[r["mean_steps"] for r in rows],"o-",label=label,color=color,alpha=.85)
    axes[0].set(title="Independent long-cycle controls all solve",xlabel="Route length in the tied graph",ylabel="Mean answer updates",xticks=[28,42,64])
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    labels=["Original\nselected graph","Targeted\nbranch deletions","Matched noncritical\nnode deletions"]
    counts=[0,2,0];totals=[1,2,32];rates=[100*k/n for k,n in zip(counts,totals)]
    bars=axes[1].bar(labels,rates,color=["#999999","#0072B2","#999999"])
    for b,k,n in zip(bars,counts,totals):axes[1].text(b.get_x()+b.get_width()/2,b.get_height()+2,f"{k}/{n}",ha="center")
    axes[1].set(title="Recovery is specific to the retained branch",ylabel="Joint three-model success (%)",ylim=(0,115));axes[1].grid(axis="y",alpha=.2)
    fig.text(.5,.01,"Left: 128 fresh paired layouts per condition, fixed η=0.5; all 3,456 model–maze outputs valid.\nRight: descriptive interventions on one selected failure, not independent population samples. Node/edge counts and S–G distance match.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1));out=S.OUT/"figures"
    for ext in ("png","pdf","svg"):fig.savefig(out/f"tie-controls-and-selected-edits.{ext}",dpi=180)
    plt.close(fig)
    S.save(out/"tie-controls-manifest.json",{"script_sha256":S.sha(__file__),"long_ties_audit_sha256":S.sha(S.OUT/"long_ties/audit.json"),
        "forensic_edits_audit_sha256":S.sha(S.OUT/"forensic_edits/audit.json"),"outputs_sha256":{f"tie-controls-and-selected-edits.{ext}":S.sha(out/f"tie-controls-and-selected-edits.{ext}") for ext in ("png","pdf","svg")}})


if __name__=="__main__":main()
