"""Render all H4bc/H4bd fresh-bank outcomes from audited public JSON."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"results/research/foureyes/fresh_pooled_donor_corrected/summary.json"
DEST = ROOT/"assets/figures"


def main():
    data = json.loads(SOURCE.read_text())
    groups = {row["arm"]:row for row in data["grouped"]}
    plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False,
                         "svg.fonttype":"none","svg.hashsalt":"fresh-pooled-donor-corrected"})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 8), gridspec_kw={"width_ratios":[1,1.25]})
    fig.subplots_adjust(left=.07,right=.97,top=.72,bottom=.35,wspace=.35)
    fig.suptitle("Fresh graphs: pooling recovers 253/256",x=.04,y=.95,ha="left",fontsize=23,fontweight="bold")
    fig.text(.04,.87,"Same 1,800 noisy observations: 15 gains and one loss versus donor masking.",fontsize=15)
    fig.text(.04,.81,"Both frozen targets pass: ≥245 exact recoveries and ≤4 paired regressions.",fontsize=12)
    arms=["bp","masked","pooled"]
    colors=["#87919a","#bd8c61","#287b83"]
    labels=["BP AdamW\n32 updates","Donor +\nmasking","Pooled\ncounts"]
    totals=[groups[a]["success"] for a in arms]
    axes[0].bar(range(3),totals,color=colors,width=.65)
    axes[0].set_xticks(range(3),labels)
    axes[0].set_ylim(0,280)
    axes[0].set_ylabel("Exact recovered graphs / 256")
    axes[0].axhline(245,color="#456268",ls="--",lw=1)
    for i,n in enumerate(totals):axes[0].text(i,n+6,str(n),ha="center",fontweight="bold")
    x=np.arange(4)
    for offset,arm,color,label in zip([-.18,.18],arms[1:],colors[1:],labels[1:]):
        y=[r["exact"] for r in groups[arm]["families"]]
        axes[1].bar(x+offset,y,width=.34,color=color,label=label.replace("\n"," "))
        for pos,n in zip(x+offset,y):axes[1].text(pos,n+1,str(n),ha="center",fontsize=10)
    axes[1].set_xticks(x,["Cycle","Cycle²","Cycle³","Nine cliques"])
    axes[1].set_ylim(0,72)
    axes[1].set_ylabel("Exact recovered graphs / 64")
    axes[1].legend(frameon=False,loc="lower center",bbox_to_anchor=(.5,1.04),ncol=2)
    for ax in axes:ax.grid(axis="y",alpha=.14);ax.set_axisbelow(True)
    fig.text(.04,.24,"All 256 fresh banks retained; every degree inferred correctly from the first 600 noisy rows.",fontsize=12)
    fig.text(.04,.18,"Three pooled graphs remain wrong: banks 105, 168 and 197. Bank 105 is the single paired loss.\n"
             "Donor alone also recovers the same 239 banks; masking adds no exact recoveries.",fontsize=11.5,linespacing=1.5)
    fig.text(.04,.10,"Synthetic graph recovery, not puzzle accuracy. BP uses a fixed training budget; compute is not matched.\n"
             "A separately reviewed audit fixes only a NumPy-version assertion; the original incomplete run is retained.",fontsize=10.5,color="#444",linespacing=1.5)
    fig.text(.04,.035,"Exact replay: 11,008 LP certificates, 33,024 rejected proof mutations, and all 256 regenerated source banks.",fontsize=10.5,color="#444")
    for ext in ["png","svg"]:
        fig.savefig(DEST/f"fresh-pooled-donor.{ext}",dpi=180,facecolor="white",metadata={"Date":None} if ext=="svg" else None)
    inputs=[SOURCE,Path(__file__)]
    receipt={"input_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
             "output_sha256":{f"fresh-pooled-donor.{ext}":hashlib.sha256((DEST/f"fresh-pooled-donor.{ext}").read_bytes()).hexdigest() for ext in ["png","svg"]}}
    (DEST/"fresh-pooled-donor-provenance.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__ == "__main__":main()
