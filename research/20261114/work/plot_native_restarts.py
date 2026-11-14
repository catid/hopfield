"""Accuracy preservation and the prespecified noninferiority bound."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as S
from export_native_restarts import manifest


def main():
    out=S.ROOT/"results/research/work";a=json.loads((out/"native-restart-summary.json").read_text())
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.2),gridspec_kw={"width_ratios":[1.5,1]},constrained_layout=True)
    ax=axes[0];keys=["native","energy","screen","length"];values=[100*a["joint_success"][k]/4096 for k in keys]
    ax.bar(range(4),values,color=["#227c9d","#c23b23","#6a8a3b","#4d7d50"],width=.65)
    for i,(value,key) in enumerate(zip(values,keys)):ax.text(i,value+1.5,f"{a['joint_success'][key]:,}/4,096",ha="center",fontsize=10)
    ax.set_xticks(range(4),["Native\n8,192 cap","Native + 3\nenergy only","Same candidates\nfeasible + energy","Same candidates\nfeasible + length"]);ax.set_ylim(0,112);ax.set_ylabel("Joint shortest-path success (%)");ax.set_title("Native candidate preserves observed accuracy",loc="left",fontweight="bold");ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    ax=axes[1];upper=a["primary"]["one_sided95_loss_upper_pp"]
    ax.hlines(0,0,upper,color="#4d7d50",lw=5);ax.scatter([0,upper],[0,0],color="#4d7d50",s=[70,35]);ax.axvline(.1,color="#c23b23",ls="--",lw=1.5)
    ax.text(upper, .16,f"Upper 95% bound\n{upper:.5f} pp",ha="center",color="#355b38");ax.text(.1,-.25,"Frozen margin\n0.1 pp",ha="center",color="#a52b20")
    ax.set_xlim(-.005,.125);ax.set_ylim(-.55,.5);ax.set_yticks([]);ax.set_xlabel("Loss probability versus native (percentage points)");ax.set_title("0 losses; 0 gains",loc="left",fontweight="bold");ax.spines["left"].set_visible(False)
    fig.suptitle("Native plus three noisy candidates: noninferiority without new recoveries",fontsize=15,fontweight="bold")
    fig.supxlabel("4,096 fresh 90×90 loop20 inputs; 3 fixed models; η=.5; candidate caps 2,048 each.\nLength selector uses candidate node count only. All native runs certify by 2,048; their prefixes are reused exactly.\n49,152 new answer trajectories; 12,288 reused prefixes. Six native joint failures remain unresolved.",fontsize=10)
    for suffix in ["png","pdf","svg"]:fig.savefig(out/f"native-restart-selection.{suffix}",dpi=180)
    manifest(out)


if __name__=="__main__":main()
