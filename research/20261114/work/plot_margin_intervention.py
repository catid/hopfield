"""Spatial perturbations versus matched expected-mean constants."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as S
from export_margin_intervention import manifest


def main():
    out=S.ROOT/"results/research/work";a=json.loads((out/"margin-intervention-summary.json").read_text());p=a["primary"]
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),gridspec_kw={"width_ratios":[1.8,1]},constrained_layout=True)
    ax=axes[0];keys=["native","full_gaussian","upward_only","constant_up","downward_only","constant_down"];values=[100*a["joint_success"][k]/4096 for k in keys]
    ax.bar(range(6),values,color=["#227c9d","#c23b23","#508a3b","#9fbd90","#75628c","#b5a6c9"],width=.66)
    for i,(value,key) in enumerate(zip(values,keys)):ax.text(i,value+1.2,f"{a['joint_success'][key]:,}\n{value:.2f}%",ha="center",fontsize=10)
    ax.set_xticks(range(6),["Native\n3","Full\n3 + Z","Upward\nmax(3, 3+Z)","Constant\n3.39894","Downward\nmin(3, 3+Z)","Constant\n2.60106"]);ax.set_ylim(0,113);ax.set_ylabel("Joint shortest-path success (%)");ax.set_title("Direction and spatial heterogeneity",loc="left",fontweight="bold");ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    ax=axes[1];low=p["one_sided95_loss_lower_pp"];high=p["one_sided95_loss_upper_pp"];point=p["observed_loss_pp"]
    ax.errorbar([point],[0],xerr=[[point-low],[high-point]],fmt="o",color="#508a3b",markersize=8,lw=3,capsize=6);ax.axvline(.1,color="#c23b23",ls="--")
    ax.text((low+high)/2,.17,f"Loss {point:.5f} pp\nBounds [{low:.5f}, {high:.5f}]",ha="center",color="#355e28");ax.text(.1,-.25,"Frozen margin\n0.1 pp",ha="center",color="#a52b20")
    ax.set_xlim(-.01,max(.15,high*1.2));ax.set_ylim(-.55,.5);ax.set_yticks([]);ax.spines["left"].set_visible(False);ax.set_xlabel("Upward-only loss versus native (pp)");ax.set_title(f"Noninferiority {p['decision']}",loc="left",fontweight="bold")
    fig.suptitle("Paired upward/downward initialization with expected-mean controls",fontsize=15,fontweight="bold")
    fig.supxlabel("4,096 fresh 90×90 loop20 inputs; 3 frozen models; η=.5; all caps 8,192; no candidate selector.\nConstants are 3 ± 1/√(2π), matching population expected means of the rectified Gaussian arms.\nExpected means are matched; realized per-input means are retained separately. Exact initial and final logits are archived.",fontsize=10)
    for suffix in ["png","pdf","svg"]:fig.savefig(out/f"margin-intervention.{suffix}",dpi=180)
    manifest(out)


if __name__=="__main__":main()
