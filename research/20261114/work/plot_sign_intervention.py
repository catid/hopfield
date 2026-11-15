"""Measured partial rescue versus the frozen ten-point hypothesis."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as S
from export_sign_intervention import manifest


def main():
    out=S.ROOT/"results/research/work";a=json.loads((out/"sign-intervention-summary.json").read_text());p=a["primary"]
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),gridspec_kw={"width_ratios":[1.5,1]},constrained_layout=True)
    ax=axes[0];keys=["native","full_gaussian","positive_only","negative_only"];values=[100*a["joint_success"][k]/2048 for k in keys]
    ax.bar(range(4),values,color=["#227c9d","#c23b23","#508a3b","#75628c"],width=.65)
    for i,(value,key) in enumerate(zip(values,keys)):ax.text(i,value+1.3,f"{a['joint_success'][key]:,}/2,048\n{value:.2f}%",ha="center",fontsize=10)
    ax.set_xticks(range(4),["Native\nu = 3","Full Gaussian\nu = 3 + Z","Negative values\nreset to 3","Positive values\nreset to 3"]);ax.set_ylim(0,113);ax.set_ylabel("Joint shortest-path success (%)");ax.set_title("Negative initial logits explain only part of collapse",loc="left",fontweight="bold");ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    ax=axes[1];point=p["observed_gain_pp"];low=p["one_sided95_net_gain_lower_pp"];high=p["one_sided95_net_gain_upper_pp"]
    ax.errorbar([point],[0],xerr=[[point-low],[high-point]],fmt="o",color="#508a3b",markersize=9,linewidth=3,capsize=6);ax.axvline(10,color="#c23b23",linestyle="--")
    ax.text(point,.2,f"Gain +{point:.3f} pp\nBounds [{low:.3f}, {high:.3f}]",ha="center",color="#355e28");ax.text(10,-.26,"Frozen target\n≥10 pp",ha="center",color="#a52b20")
    ax.set_xlim(0,12);ax.set_ylim(-.55,.5);ax.set_yticks([]);ax.spines["left"].set_visible(False);ax.set_xlabel("Reset-negative minus full-Gaussian gain (pp)");ax.set_title("10-point hypothesis falsified",loc="left",fontweight="bold")
    fig.suptitle("Paired initial-logit sign interventions on fresh cyclic mazes",fontsize=15,fontweight="bold")
    fig.supxlabel("2,048 fresh 90×90 loop20 inputs; 3 frozen models; same Gaussian draw; η=.5; all caps 8,192.\nStrict u<0 and u>0 interventions; zero values unchanged. No candidate selector.\nNegative-only perturbations change 6.002 free cells on average; they overlap 100 of 286 full-Gaussian failures.",fontsize=10)
    for suffix in ["png","pdf","svg"]:fig.savefig(out/f"sign-intervention.{suffix}",dpi=180)
    manifest(out)


if __name__=="__main__":main()
