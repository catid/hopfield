"""Separate restart energy-ranking failure from native inference accuracy."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as S
from export_restart import manifest


def main():
    out=S.ROOT/"results/research/work";a=json.loads((out/"restart-summary.json").read_text())
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12.7,5.3),gridspec_kw={"width_ratios":[1.35,1]},constrained_layout=True)
    ax=axes[0];keys=["native","energy","screen"];values=[100*a["joint_success"][k]/4096 for k in keys]
    colors=["#227c9d","#c23b23","#508a3b"];bars=ax.bar(range(3),values,color=colors,width=.62)
    for i,(value,key) in enumerate(zip(values,keys)):
        ax.text(i,value+1.5,f"{a['joint_success'][key]:,}/4,096\n{value:.3f}%",ha="center",va="bottom",fontsize=10)
    ax.set_xticks(range(3),["Single native\ninitialization","4 perturbed\nenergy only","Same 4 candidates\nfeasibility + energy"]);ax.set_ylim(0,113);ax.set_ylabel("Joint success across 3 frozen models (%)")
    ax.set_title("Screening rescues unsafe energy ranking",loc="left",fontweight="bold");ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    ax=axes[1];times=[a["native_answer_seconds"],a["four_restart_answer_seconds"]];ratio=times[1]/times[0]
    ax.bar(range(2),times,color=[colors[0],"#75628c"],width=.6)
    for i,value in enumerate(times):ax.text(i,value+4,f"{value:.2f} s",ha="center")
    ax.set_ylim(0,max(times)*1.25);ax.set_xticks([0,1],["Native answer","Four restart answers"]);ax.set_ylabel("Summed synchronized answer time (s)");ax.set_title(f"Restarts cost {ratio:.2f}× answer time",loc="left",fontweight="bold");ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    fig.suptitle("Fresh cyclic mazes: selector gain is not a gain over native inference",fontsize=15,fontweight="bold")
    p=a["primary"];fig.supxlabel(f"Same-candidate screening gain: +{p['observed_gain_pp']:.3f} pp; one-sided 95% bounds [{p['one_sided95_gain_lower_pp']:.3f}, {p['one_sided95_gain_upper_pp']:.3f}] pp.\nScreened restarts solve 2 fewer inputs than native. Feasibility uses graph/endpoints only; shortestness is evaluation-only.\n4,096 new 90×90 loop20 inputs; 3 fixed models; η=.5; u=3 versus four u=3+N(0,1) chains. Equal maximum 8,192 updates.",fontsize=10)
    for suffix in ["png","pdf","svg"]:fig.savefig(out/f"restart-selection.{suffix}",dpi=180)
    manifest(out)


if __name__=="__main__":main()
