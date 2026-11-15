"""Rare-event default comparison and its frozen effect-size target."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as S
from export_native_relaxation import manifest


def main():
    out=S.ROOT/"results/research/work";a=json.loads((out/"native-relaxation-summary.json").read_text());p=a["primary"];n=16384
    statistics=json.loads((out/"native-relaxation-statistics-verification.json").read_text());two=statistics["two_sided95_descriptive"]
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.3),gridspec_kw={"width_ratios":[1.2,1]},constrained_layout=True)
    ax=axes[0];keys=["native_lambda0","native_lambda1"];failures=[n-a["joint_success"][k] for k in keys]
    ax.bar(range(2),failures,color=["#227c9d","#508a3b"],width=.6)
    for i,key in enumerate(keys):ax.text(i,failures[i]+max(failures)*.025,f"{failures[i]} failures\n{a['joint_success'][key]:,}/{n:,} solved",ha="center",fontsize=10)
    ax.set_xticks(range(2),["Native u=3\nλ=0","Same native u=3\nλ=+1"]);ax.set_ylim(0,max(1,max(failures)*1.28));ax.set_ylabel("Joint failures across 3 frozen models");ax.set_title("Practical default on a larger fresh cohort",loc="left",fontweight="bold");ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    ax=axes[1];point=p["observed_gain_pp"];low=p["one_sided95_net_gain_lower_pp"];high=p["one_sided95_net_gain_upper_pp"]
    ax.errorbar([point],[.18],xerr=[[point-low],[high-point]],fmt="o",color="#508a3b",markersize=8,lw=3,capsize=6)
    ax.errorbar([point],[-.13],xerr=[[point-two['net_lower_pp']],[two['net_upper_pp']-point]],fmt="o",color="#777777",markersize=7,lw=2,capsize=5)
    ax.axvline(.05,color="#c23b23",ls="--");ax.axvline(0,color="#999999",lw=.8)
    ax.text(.12,.34,f"Frozen one-sided 95% bounds\n[{low:.5f}, {high:.5f}] pp",ha="center",color="#355e28",fontsize=10)
    ax.text(.12,-.35,f"Descriptive two-sided 95% interval\n[{two['net_lower_pp']:.5f}, {two['net_upper_pp']:.5f}] pp",ha="center",color="#666666",fontsize=10)
    ax.text(.05,-.58,"Frozen target ≥0.05 pp",ha="center",color="#a52b20",fontsize=10)
    ax.set_xlim(-.01,max(.22,two['net_upper_pp']+.02));ax.set_ylim(-.67,.6);ax.set_yticks([]);ax.spines["left"].set_visible(False);ax.set_xlabel("λ=+1 minus native accuracy (percentage points)");ax.set_title(f"{p['gains']} paired gains; {p['losses']} losses",loc="left",fontweight="bold")
    fig.suptitle("Native inference: fewer failed inputs with a compensated diagonal term",fontsize=15,fontweight="bold")
    fig.supxlabel("Eλ(p) = E₀(p) + λ Σfree p(1−p), with λ=1: every binary energy and the original flip certificate stay unchanged.\n16,384 fresh 90×90 loop20 inputs; 3 fixed models; u=3 in both arms; η=.5; cap8,192.\nThe prespecified one-sided lower bound clears 0.05 pp narrowly; the descriptive two-sided lower bound does not.",fontsize=10)
    for suffix in ["png","pdf","svg"]:fig.savefig(out/f"native-relaxation.{suffix}",dpi=180)
    manifest(out)


if __name__=="__main__":main()
