"""Different continuous basins despite identical binary energy rankings."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as S
from export_binary_relaxation import manifest


def main():
    out=S.ROOT/"results/research/work";a=json.loads((out/"binary-relaxation-summary.json").read_text());primary=a["primary"]
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(13.3,5.5),gridspec_kw={"width_ratios":[1.4,1]},constrained_layout=True)
    ax=axes[0];keys=["native","gaussian_lambda0","gaussian_lambda1","gaussian_lambda_minus1"];values=[100*a["joint_success"][k]/4096 for k in keys]
    ax.bar(range(4),values,color=["#227c9d","#c23b23","#508a3b","#75628c"],width=.65)
    for i,(value,key) in enumerate(zip(values,keys)):ax.text(i,value+1.2,f"{a['joint_success'][key]:,}/4,096\n{value:.2f}%",ha="center",fontsize=10)
    ax.set_xticks(range(4),["Native u=3\nλ=0","Gaussian u\nλ=0","Same Gaussian u\nλ=+1","Same Gaussian u\nλ=−1"]);ax.set_ylim(0,113);ax.set_ylabel("Joint shortest-path success (%)");ax.set_title("Changing the continuous relaxation rescues noisy starts",loc="left",fontweight="bold");ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    ax=axes[1];p=np.linspace(0,1,201)
    for strength,color,label in [(1,"#508a3b","λ=+1"),(0,"#666666","λ=0"),(-1,"#75628c","λ=−1")]:ax.plot(p,strength*p*(1-p),color=color,label=label,lw=2)
    ax.scatter([0,1],[0,0],color="black",s=45,zorder=5);ax.set_xlabel("Continuous answer p");ax.set_ylabel("Added energy per free cell: λ p(1−p)");ax.set_title("Binary corners and flip costs stay fixed",loc="left",fontweight="bold");ax.legend(frameon=False);ax.grid(alpha=.15);ax.set_ylim(-.3,.3)
    fig.suptitle("A compensated diagonal term changes trajectories, not binary energy ranking",fontsize=15,fontweight="bold")
    fig.supxlabel(f"λ=+1 versus identical-Gaussian λ=0: +{primary['observed_gain_pp']:.3f} pp; one-sided 95% bounds [{primary['one_sided95_net_gain_lower_pp']:.3f}, {primary['one_sided95_net_gain_upper_pp']:.3f}] pp. Frozen ≥5 pp target supported.\n3 frozen models; 4,096 fresh 90×90 loop20 inputs; η=.5; cap8,192; original binary certificate unchanged.\nλ=+1 still trails native overall. Every retained margin/energy diagnostic independently recomputed with maximum error0.",fontsize=10)
    for suffix in ["png","pdf","svg"]:fig.savefig(out/f"binary-relaxation.{suffix}",dpi=180)
    manifest(out)


if __name__=="__main__":main()
