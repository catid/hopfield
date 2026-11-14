"""Boundary trial and selected failure figures from retained audited evidence."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import study as S


def main():
    audit_path=S.OUT/"eta_boundary/audit.json";audit=json.loads(audit_path.read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for stratum,label,color in [("tree","90×90 trees","#0072B2"),("loops20","90×90, 20% opened loops","#D55E00"),("comb","160×160 combs","#009E73")]:
        rows=[r for r in audit["cells"] if r["stratum"]==stratum]
        axes[0].plot([r["eta"] for r in rows],[r["median_steps_including_caps"] for r in rows],"o-",label=label,color=color)
    axes[0].set(xlabel="Fixed answer step η",ylabel="Median answer updates (log scale)",title="Update reduction levels off near η = 0.8",yscale="log",xticks=[.2,.5,.8,1])
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    rows=audit["summaries"];bars=axes[1].bar([str(r["eta"]) for r in rows],[r["checked_answer_seconds"] for r in rows],color=["#0072B2","#56B4E9","#009E73","#E69F00"])
    for b,r in zip(bars,rows):axes[1].text(b.get_x()+b.get_width()/2,b.get_height()+2,f"{r['checked_answer_seconds']:.2f}",ha="center",fontsize=9)
    axes[1].set(xlabel="Fixed answer step η",ylabel="Summed checked answer time (seconds)",title="Matched GPU timing includes finite checks",ylim=(0,130));axes[1].grid(axis="y",alpha=.2)
    fig.text(.5,.01,"1,536 fresh inputs, three frozen models: all arms solve 1,535 inputs jointly. The same loop maze fails for models 1 and 2.\nη=1 loses no additional joint solves versus η=0.5; conservative one-sided 95% loss upper bound 0.7965 pp (<1 pp target).",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1));out=S.OUT/"figures"
    for ext in ("png","pdf","svg"):fig.savefig(out/f"eta-boundary.{ext}",dpi=180)
    plt.close(fig)
    path=S.OUT/"eta_boundary/failure-forensics/shortest-path-union.npz";case=np.load(path);tok=case["tokens"];fig,axes=plt.subplots(1,4,figsize=(14,5))
    panels=[(case["shortest_union"],"All shortest paths: union\n352,212 paths; 654 union cells")]
    for eta,title in [(.5,"η = 0.5: wrong certified stop\n246 cells, two branch junctions"),(.8,"η = 0.8: wrong certified stop\nOnly the two endpoints remain"),(1.,"η = 1.0: budget exhaustion\nPeriod-two oscillation detected")]:
        states=np.load(S.OUT/f"eta_boundary/failure-forensics/s2-eta{eta}-states.npz");panels.append((states["prediction"].astype(bool),title))
    for ax,(pred,title) in zip(axes,panels):
        raster=np.where(tok>0,1,0);raster[pred]=2;raster[(tok==2)|(tok==3)]=3
        ax.imshow(raster,cmap=ListedColormap(["#263238","#ECEFF1","#D55E00","#56B4E9"]),vmin=0,vmax=3,interpolation="nearest")
        endpoints=np.argwhere((tok==2)|(tok==3));ax.scatter(endpoints[:,1],endpoints[:,0],s=30,facecolors="none",edgecolors="#56B4E9",linewidths=1)
        ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
    fig.text(.5,.02,"Post-outcome forensic case: loop20 maze 38, model seed 2. Orange = selected cells; blue = endpoints.\nHidden state converged exactly. The η=0.5 and 0.8 wrong states have strictly positive single-flip margins; this is not a certificate-tolerance error.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.95))
    for ext in ("png","pdf","svg"):fig.savefig(out/f"cyclic-failure.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    S.save(out/"boundary-manifest.json",{"script_sha256":S.sha(__file__),"audit_sha256":S.sha(audit_path),
        "classification_sha256":S.sha(S.OUT/"eta_boundary/failure-classification.json"),
        "outputs_sha256":{f"{name}.{ext}":S.sha(out/f"{name}.{ext}") for name in ("eta-boundary","cyclic-failure") for ext in ("png","pdf","svg")}})


if __name__=="__main__":main()
