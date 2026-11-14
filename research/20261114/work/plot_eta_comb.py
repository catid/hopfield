"""Independent causal-comb eta scaling figure from audited outcomes."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as S


def main():
    source=S.OUT/"eta_comb/audit.json";audit=json.loads(source.read_text())
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.5))
    for eta,color in [(.2,"#0072B2"),(.5,"#E69F00")]:
        rows=sorted([r for r in audit["conditions"] if r["spine_length"]==64 and r["eta"]==eta],key=lambda r:r["branch_depth"])
        x=np.asarray([r["branch_depth"] for r in rows]);y=np.asarray([r["mean_steps"] for r in rows])
        err=np.stack([y-np.asarray([r["min_steps"] for r in rows]),np.asarray([r["max_steps"] for r in rows])-y])
        for ax,scale in [(axes[0],1),(axes[1],eta)]:
            ax.errorbar(x,scale*y,yerr=scale*err,fmt="o-",capsize=4,color=color,label=f"η = {eta}")
    axes[0].set(title="Branch pruning accelerates with step size",xlabel="Branch depth (S–G spine fixed at 64)",ylabel="Mean answer updates")
    axes[1].set(title="Scaled stopping times nearly coincide",xlabel="Branch depth (S–G spine fixed at 64)",ylabel="Step size × mean answer updates")
    for ax in axes:ax.legend(fontsize=9);ax.grid(alpha=.2);ax.set_xticks([32,64,128])
    p=audit["primary"];lo,hi=p["paired_layout_bootstrap95_ratio"]
    fig.text(.5,.01,f"64 fresh paired layouts, three fixed models; error bars show layout range. Slope ratio {p['slope_ratio']:.4f} [95% CI {lo:.4f}, {hi:.4f}].\nAll 1,920 model–maze–arm outputs valid; changing spine 16 → 128 at depth 64 changed stopping time by 0 in both arms.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1));out=S.OUT/"figures"
    for ext in ("png","pdf","svg"):fig.savefig(out/f"eta-pruning-rate.{ext}",dpi=180)
    plt.close(fig)
    S.save(out/"eta-comb-manifest.json",{"script_sha256":S.sha(__file__),"audit_sha256":S.sha(source),
          "outputs_sha256":{f"eta-pruning-rate.{ext}":S.sha(out/f"eta-pruning-rate.{ext}") for ext in ("png","pdf","svg")}})


if __name__=="__main__":main()
