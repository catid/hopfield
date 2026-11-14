"""Selected-case geometry/readout separation using only the corrected experiment."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as S


def main():
    path=S.OUT/"coupling_swap_corrected/audit.json";audit=json.loads(path.read_text());lookup={(r["case"],r["arm"]):sum(r["success_by_seed"]) for r in audit["summaries"]}
    original=lookup[("original","original_geometry_original_couplings")];fig,axes=plt.subplots(1,3,figsize=(11.5,4))
    for ax,case,title in zip(axes,["targeted_branch_A","targeted_branch_B","matched_noncritical_control"],["Targeted branch A","Targeted branch B","Matched noncritical deletion"]):
        values=np.array([[original,lookup[(case,"original_geometry_edited_couplings")]],
                         [lookup[(case,"edited_geometry_original_couplings")],lookup[(case,"edited_geometry_edited_couplings")]]])
        ax.imshow(values,vmin=0,vmax=3,cmap="viridis")
        for row in range(2):
            for col in range(2):ax.text(col,row,f"{values[row,col]}/3",ha="center",va="center",color="white" if values[row,col]<2 else "black",fontsize=16)
        ax.set(xticks=[0,1],xticklabels=["Original readout","Edited readout"],yticks=[0,1],yticklabels=["Original graph","Edited graph"],title=title)
        ax.tick_params(axis="both",length=0,labelsize=8)
    fig.text(.5,.01,"Correct shortest paths among three fixed models on one selected graph (η=0.5). Edited readout restores original geometry for both targeted edits.\nCorrected support masks preserve valid self coefficients; all 12 source cases pass exact native-logit identity transfer. Flawed first attempt excluded.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.13,1,1));out=S.OUT/"figures"
    for ext in ("png","pdf","svg"):fig.savefig(out/f"coupling-geometry-separation.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    S.save(out/"coupling-swap-manifest.json",{"script_sha256":S.sha(__file__),"audit_sha256":S.sha(path),
        "outputs_sha256":{f"coupling-geometry-separation.{ext}":S.sha(out/f"coupling-geometry-separation.{ext}") for ext in ("png","pdf","svg")}})


if __name__=="__main__":main()
