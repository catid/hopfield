"""Every valid shortest answer versus retained invalid low-energy states."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as S
from export_landscape import manifest


def main():
    out=S.ROOT/"results/research/work";data=json.loads((out/"shortest-path-energy-summary.json").read_text());rows=data["rows"]
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.4),gridspec_kw={"width_ratios":[1.7,1]},constrained_layout=True)
    ax=axes[0]
    for r in rows:
        y=r["seed"];width=r["max_valid_energy"]-r["min_valid_energy"]
        ax.barh(y,width,left=0,height=.32,color="#227c9d",alpha=.75,label="All valid shortest paths (min–max)" if y==0 else None)
        ax.scatter(r["endpoint_minus_min_valid"],y,color="#c23b23",marker="X",s=115,zorder=4,label="Disconnected endpoints" if y==0 else None)
        ax.scatter(r["retained_minus_min_valid"],y,color="#b36b00",marker="D",s=52,zorder=5,label="Native η=.5 answer" if y==0 else None)
        ax.annotate(f"{r['endpoint_minus_min_valid']:,.1f}",(r["endpoint_minus_min_valid"],y),xytext=(0,13),textcoords="offset points",ha="center",color="#9a301c",fontsize=10)
        ax.text(360,y,"valid" if y==0 else "branched",va="center",fontsize=10,color="#825300")
    ax.axvline(0,color="black",lw=.8,ls="--");ax.set_xlim(-1780,610);ax.set_ylim(-.65,2.65);ax.invert_yaxis()
    ax.set_yticks([0,1,2],["Model 0","Model 1","Model 2"]);ax.set_xlabel("Energy − minimum valid shortest-path energy")
    ax.set_title("Invalid endpoints beat every shortest path",loc="left",fontweight="bold")
    ax.legend(loc="lower left",bbox_to_anchor=(0,-.37),frameon=False,fontsize=10,ncol=1)
    ax.grid(axis="x",alpha=.15)
    with (out/"shortest-path-energy-histogram.csv").open() as f:hist=list(csv.DictReader(f))
    ax=axes[1]
    for seed,color in [(0,"#227c9d"),(1,"#8d5a97"),(2,"#508a3b")]:
        values=[v for v in hist if int(v["seed"])==seed]
        edges=[float(values[0]["delta_energy_left"])]+[float(v["delta_energy_right"]) for v in values]
        ax.stairs([int(v["paths"]) for v in values],edges,label=f"Model {seed}",color=color,lw=1.4)
    ax.set_xlabel("Energy − minimum valid energy");ax.set_ylabel("Shortest paths per energy bin");ax.set_title("Complete valid-answer distribution",loc="left",fontweight="bold");ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.suptitle("Selected cyclic failure: exhaustive energy comparison",fontsize=16,fontweight="bold")
    fig.supxlabel("352,212 verified shortest paths × 3 frozen models; retained coefficients evaluated in float64.\nOne selected graph; longer paths and other binary states are outside this enumeration.",fontsize=10)
    for suffix in ["png","pdf","svg"]:fig.savefig(out/f"shortest-path-energy-landscape.{suffix}",dpi=180)
    manifest(out)


if __name__=="__main__":main()
