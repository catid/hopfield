"""Public-only H2bf timing ratios; finite paired repeats, no confidence intervals."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2bf-veto-speed'
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'results/research/work/veto-speed';a=json.loads((D/'audit.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,5.4))
for ax,part,key,target,title in [(axes[0],a['geometry'],'paired_process_speedup',2.,'Veto geometry: CPU process time'),(axes[1],a['policy'],'paired_wall_speedup',1.15,'Complete unified48 policy: wall time')]:
    values=[r[key] for r in part['rows']];median=part['median_paired_process_speedup' if key=='paired_process_speedup' else 'median_paired_wall_speedup'];ax.scatter(range(1,5),values,s=75,color='#137c8b',zorder=3);ax.axhline(median,color='#137c8b',label=f'Median {median:.3f}x');ax.axhline(target,color='#b26e38',linestyle='--',label=f'Target {target:g}x');ax.set_xticks(range(1,5),['old / fast','fast / old','fast / old','old / fast'],rotation=15);ax.set_xlabel('Execution order within each pair');ax.set_xlim(.5,4.5);ax.set_ylim(0,max(values)*1.18);ax.set_title(title,fontsize=12,loc='left');ax.set_ylabel('Legacy time / accelerated time');ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower right',frameon=False)
fig.suptitle('Exact veto geometry: separate local and complete-policy tests',x=.02,ha='left',fontsize=14);fig.text(.02,.07,'Left: 1,024 fixed states on CPU0. Right: 256 loop80 source mazes x 3 models; two paired repetitions on each GPU.',fontsize=9);fig.text(.02,.03,'Targets decided separately. All outputs and traces match. Warmup/load excluded and reported; no new accuracy samples or timing CI.',fontsize=9);fig.tight_layout(rect=[0,.14,1,.93]);fig.savefig(D/'veto-speed.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(D/'veto-speed.svg',metadata={'Date':None});plt.close(fig)
