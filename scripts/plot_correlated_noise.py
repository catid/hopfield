"""Finite-bank structure versus iid-based noise calibration under shared errors."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
source='results/research/ripper/correlated_noise.json'
data=json.loads((ROOT/source).read_text());assert data['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'text.color':'#17243a','axes.labelcolor':'#273449','xtick.color':'#273449','ytick.color':'#273449',
    'axes.edgecolor':'#b7c1cf','svg.hashsalt':'correlated-noise-v1'})
fig,axes=plt.subplots(1,2,figsize=(13.8,7.4))
groups={group['arm']:group for group in data['grouped']}
labels=['Independent errors','Shared derangement']
colors=['#4976a8','#16887d']
for index,arm in enumerate(['iid','shared']):
    group=groups[arm]
    axes[0].bar(index,group['extra_edges'],width=.6,color=colors[index],zorder=3)
    axes[0].text(index,group['extra_edges']*1.35,f'{group["extra_edges"]:,}',ha='center',weight='bold',fontsize=14)
    values=np.array([row['eta_upper']*100 for row in data['rows'] if row['arm']==arm])
    axes[1].scatter(index+np.linspace(-.15,.15,32),values,color=colors[index],s=24,alpha=.7,zorder=3)
    axes[1].plot([index-.20,index+.20],[values.mean(),values.mean()],color=colors[index],lw=3)
    axes[1].text(index,values.max()+.5,f'Coverage {group["calibration_coverage_banks"]}/32',ha='center',weight='bold')
target=groups['iid']['extra_edges']/10
axes[0].axhline(target,color='#ba7e25',ls='--',lw=1.5)
axes[0].text(.98,target*1.14,'Frozen target: at most 10% of iid',ha='right',color='#986819',transform=axes[0].get_yaxis_transform(),fontsize=9)
axes[0].set(yscale='log',ylim=(1,80000),xticks=[0,1],xticklabels=labels,ylabel='Retained extra edge-bank pairs (log scale)')
axes[0].set_title('Empirical structure: 99.84% fewer extras',loc='left',weight='bold',pad=14)
axes[0].grid(axis='y',alpha=.15)
axes[1].axhline(5,color='#a44330',ls='--',lw=1.5)
axes[1].text(.98,5.15,'Nominal marginal error: 5%',ha='right',color='#a44330',transform=axes[1].get_yaxis_transform(),fontsize=9)
axes[1].set(ylim=(0,10),xticks=[0,1],xticklabels=labels,ylabel='Iid-based 99.5% upper bound for digit error (%)')
axes[1].set_title('Predicted calibration failure under shared repeats',loc='left',weight='bold',pad=14)
axes[1].grid(axis='y',alpha=.15)
fig.suptitle('Shared digit permutations break calibration while preserving constraints',x=.065,ha='left',fontsize=19,weight='bold')
fig.text(.065,.035,
    '32 fresh paired source banks; M1000 / N1110; 29,750 queries per arm. Both noise laws have exactly 5% marginal digit error.\n'
    'One shared Bernoulli event and uniform derangement serve both calibration repeats or all queries on a training board.\n'
    'Zero true edges lost in either arm. All 35,520 shared clean/corrupted adaptive trajectory rows agree exactly; repeat disagreements are always zero.\n'
    'The extra-edge reduction is a finite-bank empirical result. Equality invariance and calibration undercoverage follow from the specified corruption law.\n'
    f'No solver cohort or GPU work. {data["runtime"]["aggregate_cpu_s"]/60:.2f} aggregate CPU-min; {data["runtime"]["wall_s"]:.2f}s wall. BP AdamW was outside scope.',
    color='#526075',fontsize=9.3,linespacing=1.5)
fig.subplots_adjust(left=.065,right=.98,bottom=.30,top=.82,wspace=.30)
for extension in ('png','svg'):
    fig.savefig(ROOT/f'assets/figures/correlated-noise.{extension}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/correlated-noise-provenance.json').write_text(json.dumps({
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},
    'matplotlib':matplotlib.__version__,'numpy':np.__version__,
    'scope':'H1al paired source-only finite-bank extra-edge counts versus mathematically predicted calibration failure; no solver accuracy.'},indent=2)+'\n')
