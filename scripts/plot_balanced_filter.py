"""Separate exact-support recovery from the guaranteed filtering checks."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
source='results/research/ripper/balanced_filter.json'
data=json.loads((ROOT/source).read_text());assert data['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'text.color':'#17243a','axes.labelcolor':'#273449','xtick.color':'#273449','ytick.color':'#273449',
    'axes.edgecolor':'#b7c1cf','svg.hashsalt':'balanced-filter-v1'})
fig,axes=plt.subplots(1,3,figsize=(15,7.5))
groups={g['arm']:g for g in data['grouped']};labels=['Unfiltered','Balance filter'];colors=['#a44330','#347e72']
for index,arm in enumerate(['unfiltered','filtered']):
    g=groups[arm];rows=[r for r in data['rows'] if r['arm']==arm]
    for ax,key in [(axes[0],'missing_true_edges'),(axes[1],'extra_edges')]:
        values=np.array([r[key] for r in rows]);ax.bar(index,values.mean(),width=.6,color=colors[index],alpha=.85,zorder=3)
        ax.scatter(index+np.linspace(-.13,.13,32),values,color='#17243a',s=15,alpha=.6,zorder=4)
        ax.text(index,values.max()+(.35 if key=='extra_edges' else 22),f'{g[key]:,} total',ha='center',weight='bold')
    axes[2].bar(index,g['exact_graph_banks'],width=.6,color=colors[index],alpha=.85,zorder=3)
    axes[2].text(index,g['exact_graph_banks']+.8,f'{g["exact_graph_banks"]}/32',ha='center',weight='bold')
for ax in axes:
    ax.set_xticks([0,1],labels);ax.grid(axis='y',alpha=.15)
axes[0].set(ylim=(0,900),ylabel='Missing true edges per bank (of 810)')
axes[0].set_title('True-edge retention restored',loc='left',weight='bold',pad=14)
axes[1].set(ylim=(0,8.5),ylabel='Retained extra edges per bank')
axes[1].set_title('Finite residual uncertainty',loc='left',weight='bold',pad=14)
axes[2].axhline(29,color='#ba7e25',ls='--',lw=1.5)
axes[2].text(.95,29.7,'Frozen target: at least 29',ha='right',color='#986819',transform=axes[2].get_yaxis_transform(),fontsize=9)
axes[2].set(ylim=(0,33),ylabel='Banks with exact graph')
axes[2].set_title('Exact-recovery target falsified',loc='left',weight='bold',pad=14)
fig.suptitle('Perfect corruption filtering does not guarantee exact graph recovery',x=.065,ha='left',fontsize=19,weight='bold')
fig.text(.065,.035,
    '32 fresh paired source banks; M1000 / N1110; 25 training queries per board. One shared calibration fit per bank; 57,500 physical queries per pair.\n'
    'Supplied global prior: at most nine copies of any digit per completed board. Reject a queried row if one digit occurs more than nine times among 25 labels.\n'
    'All 1,961 collapsed rows rejected; no clean rows rejected; zero accepted label errors. Every filtered trajectory matches its clean accepted subsequence.\n'
    'Filtered graphs retain all true edges but have 54 extras; only 9/32 are exact, below the frozen 29/32 target. Calibration remains undercovered in every bank.\n'
    f'No solver or GPU work. {data["runtime"]["aggregate_cpu_s"]/60:.2f} CPU-min; {data["runtime"]["wall_s"]:.2f}s wall. BP AdamW outside this source-only study; not run.',
    color='#526075',fontsize=9.3,linespacing=1.5)
fig.subplots_adjust(left=.065,right=.98,bottom=.30,top=.82,wspace=.32)
for extension in ('png','svg'):
    fig.savefig(ROOT/f'assets/figures/balanced-filter.{extension}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/balanced-filter-provenance.json').write_text(json.dumps({
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},
    'matplotlib':matplotlib.__version__,'numpy':np.__version__,
    'scope':'H1an finite-bank exact recovery, missing true edges and retained extras; supplied global prior and perfect filtering separated from failed empirical target.'},indent=2)+'\n')
