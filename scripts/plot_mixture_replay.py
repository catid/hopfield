"""Paired finite-bank failure of the frozen H1at likelihood mixture."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
source='results/research/ripper/mixture_replay.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'mixture-replay-v1'})
fig,ax=plt.subplots(figsize=(11.8,7.3))
arrays=[np.array([r['extra_edges'] for r in s['rows'] if r['arm']==g['arm']]) for g in s['grouped']]
for old,new in zip(*arrays):ax.plot([0,1],[old,new],color='#b7c1cf',lw=.8,alpha=.8,zorder=1)
for index,(values,color) in enumerate(zip(arrays,['#347e72','#b65c49'])):
    ax.bar(index,values.mean(),width=.48,color=color,alpha=.82,zorder=2)
    ax.scatter(index+np.linspace(-.12,.12,32),values,color='#17243a',s=19,alpha=.65,zorder=3)
    ax.text(index,values.max()+8,f'{values.sum():,} extras across 32 banks',ha='center',weight='bold')
ax.axhline(1941/32,color='#ba7e25',ls='--',lw=1.4)
ax.text(.98,1941/32-7,'Frozen mixture ceiling: 1,941 pooled extras',ha='right',color='#986819',transform=ax.get_yaxis_transform(),fontsize=10)
ax.set(ylim=(0,185),xlim=(-.5,1.5),xticks=[0,1],xticklabels=['Original q1 = 1/9','Uniform mixture: 1/36, 1/18, 1/9, 2/9'],ylabel='Retained extra edges per bank')
ax.grid(axis='y',alpha=.15)
ax.set_title('All true edges retained in 32/32 banks; exact graphs 0/32 in both arms',loc='left',weight='bold',pad=14)
fig.suptitle('Likelihood mixture hypothesis falsified: 57.0% more residual false edges',x=.08,ha='left',fontsize=17,weight='bold')
rt=s['runtime'];fig.text(.08,.035,
    '32 reused H1as banks; masks and labels fixed. Supplied q0 = 9/1600, boundary 648000, equal component weights 1/4.\n'
    'Components remain unstopped; first crossing of their mean permanently removes a pair. The mixture adds 1,476 retained pairs and removes no new pair.\n'
    'All 103,680 original crossing paths replay exactly; independent integer / Decimal boundaries and every cumulative mixture crossing audited.\n'
    '888,000 historical training queries represented; zero new queries, calibration fits, source rows, solver inputs or GPU work.\n'
    f'Execution: {rt["aggregate_cpu_s"]:.3f} CPU-seconds / {rt["wall_s"]:.3f} seconds wall. Finite-bank post-hoc target; no fresh-source claim.\n'
    'BP AdamW is outside this source-only study; not run.',fontsize=9.0,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.97,bottom=.34,top=.80)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/mixture-replay.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/mixture-replay-provenance.json').write_text(json.dumps(dict(
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    inputs_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},
    matplotlib=matplotlib.__version__,numpy=np.__version__,scope=s['scope']),indent=2)+'\n')
