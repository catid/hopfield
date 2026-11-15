"""Paired adaptive-acquisition effect with failed-attempt cost kept explicit."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/adaptive_null.json';s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'adaptive-null-v1'})
fig,ax=plt.subplots(figsize=(11.8,7.5));old=np.array([r['fixed_mask_extras']for r in s['rows']]);new=np.array([r['extra_edges']for r in s['rows']])
for a,b in zip(old,new):ax.plot([0,1],[a,b],color='#b7c1cf',lw=.8,alpha=.7,zorder=1)
for index,(values,color)in enumerate([(old,'#a47d36'),(new,'#347e72')]):
    ax.bar(index,values.mean(),width=.48,color=color,alpha=.82,zorder=2);ax.scatter(index+np.linspace(-.12,.12,32),values,color='#17243a',s=19,alpha=.65,zorder=3)
    ax.text(index,values.max()+12,f'{values.sum():,} extras across 32 banks',ha='center',weight='bold')
ax.axhline(7114/32,color='#ba7e25',ls='--',lw=1.4);ax.text(.98,7114/32+9,'Frozen pooled ceiling 7,114 (mean 222.3)',ha='right',color='#986819',transform=ax.get_yaxis_transform(),fontsize=10)
ax.set(ylim=(0,520),xlim=(-.5,1.5),xticks=[0,1],xticklabels=['Exact null on archived masks','Exact null drives new adaptive masks'],ylabel='Retained extra edges per bank')
ax.grid(axis='y',alpha=.15);ax.set_title('All true edges retained in 32/32 banks in both arms; exact graphs 0/32 in both',loc='left',weight='bold',pad=14)
fig.suptitle('Adapting acquisition cuts extra edges by 81.8% at the same training-query budget',x=.08,ha='left',fontsize=17,weight='bold')
rt=s['runtime'];fig.text(.08,.03,
    '32 reused H1ap source banks; identical potential event/permutation and row-indexed tie streams; supplied exact q0=9/1600, q1=1/9, bound 648000.\n'
    '25 queries per row, 1,110 rows per bank: 888,000 training queries charged. No new calibration, source board, solver input or GPU work.\n'
    '34,919 of 35,520 masks change when exact-law exclusions govern acquisition from the start. Every mask, latent draw, LR state and first crossing audited.\n'
    'Original attempt FAILED its CPU gate: trigger 120.004219 CPU-s / 15.055996 s wall; final original CPU unavailable. Eight completed acquisitions preserved.\n'
    f'Separate corrected continuation: {rt["additional_aggregate_cpu_s"]:.3f} additional CPU-s / {rt["additional_execution_wall_s"]:.3f} s execution wall including preflight; 8 acquisitions reused, 24 acquired once.\n'
    'Only eager NPZ materialization changed in the auditor; all 80 arrays and 42 original artifacts verified. BP AdamW outside this source-only study; not run.',
    fontsize=9.0,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.97,bottom=.34,top=.80)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/adaptive-null.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/adaptive-null-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,numpy=np.__version__,scope=s['scope'],timing_scope='Failed original trigger is separate from measured corrected additional-budget continuation; final original CPU unavailable.'),indent=2)+'\n')
