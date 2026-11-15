"""N600 recovery confirmation and descriptive curves from public results only."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/figures'
source='results/research/ripper/recovery.json';data=json.loads((ROOT/source).read_text())
assert data['status']=='validated' and data['primary']['banks']==64
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449',
    'xtick.color':'#273449','ytick.color':'#273449','axes.edgecolor':'#b7c1cf',
    'svg.hashsalt':'hopfield-recovery-v1'})
fig,axes=plt.subplots(1,2,figsize=(12.3,6.4),gridspec_kw={'width_ratios':[1.2,1]})
arms=[('uniform','Uniform','#8997a9'),('candidate','Candidate','#447bb7'),('exponential','Exponential','#16887d')]
for arm,label,color in arms:
    rows=sorted([r for r in data['recovery_curve'] if r['arm']==arm],key=lambda r:r['n'])
    axes[0].plot([r['n'] for r in rows],[100*r['exact_graphs']/64 for r in rows],marker='o',color=color,label=label,lw=2)
axes[0].axhline(90,color='#ba7e25',ls='--',lw=1.3)
axes[0].text(990,87.5,'90% primary target',ha='right',va='top',color='#96651d',fontsize=10)
p=data['primary'];axes[0].plot([600,600],[100*p['one_sided95_lower'],100*p['observed_fraction']],color='#16887d',lw=3)
axes[0].plot([585,615],[100*p['one_sided95_lower']]*2,color='#16887d',lw=2)
axes[0].annotate(f'{p["exact_graphs"]}/64 at N600\n95% one-sided lower: {100*p["one_sided95_lower"]:.2f}%',
    xy=(600,100*p['one_sided95_lower']),xytext=(710,53),arrowprops={'arrowstyle':'-','color':'#526075'},fontsize=10,color='#116c63')
axes[0].set(xticks=[300,450,600,1000],ylim=(-3,104),xlabel='Incoming boards (25 queried digits each)',ylabel='Exact recovered graphs (%)')
axes[0].set_title('Primary N600 target remains inconclusive',loc='left',weight='bold',pad=14)
axes[0].legend(loc='upper left',frameon=False);axes[0].grid(alpha=.16)
for x,(arm,label,color) in enumerate(arms):
    r=next(r for r in data['grouped'] if r['arm']==arm);ys=100*np.array(r['solved_per_seed'])/r['n_per_seed']
    axes[1].bar(x,r['mean_accuracy_pct'],width=.65,color=color,zorder=3)
    axes[1].scatter(x+np.linspace(-.18,.18,64),ys,s=8,color='#17243a',alpha=.35,zorder=4,linewidth=0)
    axes[1].text(x,85,f'{r["mean_accuracy_pct"]:.2f}%',ha='center',color=color,weight='bold',fontsize=12)
axes[1].set(xticks=range(3),xticklabels=[a[1] for a in arms],ylabel='Solved test puzzles (%)',ylim=(0,92))
axes[1].set_title('N600 matched secondary accuracy',loc='left',weight='bold',pad=14);axes[1].grid(axis='y',alpha=.16)
fig.suptitle('Graph recovery with a fixed query budget',x=.07,ha='left',fontsize=20,weight='bold')
fig.text(.07,.025,
    '64 independent source-bank draws; no replacement within a bank. 63,454 distinct boards among 64,000 bank rows.\n'
    'Registered primary: exponential exact recovery probability ≥90% at N600. Required ≥62/64; observed 61/64.\n'
    'Other curve points and all policy comparisons are descriptive. Bar labels are means; dots are bank/search-seed results.\n'
    'Every seed uses the same 10,000 reserved test inputs; 192 cells deduplicate to 143 identical-model/seed/settings searches.\n'
    'Four chains, 262,144 moves/chain; zero wrong stops. BP AdamW is outside this acquisition study and was not run.',
    fontsize=9.2,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.09,right=.98,top=.82,bottom=.30,wspace=.34)
for ext in ('png','svg'):fig.savefig(OUT/f'graph-recovery-confirmation.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'graph-recovery-provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},'matplotlib':matplotlib.__version__,
    'numpy':np.__version__,'scope':'Audited public result only; primary inconclusive is explicitly retained.'},indent=2)+'\n')
