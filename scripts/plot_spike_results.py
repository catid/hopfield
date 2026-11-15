"""Compact completed spike study figure using audited public evidence only."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/figures'
sources=['results/research/ripper/spike.json']
data=json.loads((ROOT/sources[0]).read_text())
assert data['status']=='validated' and data['primary']['seeds']==32
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449',
    'xtick.color':'#273449','ytick.color':'#273449','axes.edgecolor':'#b7c1cf',
    'svg.hashsalt':'hopfield-spike-v1'})
fig,axes=plt.subplots(1,2,figsize=(12,5.9),gridspec_kw={'width_ratios':[1,1.15]})
p=data['primary'];totals=p['total_extra_edges'];colors=['#447bb7','#16887d']
for y,arm,color in zip([1,0],['candidate','spike'],colors):
    value=totals[arm];axes[0].barh(y,value,height=.46,color=color,zorder=3)
    axes[0].text(value+45,y,f'{value:,}',va='center',color=color,weight='bold',fontsize=13)
axes[0].set(yticks=[0,1],yticklabels=['Mixture score','Candidate policy'],ylim=(-.65,2.1),
    xlim=(0,2200),xlabel='Total false extra edges across 32 training banks')
lo,hi=np.array(p['paired_bank_bootstrap_ci95'])*100
axes[0].text(.02,.90,f'{100*p["relative_reduction"]:.1f}% fewer extra edges',
    transform=axes[0].transAxes,color='#116c63',weight='bold',fontsize=16)
axes[0].text(.02,.79,f'Paired 95% interval: {lo:.1f}% to {hi:.1f}%',
    transform=axes[0].transAxes,color='#526075',fontsize=11)
axes[0].set_title('Primary structural improvement',loc='left',weight='bold',pad=14)
axes[0].grid(axis='x',alpha=.16,zorder=0)
arms=[('uniform','Uniform','#8997a9'),('candidate','Candidate\npolicy','#447bb7'),('spike','Mixture\nscore','#16887d')]
maximum=0.
for x,(arm,label,color) in enumerate(arms):
    r=next(r for r in data['grouped'] if r['arm']==arm)
    ys=100*np.array(r['solved_per_seed'])/r['n_per_seed'];maximum=max(maximum,float(ys.max()))
    axes[1].bar(x,r['mean_accuracy_pct'],width=.65,color=color,zorder=3)
    axes[1].scatter(x+np.linspace(-.17,.17,32),ys,s=10,color='#17243a',alpha=.45,zorder=4,linewidth=0)
    axes[1].text(x,float(ys.max())+.65,f'{r["mean_accuracy_pct"]:.2f}%',ha='center',color=color,weight='bold',fontsize=12)
axes[1].set(xticks=range(3),xticklabels=[a[1] for a in arms],ylabel='Solved test puzzles (%)',ylim=(0,maximum+4.5))
axes[1].set_title('Matched secondary search accuracy',loc='left',weight='bold',pad=14)
axes[1].grid(axis='y',alpha=.16,zorder=0)
fig.suptitle('A fixed query score learns constraints more efficiently',x=.07,ha='left',fontsize=20,weight='bold')
fig.text(.07,.025,
    '32 disjoint source banks × 300 boards; each policy queries 25 digits per board before seeing current values.\n'
    'Candidate: inverse exposure. Mixture: fixed 1/2 impossible-pair prior and 1/9 possible-pair equality rate; no tuning.\n'
    'All 32 bank/search seeds use the same 10,000 reserved test puzzles in each arm; these are repetitions, not new inputs.\n'
    'Bars/labels show means on the right; dots show bank/seed results. Four chains, 262,144 moves/chain; zero wrong stops.',
    fontsize=9.4,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.14,right=.98,top=.80,bottom=.28,wspace=.40)
for ext in ('png','svg'):
    fig.savefig(OUT/f'spike-acquisition.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'spike-acquisition-provenance.json').write_text(json.dumps({
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
    'matplotlib':matplotlib.__version__,'numpy':np.__version__,'scope':'Audited tracked public result only.'},indent=2)+'\n')
