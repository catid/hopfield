"""Fresh noisy acquisition, accuracy and measured false certificates."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/figures'
source='results/research/ripper/noisy.json';data=json.loads((ROOT/source).read_text())
assert data['status']=='validated' and data['primary']['bank_seeds']==32
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'text.color':'#17243a','axes.labelcolor':'#273449','xtick.color':'#273449','ytick.color':'#273449',
    'axes.edgecolor':'#b7c1cf','svg.hashsalt':'noisy-acquisition-v1'})
arms=[('uniform_e','Uniform +\ne-process','#8997a9'),('exponential_e','Exponential +\ne-process','#16887d'),
      ('exponential_naive','Exponential +\nfirst equality','#c66a50')]
fig,axes=plt.subplots(1,2,figsize=(12.6,6.8),gridspec_kw={'width_ratios':[1.05,1]})
for x,(arm,label,color) in enumerate(arms):
    r=next(r for r in data['grouped'] if r['arm']==arm);ys=100*np.array(r['solved_per_seed'])/10000
    axes[0].bar(x,r['mean_accuracy_pct'],width=.64,color=color,zorder=3)
    axes[0].scatter(x+np.linspace(-.17,.17,32),ys,s=10,color='#17243a',alpha=.45,zorder=4,linewidth=0)
    axes[0].text(x,max(float(ys.max()),1)+2,f'{r["mean_accuracy_pct"]:.2f}%',ha='center',color=color,weight='bold',fontsize=12)
p=data['primary'];lo,hi=p['paired_two_way_bootstrap_ci95_pp']
axes[0].text(.01,.98,f'Primary gain: +{p["effect_pp"]:.2f} pp',transform=axes[0].transAxes,va='top',color='#116c63',weight='bold',fontsize=14)
axes[0].text(.01,.91,f'Paired 95% interval: {lo:.2f} to {hi:.2f} pp',transform=axes[0].transAxes,va='top',color='#526075',fontsize=10)
axes[0].set(xticks=range(3),xticklabels=[a[1] for a in arms],ylim=(0,105),ylabel='Solved test puzzles (%)')
axes[0].set_title('Fresh matched test accuracy',loc='left',weight='bold',pad=14);axes[0].grid(axis='y',alpha=.16)
for y,(arm,label,color) in zip([2,1,0],arms):
    r=next(r for r in data['structural_groups'] if r['arm']==arm);missing=r['total_missing_true_edges']/32
    axes[1].barh(y,810-missing,height=.50,color=color,zorder=3)
    axes[1].barh(y,missing,left=810-missing,height=.50,color='#e6a293',zorder=3)
    g=next(g for g in data['grouped'] if g['arm']==arm)
    invalid_capped=sum(x['budget_failures'] for x in data['rows'] if x['arm']==arm)-sum(x['correct_uncertified_returns'] for x in data['oracle_diagnostic'] if x['arm']==arm)
    axes[1].text(0,y+.33,f'Wrong certified stops: {g["wrong_stops"]:,}  |  Invalid capped returns: {invalid_capped:,}',fontsize=8.4,color='#a44330' if g['wrong_stops'] else '#526075')
    if missing:axes[1].text(810-missing/2,y,f'{missing:.1f}\nlost',ha='center',va='center',color='#6e261b',weight='bold')
    else:axes[1].text(405,y,'All 810 true edges retained',ha='center',va='center',color='white',weight='bold',fontsize=10)
axes[1].set(yticks=[0,1,2],yticklabels=[a[1] for a in reversed(arms)],ylim=(-.5,2.8),xlim=(0,835),
    xlabel='Mean true edges retained / lost per learned model')
axes[1].set_title('Measured damage from first-equality pruning',loc='left',weight='bold',pad=14)
axes[1].grid(axis='x',alpha=.16)
fig.suptitle('Active acquisition with noisy query labels',x=.06,ha='left',fontsize=21,weight='bold')
fig.text(.06,.025,
    '32 fresh source-bank/noise/tie/search seeds; 1,500 boards × 25 queried digits per bank. Exact known noise rate η = 1%; fixed q₁ = 1/9.\n'
    'Complete stored boards simulate the oracle; policies see only queried noisy labels. Every mask precedes current labels.\n'
    'The same 10,000 test inputs repeat across all 32 seeds. Bars show means; dots show seed outcomes; wrong stops and invalid capped returns are distinct.\n'
    'Ville + union bounds protect true-edge deletion at 1% per e-process model under the specified law, not solve accuracy.\n'
    'E-process extras: uniform 291 total, exponential 12 total. Four chains, 262,144 moves/chain. BP AdamW was outside scope.',
    fontsize=9.2,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.08,right=.98,top=.82,bottom=.30,wspace=.56)
for ext in ('png','svg'):fig.savefig(OUT/f'noisy-adaptive-acquisition.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'noisy-acquisition-provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},'matplotlib':matplotlib.__version__,
    'numpy':np.__version__,'scope':'Audited public noisy-acquisition data; repeated input count and guarantee scope explicit.'},indent=2)+'\n')
