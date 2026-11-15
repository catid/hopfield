"""Compare the prior acquisition result with its separate fresh-bank confirmation."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/figures'
sources=['results/research/ripper/online.json','results/research/ripper/online_confirm.json']
prior,fresh=[json.loads((ROOT/p).read_text()) for p in sources]
assert prior['status']==fresh['status']=='validated' and fresh['prior_result_not_pooled']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449',
    'xtick.color':'#273449','ytick.color':'#273449','axes.edgecolor':'#b7c1cf',
    'svg.hashsalt':'hopfield-online-confirm-v1'})
fig,axes=plt.subplots(1,2,figsize=(12,5.7),gridspec_kw={'width_ratios':[1.15,1]})
for y,data,color in [(1,prior,'#8997a9'),(0,fresh,'#16887d')]:
    p=data['primary'];point=100*p['relative_reduction'];lo,hi=np.array(p['paired_bank_bootstrap_ci95'])*100
    axes[0].errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',ms=8,lw=2.5,capsize=5,color=color)
    axes[0].text(point,y+.19,f'{point:.1f}%  [{lo:.1f}, {hi:.1f}]',ha='center',color=color,weight='bold',fontsize=11,
                 bbox={'facecolor':'white','edgecolor':'none','alpha':.94,'pad':2})
axes[0].axvline(25,color='#bd6c38',ls='--',lw=1.8)
axes[0].text(25,-.45,'Registered 25% target',ha='center',color='#a45c2e',fontsize=10,
             bbox={'facecolor':'white','edgecolor':'none','alpha':.94,'pad':2})
axes[0].set(yticks=[0,1],yticklabels=['Fresh 32 banks','Initial six banks'],ylim=(-.65,1.6),
            xlabel='Reduction in false extra edges (%)',xlim=(0,65))
axes[0].set_title('Primary graph-coverage result',loc='left',weight='bold',pad=14)
axes[0].grid(axis='x',alpha=.16)
colors=['#447bb7','#df8a3e','#16887d'];maxpoint=0.
labels=['Uniform','Exposure\nbalancing','Candidate\nfocused']
for x,arm,color in zip(range(3),['uniform','balanced','candidate'],colors):
    r=next(r for r in fresh['grouped'] if r['arm']==arm)
    ys=100*np.array(r['solved_per_seed'])/r['n_per_seed'];maxpoint=max(maxpoint,float(ys.max()))
    axes[1].bar(x,r['mean_accuracy_pct'],color=color,width=.65,zorder=3)
    axes[1].scatter(x+np.linspace(-.18,.18,len(ys)),ys,s=11,color='#17243a',alpha=.45,zorder=4,linewidth=0)
    axes[1].text(x,float(ys.max())+.2,f'{r["mean_accuracy_pct"]:.2f}%',ha='center',color=color,weight='bold')
axes[1].set(xticks=range(3),xticklabels=labels,ylabel='Solved test puzzles (%)',ylim=(0,maxpoint+1.5))
axes[1].set_title('Secondary accuracy on fresh puzzles',loc='left',weight='bold',pad=14)
axes[1].grid(axis='y',alpha=.16,zorder=0)
fig.suptitle('Choosing observations before seeing their values',x=.07,ha='left',fontsize=21,weight='bold')
decision={'supported_at_least_25percent_reduction':'supports the registered 25% target',
          'falsified_25percent_target':'does not reach the registered 25% target','inconclusive':'remains inconclusive'}[fresh['primary']['decision']]
fig.text(.07,.025,
    f'The separate 32-bank confirmation {decision}; prior six-bank evidence is not pooled.\n'
    'Every policy queries 25 cells per board across 300 boards per bank. Current values are hidden until the mask is fixed.\n'
    'Left: paired bank-bootstrap 95% intervals. Right: 32 search seeds, 10,000 new test puzzles, four chains,\n'
    '262,144 moves per chain; bars/labels show means, dots show seeds. Accuracy is secondary; every returned board was audited.',
    fontsize=9.5,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.14,right=.98,top=.80,bottom=.28,wspace=.33)
for ext in ('png','svg'):
    fig.savefig(OUT/f'online-acquisition-confirmation.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'online-confirmation-provenance.json').write_text(json.dumps({
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
    'matplotlib':matplotlib.__version__,'numpy':np.__version__,'scope':'Tracked audited public results only.'},indent=2)+'\n')
