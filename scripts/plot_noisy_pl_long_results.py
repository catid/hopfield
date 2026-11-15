"""Audited H4s likelihood/count comparison; public JSON only."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
source='results/research/foureyes/noisy_pl_long/summary.json'
data=json.loads((ROOT/source).read_text())
assert data['status']=='validated' and data['primary']['decision']=='falsified'
out=ROOT/'assets/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
 'text.color':'#17243a','axes.labelcolor':'#273449','xtick.color':'#273449','ytick.color':'#273449','svg.hashsalt':'noisy-pl-long-v1'})
arms=['adamw','paper_pl','long_pl','restricted','chernoff9']
labels=['AdamW\n32 updates','Paper SGD\n32 updates','Paper SGD\n256 updates','Restricted\ncounts','Known-noise\nChernoff']
colors=['#447bb7','#c7806c','#a44330','#16887d','#8b68a8']
fig,axes=plt.subplots(1,2,figsize=(13.6,6.2),gridspec_kw={'width_ratios':[1.15,1]})
for i,arm in enumerate(arms):
    row=next(r for r in data['grouped'] if r['arm']==arm)
    scores=[r['accuracy_pct'] for r in data['logical_bank_outcomes'] if r['arm']==arm]
    axes[0].bar(i,row['mean_accuracy_pct'],color=colors[i],width=.65,zorder=2)
    axes[0].scatter(i+np.linspace(-.17,.17,64),scores,s=11,color='#17243a',alpha=.4,zorder=3)
    axes[0].text(i,86,f"{row['mean_accuracy_pct']:.2f}%",ha='center',color=colors[i],weight='bold')
axes[0].set(xticks=range(5),xticklabels=labels,ylim=(0,100),ylabel='Returned-board accuracy (%)')
axes[0].set_title('Fixed noisy data and common search',loc='left',weight='bold',pad=16)
axes[0].grid(axis='y',alpha=.15)
struct={r['arm']:r for r in data['source_audit']['grouped']}
x=np.arange(5)
axes[1].bar(x-.17,[struct[a]['exact_banks'] for a in arms],width=.32,color='#447bb7',label='Exact graph',zorder=2)
axes[1].bar(x+.17,[struct[a]['all_true_retained'] for a in arms],width=.32,color='#16887d',label='Every true edge retained',zorder=2)
for i,a in enumerate(arms):
    for offset,key in [(-.17,'exact_banks'),(.17,'all_true_retained')]:
        y=struct[a][key];axes[1].text(i+offset,y+.7,str(y),ha='center',fontsize=10)
axes[1].set(xticks=x,xticklabels=labels,ylim=(0,78),yticks=[0,16,32,48,64],ylabel='Training banks (of 64)')
axes[1].set_title('More updates retain fewer true edges',loc='left',weight='bold',pad=16)
axes[1].legend(frameon=False,fontsize=9,loc='upper left',ncol=1)
axes[1].grid(axis='y',alpha=.15)
fig.suptitle('Eight times as many updates do not repair the noisy-label gap',x=.065,ha='left',fontsize=20,weight='bold')
fig.text(.065,.205,'256-update recipe vs counts: −61.23 pp [−69.36, −52.56]; frozen 2 pp NI target falsified.',fontsize=12,weight='bold',color='#a44330')
fig.text(.065,.055,
 '64 fresh banks × 600 boards; nine wrong free digits per board. 10,000 shared research puzzles, fixed search seed 0.\n'
 'Four chains × 262,144 moves; unit weights, zero biases and stopping threshold. 320 logical cells → 121 unique searches.\n'
 'Wrong stops / 640,000: AdamW 838; SGD32 420,000; SGD256 490,000; both count arms 0.\n'
 'AdamW is a fixed untuned control with different regularization. Dots show banks; the interval resamples banks and puzzles.',
 fontsize=9.3,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.075,right=.98,top=.82,bottom=.36,wspace=.25)
for extension in ['png','svg']:
    fig.savefig(out/f'noisy-pl-long.{extension}',dpi=190,bbox_inches='tight',facecolor='white',metadata={'Date':None} if extension=='svg' else None)
plt.close(fig)
(out/'noisy-pl-long-provenance.json').write_text(json.dumps(dict(renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 inputs_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,numpy=np.__version__),indent=2)+'\n')
