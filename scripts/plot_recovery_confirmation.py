"""Separate 64-bank and fresh 256-bank recovery results; no pooling."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/figures'
sources=['results/research/ripper/recovery.json','results/research/ripper/recovery_confirm.json']
old,new=[json.loads((ROOT/s).read_text()) for s in sources]
assert old['primary']['decision']=='inconclusive' and new['primary']['decision']=='supported_at_least_90percent_recovery'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449',
    'xtick.color':'#273449','ytick.color':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'recovery-confirmation-v1'})
fig,axes=plt.subplots(1,2,figsize=(12.5,5.7),gridspec_kw={'width_ratios':[1.25,1]})
axes[0].axvline(90,color='#ba7e25',lw=1.5,ls='--')
axes[0].text(90.2,1.55,'90% target',color='#96651d',fontsize=10)
for y,data,color in [(1,old,'#8997a9'),(0,new,'#16887d')]:
    p=data['primary'];mean=100*p['observed_fraction'];lower=100*p['one_sided95_lower']
    axes[0].plot([lower,mean],[y,y],color=color,lw=3);axes[0].plot([lower,lower],[y-.08,y+.08],color=color,lw=2)
    axes[0].scatter([mean],[y],s=70,color=color,zorder=3)
    axes[0].text(82,y+.22,f'{p["exact_graphs"]}/{p["banks"]} exact  •  lower bound {lower:.2f}%',color=color,fontsize=11,weight='bold')
axes[0].set(xlim=(81,100),ylim=(-.4,1.8),yticks=[0,1],yticklabels=['Fresh 256 banks','Earlier 64 banks'],xlabel='Exact graph recovery at N600 (%)')
axes[0].set_title('Same fixed rule and target; separate studies',loc='left',weight='bold',pad=14);axes[0].grid(axis='x',alpha=.16)
arms=[('uniform','Uniform','#8997a9'),('candidate','Candidate','#447bb7'),('exponential','Exponential','#16887d')]
for x,(arm,label,color) in enumerate(arms):
    r=next(r for r in new['structural_controls'] if r['arm']==arm);value=100*r['exact_graphs']/256
    axes[1].bar(x,value,width=.64,color=color,zorder=3)
    axes[1].text(x,value+3,f'{r["exact_graphs"]}/256',ha='center',color=color,weight='bold')
axes[1].set(xticks=range(3),xticklabels=[a[1] for a in arms],ylim=(0,108),ylabel='Exact graphs (%)')
axes[1].set_title('Fresh structural controls (descriptive)',loc='left',weight='bold',pad=14);axes[1].grid(axis='y',alpha=.16)
fig.suptitle('Independent confirmation of N600 graph recovery',x=.06,ha='left',fontsize=20,weight='bold')
fig.text(.06,.025,
    'Each bank supplies 600 boards × 25 queried digits. Exponential decay 8/9 and the ≥90% primary target remain fixed.\n'
    'Left: dots show observed fractions; line ends show exact one-sided 95% Clopper–Pearson lower bounds.\n'
    'The earlier 64-bank result remains inconclusive. The fresh 256-bank result is supported; results are not pooled.\n'
    'Fresh study: 150,504 distinct source boards among 153,600 bank rows; independent draws allow cross-bank overlap.\n'
    'Construction only: no new solver runs or test-accuracy observations. BP AdamW is outside this study and was not run.',
    fontsize=9.2,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.145,right=.98,top=.80,bottom=.31,wspace=.43)
for ext in ('png','svg'):fig.savefig(OUT/f'graph-recovery-independent-confirmation.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'graph-recovery-independent-provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},'matplotlib':matplotlib.__version__,
    'numpy':np.__version__,'scope':'Separate audited public studies, without pooling or new accuracy evidence.'},indent=2)+'\n')
