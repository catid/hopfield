"""Audited5%noise transfer: separate NI,usefulness andcertified-stop endpoints."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/figures'
source='results/research/ripper/scaled_noise.json';data=json.loads((ROOT/source).read_text());assert data['status']=='validated';p=data['primary']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'text.color':'#17243a','axes.labelcolor':'#273449','xtick.color':'#273449','ytick.color':'#273449',
    'axes.edgecolor':'#b7c1cf','svg.hashsalt':'scaled-noise-v1'})
fig,axes=plt.subplots(1,3,figsize=(16,6.2),gridspec_kw={'width_ratios':[1.35,1,1]})
arms=[('exact_short','N2220\nexact','#16887d'),('learned_short','N2220\nlearned','#447bb7'),('exact_long','N4440\nexact','#9a7f4d'),('learned_long','N4440\nlearned','#8e70ae')]
for x,(arm,label,color) in enumerate(arms):
    r=next(r for r in data['grouped'] if r['arm']==arm);ys=np.array(r['solved_per_seed'])/100
    axes[0].bar(x,r['mean_accuracy_pct'],width=.65,color=color,zorder=3)
    axes[0].scatter(x+np.linspace(-.17,.17,32),ys,s=9,color='#17243a',alpha=.4,zorder=4,linewidth=0)
    axes[0].text(x,float(ys.max())+2,f'{r["mean_accuracy_pct"]:.2f}%',ha='center',color=color,weight='bold',fontsize=11)
axes[0].set(xticks=range(4),xticklabels=[a[1] for a in arms],ylim=(0,95),ylabel='Solved test puzzles (%)',xlabel='Training boards / calibration')
axes[0].set_title('Fixed 5% replacement noise',loc='left',weight='bold',pad=14);axes[0].grid(axis='y',alpha=.16)

def interval(ax,mean,lo,hi,bound,decision,title,xlabel,left,right):
    color='#16887d' if decision.startswith('supported') else '#a44330' if decision.startswith('falsified') else '#9a7f4d'
    verdict='Supported' if decision.startswith('supported') else 'Falsified' if decision.startswith('falsified') else 'Inconclusive'
    ax.axvline(bound,ymin=.25,ymax=.70,color='#ba7e25',ls='--',lw=1.5)
    ax.plot([lo,hi],[.45,.45],color=color,lw=3);ax.scatter([mean],[.45],color=color,s=65,zorder=3)
    ax.plot([lo,lo],[.41,.49],color=color,lw=2);ax.plot([hi,hi],[.41,.49],color=color,lw=2)
    ax.text(.02,.85,verdict,transform=ax.transAxes,color=color,weight='bold',fontsize=15)
    ax.text(.02,.72,f'Mean {mean:.2f}; interval [{lo:.2f}, {hi:.2f}]',transform=ax.transAxes,color='#526075',fontsize=9.5)
    ax.set(xlim=(left,right),ylim=(0,1),yticks=[],xlabel=xlabel);ax.set_title(title,loc='left',weight='bold',pad=14);ax.grid(axis='x',alpha=.16)
lo,hi=p['paired_two_way_bootstrap_ci95_pp']
interval(axes[1],p['effect_pp'],lo,hi,-5,p['noninferiority_decision'],'N4440: NI margin −5 pp','Learned minus exact accuracy (pp)',min(lo,-5)-3,max(hi,0)+3)
lo,hi=p['learned_accuracy_ci95_pct']
interval(axes[2],p['mean_learned_accuracy_pct'],lo,hi,70,p['usefulness_decision'],'N4440: usefulness floor 70%','Learned accuracy (%)',0,100)
fig.suptitle('Doubling training under 5% replacement noise',x=.045,ha='left',fontsize=21,weight='bold')
fig.text(.045,.025,
    f'32 fresh banks; M1000 calibration; N4440 × 25 training queries. Long learned: 113,000 total queries; long exact: 111,000.\n'
    f'At N4440: {p["protected_wrong_stops"]} wrong certified stops in the two primary arms; conjunction: {p["decision"].replace("_"," ")}.\n'
    'NI and absolute-accuracy intervals each have marginal nominal 95% coverage; no simultaneous-coverage claim. N2220 prefixes are descriptive. Earlier H1ac remains falsified.\n'
    f'All seeds share 10,000 reserved test inputs. Calibration outcomes are retained: coverage {data["calibration_coverage_banks"]["learned_long"]}/32.\n'
    f'128 cells reduce to {data["unique_searches"]} unique searches. Independent replacement-noise family only. Invalid capped returns are separate. BP AdamW was outside scope.',
    fontsize=9.4,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.055,right=.98,top=.80,bottom=.31,wspace=.35)
for ext in ('png','svg'):fig.savefig(OUT/f'doubled-noisy-training.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'doubled-noisy-training-provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},'matplotlib':matplotlib.__version__,
    'numpy':np.__version__,'scope':'H1ad fixed5percent doubled-training study; NI,70percentfloor,zero wrong stops separately and conjunction.'},indent=2)+'\n')
