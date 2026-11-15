"""Audited equal-query-budget acquisition study; all arms cost57500queries."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/figures'
source='results/research/ripper/equal_budget.json';data=json.loads((ROOT/source).read_text());assert data['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'text.color':'#17243a','axes.labelcolor':'#273449','xtick.color':'#273449','ytick.color':'#273449',
    'axes.edgecolor':'#b7c1cf','svg.hashsalt':'equal-query-budget-v1'})
fig,axes=plt.subplots(1,2,figsize=(13.5,6.1),gridspec_kw={'width_ratios':[1.25,1]})
arms=[('exact','Exact 1%\nN2300','#16887d'),('fixed_upper','Fixed 2%\nN2300','#9a7f4d'),('learned_small','M1000\nN2220','#447bb7'),('learned_large','M10000\nN1500','#8e70ae')]
for x,(arm,label,color) in enumerate(arms):
    r=next(r for r in data['grouped'] if r['arm']==arm);ys=np.array(r['solved_per_seed'])/100
    axes[0].bar(x,r['mean_accuracy_pct'],width=.65,color=color,zorder=3)
    axes[0].scatter(x+np.linspace(-.17,.17,32),ys,s=9,color='#17243a',alpha=.4,zorder=4,linewidth=0)
    axes[0].text(x,float(ys.max())+2,f'{r["mean_accuracy_pct"]:.2f}%',ha='center',color=color,weight='bold',fontsize=12)
axes[0].set(xticks=range(4),xticklabels=[a[1] for a in arms],ylim=(0,94),ylabel='Solved test puzzles (%)',xlabel='M = calibration pairs; N = training boards')
axes[0].set_title('All arms receive 57,500 noisy queries',loc='left',weight='bold',pad=14);axes[0].grid(axis='y',alpha=.16)
p=data['primary'];lo,hi=p['paired_two_way_bootstrap_ci95_pp'];mean=p['effect_pp'];margin=p['noninferiority_margin_pp']
color='#16887d' if p['decision'].startswith('supported') else '#a44330' if p['decision'].startswith('falsified') else '#9a7f4d'
left=min(lo,-margin)-2;right=max(hi,0)+2
axes[1].axvline(-margin,ymin=.47,ymax=.80,color='#ba7e25',ls='--',lw=1.5)
axes[1].text(-margin+.2,.68,'−5 pp NI margin',color='#96651d',fontsize=10)
axes[1].plot([lo,hi],[.43,.43],color=color,lw=3);axes[1].scatter([mean],[.43],color=color,s=75,zorder=3)
axes[1].plot([lo,lo],[.39,.47],color=color,lw=2);axes[1].plot([hi,hi],[.39,.47],color=color,lw=2)
verdict='Supported' if p['decision'].startswith('supported') else 'Falsified' if p['decision'].startswith('falsified') else 'Inconclusive'
axes[1].text(left+.2,.86,f'{verdict}: {mean:+.3f} pp',color=color,weight='bold',fontsize=14)
axes[1].text(left+.2,.18,f'Paired 95% interval: [{lo:.2f}, {hi:.2f}] pp',color='#526075',fontsize=10)
axes[1].text(left+.2,-.04,'18,000 fewer calibration queries fund\n720 more training boards.\n'+f'Wrong certified stops in learned arms: {p["protected_wrong_stops"]}.',color='#526075',fontsize=10,linespacing=1.6,va='top')
axes[1].set(xlim=(left,right),ylim=(-.46,1.05),yticks=[],xlabel='Accuracy change: M1000/N2220 − M10000/N1500 (pp)')
axes[1].set_title('Frozen 5 pp noninferiority test',loc='left',weight='bold',pad=14);axes[1].grid(axis='x',alpha=.16)
fig.suptitle('Spending the same noisy-query budget',x=.06,ha='left',fontsize=20,weight='bold')
fig.text(.06,.025,
    '32 fresh banks. Actual independent replacement noise η = 1%; 25 training queries per board; fixed q₁ = 1/9.\n'
    'Learned-small: 2,000 calibration + 55,500 training queries. Learned-large: 20,000 + 37,500. Controls: 0 + 57,500.\n'
    'Nested calibration pairs and training prefixes; estimator receives only its own disagreement indicators. All outcomes retained.\n'
    f'All seeds share 10,000 reserved test inputs; dots are bank/search outcomes. 128 cells reduce to {data["unique_searches"]} unique searches.\n'
    'Guarantees require the declared noise family. Invalid capped returns are separate. Controls are descriptive. BP AdamW was outside scope.',
    fontsize=9.2,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.07,right=.98,top=.80,bottom=.31,wspace=.34)
for ext in ('png','svg'):fig.savefig(OUT/f'equal-query-budget.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'equal-query-budget-provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},'matplotlib':matplotlib.__version__,
    'numpy':np.__version__,'scope':'Audited H1ab equal57500query results, nested calibration and training; frozen NI plus wrong-stop gates.'},indent=2)+'\n')
