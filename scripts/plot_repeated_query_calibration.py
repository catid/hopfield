"""Audited repeated-query calibration effect and charged oracle cost."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/figures'
source='results/research/ripper/calibrated.json';data=json.loads((ROOT/source).read_text());assert data['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'text.color':'#17243a','axes.labelcolor':'#273449','xtick.color':'#273449','ytick.color':'#273449',
    'axes.edgecolor':'#b7c1cf','svg.hashsalt':'repeated-query-calibration-v1'})
fig,axes=plt.subplots(1,2,figsize=(12.3,6.1),gridspec_kw={'width_ratios':[1.05,1]})
arms=[('exact','Exact 1%\n37,500 queries','#16887d'),('learned','Learned bound\n57,500 queries','#447bb7'),('fixed_upper','Fixed 2%\n37,500 queries','#9a7f4d')]
for x,(arm,label,color) in enumerate(arms):
    r=next(r for r in data['grouped'] if r['arm']==arm);ys=np.array(r['solved_per_seed'])/100
    axes[0].bar(x,r['mean_accuracy_pct'],width=.63,color=color,zorder=3)
    axes[0].scatter(x+np.linspace(-.17,.17,32),ys,s=9,color='#17243a',alpha=.4,zorder=4,linewidth=0)
    axes[0].text(x,float(ys.max())+2,f'{r["mean_accuracy_pct"]:.2f}%',ha='center',color=color,weight='bold',fontsize=12)
axes[0].set(xticks=range(3),xticklabels=[a[1] for a in arms],ylim=(0,94),ylabel='Solved test puzzles (%)',xlabel='Queries include calibration and training')
axes[0].set_title('Calibration receives no clean labels',loc='left',weight='bold',pad=14);axes[0].grid(axis='y',alpha=.16)
p=data['primary'];lo,hi=p['paired_two_way_bootstrap_ci95_pp'];mean=p['effect_pp'];margin=p['noninferiority_margin_pp']
color='#16887d' if p['decision'].startswith('supported') else '#a44330' if p['decision'].startswith('falsified') else '#9a7f4d'
left=min(lo,-margin)-2;right=max(hi,0)+2
axes[1].axvline(-margin,ymin=.35,ymax=.80,color='#ba7e25',ls='--',lw=1.5)
axes[1].text(-margin+.2,.68,'−5 pp NI margin',color='#96651d',fontsize=10)
axes[1].plot([lo,hi],[.43,.43],color=color,lw=3);axes[1].scatter([mean],[.43],color=color,s=75,zorder=3)
axes[1].plot([lo,lo],[.39,.47],color=color,lw=2);axes[1].plot([hi,hi],[.39,.47],color=color,lw=2)
verdict='Supported' if p['decision'].startswith('supported') else 'Falsified' if p['decision'].startswith('falsified') else 'Inconclusive'
axes[1].text(left+.2,.86,f'{verdict}: {mean:.3f} pp',color=color,weight='bold',fontsize=14)
axes[1].text(left+.2,.18,f'Paired 95% interval: [{lo:.2f}, {hi:.2f}] pp',color='#526075',fontsize=10)
bounds=[c['eta_upper']*100 for c in data['calibration_results']]
axes[1].text(left+.2,-.04,f'Learned bounds: {min(bounds):.4f}%–{max(bounds):.4f}%.\nCoverage diagnostic: {data["calibration_coverage_banks"]}/32; fallback: {data["calibration_fallback_banks"]}.\nWrong certified stops in primary arms: {p["protected_wrong_stops"]}.',color='#526075',fontsize=10,linespacing=1.6,va='top')
axes[1].set(xlim=(left,right),ylim=(-.46,1.05),yticks=[],xlabel='Accuracy change: learned minus exact 1% (pp)')
axes[1].set_title('Frozen 5 pp noninferiority test',loc='left',weight='bold',pad=14);axes[1].grid(axis='x',alpha=.16)
fig.suptitle('Learning a protected noise bound from repeated queries',x=.06,ha='left',fontsize=20,weight='bold')
fig.text(.06,.025,
    '32 fresh banks. Actual independent replacement noise η = 1%; 10,000 calibration pairs; N1500 × 25 training queries.\n'
    'Learned arm pays 20,000 extra calibration queries. All arms use edge error budget .005, fixed q₁ = 1/9 and the same search.\n'
    'The learned per-model deletion guarantee also budgets .005 for calibration failure; it requires the declared noise family.\n'
    f'All seeds share 10,000 reserved test inputs; dots are bank/search outcomes. 96 cells reduce to {data["unique_searches"]} unique searches.\n'
    'Invalid capped returns are separate from wrong certified stops. Fixed-2% is descriptive. BP AdamW was outside scope.',
    fontsize=9.2,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.31,wspace=.37)
for ext in ('png','svg'):fig.savefig(OUT/f'repeated-query-noise-calibration.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'repeated-query-calibration-provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},'matplotlib':matplotlib.__version__,
    'numpy':np.__version__,'scope':'Audited H1aa public result, charged calibration queries, frozen NI and stop gates.'},indent=2)+'\n')
