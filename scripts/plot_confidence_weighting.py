"""Audited canonical accuracy, certification decomposition and frozen endpoints."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/figures'
source='results/research/ripper/confidence_weight.json';data=json.loads((ROOT/source).read_text());assert data['status']=='validated';p=data['primary']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'text.color':'#17243a','axes.labelcolor':'#273449','xtick.color':'#273449','ytick.color':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'confidence-weight-v1'})
fig,axes=plt.subplots(2,2,figsize=(14,10.2));axes=axes.ravel()
for x,arm in enumerate(['uniform','evidence','shuffled']):
    g=next(g for g in data['grouped'] if g['arm']==arm);ys=np.array(g['solved_per_seed'])/100
    axes[0].bar(x,g['correct_certified_pct'],width=.65,color='#16887d',label='Certified' if x==0 else None,zorder=3)
    axes[0].bar(x,g['correct_capped_pct'],bottom=g['correct_certified_pct'],width=.65,color='#d39a45',label='Capped return' if x==0 else None,zorder=3)
    axes[0].scatter(x+np.linspace(-.16,.16,32),ys,color='#17243a',alpha=.45,s=10,linewidth=0,zorder=4)
    axes[0].text(x,float(ys.max())+2,f'{g["mean_accuracy_pct"]:.2f}%',ha='center',weight='bold')
axes[0].set(xticks=[0,1,2],xticklabels=['Uniform','Evidence','Shuffled'],ylim=(0,100),ylabel='Canonical exact-match accuracy (%)')
axes[0].set_title('One learned fit, three weight assignments',loc='left',weight='bold',pad=12)
axes[0].legend(loc='upper left',frameon=False,ncol=2,fontsize=9);axes[0].grid(axis='y',alpha=.16)

def interval(ax,mean,lo,hi,bound,decision,title,xlabel,left,right):
    color='#16887d' if decision.startswith('supported') else '#a44330' if decision.startswith('falsified') else '#9a7f4d'
    verdict='Supported' if decision.startswith('supported') else 'Falsified' if decision.startswith('falsified') else 'Inconclusive'
    ax.axvline(bound,ymin=.23,ymax=.66,color='#ba7e25',ls='--',lw=1.5)
    ax.plot([lo,hi],[.42,.42],color=color,lw=3);ax.scatter([mean],[.42],color=color,s=65,zorder=3)
    ax.plot([lo,lo],[.38,.46],color=color,lw=2);ax.plot([hi,hi],[.38,.46],color=color,lw=2)
    ax.text(.02,.85,verdict,transform=ax.transAxes,color=color,weight='bold',fontsize=15)
    ax.text(.02,.71,f'Mean {mean:.2f}; interval [{lo:.2f}, {hi:.2f}]',transform=ax.transAxes,color='#526075',fontsize=10)
    ax.set(xlim=(left,right),ylim=(0,1),yticks=[],xlabel=xlabel);ax.set_title(title,loc='left',weight='bold',pad=12);ax.grid(axis='x',alpha=.16)
for ax,c in zip(axes[1:3],p['gain_endpoints']):
    lo,hi=c['paired_two_way_ci95_pp'];ref=c['reference']
    interval(ax,c['effect_pp'],lo,hi,10,c['decision'],f'Evidence versus {ref}: required +10 pp',f'Evidence minus {ref} accuracy (pp)',min(lo,0)-3,max(hi,10)+3)
lo,hi=p['evidence_accuracy_ci95_pct']
interval(axes[3],p['mean_evidence_accuracy_pct'],lo,hi,70,p['usefulness_decision'],'Evidence accuracy: required 70%','Canonical exact-match accuracy (%)',0,100)
fig.suptitle('Confidence weights and correct capped returns',x=.06,ha='left',fontsize=21,weight='bold')
fig.text(.06,.025,
    '32 fresh learned fits; M1000 / N2220; 57,500 queries per shared fit. Uniform 1; evidence 1 or 1/16; shuffled preserves the weight multiset.\n'
    f'{p["wrong_certified_stops"]} invalid certified stops; conjunction: {p["decision"].replace("_"," ")}. Strong penalty unit, temperature and focus threshold stay fixed.\n'
    'Three marginal nominal 95% intervals; no simultaneous-coverage claim. All seeds share 10,000 test inputs; uniqueness is not independently established.\n'
    f'Positive weights preserve the zero set; compatibility bounds the relevant certificate outcome, not capped-return accuracy. BP AdamW was outside scope.\n'
    f'{data["unique_searches"]} unique searches; {data["total_allocated_gpu_s"]/60:.2f} GPU-min including preflights. All calibration outcomes retained; coverage {data["calibration_coverage_banks"]}/32.',
    fontsize=9.2,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.065,right=.98,top=.89,bottom=.21,hspace=.48,wspace=.23)
for ext in ('png','svg'):fig.savefig(OUT/f'confidence-weighted-inference.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'confidence-weighted-inference-provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},'matplotlib':matplotlib.__version__,'numpy':np.__version__,
    'scope':'H1ah: evidence-uniform and evidence-shuffled 10pp endpoints,70percent canonical accuracy floor,invalid-certificate gate; certified/capped decomposition.'},indent=2)+'\n')
