"""Frozen conservative-calibration noninferiority result, public data only."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/figures'
source='results/research/ripper/noise_bound.json';data=json.loads((ROOT/source).read_text())
assert data['status']=='validated' and data['primary']['decision']=='falsified_10pp_noninferiority'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'text.color':'#17243a','axes.labelcolor':'#273449','xtick.color':'#273449','ytick.color':'#273449',
    'axes.edgecolor':'#b7c1cf','svg.hashsalt':'noise-bound-v1'})
fig,axes=plt.subplots(1,2,figsize=(12.3,5.9),gridspec_kw={'width_ratios':[1.1,1]})
arms=[('exact','1% exact','#16887d'),('upper','2% upper\nbound','#447bb7'),('under','0.5% underbound\n(no guarantee)','#b58a44')]
for x,(arm,label,color) in enumerate(arms):
    r=next(r for r in data['grouped'] if r['arm']==arm);ys=np.array(r['solved_per_seed'])/100
    axes[0].bar(x,r['mean_accuracy_pct'],width=.63,color=color,zorder=3)
    axes[0].scatter(x+np.linspace(-.17,.17,32),ys,s=9,color='#17243a',alpha=.4,zorder=4,linewidth=0)
    axes[0].text(x,float(ys.max())+2,f'{r["mean_accuracy_pct"]:.2f}%',ha='center',color=color,weight='bold',fontsize=12)
axes[0].set(xticks=range(3),xticklabels=[a[1] for a in arms],ylim=(0,94),ylabel='Solved test puzzles (%)',xlabel='Assumed noise rate (actual rate is 1%)')
axes[0].set_title('Same queries and fixed search budget',loc='left',weight='bold',pad=14);axes[0].grid(axis='y',alpha=.16)
p=data['primary'];lo,hi=p['paired_two_way_bootstrap_ci95_pp'];mean=p['effect_pp']
axes[1].axvline(-10,color='#ba7e25',ls='--',lw=1.5)
axes[1].text(-9.5,.70,'−10 pp NI margin',color='#96651d',fontsize=10)
axes[1].plot([lo,hi],[.45,.45],color='#a44330',lw=3);axes[1].scatter([mean],[.45],color='#a44330',s=75,zorder=3)
axes[1].plot([lo,lo],[.41,.49],color='#a44330',lw=2);axes[1].plot([hi,hi],[.41,.49],color='#a44330',lw=2)
axes[1].text(-29,.86,f'NI target falsified: {mean:.3f} pp',color='#a44330',weight='bold',fontsize=14)
axes[1].text(-29,.20,f'Paired 95% interval: [{lo:.2f}, {hi:.2f}] pp',color='#526075',fontsize=10)
axes[1].text(-29,-.06,'Zero wrong certified stops in both protected arms.\nAll true edges retained in 32/32 protected models.\nExtra edges: 160 with the upper bound; 17 with exact η.',color='#526075',fontsize=10,linespacing=1.6,va='top')
axes[1].set(xlim=(-30,1),ylim=(-.45,1.05),yticks=[],xlabel='Accuracy change: assumed 2% minus exact 1% (pp)')
axes[1].set_title('Protection does not ensure noninferior accuracy',loc='left',weight='bold',pad=14);axes[1].grid(axis='x',alpha=.16)
fig.suptitle('The accuracy cost of conservative noise calibration',x=.06,ha='left',fontsize=20,weight='bold')
fig.text(.06,.025,
    '32 fresh banks × 1,500 boards × 25 queried noisy digits. Actual independent noise η = 1%; fixed q₁ = 1/9.\n'
    'N1500 and the 10 pp margin were fixed before the implementation pilot. The earlier exact-calibration result is unchanged.\n'
    'All seeds share the same 10,000 reserved test inputs; dots are bank/search outcomes. 96 cells reduce to 77 unique searches.\n'
    'The 0.5% underbound lost one true edge and produced no wrong stops here; these observations do not give it protection.\n'
    'Invalid capped returns are separate from wrong certified stops. Four chains, 262,144 moves/chain. BP AdamW was outside scope.',
    fontsize=9.2,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.31,wspace=.37)
for ext in ('png','svg'):fig.savefig(OUT/f'conservative-noise-calibration.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'noise-bound-provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},'matplotlib':matplotlib.__version__,
    'numpy':np.__version__,'scope':'Audited public conservative-calibration result; failed NI and passed stop condition are separate.'},indent=2)+'\n')
