"""Plot audited count-energy control and Sudoku budget curve from public exports."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/figures'
sources = ['results/research/foureyes/count_budget/summary.json',
           'results/research/foureyes/budget/summary.json',
           'results/research/ripper/finaltest.json']
count, learned, short = [json.loads((ROOT / name).read_text()) for name in sources]
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12, 'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.titleweight':'bold', 'text.color':'#17243a',
                     'axes.labelcolor':'#273449', 'xtick.color':'#273449', 'ytick.color':'#273449',
                     'axes.edgecolor':'#b7c1cf', 'svg.hashsalt':'hopfield-count-v1'})
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={'width_ratios':[1,1.25]})
original = next(r for r in short['grouped'] if r['k']==10 and r['arm']=='original_general')
values = [original['mean_accuracy_pct'], learned['grouped'][0]['mean_accuracy_pct'], count['grouped'][0]['mean_accuracy_pct']]
axes[0].bar(np.arange(3), values, color=['#8997a9','#447bb7','#16887d'], width=.65, zorder=3)
axes[0].set(ylim=(0,100), ylabel='Solved puzzles (%)', xticks=np.arange(3),
            xticklabels=['Original\nCRAM','Transformed\nlearned','Count\nenergy'])
axes[0].set_title('At 262,144 moves / chain', loc='left', fontsize=14, pad=13)
for i,v in enumerate(values):
    axes[0].text(i,v+2,f'{v:.2f}%',ha='center',weight='bold',fontsize=12)
axes[0].grid(axis='y',alpha=.17,zorder=0)
for d,label,color,marker in [(learned,'Transformed learned','#447bb7','o'),(count,'No-optimizer counts','#16887d','s')]:
    xs=[r['cap'] for r in d['grouped']]
    ys=[10000-np.mean(r['per_seed_solved']) for r in d['grouped']]
    axes[1].plot(xs,ys,marker=marker,color=color,lw=2,label=label)
axes[1].set(xscale='log',yscale='log',xlabel='Moves per chain',ylabel='Unsolved puzzles per 10,000\n(mean across six search seeds)',ylim=(2,3000))
axes[1].set_title('Longer budget, same fixed energies',loc='left',fontsize=14,pad=13)
axes[1].legend(frameon=False,fontsize=10)
axes[1].grid(which='major',alpha=.17)
last=count['grouped'][-1]
axes[1].text(.05,.12,f"At 16,777,216 moves:\n{last['mean_accuracy_pct']:.4f}% solved", transform=axes[1].transAxes,
             weight='bold',fontsize=12,color='#16887d',linespacing=1.6)
fig.suptitle('A simple count energy matches the learned model',x=.07,ha='left',fontsize=21,weight='bold')
fig.text(.07,.015,'10,000 reserved official-test puzzles · four chains · six search seeds · zero count-energy wrong stops\nTraining counts: 10 base solutions × 100 stored variants. Spatial augmentation supplies task knowledge.\n16,777,216-move comparison: count minus learned +0.015 pp; frozen 0.1 pp noninferiority margin passed.',fontsize=10,color='#526075')
fig.subplots_adjust(left=.07,right=.98,top=.79,bottom=.24,wspace=.36)
OUT.mkdir(parents=True,exist_ok=True)
for ext in ['png','svg']:
    fig.savefig(OUT/f'count-energy.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'count-provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
    'matplotlib':matplotlib.__version__,'numpy':np.__version__},indent=2)+'\n')
