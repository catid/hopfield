"""Public-only figure: clue-only recovery and the registered mask randomization."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'assets/figures'
sources = ['results/research/ripper/clues.json','results/research/ripper/masks.json',
           'results/research/ripper/masks_draws.csv']
clues,masks = [json.loads((ROOT/p).read_text()) for p in sources[:2]]
assert clues['status']==masks['status']=='validated'
joint = np.zeros(1024,np.int64)
with (ROOT/sources[2]).open() as f:
    for r in csv.DictReader(f):
        if int(r['n'])==300: joint[int(r['draw'])] += int(r['extra_edges'])
assert joint.mean()==masks['primary']['simulated_mean_total']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449',
    'xtick.color':'#273449','ytick.color':'#273449','axes.edgecolor':'#b7c1cf',
    'svg.hashsalt':'hopfield-clue-mask-v1'})
fig,axes = plt.subplots(1,2,figsize=(12,5.7))
ns = [100,300,1000,3000]
for arm,label,color in [('full','Complete solutions','#447bb7'),('clues','Given clues only','#16887d')]:
    vals = np.array([[next(r['extra_edges'] for r in clues['structure']
        if r['arm']==arm and r['seed']==s and r['n']==n) for n in ns] for s in range(6)])
    axes[0].plot(ns,vals.mean(0),color=color,lw=2,marker='o',label=label)
    for s in range(6):
        axes[0].scatter(np.array(ns)*np.linspace(.95,1.05,6)[s],vals[s],s=16,color=color,alpha=.55)
axes[0].set(xscale='log',yscale='symlog',xlabel='Training puzzles in the fixed bank',
            ylabel='False extra edges',ylim=(-.3,1500),xticks=ns,xticklabels=[str(n) for n in ns])
axes[0].set_yticks([0,1,10,100,1000],['0','1','10','100','1,000'])
axes[0].set_title('Hidden training digits are unnecessary',loc='left',weight='bold',pad=14)
axes[0].legend(frameon=False,fontsize=10,loc='upper right')
axes[0].text(.05,.17,'All six clue banks exact at 1,000\nand at the primary 3,000-puzzle endpoint',
             transform=axes[0].transAxes,color='#116c63',fontsize=10.5,linespacing=1.5,
             bbox={'facecolor':'white','edgecolor':'none','alpha':.92,'pad':2})
axes[0].grid(alpha=.15)
axes[1].hist(joint,bins=np.arange(340,506,6),color='#89bdb5',edgecolor='white')
observed = masks['primary']['official_total']
axes[1].axvline(observed,color='#bd6c38',lw=2,linestyle='--')
axes[1].text(observed+3,100,'Official masks\n440 extra edges',color='#a45c2e',fontsize=11,ha='left')
axes[1].text(.05,.89,f"One-sided p = {masks['primary']['one_sided_finite_simulation_p']:.3f}\nOfficial-mask advantage not supported",
             transform=axes[1].transAxes,fontsize=10.5,linespacing=1.5,
             bbox={'facecolor':'white','edgecolor':'none','alpha':.92,'pad':2})
axes[1].set(xlabel='Extra edges summed over six N=300 banks',ylabel='Joint randomization draws',ylim=(0,145))
axes[1].set_title('Uniform masks provide a matched control',loc='left',weight='bold',pad=14)
axes[1].grid(axis='y',alpha=.15)
fig.suptitle('Learning constraints from incomplete observations',x=.065,ha='left',fontsize=21,weight='bold')
fig.text(.065,.025,
    'Left: six raw banks; only jointly given equal digits remove an edge. No hidden-label calibration or geometry input.\n'
    'Right: 1,024 fixed joint draws; complete boards and per-board clue counts held constant, cell masks randomized.\n'
    'At 3,000 puzzles, all 6,144 randomized banks recover the exact graph. This is conditional mask randomization,\n'
    'not uniform Sudoku generation. The registered one-sided test does not establish equivalence or a reverse effect.',
    fontsize=9.5,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.065,right=.98,top=.80,bottom=.28,wspace=.30)
for ext in ('png','svg'):
    fig.savefig(OUT/f'clue-mask-results.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'clue-mask-provenance.json').write_text(json.dumps({
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
    'matplotlib':matplotlib.__version__,'numpy':np.__version__,'scope':'Tracked public results only.'},indent=2)+'\n')
