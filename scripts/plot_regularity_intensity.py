"""Public H4ag paired corruption-intensity boundary, no archive inputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'results/research/foureyes/regularity_intensity'
s=json.loads((PUBLIC/'summary.json').read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'h4ag-intensity'})
fig,(ax,bx)=plt.subplots(1,2,figsize=(13,6.5));x=np.arange(4)
conditions=['hot18_weight8','hot18_weight32'];colors=['#257d77','#c55b4d']
for k,c in enumerate(conditions):
 values=[next(r['exact_graphs'] for r in s['grouped'] if r['condition']==c and r['arm']==arm) for arm in ['adamw','canonical','restricted','regularity']]
 ax.bar(x+(k-.5)*.35,values,width=.35,color=colors[k],label=f'Hot weight {8 if k==0 else 32}')
 for pos,value in zip(x+(k-.5)*.35,values):ax.text(pos,value+1,str(value),ha='center',fontsize=10)
 margins=sorted(r['separation_margin'] for r in s['diagnostics'] if r['condition']==c)
 bx.plot(range(1,65),margins,color=colors[k],lw=2,label=f'Hot weight {8 if k==0 else 32}')
ax.set(xticks=x,xticklabels=['BP AdamW\n32 updates','Canonical PL\n256 updates','Restricted\ncount gap','Equal-degree\nthreshold'],ylabel='Exact graphs / 64 paired banks',ylim=(0,72))
ax.legend(loc='upper left',frameon=False);ax.set_title('The stronger-corruption target fails',loc='left',fontweight='bold')
bx.axhline(0,color='#738196',lw=1,ls='--');bx.set(xlabel='Banks sorted separately by margin',ylabel='Minimum nonedge count − maximum true-edge count')
bx.set_title('60/64 strong-noise banks cannot be thresholded',loc='left',fontweight='bold',fontsize=11)
bx.text(.03,.05,'A positive margin permits an exact threshold.\nNonpositive margins make every threshold fail.',transform=bx.transAxes,fontsize=10,bbox={'facecolor':'white','edgecolor':'#ddd','pad':6})
fig.suptitle('Concentrating the same nine corruptions breaks count separation',x=.06,ha='left',fontsize=17,fontweight='bold')
fig.text(.06,.06,'Same 64 clean banks × 2,400 boards; new paired heatmaps, race clocks, digit offsets and augmentation draws.\nExactly nine free cells change per board. Hot cells: 18; cold weight: 1. No new sources or puzzle-solving evaluation.\nPrimary: weight32 ≥61/64 exact and zero missing edges — falsified. Paired weight8: 64/64 exact — supported.\nEqual degree is supplied; its value is not. Finite, different-work fitting controls; no optimizer-superiority claim.',fontsize=10,linespacing=1.5,color='#415062')
fig.subplots_adjust(left=.07,right=.98,bottom=.30,top=.83,wspace=.36)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/regularity-intensity.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'Matplotlib'})
plt.close(fig)
files=['scripts/plot_regularity_intensity.py','results/research/foureyes/regularity_intensity/summary.json','assets/figures/regularity-intensity.png','assets/figures/regularity-intensity.svg']
(ROOT/'assets/figures/regularity-intensity-provenance.json').write_text(json.dumps({'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
