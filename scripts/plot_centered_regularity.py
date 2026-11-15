"""Public H4ai count-centering outcomes and remaining separation failures."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'results/research/foureyes/centered_regularity/summary.json').read_text())
old=json.loads((ROOT/'results/research/foureyes/regularity_intensity/summary.json').read_text())
lp=json.loads((ROOT/'results/research/foureyes/intensity_degree/summary.json').read_text())
c='hot18_weight32';part=[r for r in s['diagnostics'] if r['condition']==c]
values=[next(r['exact_graphs'] for r in old['grouped'] if r['condition']==c and r['arm']=='adamw'),next(r['exact_graphs'] for r in old['grouped'] if r['condition']==c and r['arm']=='regularity')]+[next(r['exact'] for r in s['grouped'] if r['condition']==c and r['arm']==arm) for arm in ['centered_regularity','centered_top810']]+[next(r['success'] for r in lp['grouped'] if r['condition']==c)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'h4ai-centering'})
fig,(ax,bx)=plt.subplots(1,2,figsize=(13,6.5))
ax.barh(range(5),values,color=['#8b98a9','#8b98a9','#337c98','#c28845','#197c73']);ax.invert_yaxis();ax.set_yticks(range(5),['BP AdamW32 (archived)','Raw regularity (archived)','Centered regularity','Centered lowest810','Degree20 LP (archived)']);ax.set(xlim=(0,72),xlabel='Exact graphs / 64 strong-noise banks')
for i,v in enumerate(values):ax.text(v+1,i,str(v),va='center',fontweight='bold')
ax.set_title('Centering helps; the frozen target still fails',loc='left',fontsize=12,fontweight='bold')
bx.plot(range(1,65),sorted(r['raw_separation_margin'] for r in part),label='Raw count margin',color='#8b98a9',lw=2)
bx.plot(range(1,65),sorted(r['centered_separation_margin']/79 for r in part),label='Centered margin ÷79',color='#337c98',lw=2)
bx.axhline(0,color='#777',ls='--',lw=1);bx.set(xlabel='Banks sorted separately by margin',ylabel='Minimum nonedge − maximum true-edge score')
bx.legend(loc='lower right',frameon=False);bx.set_title('31/64 centered banks remain inseparable',loc='left',fontsize=12,fontweight='bold')
fig.suptitle('Removing additive cell effects improves recovery without supplying degree20',x=.04,ha='left',fontsize=15.5,fontweight='bold')
fig.text(.04,.045,'Same128 archived count matrices; no new observations. Centered score:79cᵢⱼ−sᵢ−sⱼ, shifted by one common constant.\nThe primary keeps the unchanged equal-degree threshold rule, without degree20, edge count810 or a density guard.\nWeight32:33/64 exact and25,079 missing — target falsified. Weight8:64/64 exact. All182,384 candidate cuts audited.\nCentered top810 and archived degree20 LP supply additional structure; these are separate controls, not equivalent assumptions.',fontsize=10,linespacing=1.5,color='#415062')
fig.subplots_adjust(left=.20,right=.98,top=.82,bottom=.29,wspace=.40)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/centered-regularity.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'Matplotlib'})
plt.close(fig)
files=['scripts/plot_centered_regularity.py','results/research/foureyes/centered_regularity/summary.json','results/research/foureyes/regularity_intensity/summary.json','results/research/foureyes/intensity_degree/summary.json','assets/figures/centered-regularity.png','assets/figures/centered-regularity.svg']
(ROOT/'assets/figures/centered-regularity-provenance.json').write_text(json.dumps({'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
