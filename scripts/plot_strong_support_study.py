"""H4aa certificate restoration and its failed prospective gain threshold."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE='results/research/foureyes/strong_support_study/summary.json'
PARENT='results/research/foureyes/canonical_confirm/summary.json'
s=json.loads((ROOT/SOURCE).read_text());parent=json.loads((ROOT/PARENT).read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'strong-support-study-v1','svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(12.8,6.8));groups={g['arm']:g for g in parent['grouped']}
axes[0].bar([0,3],[groups['adamw']['accuracy_pct'],groups['restricted']['accuracy_pct']],color='#8a94a3',label='Archived separate fit: total correct')
for i,g in enumerate(s['grouped'],1):
    cert=g['correct_certified']/3200;cap=g['correct_capped']/3200
    axes[0].bar(i,cert,color='#287f77',label='Correct certificate'if i==1 else None)
    axes[0].bar(i,cap,bottom=cert,color='#9fc4df',label='Correct capped return'if i==1 else None)
for i,v in enumerate([groups['adamw']['accuracy_pct']]+[g['accuracy_pct']for g in s['grouped']]+[groups['restricted']['accuracy_pct']]):axes[0].text(i,v+1.5,f'{v:.2f}%',ha='center',fontsize=10)
axes[0].set(xticks=[0,1,2,3],xticklabels=['BP AdamW','Weak 1/256','Delete weak','Counts'],ylim=(0,95),ylabel='Exact-match accuracy (%)',title='Deletion restores valid certificates')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.06,.23),ncol=3,fontsize=9,frameon=False)
e=s['primary'];lo,hi=e['gain95_interval_pp'];gain=e['gain_pp']
axes[1].errorbar([gain],[0],xerr=[[gain-lo],[hi-gain]],fmt='o',color='#287f77',capsize=6,markersize=9)
axes[1].axvline(5,color='#b34b3d',linestyle='--',label='Frozen +5-point target');axes[1].axvline(0,color='#8a94a3',linewidth=.8)
axes[1].text(gain,.23,f'+{gain:.3f} points\n95% CI [{lo:.3f}, {hi:.3f}]',ha='center')
axes[1].set(xlim=(-.3,6),ylim=(-.5,.65),yticks=[],xlabel='Delete weak minus weak 1/256 (points)',title='Primary target: falsified');axes[1].legend(loc='lower right',fontsize=10,frameon=False)
for ax in axes:ax.grid(axis='y'if ax is axes[0]else'x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Removing weak constraints helps, but misses the target',x=.065,ha='left',fontsize=18,weight='bold')
fig.text(.065,.065,'Same 64 fits and 5,000 seen inputs; strong weights stay 1. Four chains, 262,144 moves per chain.\n'
         'Canonical targets compatible with zero energy: 0% → 84.75%. No wrong certificates observed; this search alone does not prove soundness.\n'
         '5,000 paired bank/input bootstrap draws. 59 new searches, 41.306 seconds; 64 archived baselines.\n'
         'BP AdamW/counts are separately fitted controls. Original audit failure and stop-index correction retained; no inference rerun.',fontsize=10,color='#4d5967',linespacing=1.5)
fig.subplots_adjust(left=.065,right=.98,bottom=.32,top=.84,wspace=.28)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/strong-support-study.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
(ROOT/'assets/figures/strong-support-study-provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in [SOURCE,PARENT]}},indent=2)+'\n')
