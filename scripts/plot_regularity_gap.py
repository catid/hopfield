"""Public H4ae graph recovery and all sixteen selected count thresholds."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'results/research/foureyes/regularity_gap/summary.json'
s=json.loads(source.read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'regularity-gap-v1'})
fig,(left,right)=plt.subplots(1,2,figsize=(13.6,7.3),gridspec_kw={'width_ratios':[1,1.3]})
values=[next(row['exact_graphs'] for row in s['grouped'] if row['arm']==arm) for arm in ['adamw','canonical','restricted','regularity']]
left.bar(range(4),values,color=['#738196','#99a9b6','#bb694e','#347e72'],width=.65)
for x,value in enumerate(values):left.text(x,value+.35,str(value),ha='center',weight='bold')
left.set(xticks=range(4),xticklabels=['BP AdamW','Canonical\nPL','Gap\ncounts','Equal-degree\ncounts'],ylim=(0,19),yticks=[0,4,8,12,16],ylabel='Exact graphs / 16 groups')
left.set_title('A   Identical newly corrupted observations',loc='left',pad=15,fontsize=12)
d=s['diagnostics'];x=np.arange(1,17)
lower=np.array([row['true_count_max'] for row in d]);upper=np.array([row['nonpeer_count_min'] for row in d])
right.fill_between(x,lower,upper,color='#347e72',alpha=.2,label='Separating threshold interval')
right.plot(x,lower,color='#347e72',lw=1);right.plot(x,upper,color='#347e72',lw=1)
right.scatter(x,[r['restricted_threshold'] for r in s['fit_audits']],color='#bb694e',marker='x',s=45,label='Original largest gap',zorder=3)
right.scatter(x,[r['selected']['cut'] for r in s['selection_records']],color='#347e72',s=32,label='Equal-degree rule',zorder=4)
right.set(xlim=(.6,16.4),ylim=(40,250),xticks=[1,4,8,12,16],xlabel='Source quartet / independent cell profile',ylabel='Pair-equality count')
right.set_title('B   Equal degree finds every separating cut',loc='left',pad=15,fontsize=12)
right.legend(frameon=True,facecolor='white',edgecolor='none',fontsize=9,loc='center right',bbox_to_anchor=(.99,.43))
for ax in [left,right]:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('An equal-degree prior repairs concentrated-noise threshold selection',x=.065,ha='left',fontsize=18,weight='bold',y=.97)
fig.text(.065,.886,'Frozen target supported: 16/16 exact graphs, zero missing edges; original gap heuristic recovers 6/16.',fontsize=11.8,color='#27675c')
fig.subplots_adjust(left=.07,right=.97,bottom=.34,top=.77,wspace=.29)
fig.text(.065,.12,'Same 64 clean banks in 16 quartets; 2,400 board appearances each. Sixteen new independent 18-cell profiles and new corruptions.\nMinimize degree dispersion at count cuts no higher than the median; degree 20 and graph size 810 are not supplied.\nAll 2,408 candidates audited; the unique regular candidate is the true graph in every group. No minimum-degree or density guard.\nEqual degree is a structural prior. Reused clean sources, different training work; no fresh-source or inference-accuracy claim.',fontsize=10,linespacing=1.6)
outputs={}
for ext in ['png','svg']:
    path=ROOT/f'assets/figures/regularity-gap.{ext}'
    fig.savefig(path,dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    outputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
plt.close(fig)
receipt={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':{str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest()},'output_sha256':outputs,'matplotlib':matplotlib.__version__}
(ROOT/'assets/figures/regularity-gap-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(outputs))
