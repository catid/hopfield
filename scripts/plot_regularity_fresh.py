"""Public H4af graph recovery and all sixty-four selected count thresholds."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'results/research/foureyes/regularity_fresh/summary.json'
s=json.loads(source.read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'regularity-fresh-v1'})
fig,(left,right)=plt.subplots(1,2,figsize=(13.6,7.3),gridspec_kw={'width_ratios':[1,1.3]})
values=[next(row['exact_graphs'] for row in s['grouped'] if row['arm']==arm) for arm in ['adamw','canonical','restricted','regularity']]
left.bar(range(4),values,color=['#738196','#99a9b6','#bb694e','#347e72'],width=.65)
for x,value in enumerate(values):left.text(x,value+1.3,str(value),ha='center',weight='bold')
left.set(xticks=range(4),xticklabels=['BP AdamW','Canonical\nPL','Gap\ncounts','Equal-degree\ncounts'],ylim=(0,74),yticks=[0,16,32,48,64],ylabel='Exact graphs / 64 banks')
left.set_title('A   Identical newly corrupted observations',loc='left',pad=15,fontsize=12)
d=s['diagnostics'];x=np.arange(1,65)
lower=np.array([row['true_count_max'] for row in d]);upper=np.array([row['nonpeer_count_min'] for row in d])
right.fill_between(x,lower,upper,color='#347e72',alpha=.2,label='Separating threshold interval')
right.plot(x,lower,color='#347e72',lw=1);right.plot(x,upper,color='#347e72',lw=1)
right.scatter(x,[r['restricted_threshold'] for r in s['fit_audits']],color='#bb694e',marker='x',s=22,label='Original largest gap',zorder=3)
right.scatter(x,[r['selected']['cut'] for r in s['selection_records']],color='#347e72',s=14,label='Equal-degree rule',zorder=4)
right.set(xlim=(.6,64.4),ylim=(40,250),xticks=[1,16,32,48,64],xlabel='New source bank / independent cell profile',ylabel='Pair-equality count')
right.set_title('B   Equal degree finds every separating cut',loc='left',pad=15,fontsize=12)
right.legend(frameon=True,facecolor='white',edgecolor='none',fontsize=9,loc='center right',bbox_to_anchor=(.99,.43))
for ax in [left,right]:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Equal-degree recovery confirms on 64 new source banks',x=.065,ha='left',fontsize=18,weight='bold',y=.97)
fig.text(.065,.886,'Frozen target supported: 64/64 exact graphs, zero missing edges; original gap heuristic recovers 26/64.',fontsize=11.8,color='#27675c')
fig.subplots_adjust(left=.07,right=.97,bottom=.34,top=.77,wspace=.29)
fig.text(.065,.12,'64 independent banks, each with 2,400 distinct source rows; 150,618 unique identities, all disjoint from the H4y motivating sources.\nMinimize degree dispersion at count cuts no higher than the median; degree 20 and graph size 810 are not supplied.\nAll 9,581 candidates audited; the unique regular candidate is the true graph in every bank. No minimum-degree or density guard.\nEqual degree is a structural prior. Same official corpus and stated noise law, different training work; no new inference-accuracy claim.',fontsize=10,linespacing=1.6)
outputs={}
for ext in ['png','svg']:
    path=ROOT/f'assets/figures/regularity-fresh.{ext}'
    fig.savefig(path,dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    outputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
plt.close(fig)
receipt={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':{str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest()},'output_sha256':outputs,'matplotlib':matplotlib.__version__}
(ROOT/'assets/figures/regularity-fresh-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(outputs))
