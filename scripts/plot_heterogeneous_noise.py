"""H4ad paired profile outcome and postfit threshold diagnosis."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'results/research/foureyes/heterogeneous_noise/summary.json'
s=json.loads(source.read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'heterogeneous-noise-v1'})
fig,(left,right)=plt.subplots(1,2,figsize=(13.6,7.3),gridspec_kw={'width_ratios':[1,1.3]})
for offset,condition,color,label in [(-.18,'uniform','#738196','Uniform locations'),(.18,'hot18_weight8','#bb694e','18 hot cells, weight 8')]:
    values=[next(r['exact_graphs'] for r in s['grouped'] if r['condition']==condition and r['arm']==arm) for arm in ['adamw','canonical','restricted']]
    x=np.arange(3)+offset;left.bar(x,values,width=.34,color=color,label=label)
    for xx,value in zip(x,values):left.text(xx,value+.35,str(value),ha='center',weight='bold')
left.set(xticks=range(3),xticklabels=['BP AdamW','Canonical PL','Counts'],ylim=(0,19),yticks=[0,4,8,12,16],ylabel='Exact graphs / 16 groups')
left.set_title('A   Same nine changed free digits per board',loc='left',pad=15,fontsize=12)
left.legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.15))
d=[r for r in s['diagnostics'] if r['condition']=='hot18_weight8'];x=np.arange(1,17)
lower=np.array([r['true_count_max'] for r in d]);upper=np.array([r['nonpeer_count_min'] for r in d])
threshold=[r['restricted_threshold'] for r in s['fit_audits'] if r['condition']=='hot18_weight8']
right.fill_between(x,lower,upper,color='#347e72',alpha=.25,label='A separating threshold exists')
right.plot(x,lower,color='#347e72',lw=1);right.plot(x,upper,color='#347e72',lw=1)
right.scatter(x,threshold,color='#bb694e',s=35,zorder=3,label='Heuristic-selected threshold')
right.set(xlim=(.6,16.4),ylim=(40,245),xticks=[1,4,8,12,16],xlabel='Fixed source quartet',ylabel='Pair-equality count')
right.set_title('B   Seven choices fall below the separating gap',loc='left',pad=15,fontsize=12)
right.legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.15))
for ax in [left,right]:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Where the noise lands changes graph recovery',x=.065,ha='left',fontsize=19,weight='bold',y=.97)
fig.text(.065,.886,'Frozen count-robustness target FAILED: 9/16 exact graphs and 2,240 missing true edge-group pairs.',fontsize=11.8,color='#a04c35')
fig.subplots_adjust(left=.07,right=.97,bottom=.36,top=.77,wspace=.29)
fig.text(.065,.14,'Same 64 archived banks in 16 quartets; 2,400 board appearances per group, paired random clocks and digit offsets.\nAll observations still admit an exact count threshold. Seven failed count graphs isolate all 18 hot cells; invalid zero-energy witnesses are verified.\nSingle supplied corruption profile, unchanged estimators. Different training work; no fresh-source, optimizer-superiority or solver-accuracy claim.',fontsize=10,linespacing=1.6)
out=ROOT/'assets/figures'
for ext in ['png','svg']:fig.savefig(out/f'heterogeneous-noise.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
plt.close(fig)
receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},output_sha256={f'heterogeneous-noise.{ext}':hashlib.sha256((out/f'heterogeneous-noise.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
(out/'heterogeneous-noise-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))
