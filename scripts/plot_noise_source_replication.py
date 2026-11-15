"""Public-data-only source/noise recovery curve; prefixes are descriptive."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'results/research/foureyes/noise_source_replication/summary.json'
s=json.loads(source.read_text());rows=s['grouped'];x=np.arange(len(rows));y=np.array([100*r['recovery_fraction'] for r in rows])
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'hopfield-noise-source-v1'})
fig,axes=plt.subplots(1,2,figsize=(12,4.9),gridspec_kw={'width_ratios':[1.2,1]})
fig.suptitle('Count recovery survives noise across fresh training banks',fontweight='bold',fontsize=16,y=.99)
ax=axes[0];colors=['#94a3b8','#94a3b8','#94a3b8','#047857'];ax.bar(x,y,color=colors,width=.65)
for i,r in enumerate(rows):ax.text(i,y[i]+2.7,f"{r['exact_graphs']}/512",ha='center',fontsize=10,fontweight='bold')
r=rows[-1];ax.errorbar([3],[100],yerr=[[100*(1-r['lower_one_sided95'])],[0]],fmt='none',capsize=4,color='#111827')
ax.set(xticks=x,xticklabels=[f"{r['n']:,}" for r in rows],xlabel='Training boards per independently sampled bank',ylabel='Exact graph recovery (%)',ylim=(0,115))
ax.set_title('512 independent source + corruption draws',fontsize=11)
ax.text(3,50,'Primary\n≥99%\npasses',ha='center',color='white',fontweight='bold',fontsize=11)
ax=axes[1];ax.axis('off')
ax.text(.02,.94,'What changes at the boundary?',fontsize=14,fontweight='bold',va='top')
ax.text(.02,.75,'300 boards',fontsize=12,fontweight='bold');ax.text(.02,.64,'21 exact fits; 84 have any separating threshold.\nMost failures reflect overlapping pair counts.',fontsize=10.5)
ax.text(.02,.45,'600 boards',fontsize=12,fontweight='bold');ax.text(.02,.34,'493 exact fits; all 512 have a separator.\nThe remaining 19 failures choose the wrong gap.',fontsize=10.5)
ax.text(.02,.15,'3,000 boards: 512/512 exact',fontsize=12,fontweight='bold',color='#047857');ax.text(.02,.04,'One-sided 95% lower recovery bound: 99.4166%.',fontsize=10.5)
fig.text(.5,.025,'Nine free digits changed per board; original clues preserved. Fixed official corpus; source banks sampled independently.\nCross-bank overlap allowed. Smaller prefixes descriptive. Graph recovery, not solver accuracy.',ha='center',fontsize=9,color='#475569')
fig.tight_layout(rect=(0,.12,1,.94),w_pad=3)
out=ROOT/'assets/figures'
for ext in ['png','svg']:fig.savefig(out/f'noise-source-replication.{ext}',dpi=180,metadata={'Date':None} if ext=='svg' else None)
(out/'noise-source-replication-provenance.json').write_text(json.dumps(dict(public_data_only=True,input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]}),indent=2)+'\n')
