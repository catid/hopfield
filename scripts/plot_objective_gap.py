"""Public H4ao exact objective-gap diagnosis."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/research/foureyes/objective_gap';s=json.loads((BASE/'summary.json').read_text());rows=json.loads((BASE/'rows.json').read_text());fail=[r for r in rows if r['parent_classification']!='success']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'h4ao-exact-objective-gap-v1'})
fig,(a,b)=plt.subplots(1,2,figsize=(13,6.5),gridspec_kw={'width_ratios':[1.25,1]})
colors=['#ae563c','#cf9d53','#768794'];labels=['Wrong integral optimum','Fractional optimum','True graph co-optimal'];values=[sum(r['gap_twice']>0 and r['parent_classification']==c for r in fail) for c in ['resolved_failure_nonexact','resolved_failure_fractional']]+[sum(r['gap_twice']==0 for r in fail)]
a.barh(range(3),values,color=colors,height=.55);a.invert_yaxis();a.set_yticks(range(3),labels);a.set(xlim=(0,375),xlabel='Cases among 394 selected failures');a.set_title('385 cases strictly exclude the true graph',loc='left',weight='bold',fontsize=12)
for i,v in enumerate(values):a.text(v+5,i,str(v),va='center',weight='bold')
for i,n in enumerate([300,600,1200]):
 part=[r for r in fail if r['rows']==n];strict=sum(r['gap_twice']>0 for r in part);ties=len(part)-strict
 b.barh(i,strict,color=colors[0],height=.55,label='Truth strictly suboptimal' if i==0 else None);b.barh(i,ties,left=strict,color=colors[2],height=.55,label='Truth co-optimal' if i==0 else None);b.text(len(part)+4,i,f'{strict} + {ties}',va='center',fontsize=10)
b.invert_yaxis();b.set_yticks(range(3),['300 rows','600 rows','1,200 rows']);b.set(xlim=(0,295),xlabel='Failed cases across four families');b.set_title('Only nine failures leave the truth tied',loc='left',weight='bold',fontsize=12);b.legend(loc='lower right',fontsize=9)
fig.suptitle('Most graph failures come from the empirical objective',x=.03,y=.97,ha='left',fontsize=18,weight='bold')
fig.text(.03,.88,'385/394 failures (97.72%) have strictly positive true-graph cost gaps; the frozen 90% target passes.',fontsize=11.7)
fig.text(.03,.19,'Exact comparison at the selected, correct degree. All 768 selected primal/dual certificates verified.\nAll 374 successful outputs have zero gap; 2,304 certificate mutations rejected.\nReused synthetic observations, no new labels or LPs. This diagnoses the empirical relaxed LP,\nnot information-theoretic impossibility. BP AdamW is outside this source-only diagnostic.',fontsize=10.7,linespacing=1.5,va='top',color='#405169')
fig.subplots_adjust(left=.205,right=.97,bottom=.32,top=.75,wspace=.38)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/objective-gap.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'Matplotlib'})
plt.close(fig)
files=['scripts/plot_objective_gap.py',str((BASE/'summary.json').relative_to(ROOT)),str((BASE/'rows.json').relative_to(ROOT)),'assets/figures/objective-gap.png','assets/figures/objective-gap.svg'];(ROOT/'assets/figures/objective-gap-provenance.json').write_text(json.dumps({'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
