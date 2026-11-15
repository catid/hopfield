"""H4ap paired graph recovery and retained regressions from public evidence."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'results/research/foureyes/bootstrap_consensus/summary.json';s=json.loads(source.read_text());groups=s['grouped']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'h4ap-bootstrap-consensus-v1'})
fig,(a,b)=plt.subplots(1,2,figsize=(13,6.5),gridspec_kw={'width_ratios':[1.3,1]});y=np.arange(4);labels=['Cycle, degree 2','Cycle square, degree 4','Cycle cube, degree 6','Nine cliques, degree 8'];old=[r['baseline_exact'] for r in groups];new=[r['success'] for r in groups]
for values,offset,color,label in [(old,-.17,'#668397','Original count LP'),(new,.17,'#b26042','32-bootstrap consensus')]:
 a.barh(y+offset,values,height=.3,color=color,label=label)
 for i,v in enumerate(values):a.text(v+.7,i+offset,str(v),va='center',fontsize=10)
a.invert_yaxis();a.set_yticks(y,labels);a.set(xlim=(0,64),xlabel='Exact graphs / 64');a.set_title('129 → 108 exact graphs across all families',loc='left',weight='bold',fontsize=12);a.legend(loc='lower right',fontsize=9)
gains=sum(r['gains'] for r in groups);losses=sum(r['losses'] for r in groups);b.barh([0,1],[gains,losses],color=['#20806c','#b26042'],height=.5);b.invert_yaxis();b.set_yticks([0,1],['New recoveries','Lost recoveries']);b.set(xlim=(0,38),xlabel='Paired changes / 256 banks');b.set_title('Regressions outnumber recoveries',loc='left',weight='bold',fontsize=12)
for i,v in enumerate([gains,losses]):b.text(v+.6,i,str(v),va='center',weight='bold')
fig.suptitle('Bootstrap consensus worsens exact graph recovery',x=.03,y=.97,ha='left',fontsize=18,weight='bold');fig.text(.03,.88,'Both frozen targets fail: 108 exact graphs < 145 required; 31 losses > 8 allowed.',fontsize=12)
fig.text(.03,.20,'All 256 examined 600-row synthetic banks retained; original inferred degree fixed in every bootstrap.\n8,448 LPs certified; 25,344 mutations rejected. Six final fractional failures, zero unresolved banks.\n32 bootstrap resamples add no source data or labels. Fractional bootstrap solutions contribute half-votes.\nBP AdamW was not rerun. This is an archived-source intervention, not a fresh confirmation.',fontsize=10.6,linespacing=1.5,va='top',color='#405169');fig.subplots_adjust(left=.19,right=.97,bottom=.33,top=.75,wspace=.5)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/bootstrap-consensus.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'Matplotlib'})
plt.close(fig)
files=['scripts/plot_bootstrap_consensus.py',str(source.relative_to(ROOT)),'assets/figures/bootstrap-consensus.png','assets/figures/bootstrap-consensus.svg'];(ROOT/'assets/figures/bootstrap-consensus-provenance.json').write_text(json.dumps({'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
