"""H1bi exact curvature and graph outcomes; no refitting."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'results/research/ripper/partial_degree/summary.json';s=json.loads(source.read_text())
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#26334b','text.color':'#26334b','axes.edgecolor':'#b4c0d0','savefig.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(14,8));fig.subplots_adjust(left=.07,right=.97,bottom=.34,top=.78,wspace=.3)
fig.suptitle('Partial observations retain an identifiable degree-20 curvature peak',x=.07,y=.965,ha='left',fontsize=21,fontweight='bold')
fig.text(.07,.89,'Aligned targets supported: degree 20 on 32/32 (target 31); exact graph on 28/32 (target 27).',fontsize=14)
colors={'aligned':'#328273','permuted':'#92769e'};degrees=np.arange(1,40)
for arm in ['aligned','permuted']:
    v=np.array([r['curvature_twice'] for r in s['rows'] if r['arm']==arm],float);v/=v.max(1)[:,None]
    axes[0].fill_between(degrees,v.min(0),v.max(0),color=colors[arm],alpha=.14)
    axes[0].plot(degrees,v.mean(0),color=colors[arm],label=arm.capitalize(),linewidth=2)
axes[0].set(xlabel='Candidate common degree',ylabel="Curvature / each bank's maximum",xlim=(1,39),ylim=(0,1.06),xticks=[1,10,20,30,39]);axes[0].legend(frameon=False,loc='upper left');axes[0].set_title('All 64 unique maxima occur at degree 20',loc='left',fontsize=13)
axes[0].annotate('20',xy=(20,1),xytext=(24,.94),arrowprops={'arrowstyle':'-','color':'#52647c'})
axes[0].grid(axis='y',alpha=.15)
bottom=np.zeros(2)
for key,label,color in [('success','Certified integral exact','#328273'),('resolved_failure_nonexact','Certified integral nonexact','#af8555'),('resolved_failure_fractional','Certified fractional','#92769e')]:
    vals=np.array([g[key] for g in s['grouped']]);axes[1].bar([0,1],vals,bottom=bottom,color=color,width=.55,label=label)
    for i,v in enumerate(vals):
        if v:axes[1].text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white',weight='bold')
    bottom+=vals
axes[1].plot([-.27,.27],[27,27],color='#bd6c69',linestyle='--',linewidth=1.3);axes[1].text(-.27,27.5,'Aligned target: 27',fontsize=10,color='#ad5b59',bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1})
axes[1].set(xticks=[0,1],xticklabels=['Aligned','Permuted\n(descriptive)'],ylim=(0,34),yticks=[0,8,16,24,32],ylabel='Banks / 32');axes[1].set_title('Same outcomes as supplied degree 20',loc='left',fontsize=13);fig.legend(*axes[1].get_legend_handles_labels(),frameon=False,loc='upper center',bbox_to_anchor=(.5,.28),ncol=3,fontsize=10)
fig.text(.07,.22,'All 2,560 fresh LPs certified; 64 analytic zero witnesses; 7,680 mutated proofs rejected. No unresolved banks.\n'
         'Examined H1bh banks reused; no new labels or sources. Common-degree grid 0–40 and historical donor feedback remain priors.\n'
         'Left: per-bank normalization, line = mean, band = full range. Right: returned fractional optima retained as failures.\n'
         '24.277 CPU-s / 3.164 wall-s, eight CPU workers. Each alternative still costs 720k recipient + 240k donor labels.\n'
         'BP AdamW outside this source-only study; not run. Both archived top-810 controls: 0/32 exact.\n'
         'This adaptive follow-up does not reverse H1bh’s failed fresh-recipient targets or claim improved graph recovery.',
         fontsize=10.5,color='#596a83',va='top',linespacing=1.5)
out=ROOT/'assets/figures';out.mkdir(exist_ok=True)
for ext in ['png','svg']:fig.savefig(out/f'partial-degree.{ext}',dpi=170)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
provenance={'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),'script_sha256':sha(Path(__file__)),'png_sha256':sha(out/'partial-degree.png'),'svg_sha256':sha(out/'partial-degree.svg'),'status':'rendered_from_frozen_summary_without_refitting'}
(out/'partial-degree-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
