"""H4aj observed degree selection and exact graph recovery, public data only."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'results/research/foureyes/degree_curvature/summary.json').read_text())
old=json.loads((ROOT/'results/research/foureyes/regularity_intensity/summary.json').read_text())
center=json.loads((ROOT/'results/research/foureyes/centered_regularity/summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'h4aj-curvature'})
fig,(ax,bx)=plt.subplots(1,2,figsize=(13,6.4))
for c,label,color in [('hot18_weight8','Hot weight8','#317e98'),('hot18_weight32','Hot weight32','#25816a')]:
 curves=np.array([r['curvature_twice'] for r in s['rows'] if r['condition']==c])/2
 ax.fill_between(np.arange(1,40),curves.min(0),curves.max(0),alpha=.18,color=color)
 ax.plot(np.arange(1,40),np.median(curves,axis=0),lw=2,color=color,label=label)
ax.set(xlabel='Candidate common degree (frozen range1–39)',ylabel='Exact LP cost second difference');ax.legend(frameon=False,loc='upper right');ax.set_title('Every bank selects20 from the cost curve',loc='left',fontsize=12,fontweight='bold')
ax.text(.03,.94,'Lines: median across64 banks\nShading: full observed range',transform=ax.transAxes,va='top',fontsize=9)
c='hot18_weight32';values=[next(r['exact_graphs'] for r in old['grouped'] if r['condition']==c and r['arm']=='adamw'),next(r['exact_graphs'] for r in old['grouped'] if r['condition']==c and r['arm']=='regularity'),next(r['exact'] for r in center['grouped'] if r['condition']==c and r['arm']=='centered_regularity'),next(r['success'] for r in s['grouped'] if r['condition']==c)]
bx.barh(range(4),values,color=['#8b98a9','#8b98a9','#317e98','#25816a']);bx.invert_yaxis();bx.set_yticks(range(4),['BP AdamW32\n(archived)','Raw regularity\n(archived)','Centered regularity\n(archived)','LP curvature\n(new)']);bx.set(xlim=(0,72),xlabel='Exact graphs /64 strong-noise banks')
for i,v in enumerate(values):bx.text(v+.8,i,str(v),va='center',fontweight='bold')
bx.set_title('64/64 exact without supplying degree20',loc='left',fontsize=12,fontweight='bold')
fig.suptitle('Infer a bounded common degree from exact LP value curvature',x=.04,ha='left',fontsize=17,fontweight='bold')
fig.text(.04,.04,'All degrees0–40 considered; select the largest discrete curvature at1–39, ties lower degree. Equal degree and this grid remain priors.\nSame128 examined count matrices; no new source data or puzzle-solving claim. Degree20 is an observed selection, not a supplied value.\n5,120 positive-degree optimality proofs +128 analytic degree-zero proofs;15,360 corrupted certificates rejected.\nAll128 selected graphs exact. Every curvature maximum unique; optima themselves are not claimed unique. Fresh confirmation remains needed.',fontsize=10,linespacing=1.5,color='#415062')
fig.subplots_adjust(left=.07,right=.98,bottom=.28,top=.82,wspace=.50)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/degree-curvature.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'Matplotlib'})
plt.close(fig)
files=['scripts/plot_degree_curvature.py','results/research/foureyes/degree_curvature/summary.json','results/research/foureyes/regularity_intensity/summary.json','results/research/foureyes/centered_regularity/summary.json','assets/figures/degree-curvature.png','assets/figures/degree-curvature.svg']
(ROOT/'assets/figures/degree-curvature-provenance.json').write_text(json.dumps({'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
