"""Public H4al recovery and degree-selection plots."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'results/research/foureyes/varying_degree'
s=json.loads((base/'degree/summary.json').read_text());c=json.loads((base/'controls/summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'h4al-varying-degree-v1'})
fig,(ax,bx)=plt.subplots(1,2,figsize=(14,7.2),gridspec_kw={'width_ratios':[1.15,1]})
labels=['BP AdamW (32 steps)','Canonical PL (256 steps)','Restricted count gap','Raw regularity','Centered regularity','LP degree curvature'];ys=np.arange(6)
for offset,family,color,label in [(-.18,'cycle81','#287d9c','81-cycle: degree 2'),(.18,'cliques9','#d18b30','Nine cliques: degree 8')]:
 rows=[r for r in s['rows'] if r['condition']==family]
 values=[next(r['exact_graphs'] for r in c['grouped'] if r['arm']==arm and r['condition']==family) for arm in ['adamw','canonical','restricted','regularity']]+[sum(r['centered_control']['exact'] for r in rows),sum(r['classification']=='success' for r in rows)]
 ax.barh(ys+offset,values,height=.32,color=color,label=label)
 for y,v in zip(ys+offset,values):ax.text(v+.7,y,str(v),va='center',fontsize=9)
 v=np.array([r['curvature_twice'] for r in rows],np.int64)/2
 bx.fill_between(np.arange(1,40),v.min(0),v.max(0),color=color,alpha=.15)
 bx.plot(np.arange(1,40),np.median(v,axis=0),color=color,label=label,lw=2)
ax.invert_yaxis();ax.set_yticks(ys,labels);ax.set(xlim=(0,70),xlabel='Exact recovered graphs / 64 banks per family');ax.legend(loc='upper right',fontsize=9)
ax.set_title('Same noise and fixed estimator recipes',loc='left',fontsize=12,weight='bold')
bx.set(xlabel='Candidate common degree (1–39)',ylabel='Exact LP cost second difference');bx.set_title('Every bank selects its true degree',loc='left',fontsize=12,weight='bold');bx.legend(fontsize=9)
fig.suptitle('One selection rule recovers degree-2 and degree-8 synthetic graphs',x=.035,y=.96,ha='left',fontsize=17,weight='bold')
fig.text(.035,.885,'64/64 exact in each family. Equal degree and the bounded 0–40 grid remain supplied priors.',fontsize=12)
fig.text(.035,.17,'128 independently seeded graphs × 2,400 uniform proper 9-colorings; graph orders permuted separately.\nNine digits changed per row; 18 hot cells weighted 32×. All 5,120 LPs have exact certificates.\nCurves show median with full-range shading. Both frozen ≥61/64 graph targets supported.\nThese are two specified synthetic families; optimizer controls have fixed unequal work. This measures graph recovery, not puzzle solving.',fontsize=10.3,linespacing=1.6,va='top',color='#405169')
fig.subplots_adjust(left=.205,right=.98,bottom=.28,top=.77,wspace=.42)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/varying-degree.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'Matplotlib'})
plt.close(fig)
files=['scripts/plot_varying_degree.py','results/research/foureyes/varying_degree/degree/summary.json','results/research/foureyes/varying_degree/controls/summary.json','assets/figures/varying-degree.png','assets/figures/varying-degree.svg']
(ROOT/'assets/figures/varying-degree-provenance.json').write_text(json.dumps({'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
