"""Fresh-source H4ak results from public audited data only."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'results/research/foureyes/fresh_curvature'
s=json.loads((base/'degree/summary.json').read_text());c=json.loads((base/'controls/summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'h4ak-fresh-curvature-v1'})
fig,(ax,bx)=plt.subplots(1,2,figsize=(14,7.2),gridspec_kw={'width_ratios':[1.15,1]})
arms=['adamw','canonical','restricted','regularity'];values=[next(r['exact_graphs'] for r in c['grouped'] if r['arm']==a) for a in arms]+[s['centered_exact'],s['grouped'][0]['success']]
labels=['BP AdamW (32 steps)','Canonical PL (256 steps)','Restricted count gap','Raw regularity','Centered regularity','Inferred degree: LP curvature']
ax.barh(range(6),values,color=['#8b98a9']*3+['#7e9baf','#307e99','#18806a']);ax.invert_yaxis();ax.set_yticks(range(6),labels);ax.set(xlim=(0,71),xlabel='Exact recovered graphs / 64 fresh banks')
for i,v in enumerate(values):ax.text(v+1,i,str(v),va='center',weight='bold')
ax.set_title('Same corrupted observations, different estimators',loc='left',fontsize=12,weight='bold')
v=np.array([r['curvature_twice'] for r in s['rows']],np.int64)/2
bx.fill_between(np.arange(1,40),v.min(0),v.max(0),color='#18806a',alpha=.18);bx.plot(np.arange(1,40),np.median(v,axis=0),color='#18806a',lw=2)
bx.set(xlabel='Candidate common degree (1–39)',ylabel='Exact LP cost second difference');bx.set_title('Every fresh bank selects degree 20',loc='left',fontsize=12,weight='bold')
bx.text(.03,.95,'Median curve\nShading: full range\n2,560 LPs certified',transform=bx.transAxes,va='top',fontsize=10)
fig.suptitle('Fresh confirmation: 64/64 graphs recovered under concentrated label noise',x=.035,y=.96,ha='left',fontsize=17,weight='bold')
fig.text(.035,.885,'No numerical degree or edge count supplied. Equal degree and the bounded 0–40 grid remain priors.',fontsize=12)
fig.text(.035,.17,'64 independently seeded banks × 2,400 distinct rows; 147,012 unique IDs after excluding 2,134,884 declared prior IDs.\nNine free digits changed per row, with 18 hot cells weighted 32×. Cross-bank overlap retained; same official training corpus.\nPrimary ≥61 exact graphs supported. All 64 degree-zero witnesses pass; 7,680 altered certificates rejected.\nFixed optimizer recipes are descriptive controls, not matched-compute or optimum comparisons. This measures graph recovery, not puzzle solving.',fontsize=10.3,linespacing=1.6,va='top',color='#405169')
fig.subplots_adjust(left=.205,right=.98,bottom=.28,top=.77,wspace=.42)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/fresh-curvature.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'Matplotlib'})
plt.close(fig)
files=['scripts/plot_fresh_curvature.py','results/research/foureyes/fresh_curvature/degree/summary.json','results/research/foureyes/fresh_curvature/controls/summary.json','assets/figures/fresh-curvature.png','assets/figures/fresh-curvature.svg']
(ROOT/'assets/figures/fresh-curvature-provenance.json').write_text(json.dumps({'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
