"""H4ah public comparison of local degree information versus edge count."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'results/research/foureyes/intensity_degree/summary.json').read_text())
old=json.loads((ROOT/'results/research/foureyes/regularity_intensity/summary.json').read_text())
c='hot18_weight32';row=next(r for r in s['grouped'] if r['condition']==c)
values=[next(r['exact_graphs'] for r in old['grouped'] if r['condition']==c and r['arm']==arm) for arm in ['adamw','canonical','restricted','regularity']]+[row['top810_exact'],row['success']]
labels=['BP AdamW32 (archived)','Canonical PL256 (archived)','Restricted count gap (archived)','Equal-degree threshold (archived)','Lowest-count 810 edges','Degree20 at every cell, exact LP']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'h4ah-degree'})
fig,ax=plt.subplots(figsize=(12,6.3));ax.barh(range(6),values,color=['#8b98a9']*4+['#c28845','#197c73']);ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set(xlim=(0,72),xticks=[0,16,32,48,64],xlabel='Exact graphs / 64 examined strong-noise banks')
for i,v in enumerate(values):ax.text(v+.8,i,f'{v}/64',va='center',fontweight='bold')
ax.set_title('Per-cell degree constraints repair graphs that count thresholds cannot',loc='left',fontsize=15,fontweight='bold',pad=16)
fig.text(.04,.025,'Same raw equality counts, no new observations. Degree20 is supplied; graph geometry is used only to evaluate.\nAll128 LPs across both noise levels are integral, with exact primal-dual proofs. Top810 supplies edge count alone.\nAdaptive source-cohort repair; no fresh-source, puzzle-solving, optimizer-superiority or unique-optimum claim.',fontsize=10,linespacing=1.5,color='#415062')
fig.tight_layout(rect=[0,.15,1,1])
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/intensity-degree.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'Matplotlib'})
plt.close(fig)
files=['scripts/plot_intensity_degree.py','results/research/foureyes/intensity_degree/summary.json','results/research/foureyes/regularity_intensity/summary.json','assets/figures/intensity-degree.png','assets/figures/intensity-degree.svg']
(ROOT/'assets/figures/intensity-degree-provenance.json').write_text(json.dumps({'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
