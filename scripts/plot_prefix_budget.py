"""H4an public synthetic sample-budget curves, including contextual parents."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'results/research/foureyes/prefix_budget/degree';s=json.loads((base/'summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'h4an-prefix-budget-v1'})
fig,(ax,bx)=plt.subplots(1,2,figsize=(14,7),gridspec_kw={'width_ratios':[1.2,1]})
families=['cycle81','power2','power3','cliques9'];labels=['Cycle, degree 2','Cycle square, degree 4','Cycle cube, degree 6','Nine cliques, degree 8'];colors=['#287d9c','#18806a','#9467bd','#d18b30'];parents=[]
for i,(family,label,color) in enumerate(zip(families,labels,colors)):
 rows=[r for r in s['grouped'] if r['family']==family];values=[r['success'] for r in rows]
 parent='varying_degree' if family in ['cycle81','cliques9'] else 'cycle_power';path=ROOT/f'results/research/foureyes/{parent}/degree/summary.json';old=json.loads(path.read_text());context=next(r['success'] for r in old['grouped'] if r['condition']==family);parents.append(str(path.relative_to(ROOT)))
 ax.plot(range(3),values,'o-',color=color,label=label,lw=2);ax.plot([2,3],[values[-1],context],'o--',color=color,alpha=.65)
 bx.barh(i,values[1],color=color,height=.5);bx.scatter(64,i,s=80,facecolors='none',edgecolors=color,linewidths=2)
 bx.text(values[1]+1,i,str(values[1]),va='center',weight='bold')
ax.axhline(61,ls=':',color='#666666',lw=1);ax.text(.05,62.4,'600-row target: 61',fontsize=9,color='#666666');ax.set(xticks=range(4),xticklabels=['300','600','1,200','2,400*'],ylim=(-2,69),xlabel='Noisy rows per bank',ylabel='Exact selected graphs / 64');ax.legend(loc='lower right',fontsize=9)
ax.set_title('Less data changes graph recovery sharply',loc='left',weight='bold',fontsize=12)
bx.invert_yaxis();bx.set_yticks(range(4),labels);bx.set(xlim=(0,70),xlabel='Exact graphs / 64 at 600 rows');bx.set_title('All degrees are right; many graphs are wrong',loc='left',weight='bold',fontsize=12)
bx.text(0,-.22,'Hollow markers: correct degrees; bars: exact graphs',transform=bx.transAxes,fontsize=9,va='top')
fig.suptitle('Correct degree does not establish exact graph recovery',x=.04,y=.96,ha='left',fontsize=18,weight='bold')
fig.text(.04,.88,'All 768 grids select the true degree. All four frozen 600-row graph targets fail.',fontsize=12)
fig.text(.04,.17,'Literal 300/600/1,200-row prefixes of 256 examined synthetic banks; same fixed concentrated corruption.\n*2,400-row points reuse earlier audited results as context; dashed segments distinguish that prior work.\n30,720 LPs certified; 54 selected fractional failures retained. Three individual nonmonotonic cases also retained.\nSource intervention, not fresh confirmation. BP AdamW was not rerun. Equal degree and the bounded 0–40 grid remain priors.',fontsize=10.4,linespacing=1.55,va='top',color='#405169')
fig.subplots_adjust(left=.08,right=.97,bottom=.29,top=.76,wspace=.5)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/prefix-budget.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'Matplotlib'})
plt.close(fig)
files=['scripts/plot_prefix_budget.py','results/research/foureyes/prefix_budget/degree/summary.json',*sorted(set(parents)),'assets/figures/prefix-budget.png','assets/figures/prefix-budget.svg']
(ROOT/'assets/figures/prefix-budget-provenance.json').write_text(json.dumps({'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
