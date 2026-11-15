"""H2bj source outcome and dual-certificate margin context."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2bj-certificate-fallback'
D=Path(__file__).resolve().parents[3]/'results/research/work/certificate-fallback'
a=json.loads((D/'audit.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11.5,5.5))
names=['BP AdamW','LP96','LP96 + BP fallback','Same, dual certificate']
values=list(a['joint_success'].values());axes[0].barh(names,[4096-v for v in values],color=['#777777','#986caa','#137c8b','#137c8b']);axes[0].invert_yaxis()
for i,v in enumerate(values):axes[0].text(4096-v+3,i,f'{4096-v} failure'+('' if 4096-v==1 else 's')+f' ({v}/4096 solved)',va='center',fontsize=9)
axes[0].set_xlim(0,255);axes[0].set_xlabel('Joint source failures');axes[0].set_title('Known source rescue, verified execution',loc='left',fontsize=10)
x=[0,1,2];bp=[r['bp_original_margin'] for r in a['all_routed_cases']];lp=[r['lp_original_margin'] for r in a['all_routed_cases']]
axes[1].scatter(x,bp,label='Original BP margin',color='#777777',s=60);axes[1].scatter(x,lp,label='Original LP margin',color='#137c8b',s=60)
axes[1].set_yscale('log');axes[1].set_ylim(.02,9);axes[1].set_xticks(x,['Seed 0','Seed 1','Seed 2']);axes[1].set_ylabel('Positive minimum original flip margin (log scale)')
for i,v in enumerate(lp):axes[1].annotate(f'{v:.5f}',(i,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
axes[1].legend(frameon=False,loc='center right',fontsize=9);axes[1].set_title('New cross-LP local-certificate endpoint supported',loc='left',fontsize=10)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('Numerical-only fallback repairs the retained source cap failure',x=.02,ha='left',fontsize=13)
fig.text(.02,.085,'All 12,288 model-inputs routed: 12,285 LP outputs unchanged; 3 actual BP subset replays on input 4078.',fontsize=9)
fig.text(.02,.05,'No labels, shortest distances or graph checks enter routing. All 96 original-batch gate states and 3 subset outputs match.',fontsize=9)
fig.text(.02,.015,'Zero fresh accuracy evidence; H2bh 99.9% endpoint stays inconclusive. Local margins do not prove shortestness or global optimality.',fontsize=9)
fig.tight_layout(rect=[0,.14,1,.94]);fig.savefig(D/'certificate-fallback.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(D/'certificate-fallback.svg',metadata={'Date':None});plt.close(fig)
