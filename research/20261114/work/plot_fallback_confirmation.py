"""H2bk fresh results with unused fallback branch explicitly visible."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2bk-fresh-unused-fallback'
D=Path(__file__).resolve().parents[3]/'results/research/work/fallback-confirmation';a=json.loads((D/'audit.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,5.7));names=['BP AdamW','LP96','LP96 + BP fallback','Same, dual certificate'];values=list(a['joint_success'].values())
axes[0].barh(names,[4096-v for v in values],color=['#777777','#137c8b','#137c8b','#137c8b']);axes[0].invert_yaxis()
for i,v in enumerate(values):axes[0].text(4096-v+3,i,f'{4096-v} failures; {v}/4096 solved',va='center',fontsize=9)
axes[0].set_xlim(0,285);axes[0].set_xlabel('Joint failures on fresh mazes');axes[0].set_title('Every model must succeed on each maze',loc='left',fontsize=10)
for y,(key,label) in enumerate([('hybrid_absolute','Main hybrid'),('strict_absolute','Dual-certificate hybrid')]):
    e=a['primary_endpoints'][key];point=e['observed_success_pp'];lo=e['one_sided95_success_lower_pp'];hi=e['one_sided95_success_upper_pp']
    axes[1].errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',capsize=5,color='#137c8b',markersize=7)
    axes[1].text(lo,y+.13,f'95% lower bound {lo:.6f}%',fontsize=9)
axes[1].axvline(99.9,color='#b26e38',linestyle='--',label='Prespecified 99.9% target');axes[1].set_yticks([0,1],['Main hybrid','Dual-certificate hybrid']);axes[1].set_ylim(-.4,1.6);axes[1].set_xlim(99.85,100.025);axes[1].set_xlabel('Joint success (%)');axes[1].set_title('Separate absolute endpoints: both supported',loc='left',fontsize=10);axes[1].set_xticks([99.85,99.90,99.95,100.00],['99.85','99.90','99.95','100.00']);axes[1].legend(frameon=True,facecolor='white',edgecolor='none',framealpha=1,loc='upper left',fontsize=9)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('Fresh confirmation: LP96 solves every input; fallback is unused',x=.02,ha='left',fontsize=13)
fig.text(.02,.09,'0 routed model-inputs; all 12,288 LP96 outputs accepted. Both hybrid arms reuse those outputs exactly.',fontsize=10,weight='bold')
fig.text(.02,.055,f"BP noninferiority supported: 191 gains, 0 losses. Charged readiness overhead {a['cost_endpoint']['main_overhead_percent']:.3f}% (target <=10%).",fontsize=9)
fig.text(.02,.02,'No fresh conditional BP-rescue evidence. Separate confidence decisions, no familywise claim or pooling with H2bh/H2bj.',fontsize=9)
fig.tight_layout(rect=[0,.16,1,.94]);fig.savefig(D/'fallback-confirmation.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(D/'fallback-confirmation.svg',metadata={'Date':None});plt.close(fig)
