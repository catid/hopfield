"""H2br joint accuracy and both separately measured complete-cost endpoints."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2br-endpoint-dispatch'
d=Path(__file__).resolve().parents[3]/'results/research/work/bp-first-endpoint-dispatch'
a=json.loads((d/'audit.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(16,5.8))
arms=['BP_AdamW','LP96','cascade','dispatcher']
names=['BP AdamW executed prefix','LP96 standalone','Always BP → geometry → LP96','Manhattan182 dispatcher']
colors=['#777777','#b16c37','#137c8b','#6d479d']
values=[a['joint_success'][arm] for arm in arms]
axes[0].barh(names,values,color=colors);axes[0].invert_yaxis()
for i,v in enumerate(values):axes[0].text(1700,i,f'{v}/4096',va='center',color='white',fontsize=10)
axes[0].set_xlim(0,4300);axes[0].set_xlabel('Fresh mazes jointly solved, all three models')
axes[0].set_title('Fixed block0 accuracy; 4,096 IID mixture mazes',loc='left',fontsize=10)
for ax,control,label,color in zip(axes[1:],['LP96','BP_geometry_LP96'],['LP96','Always BP cascade'],colors[1:3]):
    endpoint=a['cost_endpoints'][control];rows=endpoint['blocks'];xx=list(range(6))
    ax.bar([i-.18 for i in xx],[r['control_seconds'] for r in rows],width=.36,color=color,label=label)
    ax.bar([i+.18 for i in xx],[r['dispatcher_seconds'] for r in rows],width=.36,color=colors[3],label='Dispatcher')
    for i,r in enumerate(rows):ax.text(i,max(r['control_seconds'],r['dispatcher_seconds'])*1.03,f"{100*r['saving']:+.1f}%",ha='center',fontsize=8)
    ax.set_ylim(0,max(max(r['control_seconds'],r['dispatcher_seconds']) for r in rows)*1.23)
    ax.set_xticks(xx,[f'G{int(i>=3)} b{i}' for i in xx],rotation=20)
    ax.set_ylabel('Complete answer seconds');ax.legend(loc='upper center',fontsize=8,frameon=False)
    ax.set_title(f"Versus {label}: {100*endpoint['median_saving']:+.2f}% median saving\n≥5% cost target {endpoint['decision']}",loc='left',fontsize=10)
for ax in axes:ax.spines[['top','right']].set_visible(False)
absolute=a['primary_endpoints']['absolute_dispatch_joint_success_999'];ni=a['primary_endpoints']['dispatch_LP96_noninferiority']
fig.suptitle('Token-only endpoint dispatch on a fresh 50:50 endpoint mixture',x=.02,ha='left',fontsize=13)
fig.text(.02,.11,f"Dispatcher ≥99.9%: {absolute['decision']}; one-sided 95% lower {absolute['one_sided95_success_lower_pp']:.4f}%. LP96 noninferiority: {ni['decision']}.",fontsize=9)
fig.text(.02,.075,'Each maze independently chooses uniform open-cell or opposite-region endpoints; no quota, filtering or replacement.',fontsize=9)
fig.text(.02,.04,'Complete costs charge readiness, extraction, transfer, solving, certificates and mapping. BP is an executed prefix; barriers and archives excluded.',fontsize=9)
fig.text(.02,.005,'Separate accuracy and two cost endpoints; no simultaneous confidence or causal distance claim. Geometry cannot guarantee shortestness.',fontsize=9)
fig.tight_layout(rect=[0,.15,1,.94])
fig.savefig(d/'bp-first-endpoint-dispatch.png',dpi=160,metadata={'Software':'Matplotlib'})
fig.savefig(d/'bp-first-endpoint-dispatch.svg',metadata={'Date':None});plt.close(fig)
