"""H2bl fixed-source rejection failure and separate retention result."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2bl-cross-certificate-negative'
D=Path(__file__).resolve().parents[3]/'results/research/work/cross-certificate-screen';a=json.loads((D/'audit.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,5.5))
names=['BP AdamW own certificate','LP certificate on BP board','Both certificates (dual)'];keys=['bp_own_accept','lp_cross_accept','dual_accept'];bad=[a['control_confusion_matrices'][k]['accepted_graph_failures'] for k in keys]
axes[0].barh(names,bad,color=['#777777','#a34c44','#a34c44']);axes[0].invert_yaxis()
for i,v in enumerate(bad):axes[0].text(v+5,i,f'{v}/510',va='center',fontsize=10)
axes[0].set_xlim(0,615);axes[0].set_xlabel('Wrong BP paths accepted');axes[0].set_title('A second local certificate leaves 509 errors accepted',loc='left',fontsize=10)
values=[a['primary_endpoints']['error_rejection']['observed_percent'],a['primary_endpoints']['valid_retention']['observed_percent']];targets=[90,99]
axes[1].barh(['Wrong-path rejection','Valid-path retention'],values,color=['#a34c44','#137c8b']);axes[1].invert_yaxis()
for y,v,target,label in zip([0,1],values,targets,['1/510; target falsified','11778/11778; target supported']):
    axes[1].plot([target,target],[y-.36,y+.36],color='#b26e38',linestyle='--');axes[1].text(3,y,label,va='center',fontsize=9,color='black' if y==0 else 'white');axes[1].text(target,y-.42,f'{target}% target',ha='center',fontsize=9)
axes[1].set_xlim(0,105);axes[1].set_xlabel('Finite source percentage');axes[1].set_title('Separate prespecified 90% / 99% targets',loc='left',fontsize=10)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('Original LP local certificates do not screen BP graph errors',x=.02,ha='left',fontsize=13)
fig.text(.02,.085,'All 12,288 archived H2bh BP boards. Margins and acceptance flags sealed before graph labels were joined.',fontsize=9)
fig.text(.02,.05,'Smallest LP margin among the 509 accepted errors: +0.015162. Threshold remains -0.000001; no tolerance tuning.',fontsize=9)
fig.text(.02,.015,'Zero fresh trajectories or population evidence. Local certificates alone do not establish path validity or shortestness.',fontsize=9)
fig.tight_layout(rect=[0,.15,1,.94]);fig.savefig(D/'cross-certificate-screen.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(D/'cross-certificate-screen.svg',metadata={'Date':None});plt.close(fig)
