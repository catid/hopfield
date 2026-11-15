"""Deterministic fresh H2ay failures, cost and separately decided endpoints."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2ax-density-transfer'
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'results/research/work/density-transfer';a=json.loads((ROOT/'results/research/work/density-transfer-summary.json').read_text())
labels=['BP AdamW native','LP binary','Connectivity veto','Lexical cycle256','Lexical256 → lexical512','Lexical256 → energy512'];arms=['pl','old_binary','connectivity_veto','lexical256','lexical512_from_lexical256','energy512_from_lexical256'];colors=['#778899','#b0bec5','#a5c8d0','#659caa','#137c8b','#4c9f70']
fig,axes=plt.subplots(1,2,figsize=(12,5.8),gridspec_kw={'width_ratios':[1,1]});failures=[4096-a['joint_success'][k] for k in arms];cost=[a['answer_seconds_by_arm'][k] for k in arms]
for ax,values,title in zip(axes,[failures,cost],['Joint failures / 4,096 fresh mazes','Answer seconds, complete prefixes charged']):
 ax.barh(range(6),values,color=colors);ax.invert_yaxis();ax.set_yticks(range(6),labels if ax is axes[0] else []);ax.set_xlim(0,max(values)*1.2+1)
 for i,v in enumerate(values):ax.text(v+max(values)*.02,i,str(v) if ax is axes[0] else f'{v:.2f}',va='center',fontsize=10)
 ax.set_title(title,fontsize=11,loc='left');ax.spines[['top','right']].set_visible(False)
end=a['primary_endpoints'];text=' | '.join(label+': '+end[key]['decision'] for key,label in [('lexical512_gain_vs_lexical256','Gain ≥0.2 pp'),('lexical512_regression_probability','Regression ≤0.1%'),('lexical512_noninferiority_vs_bp','BP net NI ≥−0.1 pp')]);fig.suptitle('Density shift: 180×180 mazes with 50% of closed connectors opened',fontsize=14,x=.02,ha='left');fig.text(.02,.065,text,fontsize=10);fig.text(.02,.025,'All three fixed models must solve each maze. Both cap512 arms share the lexical256 start. Three endpoint decisions are separate.',fontsize=9);fig.tight_layout(rect=[0,.12,1,.94]);fig.savefig(D/'fresh-density-transfer.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(D/'fresh-density-transfer.svg',metadata={'Date':None});plt.close(fig)
