"""Deterministic public-only H2be fresh failures, full-prefix costs and endpoints."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2be-loop80-transfer'
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'results/research/work/loop80-transfer';a=json.loads((ROOT/'results/research/work/loop80-transfer-summary.json').read_text())
labels=['BP AdamW native','LP binary','Connectivity veto','Original lexical16','Unified16','Unified48'];arms=['pl','old_binary','connectivity_veto','original_lexical16','unified16','unified48'];colors=['#778899','#b0bec5','#a5c8d0','#659caa','#4c9f70','#137c8b']
fig,axes=plt.subplots(1,2,figsize=(12,6.6),gridspec_kw={'width_ratios':[1,1]});failures=[4096-a['joint_success'][k] for k in arms];cost=[a['answer_seconds_by_arm'][k] for k in arms]
for ax,values,title in zip(axes,[failures,cost],['Joint failures / 4,096 fresh mazes','Answer seconds, full prefixes charged']):
    ax.barh(range(6),values,color=colors);ax.invert_yaxis();ax.set_yticks(range(6),labels if ax is axes[0] else []);ax.set_xlim(0,max(values)*1.22+1)
    for i,v in enumerate(values):ax.text(v+max(values)*.02,i,str(v) if ax is axes[0] else f'{v:.2f}',va='center',fontsize=10)
    ax.set_title(title,fontsize=11,loc='left');ax.spines[['top','right']].set_visible(False)
end=a['primary_endpoints'];items=[('unified48_gain_vs_original16','Gain vs original16 >=1 pp'),('unified48_regression_probability','Regression <=0.1%'),('unified48_noninferiority_vs_bp','BP net NI >=-0.1 pp'),('unified48_gain_vs_unified16','Gain vs unified16 >=0.5 pp')]
for row,pair in enumerate([items[:2],items[2:]]):fig.text(.02,.13-.035*row,' | '.join(label+': '+end[key]['decision'] for key,label in pair),fontsize=10)
fig.text(.02,.06,'Absolute joint success >=99.9%: '+end['unified48_absolute_sufficiency']['decision'],fontsize=10)
fig.suptitle('Density80 tests the fixed cycle-budget boundary',fontsize=14,x=.02,ha='left');fig.text(.02,.025,'180x180 loop80; all three fixed models must solve each maze. Five separate endpoint decisions; no source-cohort pooling.',fontsize=9);fig.tight_layout(rect=[0,.18,1,.94]);fig.savefig(D/'loop80-transfer.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(D/'loop80-transfer.svg',metadata={'Date':None});plt.close(fig)
