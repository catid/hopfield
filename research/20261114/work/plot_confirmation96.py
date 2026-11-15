"""Deterministic public-only H2bh fresh accuracy and measured charged costs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2bh-fresh96-confirmation'
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'results/research/work/confirmation96';a=json.loads((D.parent/'confirmation96-summary.json').read_text());labels=['BP AdamW native','LP total48','LP total64','LP total96'];arms=['pl','unified48','unified64','unified96'];colors=['#778899','#b26e38','#4c91a0','#137c8b'];fig,axes=plt.subplots(1,2,figsize=(11.5,6.2));failures=[4096-a['joint_success'][arm] for arm in arms];cost=[a['answer_seconds_by_arm'][arm] for arm in arms]
for ax,values,title in zip(axes,[failures,cost],['Joint failures / 4,096 fresh mazes','Answer seconds, full prefixes and own returns']):
    ax.barh(range(4),values,color=colors);ax.invert_yaxis();ax.set_yticks(range(4),labels if ax is axes[0] else []);ax.set_xlim(0,max(values)*1.22+1)
    for i,v in enumerate(values):ax.text(v+max(values)*.02,i,str(v) if ax is axes[0] else f'{v:.2f}',va='center',fontsize=10)
    ax.set_title(title,fontsize=11,loc='left');ax.spines[['top','right']].set_visible(False)
end=a['primary_endpoints'];items=[('absolute96','96 joint success >=99.9%'),('gain96_vs48','96 gain vs48 >=0.5 pp'),('regression96_vs48','96 regression vs48 <=0.1%'),('BP_noninferiority96','96 BP net NI >=-0.1 pp')]
for row,pair in enumerate([items[:2],items[2:]]):fig.text(.02,.17-.035*row,' | '.join(label+': '+end[key]['decision'] for key,label in pair),fontsize=9)
c=a['cost_endpoint'];fig.text(.02,.095,f"Measured charged96 overhead vs48: {c['measured_charged_answer_overhead_percent']:.3f}% (<=10% target: {c['decision']}). No timing confidence interval.",fontsize=9)
fig.suptitle('Fresh loop80 test of the fixed 96-round decoder',fontsize=14,x=.02,ha='left');fig.text(.02,.045,'180x180; three fixed models must all solve each input. Four accuracy endpoints and one cost endpoint decided separately.',fontsize=9);fig.text(.02,.015,'Total64 is descriptive. No source-cohort pooling; internal binary/veto/16-prefix states are retained audit evidence.',fontsize=9);fig.tight_layout(rect=[0,.23,1,.93]);fig.savefig(D/'confirmation96.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(D/'confirmation96.svg',metadata={'Date':None});plt.close(fig)
