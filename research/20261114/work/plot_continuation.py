"""Public H2bg source-only outcomes and archived residual cycle ranks."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2bg-source-continuation'
D=Path(__file__).resolve().parents[3]/'results/research/work/continuation'
a=json.loads((D/'audit.json').read_text());diag=json.loads((D/'source-diagnosis/geometry-progress.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11.5,5.3))
labels=['BP AdamW','LP48 / 256','LP64 / 256','LP96 / 256','LP96 / 512'];failures=[4096-a['bp_source_joint_success'],4096-a['source_joint_success']]+[4096-v for v in a['joint_success'].values()];colors=['#777777','#b26e38','#4c91a0','#137c8b','#137c8b'];bars=axes[0].bar(range(5),failures,color=colors)
for bar,value in zip(bars,failures):axes[0].text(bar.get_x()+bar.get_width()/2,value+4,str(value),ha='center',fontsize=10)
axes[0].set_xticks(range(5),labels,rotation=20,ha='right');axes[0].set_ylim(0,220);axes[0].set_ylabel('Joint failures among 4,096 source inputs');axes[0].set_title('Retained source cohort; three fixed models',loc='left',fontsize=11)
hist=diag['final_cycle_rank_histogram'];ranks=list(map(int,hist));counts=list(hist.values());axes[1].bar(ranks,counts,width=.85,color=['#4c91a0' if r<=16 else '#b26e38' for r in ranks]);axes[1].axvline(16.5,color='#444444',linestyle='--',label='16 extra moves: total64');axes[1].axvline(48,color='#137c8b',linestyle='--',label='48 extra moves: total96');axes[1].set_xlim(0,50);axes[1].set_ylim(0,max(counts)+4);axes[1].set_xlabel('Independent cycles remaining after total48');axes[1].set_ylabel('Archived capped model states');axes[1].set_title('All 148 caps had legal candidates of size <=55',loc='left',fontsize=11);axes[1].legend(frameon=False,fontsize=9)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('More rounds repair the archived loop80 failures',x=.02,ha='left',fontsize=14);fig.text(.02,.065,'Total64 rescues 106/148 caps; total96 rescues 148/148. Limit512 changes no choices, boards, flags or outcomes.',fontsize=9);fig.text(.02,.025,'Source intervention only: no fresh inputs, confidence interval or generalization claim. Original H2be result remains unchanged.',fontsize=9);fig.tight_layout(rect=[0,.13,1,.93]);fig.savefig(D/'continuation.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(D/'continuation.svg',metadata={'Date':None});plt.close(fig)
