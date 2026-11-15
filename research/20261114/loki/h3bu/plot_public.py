"""Show missing coverage and the binding resource limit, not a partial primary."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];P=ROOT/'results/research/loki/h3bu'


def main():
    path=P/'summary.json';s=json.loads(path.read_text());protocol=ROOT/'research/20261114/loki/h3bu/protocol.json';p=json.loads(protocol.read_text());assert s['status']=='incomplete' and s['primary_decision'] is None;plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bu-incomplete'});fig,(ax,resources)=plt.subplots(2,1,figsize=(13,8.5),gridspec_kw={'height_ratios':[1,2]});fig.subplots_adjust(left=.21,right=.94,bottom=.28,top=.70,hspace=.55)
    fig.suptitle('Deeper trees hit the state cap; the study is incomplete',x=.04,y=.96,ha='left',fontsize=19,fontweight='bold');fig.text(.04,.89,'NO PRIMARY DECISION: the frozen 2% total-call hypothesis requires all 4,096 cases.',fontsize=13,fontweight='bold');fig.text(.04,.82,'1,779 cases completed; 2,317 remain pending. All planned cases and partial evidence are retained.\n4,402 actual calls were paid before the stop; this prefix is not a full-cohort comparison.',fontsize=11.6,linespacing=1.5)
    ax.barh([0],[s['completed_cases']],color='#557f9c',height=.5,label='Completed');ax.barh([0],[s['pending_cases']],left=[s['completed_cases']],color='#d8dbde',height=.5,label='Pending');ax.set_xlim(0,4096);ax.set_yticks([0],['Case coverage']);ax.set_xticks([0,1024,2048,3072,4096]);ax.text(s['completed_cases']/2,0,'1,779',ha='center',va='center',color='white',fontweight='bold');ax.text(s['completed_cases']+s['pending_cases']/2,0,'2,317 pending',ha='center',va='center',fontweight='bold');ax.legend(frameon=False,ncol=2,bbox_to_anchor=(0,1.55),loc='upper left')
    values=[s['global_memo_states']/2000000,s['global_split_evaluations']/50000000,s['aggregate_CPU_seconds']/600,s['wall_seconds']/900];labels=['Memo states','Split evaluations','Aggregate CPU time','Wall time'];texts=['2,000,000 / 2,000,000','38,546,310 / 50,000,000','217.976 / 600 seconds','235.846 / 900 seconds'];resources.barh(range(4),values,color=['#a74d47','#718c9c','#718c9c','#718c9c'],height=.57);resources.set_yticks(range(4),labels);resources.invert_yaxis();resources.set_xlim(0,1.2);resources.set_xticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%']);resources.axvline(1,color='#a74d47',ls='--',lw=1)
    for i,text in enumerate(texts):resources.text(.02,i,text,va='center',fontsize=10.9,color='white' if values[i]>.6 else '#17222b',fontweight='bold')
    resources.set_xlabel('Fraction of the declared limit');fig.text(.04,.14,'External full archive audited: 1,779 completed routes, 2,361 tree proofs and 14,075 counterfactual nodes.\nThe interrupted tree has 483 charged states and 3,240 splits without a complete certificate.',fontsize=11.3,linespacing=1.5);fig.text(.04,.055,'BP AdamW: not measured. CPU exact search; 0 GPU seconds. No budget extension or partial-cohort primary.\nA separate optimization assessment does not complete or extend this capped experiment.',fontsize=10.9,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(P/f'budget-capped-depth.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,protocol,Path(__file__)]},output_sha256={f'budget-capped-depth.{ext}':hashlib.sha256((P/f'budget-capped-depth.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(P/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
