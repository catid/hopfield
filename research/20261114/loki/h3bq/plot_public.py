"""Two separate fresh finite-cohort decisions, each with its own fixed target."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bq'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';c=s['continuation_calls'];F,P,U=[c[k] for k in ['firstlex','posterior','uniform']]
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bq-public'})
    fig,axes=plt.subplots(1,2,figsize=(13,8.5));fig.subplots_adjust(left=.09,right=.95,bottom=.32,top=.72,wspace=.3)
    fig.suptitle('Fresh banks: uniform trees meet both declared cost targets',x=.04,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.04,.89,'Two separate finite-cohort decisions: ≥2% saving over first-lex; ≥5% over posterior weighting.',fontsize=12.3,fontweight='bold')
    fig.text(.04,.82,'1,408 fresh eight-positive banks for the same 22 selected truths. Shared first four MAP queries.\nAll three policies identify the same 1,368 cases; 40 infeasible cases and 297 prior successes are retained.',fontsize=11.4,linespacing=1.5)
    for ax,baseline,key,title,ratio in [(axes[0],F,'firstlex','First-lex comparator',.98),(axes[1],P,'posterior','Posterior-weighted comparator',.95)]:
        reduction=100*(1-U/baseline);limit=int(ratio*baseline);ax.bar([0,1],[baseline,U],width=.57,color=['#557f9c','#60957b']);ax.set_xticks([0,1],[title.replace(' comparator',''),'Uniform optimum']);ax.set_ylim(0,3400);ax.set_ylabel('Actual continuation calls');ax.set_title(f'{reduction:.2f}% reduction — SUPPORTED',fontsize=12.5,pad=13)
        ax.axhline(limit,ls='--',color='#6b5454',lw=1.2);ax.text(.5,limit-230,f'Passing target ≤{limit:,} calls',ha='center',fontsize=10.4,color='#5b4545',bbox=dict(facecolor='white',edgecolor='none',pad=2))
        for i,v in enumerate([baseline,U]):ax.text(i,v+70,f'{v:,}',ha='center',fontsize=12.5,fontweight='bold')
        ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    fig.text(.04,.24,'The 2% first-lex target was informed by H3bp; its old 5% target is not supported by this 3.98% saving.',fontsize=11.9,fontweight='bold')
    fig.text(.04,.17,'5,577 shared MAP + 2,761 first-lex + 2,882 posterior + 2,651 uniform = 13,871 newly charged calls.\n4,408 continuation calls reacquire another policy’s same-case query; none is free.',fontsize=11.3,linespacing=1.5)
    fig.text(.04,.085,'Fresh source-bank replication on selected truths; no new-rule or population guarantee.\nEqual per-case accuracy is a required correctness invariant, not a separate scientific success.',fontsize=10.9,color='#444',linespacing=1.5)
    fig.text(.04,.03,f"BP AdamW: not measured. 11,264 new positive appearances. Study {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.7,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'fresh-three-policy-cost.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,Path(__file__)]},output_sha256={f'fresh-three-policy-cost.{ext}':hashlib.sha256((PUBLIC/f'fresh-three-policy-cost.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
