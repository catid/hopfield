"""Expose the failed total-call endpoint and separate the guard mechanism."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bt'


def main():
    path=PUBLIC/'summary.json';baseline=ROOT/'results/research/loki/h3br/summary.json';previous=ROOT/'results/research/loki/h3bs/summary.json';s=json.loads(path.read_text());old=json.loads(baseline.read_text());prior=json.loads(previous.read_text());assert s['status']=='validated';maps=[old['shared_MAP_calls'],prior['new_MAP_calls'],s['new_MAP_calls']];trees=[old['continuation_calls']['posterior'],prior['new_tree_calls'],s['new_tree_calls']];totals=[s['old_policy_calls'],prior['new_policy_calls'],s['new_policy_calls']];saving=100*(1-totals[2]/totals[0])
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bt-public'})
    fig,ax=plt.subplots(figsize=(13.4,8.8));fig.subplots_adjust(left=.10,right=.94,bottom=.33,top=.73)
    fig.suptitle('Exact cost guard saves 0.62%; the 2% target fails',x=.04,y=.96,ha='left',fontsize=20,fontweight='bold')
    fig.text(.04,.89,f'PRIMARY FALSIFIED: {totals[2]:,} newly paid calls versus {totals[0]:,}; {saving:.2f}% fewer. Required ≤10,147.',fontsize=13,fontweight='bold')
    fig.text(.04,.82,'All 4,096 examined H3br cases retained. Every actual MAP/tree answer reacquired and charged.\nIdentification unchanged: 4,083 exact cases and 13 retained failures.',fontsize=11.6,linespacing=1.5)
    ax.bar(range(3),maps,width=.56,color='#557f9c',label='MAP calls');ax.bar(range(3),trees,bottom=maps,width=.56,color='#bb805f',label='Tree calls');ax.set_xticks(range(3),['Original policy\narchived H3br comparator','Feasibility-only handoff\narchived H3bs, descriptive','Exact cost guard\nnew H3bt execution']);ax.set_ylabel('Total actual membership calls');ax.set_ylim(0,13200);ax.axhline(s['primary']['passing_maximum'],ls='--',color='#744848',lw=1.2);ax.text(1,9500,'Passing target ≤10,147',ha='center',fontsize=10.6,color='#744848',bbox=dict(facecolor='white',edgecolor='none',pad=2));ax.legend(loc='upper left',frameon=False,ncol=2,bbox_to_anchor=(0,1.03))
    for i,v in enumerate(totals):ax.text(i,v+350,f'{v:,}',ha='center',fontweight='bold',fontsize=13)
    ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    fig.text(.04,.25,'2,291 rejected guards; 3,070 accepted, including 2,388 exact expected-cost ties.',fontsize=12.4,fontweight='bold')
    fig.text(.04,.18,'H3bt: 5,515 MAP + 4,776 tree calls. Saving: 64 versus H3br; 646 versus H3bs (descriptive).\n9,450 calls reacquire archived same-case queries; 841 query boards were absent from the archived case.',fontsize=11.3,linespacing=1.5)
    fig.text(.04,.095,'Expected nonincrease under the declared posterior is a correctness property; realized 2% saving is not guaranteed.\nArchived-cohort intervention, no new positives or independent confirmation. Finite model class, noiseless oracle.',fontsize=10.9,color='#444',linespacing=1.5)
    fig.text(.04,.03,f"BP AdamW: not measured. All online work included: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.8,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'guarded-handoff-total-cost.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,baseline,previous,Path(__file__)]},output_sha256={f'guarded-handoff-total-cost.{ext}':hashlib.sha256((PUBLIC/f'guarded-handoff-total-cost.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
