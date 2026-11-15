"""Show increased total query cost and the MAP-to-tree substitution explicitly."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bs'


def main():
    path=PUBLIC/'summary.json';parent=ROOT/'results/research/loki/h3br/summary.json';s=json.loads(path.read_text());old=json.loads(parent.read_text());assert s['status']=='validated';maps=[old['shared_MAP_calls'],s['new_MAP_calls']];trees=[old['continuation_calls']['posterior'],s['new_tree_calls']];totals=[s['old_policy_calls'],s['new_policy_calls']];increase=100*(totals[1]/totals[0]-1)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bs-public'})
    fig,ax=plt.subplots(figsize=(12.8,8.5));fig.subplots_adjust(left=.11,right=.93,bottom=.33,top=.73)
    fig.suptitle('Earlier certified handoff costs more total membership calls',x=.04,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.04,.89,f'PRIMARY FALSIFIED: {totals[1]:,} calls versus {totals[0]:,}; +{increase:.2f}%. Required ≤10,147.',fontsize=13,fontweight='bold')
    fig.text(.04,.82,'All 4,096 archived matched-prior cases retained; every new MAP and tree answer independently paid.\nExactly 4,083 identified cases and all 13 failures preserved, as required by the restriction argument.',fontsize=11.4,linespacing=1.5)
    ax.bar([0,1],maps,width=.54,color='#557f9c',label='MAP calls');ax.bar([0,1],trees,bottom=maps,width=.54,color='#bb805f',label='Tree calls');ax.set_xticks([0,1],['Original: up to four MAP + posterior tree\narchived H3br','Early: switch at first feasible tree\nnew H3bs execution']);ax.set_ylabel('Total actual membership calls');ax.set_ylim(0,13300);ax.axhline(s['primary']['passing_maximum'],ls='--',color='#744848',lw=1.2)
    ax.text(.5,9700,'Passing target ≤10,147',ha='center',fontsize=10.8,color='#744848',bbox=dict(facecolor='white',edgecolor='none',pad=2));ax.legend(loc='upper left',frameon=False,ncol=2,bbox_to_anchor=(0,1.02))
    for i,v in enumerate(totals):ax.text(i,v+350,f'{v:,}',ha='center',fontweight='bold',fontsize=13)
    ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    fig.text(.04,.24,'6,709 fewer MAP calls were outweighed by 7,291 additional tree calls: net 582 more calls.',fontsize=12.2,fontweight='bold')
    fig.text(.04,.17,'Original: 9,933 MAP + 422 tree. Early handoff: 3,224 MAP + 7,713 tree.\n7,578 calls reacquire archived same-case queries; 3,359 query boards were absent from the archived case.',fontsize=11.4,linespacing=1.5)
    fig.text(.04,.085,'Feasibility preserves identification, but does not guarantee cheaper cost than the remaining MAP policy.\nExamined archived-cohort intervention; no independent confirmation, new source positives or accuracy-gain claim.',fontsize=10.9,color='#444',linespacing=1.5)
    fig.text(.04,.03,f"BP AdamW: not measured. Online compilation included: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.8,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'early-handoff-total-cost.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,parent,Path(__file__)]},output_sha256={f'early-handoff-total-cost.{ext}':hashlib.sha256((PUBLIC/f'early-handoff-total-cost.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
