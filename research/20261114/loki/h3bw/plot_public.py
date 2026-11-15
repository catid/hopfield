"""Complete total-query endpoint, with declared target and archived-cohort limits."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];P=ROOT/'results/research/loki/h3bw'


def main():
    path=P/'summary.json';baseline=ROOT/'results/research/loki/h3br/summary.json';guard=ROOT/'results/research/loki/h3bt/summary.json';s=json.loads(path.read_text());old=json.loads(baseline.read_text());prior=json.loads(guard.read_text());assert s['status']=='validated';saving=100*(1-s['new_policy_calls']/s['old_policy_calls']);maps=[old['shared_MAP_calls'],prior['new_MAP_calls'],s['new_MAP_calls']];trees=[old['continuation_calls']['posterior'],prior['new_tree_calls'],s['new_tree_calls']];totals=[s['old_policy_calls'],prior['new_policy_calls'],s['new_policy_calls']];plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bw-complete'});fig,ax=plt.subplots(figsize=(13.5,8.9));fig.subplots_adjust(left=.11,right=.94,bottom=.34,top=.72)
    fig.suptitle(f'Deeper budget-matched trees save {saving:.2f}% of total calls',x=.04,y=.96,ha='left',fontsize=20,fontweight='bold');fig.text(.04,.89,f"PRIMARY {s['decision'].upper()}: {s['new_policy_calls']:,} new calls versus {s['old_policy_calls']:,}; target ≤10,147.",fontsize=13,fontweight='bold');fig.text(.04,.82,'All 4,096 archived cases completed; every actual MAP/tree answer independently paid.\nAll exact flags unchanged: 4,083 identified cases and all 13 baseline failures retained.',fontsize=11.5,linespacing=1.5)
    ax.bar(range(3),maps,width=.56,color='#557f9c',label='MAP calls');ax.bar(range(3),trees,bottom=maps,width=.56,color='#bb805f',label='Tree calls');ax.set_xticks(range(3),['Original depth-four continuation\narchived H3br comparator','Guarded depth-four handoff\narchived H3bt, descriptive','Depth = 8 − used MAP\nnew H3bw execution']);ax.set_ylabel('Total actual membership calls');ax.set_ylim(0,12700);ax.axhline(s['primary']['passing_maximum'],color='#744848',ls='--',lw=1.2);ax.text(1,9600,'Passing target ≤10,147',ha='center',fontsize=10.7,color='#744848',bbox=dict(facecolor='white',edgecolor='none',pad=2));ax.legend(loc='upper left',frameon=False,ncol=2,bbox_to_anchor=(0,1.04));ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    for i,value in enumerate(totals):ax.text(i,value+320,f'{value:,}',ha='center',fontweight='bold',fontsize=13)
    fig.text(.04,.25,f"H3bw: {s['new_MAP_calls']:,} MAP + {s['new_tree_calls']:,} tree calls; {s['old_policy_calls']-s['new_policy_calls']:,} fewer than the original policy.",fontsize=12.3,fontweight='bold');fig.text(.04,.18,'Expected nonincrease and preserved identification are correctness properties, separate from this finite-cohort endpoint.\nAll hypothetical baseline branches must identify before handoff; every recomputed Huffman merge is charged.',fontsize=11.2,linespacing=1.5);fig.text(.04,.095,'Same examined archived cohort; no new positives or independent confirmation. H3bu and H3bv remain incomplete.\nComplete proof archive retained outside Git; the compact checkout includes its audited identities and paid-query ledger.',fontsize=10.8,color='#444',linespacing=1.5);fig.text(.04,.03,f"BP AdamW: not measured. {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. Original eight-call limit retained.",fontsize=10.8,color='#444')
    for ext in ['png','svg']:fig.savefig(P/f'inline-bound-total-cost.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,baseline,guard,Path(__file__)]},output_sha256={f'inline-bound-total-cost.{ext}':hashlib.sha256((P/f'inline-bound-total-cost.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(P/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
