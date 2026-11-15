"""Show the missed frozen target alongside the modest observed cost reduction."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4]
PUBLIC=ROOT/'results/research/loki/h3bp'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated'
    values=[s['old_continuation_calls'],s['posterior_weighted_continuation_calls'],s['new_continuation_calls']]
    reduction=100*(1-values[2]/values[0]);expected=100*s['descriptive_uniform_expected_relative_reduction']['display_float']
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bp-public'})
    fig,ax=plt.subplots(figsize=(12.8,8.5));fig.subplots_adjust(left=.11,right=.93,bottom=.34,top=.72)
    fig.suptitle('Uniform trees save calls, but miss the required 5% reduction',x=.04,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.04,.895,f'PRIMARY FALSIFIED: {values[2]:,} continuation calls; the frozen target was ≤2,588.',fontsize=13,fontweight='bold')
    fig.text(.04,.82,'Same 1,408 archived eight-positive cases and first four MAP queries; all cases retained.\n439 feasible candidate sets optimized; 1,354 exact cases preserved as a correctness invariant.',fontsize=11.7,linespacing=1.5)
    ax.bar(range(3),values,color=['#557f9c','#b87961','#6d9b86'],width=.56)
    ax.set_xticks(range(3),['First-lex tree\narchived H3bm','Posterior optimum\narchived H3bn','Uniform optimum\nnew H3bp execution'])
    ax.set_ylabel('Actual continuation membership calls');ax.set_ylim(0,3400)
    ax.axhline(s['actual_continuation_threshold'],ls='--',color='#704040',lw=1.3)
    ax.text(2.43,2480,'Passing target\n≤2,588',ha='right',va='top',fontsize=10.5,color='#704040',bbox=dict(facecolor='white',edgecolor='none',pad=2))
    for i,v in enumerate(values):ax.text(i,v+65,f'{v:,}',ha='center',fontsize=13,fontweight='bold')
    ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    fig.text(.04,.245,f'Observed reduction versus first-lex: {values[0]-values[2]} calls ({reduction:.2f}%); required: at least 5%.',fontsize=12.3,fontweight='bold')
    fig.text(.04,.17,f'Uniform expected cost fell {expected:.2f}%; the {values[1]-values[2]}-call benefit versus posterior weighting is descriptive.\n5,577 inherited MAP calls + 2,648 newly charged calls = 8,225 total policy calls.',fontsize=11.5,linespacing=1.5)
    fig.text(.04,.085,'Archived-source intervention; no independent confirmation or new positive examples.\n54 known infeasible cases and 287 prior successes need no new calls; no fallback or expanded query budget.',fontsize=10.8,color='#444',linespacing=1.5)
    fig.text(.04,.03,f"BP AdamW: not measured. Study {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.8,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'uniform-actual-cost.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,Path(__file__)]},output_sha256={f'uniform-actual-cost.{ext}':hashlib.sha256((PUBLIC/f'uniform-actual-cost.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
