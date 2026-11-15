"""Deterministic figure separating realized cost from correctness invariants."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];P=ROOT/'results/research/loki/h3ca'


def main():
    src=P/'summary.json';s=json.loads(src.read_text());H,T=s['H_calls'],s['T_calls'];d=s['descriptive']
    saving=100*(H-T)/H
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'h3ca-tree-weights'})
    fig,axes=plt.subplots(1,2,figsize=(13,9));fig.subplots_adjust(left=.08,right=.97,bottom=.34,top=.73,wspace=.35)
    fig.suptitle(f"Tree-weight intervention: {saving:.2f}% fewer paid queries",x=.04,y=.96,ha='left',fontsize=20,fontweight='bold')
    fig.text(.04,.895,f"Prespecified 2% cost endpoint {s['decision']}. Exact identification agrees on every pair.",fontsize=13)
    fig.text(.04,.82,'All 1,024 fresh uniform-truth pairs retained. Both arms use hierarchical MAP guidance.\n'
             'Only T’s tree objective uses the matched posterior.',fontsize=11.5,linespacing=1.5)
    ax=axes[0];maps=[s['MAP_calls'][a] for a in ['H','T']];trees=[s['tree_calls'][a] for a in ['H','T']]
    ax.bar([0,1],maps,width=.5,color='#557f9c',label='MAP calls');ax.bar([0,1],trees,bottom=maps,width=.5,color='#bb805f',label='Tree calls')
    ax.set_xticks([0,1],['H: hierarchical\ntree weights','T: matched\ntree weights']);ax.set_ylabel('Actual membership calls')
    ax.set_ylim(0,H*1.23);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    ax.axhline(s['primary']['passing_maximum'],color='#744848',ls='--',lw=1.1)
    ax.legend(loc='upper left',frameon=False,ncol=2,bbox_to_anchor=(-.04,1.07))
    for i,n in enumerate([H,T]):ax.text(i,n+H*.025,f'{n:,}',ha='center',fontweight='bold')
    ax=axes[1];diff=d['actual_query_difference_T_minus_H'];xs=[int(k) for k in diff];ys=list(diff.values())
    ax.bar(xs,ys,color=['#538c7b' if k<0 else '#bb805f' if k>0 else '#84929c' for k in xs])
    ax.set_xticks(xs);ax.set_xlabel('Per-case calls: T minus H');ax.set_ylabel('Cases');ax.set_ylim(0,max(ys)*1.18)
    for x,y in zip(xs,ys):ax.text(x,y+max(ys)*.025,str(y),ha='center',fontsize=10)
    fig.text(.04,.255,f"Both exact: {s['H_exact']:,}/1,024      Neither exact: 16      Discordant exact flags: 0",fontsize=13,fontweight='bold')
    fig.text(.04,.19,f"Realized cases: T uses fewer calls in {d['T_fewer']}, more in {d['T_more']}, and ties in {d['tied']}.\n"
             'All 16 shared failures stop above 16 candidates after four MAP calls; none reaches a tree.',fontsize=11.5,linespacing=1.5)
    fig.text(.04,.12,f"Matched expected tree cost: {d['strictly_smaller_matched_expectation']} strict improvements and {d['equal_matched_expectation']} ties at {d['common_handoffs']} handoffs.\n"
             'This audited expectation invariant is distinct from the realized 2% cohort endpoint.',fontsize=11,linespacing=1.5)
    fig.text(.04,.055,f"BP AdamW: unmeasured. {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.\n"
             'Finite supplied-class result; full optimality proofs retained outside Git.',fontsize=10.5,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(P/f'tree-weight-cost.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [src,Path(__file__)]},
                 output_sha256={f'tree-weight-cost.{ext}':hashlib.sha256((P/f'tree-weight-cost.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (P/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
