"""Exposure-heuristic result figure from tracked source and outcome exports."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3r'
def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3r-public'})
    fig,axes=plt.subplots(1,3,figsize=(15,7));fig.subplots_adjust(left=.055,right=.98,bottom=.35,top=.75,wspace=.32)
    fig.suptitle('Source-only exposure filtering recovers the target graph',x=.055,y=.96,ha='left',fontsize=19,fontweight='bold')
    fig.text(.055,.89,f"Frozen filtered-model noninferiority: {s['decision']}. Exact target energy identity with the supplied N15 oracle.",fontsize=12,fontweight='bold')
    fig.text(.055,.83,'5,180 unique N14 source patterns · 4,096 fresh N15 search inputs · fixed threshold ln(364 / 0.01)',fontsize=11)
    colors=['#777','#2878b5','#19876b','#8c5eb5'];arms=[None,'n14-full','exposure-filtered','true-graph-oracle'];labels=['BP AdamW\nnot run','All source\nzeros','Exposure\nfiltered','Supplied\noracle']
    for k,arm in enumerate(arms):
        if arm is None:
            for ax in axes[:2]:ax.text(k,45,'Not\nmeasured',ha='center',va='center',color=colors[k],fontsize=8)
            continue
        row=next(r for r in s['rows'] if r['arm']==arm)
        for ax,key in zip(axes[:2],['certificate_first_accuracy_pct','coverage_pct']):
            v=row[key];ax.bar(k,v,width=.62,color=colors[k],alpha=.87,hatch='//' if arm=='true-graph-oracle' else None);ax.annotate(f'{v:.2f}',(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8.5)
    for ax,title in zip(axes[:2],['A   Accuracy from 20 chains','B   Distinct-solution coverage']):
        ax.set_xticks(range(4),labels,fontsize=8);ax.set_ylim(0,114);ax.set_xlim(-.55,3.55);ax.set_ylabel('Percent');ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    d=s['source_support_diagnostic'];offsets=d['all_offsets'];examples=[next(x for x in offsets if (x['dr'],x['dc'])==o) for o in [(13,13),(12,13),(1,2)]]
    for k,(row,color) in enumerate(zip(examples,['#ca8a04','#2878b5','#19876b'])):
        v=row['expected_count'];axes[2].bar(k,v,width=.6,color=color);axes[2].annotate(f'{v:.3f}',(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8.5)
    axes[2].axhline(d['threshold'],color='#b44',ls='--',label=f"Fixed threshold {d['threshold']:.3f}")
    axes[2].set_xticks(range(3),['Diagonal\nendpoint','Source-global\nextra','True knight\noffset'],fontsize=8);axes[2].set_yscale('log');axes[2].set_ylim(.7,10000)
    axes[2].set_ylabel('Expected co-occurrences under independence');axes[2].set_title('C   Source exposure examples',loc='left',fontweight='bold',pad=12);axes[2].legend(loc='upper left',fontsize=8,frameon=False);axes[2].grid(axis='y',alpha=.18);axes[2].set_axisbelow(True)
    fig.text(.055,.27,'The filter removes four spurious offsets. Two removed diagonal entries remain enforced by the unchanged lines.',fontsize=10.8,fontweight='bold')
    fig.text(.055,.205,'Six of 60 finite zero offsets fall below threshold; all four true knight offsets remain. The final graph has no false or missed attacks.\nThe supplied-oracle bar is mapped from exact pre-inference identity; six distinct searches ran, and three duplicate jobs were avoided.',fontsize=10,linespacing=1.5)
    fig.text(.055,.135,'Exposure is N × sum of products of source cell marginals over valid placements. No radius cutoff or target-label tuning is used.\nThe independence model is a heuristic: its threshold is not a calibrated p-value, coverage claim or Bonferroni guarantee for constrained data.',fontsize=9.5,color='#444',linespacing=1.5)
    fig.text(.055,.07,'The rule was frozen before exposure scores were computed. The target universe was already enumerated and structurally censused;\nfresh means disjoint from earlier search cohorts. Original and certificate-first scores remain separate in the tracked tables.',fontsize=9.5,color='#444',linespacing=1.5)
    fig.text(.055,.025,'One complete unique-pattern source universe; three repeated search seeds do not establish reliability across source draws or data multiplicities.',fontsize=9,color='#444')
    for ext in ['png','svg']:fig.savefig(DEST/f'exposure-filter.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},output_sha256={f'exposure-filter.{ext}':hashlib.sha256((DEST/f'exposure-filter.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))
if __name__=='__main__':main()
