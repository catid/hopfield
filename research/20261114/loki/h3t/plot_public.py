"""Queen-plus-camel family-transfer figure from tracked outcomes and source exposures."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3t'
def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3t-public'})
    fig,axes=plt.subplots(1,3,figsize=(15,7));fig.subplots_adjust(left=.055,right=.98,bottom=.39,top=.75,wspace=.32)
    fig.suptitle('Large source exposure does not make an absence transferable',x=.055,y=.96,ha='left',fontsize=18,fontweight='bold')
    p=s['primary'];fig.text(.055,.89,f"Frozen queen-plus-camel noninferiority: {s['decision']}. Filtered-minus-oracle accuracy {p['accuracy_difference_pp']:+.2f} pp; margin −0.1 pp.",fontsize=11.5,fontweight='bold')
    fig.text(.055,.83,'176 complete N12 source patterns · 4,096 fresh N13 inputs · unchanged exposure rule and homogeneous lines',fontsize=11)
    colors=['#777','#2878b5','#19876b','#8c5eb5'];arms=[None,'n12-full','exposure-filtered','true-graph-oracle'];labels=['BP AdamW\nnot run','All source\nzeros','Exposure\nfiltered','Supplied\noracle']
    for k,arm in enumerate(arms):
        if arm is None:
            for ax in axes[:2]:ax.text(k,45,'Not\nmeasured',ha='center',va='center',color=colors[k],fontsize=8)
            continue
        row=next(r for r in s['rows'] if r['arm']==arm)
        for offset,key,alpha,hatch in [(-.16,'original_accuracy_pct',.35,'//'),(.16,'certificate_first_accuracy_pct',.9,None)]:
            v=row[key];axes[0].bar(k+offset,v,width=.29,color=colors[k],alpha=alpha,hatch=hatch);axes[0].annotate(f'{v:.1f}',(k+offset,v),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7)
        v=row['coverage_pct'];axes[1].bar(k,v,width=.62,color=colors[k],alpha=.87);axes[1].annotate(f'{v:.2f}',(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8.5)
    for ax,title in zip(axes[:2],['A   Accuracy from 20 chains','B   Distinct-solution coverage']):
        ax.set_xticks(range(4),labels,fontsize=8);ax.set_ylim(0,114);ax.set_xlim(-.55,3.55);ax.set_ylabel('Percent');ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    d=s['source_support_diagnostic'];offsets=d['all_offsets'];examples=[next(x for x in offsets if (x['dr'],x['dc'])==o) for o in [(10,11),(8,9),(2,11),(2,8)]]
    for k,row in enumerate(examples):
        v=row['expected_count'];axes[2].bar(k,v,width=.6,color='#ca8a04' if row['finite_zero_retained'] else '#777');axes[2].annotate(f'{v:.3f}',(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    axes[2].axhline(d['threshold'],color='#b44',ls='--',label=f"Threshold {d['threshold']:.3f}")
    axes[2].set_xticks(range(4),['(10,11)\nremoved','(8,9)\nremoved','(2,11)\nretained','(2,8)\nretained'],fontsize=8);axes[2].set_yscale('log');axes[2].set_ylim(.1,200)
    axes[2].set_ylabel('Expected count under independent occupancy');axes[2].set_title('C   Extra source-zero offsets',loc='left',fontweight='bold',pad=12);axes[2].legend(loc='upper left',fontsize=8,frameon=False);axes[2].grid(axis='y',alpha=.18);axes[2].set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor='#777',alpha=.35,hatch='//',label='Original lowest-energy selection'),Patch(facecolor='#777',alpha=.9,label='Certificate-first selection')],loc='lower left',bbox_to_anchor=(.05,.29),ncol=2,frameon=False,fontsize=9)
    fig.text(.055,.25,'Eight source-global zero offsets pass the fixed exposure threshold and retain 308 false target exclusions.',fontsize=10.8,fontweight='bold')
    fig.text(.055,.19,'The filter removes 104 false pairs but misses no true queen or camel attack. All 16 extra source zeros have valid target witnesses.\nThis family uses camel moves (1,3)/(3,1), not knight moves; source and target universes were independently enumerated.',fontsize=10,linespacing=1.5)
    fig.text(.055,.12,'The rule and threshold were frozen before exposure inspection. Independent-cell exposure is a heuristic, without calibrated p-values\nor multiple-testing guarantees. No radius cutoff, threshold adjustment or target-based pruning follows this result.',fontsize=9.5,color='#444',linespacing=1.5)
    fig.text(.055,.055,'New family and smaller board sizes than H3r: this is an unchanged-rule transfer test, not an isolated family-only causal comparison.\nAll nine model–seed searches are distinct. Certificate correctness, availability and finite search success are reported separately.',fontsize=9.3,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(DEST/f'camel-family.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},output_sha256={f'camel-family.{ext}':hashlib.sha256((DEST/f'camel-family.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))
if __name__=='__main__':main()
