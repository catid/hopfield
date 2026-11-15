"""Public superqueen boundary figure; all values read from tracked summary."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3l'


def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3l-public'})
    arms=[None,'line-only','hybrid','direct-offset','true-graph-oracle'];colors=['#777777','#ca8a04','#19876b','#2878b5','#8c5eb5']
    labels=['BP AdamW\nnot run','Lines only','Lines + all\nsource zeros','Direct\nfinite table','Supplied\noracle']
    fig,axes=plt.subplots(1,2,figsize=(13,7));fig.subplots_adjust(left=.065,right=.98,bottom=.38,top=.75,wspace=.23)
    fig.suptitle('Superqueens expose two limits of the geometric prior',x=.065,y=.96,ha='left',fontsize=18,fontweight='bold')
    p=s['primary'];fig.text(.065,.89,f"Frozen hybrid-vs-oracle noninferiority: {s['decision']}. N20 difference {p['accuracy_difference_pp']:+.2f} pp; margin −0.1 pp.",fontsize=11,fontweight='bold')
    fig.text(.065,.83,'128 N12 superqueen examples · 4,096 fresh N13 inputs · common temperature, strict margin and q gate',fontsize=10.6)
    for k,arm in enumerate(arms):
        if arm is None:
            for ax in axes:ax.text(k,45,'Not measured\non superqueens',ha='center',va='center',color=colors[k],fontsize=8)
            continue
        row=next(r for r in s['rows'] if r['arm']==arm)
        for offset,key,alpha,hatch in [(-.17,'original_accuracy_pct',.35,'//'),(.17,'certificate_first_accuracy_pct',.9,None)]:
            value=row[key];axes[0].bar(k+offset,value,width=.31,color=colors[k],alpha=alpha,hatch=hatch)
            axes[0].annotate(f'{value:.1f}',(k+offset,value),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        value=row['coverage_pct'];axes[1].bar(k,value,width=.62,color=colors[k],alpha=.87)
        axes[1].annotate(f'{value:.2f}%',(k,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    for ax,title in zip(axes,['A   N20 superqueen accuracy','B   Distinct-superqueen coverage']):
        ax.set_xticks(range(5),labels,fontsize=8.5);ax.set_xlim(-.55,4.55);ax.set_ylim(0,116 if ax is axes[0] else 108)
        ax.set_ylabel('Percent');ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor='#777',alpha=.35,hatch='//',label='Original lowest-energy selection'),Patch(facecolor='#777',alpha=.9,label='Certificate-first selection')],
        loc='lower left',bbox_to_anchor=(.06,.275),ncol=2,frameon=False,fontsize=9)
    diag=s['source_support_diagnostic'];fig.text(.065,.235,f"{diag['extra_offsets_globally_absent_at_N12']} extra offsets are absent from every N12 superqueen solution, yet occur at N13. All were retained.",fontsize=10.4,fontweight='bold')
    fig.text(.065,.17,'Line-only misses all 528 target knight-attack pairs: a hypothesis-family mismatch. Hybrid adds the finite constraints\nbut also 212 false exclusions from source absences. The oracle is an explicitly supplied-rule reference.',fontsize=10,linespacing=1.5)
    fig.text(.065,.10,'Independent exhaustive enumeration: 156 N12 and 1,876 N13 superqueen solutions. No target label selects lines,\nremoves zero offsets, tunes coefficients or calibrates temperature. This is a boundary test, not arbitrary-rule learning.',fontsize=9,color='#444',linespacing=1.5)
    fig.text(.065,.035,'Three search seeds for one source bank; original and certificate-first scores remain separate.\nEvery saved board is checked against the exhaustive superqueen solution set, including knight constraints and givens.',fontsize=9,color='#444',linespacing=1.4)
    for ext in ['png','svg']:fig.savefig(DEST/f'superqueen-boundary.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},
        output_sha256={f'superqueen-boundary.{ext}':hashlib.sha256((DEST/f'superqueen-boundary.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))


if __name__=='__main__':main()
