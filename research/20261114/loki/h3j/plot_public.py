"""Deterministic public figure for zero-shot N13 line extrapolation."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3j'


def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3j-public'})
    arms=[None,'direct-offset','line-extrapolation','true-graph-oracle'];colors=['#777777','#2878b5','#19876b','#8c5eb5']
    labels=['BP AdamW\nnot run','Direct table\nN12 → N13','Fitted lines\nN12 → N13','Oracle reference\nidentity mapping']
    fig,axes=plt.subplots(1,2,figsize=(13,7));fig.subplots_adjust(left=.065,right=.98,bottom=.38,top=.75,wspace=.23)
    fig.suptitle('A fitted geometric rule transfers from 12-Queens to 13-Queens',x=.065,y=.96,ha='left',fontsize=17.5,fontweight='bold')
    p=s['primary'];fig.text(.065,.89,f"Frozen primary supported: +{p['accuracy_difference_pp']:.2f} pp certificate-first N20; 95% lower bound +{p['accuracy_one_sided_95_lower_pp']:.2f} pp.",fontsize=11,fontweight='bold')
    fig.text(.065,.83,'128 N12 training patterns · no N13 training · 4,096 fresh inputs · 20 chains × 3 search seeds',fontsize=11)
    for k,arm in enumerate(arms):
        if arm is None:
            for ax in axes:ax.text(k,45,'Not measured\nat N13',ha='center',va='center',color=colors[k],fontsize=9)
            continue
        row=next(r for r in s['rows'] if r['arm']==arm)
        for offset,key,alpha,hatch in [(-.17,'original_accuracy_pct',.35,'//'),(.17,'certificate_first_accuracy_pct',.9,None)]:
            value=row[key];axes[0].bar(k+offset,value,width=.31,color=colors[k],alpha=alpha,hatch=hatch)
            axes[0].annotate(f'{value:.2f}',(k+offset,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        value=row['coverage_pct'];axes[1].bar(k,value,width=.62,color=colors[k],alpha=.85,hatch='..' if arm=='true-graph-oracle' else None)
        axes[1].annotate(f'{value:.2f}%',(k,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    for ax,title in zip(axes,['A   N20 accuracy and output selection','B   Distinct-completion coverage']):
        ax.set_xticks(range(4),labels,fontsize=9);ax.set_xlim(-.55,3.55);ax.set_ylim(0,116 if ax is axes[0] else 100)
        ax.set_ylabel('Percent');ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor='#777',alpha=.35,hatch='//',label='Original lowest-energy selection'),Patch(facecolor='#777',alpha=.9,label='Certificate-first selection')],
        loc='lower left',bbox_to_anchor=(.06,.275),ncol=2,frameon=False,fontsize=9)
    fig.text(.065,.235,'Direct transfer produced 26,301 wrong stopped chains. Every true conflict uses a target attack pair absent from its table.',fontsize=10.3,fontweight='bold')
    fig.text(.065,.17,'Fitted lines and the supplied oracle have exactly identical solver parameters. Oracle outputs are derived by an explicit\npre-inference identity mapping, not independent replication; six unique searches were run rather than nine.',fontsize=10,linespacing=1.5)
    fig.text(.065,.10,'The primitive linear-rule family, coordinate geometry and q=13 are supplied priors. All arms share the inherited N12\ntemperature unit and strict local margin. Target solutions enter input reservation and evaluation only.',fontsize=9,color='#444',linespacing=1.5)
    fig.text(.065,.035,'Uniform unique valid 1/2/3-clue sampling: 2 / 140 / 3,954 inputs. One N12 bank draw; three repeated search seeds.\nThe fitted energy has zero target graph errors; direct transfer has 56 false exclusions and 28 missed attack pairs.',fontsize=9,color='#444',linespacing=1.4)
    for ext in ['png','svg']:fig.savefig(DEST/f'zero-shot-n13.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},
        output_sha256={f'zero-shot-n13.{ext}':hashlib.sha256((DEST/f'zero-shot-n13.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))


if __name__=='__main__':main()
