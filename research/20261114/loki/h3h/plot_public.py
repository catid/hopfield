"""Render H3h from its tracked public export only."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.hashsalt'] = 'hopfield-h3h-v1'

ROOT=Path(__file__).resolve().parents[4]
DEST=ROOT/'results/research/loki/h3h'


def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(13,7))
    fig.subplots_adjust(left=.07,right=.97,bottom=.39,top=.76,wspace=.22)
    fig.suptitle('Displacement sharing prevents severe support overfitting',x=.07,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.07,.89,'12-Queens · same count-energy design · 8,192 bank-incompatible inputs · 20 chains × 3 seeds',fontsize=11)
    fig.text(.07,.83,'Frozen 32-pattern primary: +97.39 pp N20 from pooling; one-sided 95% lower bound +97.09 pp.',fontsize=11,fontweight='bold')
    cfg=[('context','#c44e52','BP AdamW 4,000 (512 only)'),('exclusion','#19876b','Displacement-pooled counts (cached)'),('pair-count','#2878b5','Individual-pair counts (new)')]
    for ax,metric,title in zip(axes,['certificate_first_accuracy_pct','coverage_pct'],['A   N20 accuracy','B   Distinct-completion coverage']):
        for arm,color,label in cfg:
            rows=[s['context_bp_adamw']] if arm=='context' else sorted([r for r in s['rows'] if r['arm']==arm],key=lambda r:r['bank'])
            ax.plot([r['bank'] for r in rows],[r[metric] for r in rows],marker='s' if arm=='context' else 'o',color=color,label=label,
                    linewidth=2,linestyle='none' if arm=='context' else '-')
            if arm!='context':
                for r in rows:
                    dy=8 if arm=='exclusion' else -16 if r[metric]>5 else 8
                    ax.annotate(f"{r[metric]:.2f}%",(r['bank'],r[metric]),xytext=(0,dy),textcoords='offset points',ha='center',color=color,fontsize=9)
        ax.set_xscale('log',base=2);ax.set_xticks([32,128,512],['32','128','512']);ax.set_xlim(24,680);ax.set_ylim(-6,115 if metric.startswith('certificate') else 90)
        ax.set_xlabel('Unique complete training patterns');ax.set_ylabel('Percent');ax.grid(axis='y',alpha=.18)
        ax.set_title(title,loc='left',fontweight='bold',pad=12)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.065,.26),ncol=3,frameon=False,fontsize=9)
    fig.text(.07,.215,'At 32 patterns: 204 false initial certificates → 12,240 wrong stopped chains. All remain failures.',fontsize=10.5,fontweight='bold')
    fig.text(.07,.165,'Post hoc valid-certificate census: individual pairs admit answers for 0 / 2 / 7,226 inputs at 32 / 128 / 512 patterns.\nThese counts match successful inputs in every seed. Removing only false exclusions rejects every wrong certificate.',fontsize=10,linespacing=1.5)
    fig.text(.07,.095,'Adaptive reuse of H3g, not fresh confirmation. The cohort excludes every completion in the largest training bank;\nit contains 0 one-clue, 16 two-clue and 8,176 three-clue inputs. BP AdamW is a 512-pattern reference only.',fontsize=9,color='#444',linespacing=1.5)
    fig.text(.07,.03,'Allowed-pair reward and supplied geometry are explicit priors. One nested bank draw; seeds repeat search.\nA cardinality stop gate can remove the initial false certificates but cannot create valid certificates absent from this energy.',fontsize=9,color='#444',linespacing=1.4)
    for ext in ['png','svg']:fig.savefig(DEST/f'offset-sharing-ablation.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [source,Path(__file__)]},
        output_sha256={f'offset-sharing-ablation.{ext}':hashlib.sha256((DEST/f'offset-sharing-ablation.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['output_sha256']))


if __name__=='__main__':main()
