"""Reproducible H3i figure from tracked public summary only."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3i'


def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3i-public'})
    colors=['#777777','#2878b5','#19876b','#ca8a04','#8c5eb5']
    arms=[None,'raw128','augmented128','raw1024','offset128']
    names=['BP AdamW\nnot run','Raw 128\n128 patterns','8 views of 128\n996 patterns','Raw 1,024\n1,024 patterns','Offset pooling\n128 patterns']
    fig,axes=plt.subplots(1,2,figsize=(13,6.7));fig.subplots_adjust(left=.065,right=.98,bottom=.34,top=.75,wspace=.22)
    fig.suptitle('Dihedral augmentation repairs much of the missing support',x=.065,y=.96,ha='left',fontsize=18,fontweight='bold')
    p=s['primary'];fig.text(.065,.89,f"Fresh 12-Queens cohort: augmented-vs-raw128 N20 gain {p['accuracy_difference_pp']:.2f} pp; frozen primary {s['decision']}.",fontsize=11,fontweight='bold')
    fig.text(.065,.83,'Same strict local margin and q-cardinality gate · 4,096 inputs · 20 chains × 3 search seeds',fontsize=11)
    for ax,metric,title in zip(axes,['certificate_first_accuracy_pct','coverage_pct'],['A   N20 accuracy','B   Distinct-completion coverage']):
        for k,arm in enumerate(arms):
            if arm is None:
                ax.text(k,45,'Not measured\non this cohort',ha='center',va='center',color=colors[k],fontsize=8.5)
                continue
            r=next(x for x in s['rows'] if x['arm']==arm);value=r[metric]
            ax.bar(k,value,width=.62,color=colors[k],alpha=.88)
            ax.annotate(f'{value:.2f}%',(k,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
            if value==0:ax.scatter([k],[0],color=colors[k],s=15,clip_on=False)
        ax.set_xticks(range(5),names,fontsize=8.5);ax.set_ylim(0,112 if metric.startswith('certificate') else 92)
        ax.set_xlim(-.55,4.55);ax.set_ylabel('Percent');ax.set_title(title,loc='left',fontweight='bold',pad=12)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.text(.065,.24,'Eight rotations/reflections of 128 raw patterns give 996 unique views; 1,024 independent raw patterns are a descriptive control.',fontsize=10)
    fig.text(.065,.18,f"No wrong stopped chains in any arm. Exact 95% upper bound for any wrong stop per input: {100*p['any_wrong_stop_probability_exact95_upper']:.4f}%\n(primary limit 0.1%; all chains and seeds clustered by input). BP AdamW was not rerun on this new cohort.",fontsize=10,linespacing=1.5)
    fig.text(.065,.105,'Every valid completion is outside the union of all 1,839 training patterns. Inputs exclude prior H3e/H3g evaluations;\nthis conditioned cohort has 2 two-clue and 4,094 three-clue inputs, with no one-clue cases.',fontsize=9,color='#444',linespacing=1.5)
    fig.text(.065,.035,'Fixed augmentation supplies geometry; count energy retains an explicit allowed-pair reward. One fresh bank draw; seeds repeat search.\nThe raw 1,024 and offset controls are descriptive; this experiment does not claim equal independent data across all arms.',fontsize=9,color='#444',linespacing=1.4)
    for ext in ['png','svg']:fig.savefig(DEST/f'dihedral-repair.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},
        output_sha256={f'dihedral-repair.{ext}':hashlib.sha256((DEST/f'dihedral-repair.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))


if __name__=='__main__':main()
