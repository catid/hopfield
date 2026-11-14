"""Plot the complete-pattern-holdout study from tracked public JSON only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.hashsalt'] = 'hopfield-h3g-v1'

ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3g'


def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    colors={'adamw-decay':'#c44e52','adam-nodecay':'#2878b5','sgd-pre':'#ca8a04','exclusion':'#19876b'}
    names={'adamw-decay':'BP AdamW 4000 (512 only)','adam-nodecay':'BP Adam 250','sgd-pre':'SGD + precond. 4000','exclusion':'Count + fixed energy'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(13,5.9),gridspec_kw={'width_ratios':[1.35,1]})
    fig.subplots_adjust(left=.065,right=.98,bottom=.28,top=.74,wspace=.22)
    fig.suptitle('12-Queens with complete solution patterns held out',x=.065,y=.96,ha='left',fontsize=19,fontweight='bold')
    fig.text(.065,.885,'Every valid answer to every test input is absent from the largest 512-pattern training bank.',fontsize=11,fontweight='bold')
    fig.text(.065,.825,'8,192 shared inputs · 20 chains × 3 seeds · certificate-first selection · no tuning on the new split',fontsize=10.5)
    arms=['adamw-decay','adam-nodecay','sgd-pre','exclusion'];ax=axes[0]
    for i,arm in enumerate(arms):
        row=next(r for r in s['rows'] if r['bank']==512 and r['arm']==arm)
        ax.bar(i,row['coverage_pct'],color=colors[arm],width=.65,alpha=.83)
        ax.annotate(f"{row['coverage_pct']:.2f}% cov.\n{row['certificate_first_accuracy_pct']:.3f}% N20",(i,row['coverage_pct']),
                    xytext=(0,7),textcoords='offset points',ha='center',fontsize=8.5,linespacing=1.4)
    ax.bar(4,0,color='#777777',width=.65);ax.scatter([4],[0],color='#777777',s=18,clip_on=False)
    ax.annotate('0% coverage\n0% N20',(4,0),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8.5)
    ax.set_xticks(range(5),['BP AdamW\n4,000','BP Adam\n250','SGD + precond.\n4,000','Count + fixed\nenergy','Training-bank\nretrieval'])
    ax.set_ylim(0,93);ax.set_ylabel('Distinct-completion coverage (%)');ax.set_title('A   Primary bank: 512 training patterns',loc='left',fontweight='bold',pad=12)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax=axes[1]
    for arm in arms:
        rows=sorted([r for r in s['rows'] if r['arm']==arm],key=lambda r:r['bank'])
        ax.plot([r['bank'] for r in rows],[r['certificate_first_accuracy_pct'] for r in rows],marker='o',
                color=colors[arm],label=names[arm],linestyle='-' if len(rows)>1 else 'none')
    ax.set_xscale('log',base=2);ax.set_xticks([32,128,512],['32','128','512']);ax.set_ylim(0,105)
    ax.set_xlabel('Unique complete training patterns');ax.set_ylabel('Certificate-first N20 accuracy (%)')
    ax.set_title('B   Nested smaller-bank controls',loc='left',fontweight='bold',pad=12);ax.grid(axis='y',alpha=.18)
    ax.legend(loc='lower right',frameon=False,fontsize=8.5)
    p=s['primary']
    fig.text(.065,.16,f"Frozen 512-pattern count-vs-Adam250 hypothesis: {s['decision']}. Accuracy difference {p['accuracy_difference_pp']:+.3f} pp; coverage {p['coverage_difference_pp']:+.3f} pp.",fontsize=10.5)
    fig.text(.065,.10,'Test sampling is conditioned on incompatibility with the largest bank: 0 one-clue, 16 two-clue and 8,176 three-clue inputs.\nIt excludes the earlier H3e test inputs. Each training pattern contributes all 298 one/two/three-clue masks; lower banks are secondary.',fontsize=9,color='#444',linespacing=1.5)
    fig.text(.065,.035,'Count support uses training patterns only; its fixed energy design and supplied offset geometry remain explicit priors.\nOne nested bank draw and one deterministic model per setting; seeds repeat search. Earlier studies examined the full mathematical solution universe.',fontsize=9,color='#444',linespacing=1.4)
    for extension in ['png','svg']:fig.savefig(DEST/f'pattern-holdout.{extension}',dpi=180,facecolor='white',metadata={'Date':None} if extension=='svg' else None)
    receipt=dict(input_sha256={str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest(),
                              str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        output_sha256={f'pattern-holdout.{extension}':hashlib.sha256((DEST/f'pattern-holdout.{extension}').read_bytes()).hexdigest() for extension in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))


if __name__=='__main__':main()
